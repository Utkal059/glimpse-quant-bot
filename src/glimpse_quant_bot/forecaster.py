"""Asset-aware ensemble probability forecaster.

Three independent views of the terminal-price distribution are pooled linearly
(a weighted mixture of CDFs, i.e. Bayesian model averaging with fixed priors):

``options_rnd``
    Breeden-Litzenberger risk-neutral density from the Deribit chain
    (:mod:`glimpse_quant_bot.options_density`). BTC and ETH only.

``har_microstructure``
    Variance from a log-HAR model (Corsi 2009) on Garman-Klass bar variances,
    iterated forward to the market's expiry; shape from a Fernandez-Steel
    skewed Student-t whose tail index comes from the sample kurtosis (relaxed
    toward Gaussian with horizon), whose skew comes from perpetual funding
    (crowded longs -> left skew) and whose location carries a small,
    fast-decaying drift from L2 order-book imbalance.

``gjr_garch`` / ``timesfm``
    GJR-GARCH(1,1) fitted by Gaussian QML and propagated by filtered
    historical simulation (bootstrapped standardised residuals); replaced by
    Google TimesFM 2.5 quantiles when ``glimpse-markets[forecasting]`` is
    installed.

Weights are 50/30/20 for assets with listed options and 60/40 otherwise, and
are re-normalised whenever a component is unavailable, so bucket probabilities
always sum to one. The pooled quantile function is mapped onto Glimpse buckets
with the SDK's own ``bucket_probabilities`` / ``probability_between``.

The cross-model Jensen-Shannon divergence is reported as ``disagreement`` and
shrinks the Kelly fraction: when the models disagree about where the mass is,
the bot bets smaller.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import structlog
from glimpse_markets.forecasting.signal import bucket_probabilities as sdk_bucket_probabilities
from glimpse_markets.models import QuoteOutcome
from glimpse_markets.money import PriceUnits
from scipy.optimize import minimize
from scipy.special import gammaln, stdtr

from glimpse_quant_bot.calibration import CalibrationView, apply_temperature
from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.external_feeds import Candle, MarketDataHub
from glimpse_quant_bot.flow_toxicity import (
    FlowSignal,
    OfiTracker,
    book_levels,
    ofi_drift,
    toxic_tail_index,
    toxic_variance_multiplier,
    toxicity_from_candles,
)
from glimpse_quant_bot.lmsr import LmsrBook
from glimpse_quant_bot.options_density import (
    YEAR_SECONDS,
    VolSurface,
    rectify_cdf,
    risk_neutral_distribution,
)
from glimpse_quant_bot.pricing_kernel import physical_distribution
from glimpse_quant_bot.types import AssetType, BucketSignal, TopicMeta

log = structlog.get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
Cdf = Callable[[FloatArray], FloatArray]

QUANTILE_LEVELS: tuple[float, ...] = tuple((i + 1) / 400.0 for i in range(399))
"""Fine quantile grid handed to the SDK's ``bucket_probabilities``."""

GRID_POINTS = 1601
GRID_SIGMAS = 8.0
MAX_MODEL_STEPS = 720

FUNDING_REFERENCE = 3.0e-4
"""A 0.03% funding payment per interval is treated as strongly one-sided."""
FUNDING_SKEW = 0.08
IMBALANCE_DRIFT = 0.15
IMBALANCE_HORIZON_SECONDS = 300.0
"""Order-book imbalance is only informative over minutes; its drift decays beyond this."""


class ForecastError(RuntimeError):
    """A forecast could not be produced for this topic."""


# ---------------------------------------------------------------------------
# Realised variance and HAR
# ---------------------------------------------------------------------------


def garman_klass_variance(candles: Sequence[Candle]) -> FloatArray:
    """Garman-Klass (1980) range-based variance estimate for each bar."""
    opens = np.array([bar.open for bar in candles])
    highs = np.array([bar.high for bar in candles])
    lows = np.array([bar.low for bar in candles])
    closes = np.array([bar.close for bar in candles])
    variance: FloatArray = np.maximum(
        0.5 * np.log(highs / lows) ** 2 - (2.0 * math.log(2.0) - 1.0) * np.log(closes / opens) ** 2,
        0.0,
    )
    return variance


def close_returns(candles: Sequence[Candle]) -> FloatArray:
    closes = np.array([bar.close for bar in candles])
    returns: FloatArray = np.diff(np.log(closes))
    return returns


def excess_kurtosis(returns: FloatArray) -> float:
    if returns.size < 20:
        return 3.0
    centred = returns - float(returns.mean())
    second = float((centred**2).mean())
    if second <= 0.0:
        return 3.0
    return float((centred**4).mean()) / second**2 - 3.0


@dataclass(frozen=True, slots=True)
class HarFit:
    """Log-HAR regression ``x_{t+1} = c + b1 x_t + b2 mean_day(x) + b3 mean_week(x)``
    on ``x = ln(bar variance)``."""

    intercept: float
    beta_bar: float
    beta_day: float
    beta_week: float
    smear: float
    """Ratio calibration ``sum(variance) / sum(exp(x_hat))``: makes the level of the
    back-transformed forecast unbiased in-sample. (Duan's ``E[exp(residual)]``
    overshoots here because log bar-variance residuals are strongly left-skewed.)"""
    history: FloatArray
    day: int
    week: int
    floor: float
    cap: float
    fallback: bool


def fit_har(bar_variance: FloatArray, day: int, week: int) -> HarFit:
    positive = bar_variance[bar_variance > 0.0]
    median = float(np.median(positive)) if positive.size else 1.0e-8
    floor = max(median * 0.05, 1.0e-14)
    cap = median * 50.0
    x = np.log(np.maximum(bar_variance, floor))
    n = int(x.size)

    if n < week + 40:
        decay = 0.94 ** np.arange(n - 1, -1, -1, dtype=np.float64)
        level = float((np.maximum(bar_variance, floor) * decay).sum() / decay.sum())
        return HarFit(
            intercept=math.log(level),
            beta_bar=0.0,
            beta_day=0.0,
            beta_week=0.0,
            smear=1.0,
            history=np.full(week, math.log(level)),
            day=day,
            week=week,
            floor=floor,
            cap=cap,
            fallback=True,
        )

    idx = np.arange(week - 1, n - 1)
    running = np.concatenate([[0.0], np.cumsum(x)])
    mean_day = (running[idx + 1] - running[idx + 1 - day]) / day
    mean_week = (running[idx + 1] - running[idx + 1 - week]) / week
    design = np.column_stack([np.ones(idx.size), x[idx], mean_day, mean_week])
    target = x[idx + 1]
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    intercept, b1, b2, b3 = (float(value) for value in beta)
    persistence = b1 + b2 + b3
    if not math.isfinite(persistence) or persistence >= 0.995 or persistence <= -0.5:
        shrink = 0.995 / persistence if persistence >= 0.995 else 0.0
        b1, b2, b3 = b1 * shrink, b2 * shrink, b3 * shrink
        intercept = float(target.mean()) * (1.0 - (b1 + b2 + b3))
    fitted = intercept + b1 * x[idx] + b2 * mean_day + b3 * mean_week
    smear = float(np.clip(np.exp(target).sum() / np.exp(fitted).sum(), 0.5, 5.0))
    return HarFit(
        intercept=intercept,
        beta_bar=b1,
        beta_day=b2,
        beta_week=b3,
        smear=smear,
        history=x[-week:].copy(),
        day=day,
        week=week,
        floor=floor,
        cap=cap,
        fallback=False,
    )


def har_variance_path(fit: HarFit, steps: int) -> FloatArray:
    """Expected variance of each of the next ``steps`` bars (iterated forecast)."""
    week = fit.week
    values = np.empty(week + steps)
    values[:week] = fit.history
    for step in range(steps):
        i = week + step
        values[i] = (
            fit.intercept
            + fit.beta_bar * values[i - 1]
            + fit.beta_day * float(values[i - fit.day : i].mean())
            + fit.beta_week * float(values[i - week : i].mean())
        )
    path: FloatArray = np.clip(np.exp(values[week:]) * fit.smear, fit.floor, fit.cap)
    return path


# ---------------------------------------------------------------------------
# GJR-GARCH(1,1) with filtered historical simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GjrGarchFit:
    """``h_t = omega + (alpha + gamma * 1[e_{t-1} < 0]) e_{t-1}^2 + beta h_{t-1}``
    on percent returns."""

    omega: float
    alpha: float
    gamma: float
    beta: float
    next_variance: float
    residuals: FloatArray
    """Standardised residuals (zero mean, unit variance) used for the bootstrap."""
    scale: float

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta


def _gjr_nll(params: Sequence[float], eps: list[float], initial_variance: float) -> float:
    omega, alpha, gamma, beta = (float(value) for value in params)
    persistence = alpha + 0.5 * gamma + beta
    if persistence >= 0.9995 or omega <= 0.0:
        return 1.0e10 + 1.0e6 * max(persistence, 0.0)
    h = initial_variance
    total = 0.0
    for e in eps:
        total += math.log(h) + e * e / h
        h = omega + (alpha + (gamma if e < 0.0 else 0.0)) * e * e + beta * h
    return 0.5 * total


def fit_gjr_garch(returns: FloatArray, scale: float = 100.0) -> GjrGarchFit:
    eps_array = (returns - float(returns.mean())) * scale if returns.size else returns
    variance = float(eps_array.var()) if eps_array.size else 0.0
    if eps_array.size < 100 or variance <= 0.0:
        level = variance if variance > 0.0 else 1.0e-4
        z = eps_array / math.sqrt(level) if eps_array.size else np.array([-1.0, 1.0])
        return GjrGarchFit(level, 0.0, 0.0, 0.0, level, _standardise(z), scale)

    eps = [float(value) for value in eps_array]
    start = np.array([0.05 * variance, 0.05, 0.05, 0.88])
    bounds = [(1.0e-8 * variance, 10.0 * variance), (0.0, 0.5), (0.0, 0.6), (0.0, 0.9995)]
    result = minimize(_gjr_nll, start, args=(eps, variance), method="L-BFGS-B", bounds=bounds)
    best = result.x if math.isfinite(float(result.fun)) and float(result.fun) < 1.0e9 else start
    omega, alpha, gamma, beta = (float(value) for value in best)

    h = variance
    conditional = np.empty(len(eps))
    for i, e in enumerate(eps):
        conditional[i] = h
        h = omega + (alpha + (gamma if e < 0.0 else 0.0)) * e * e + beta * h
    return GjrGarchFit(
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        next_variance=h,
        residuals=_standardise(eps_array / np.sqrt(conditional)),
        scale=scale,
    )


def _standardise(values: FloatArray) -> FloatArray:
    spread = float(values.std())
    if spread <= 0.0:
        return np.array([-1.0, 1.0])
    standardised: FloatArray = (values - float(values.mean())) / spread
    return standardised


def simulate_gjr_paths(fit: GjrGarchFit, steps: int, n_paths: int, seed: int) -> FloatArray:
    """Cumulative log returns, shape ``(steps + 1, n_paths)``; row 0 is zero."""
    rng = np.random.default_rng(seed)
    h = np.full(n_paths, fit.next_variance)
    cumulative = np.zeros(n_paths)
    out = np.zeros((steps + 1, n_paths))
    pool = fit.residuals
    for step in range(1, steps + 1):
        shocks = np.sqrt(h) * pool[rng.integers(0, pool.size, n_paths)]
        cumulative = cumulative + shocks
        out[step] = cumulative
        h = fit.omega + (fit.alpha + fit.gamma * (shocks < 0.0)) * shocks**2 + fit.beta * h
    scaled: FloatArray = out / fit.scale
    return scaled


def terminal_log_returns(paths: FloatArray, horizon_bars: float) -> FloatArray:
    """Simulated log returns at a fractional horizon, re-centred to a martingale."""
    steps = paths.shape[0] - 1
    if horizon_bars >= steps:
        raw = paths[steps] * math.sqrt(horizon_bars / steps)
    else:
        whole = math.floor(horizon_bars)
        fraction = horizon_bars - whole
        raw = paths[whole] + math.sqrt(fraction) * (paths[whole + 1] - paths[whole])
    centred: FloatArray = raw - math.log(float(np.exp(raw).mean()))
    return centred


# ---------------------------------------------------------------------------
# Skewed Student-t
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SkewStudentT:
    """Fernandez-Steel (1998) two-piece skewed Student-t, standardised to zero
    mean and unit variance. ``gamma > 1`` skews right, ``gamma < 1`` left.

    Chosen over a Cornish-Fisher expansion because its CDF is monotone for every
    parameter value, whereas Cornish-Fisher quantiles can cross for the skew and
    kurtosis levels seen in crypto returns.
    """

    nu: float
    gamma: float

    def _moments(self) -> tuple[float, float]:
        nu, g = self.nu, self.gamma
        abs_moment = (
            2.0
            * math.sqrt(nu)
            * math.exp(float(gammaln((nu + 1.0) / 2.0)) - float(gammaln(nu / 2.0)))
            / (math.sqrt(math.pi) * (nu - 1.0))
        )
        mean = abs_moment * (g - 1.0 / g)
        variance = (nu / (nu - 2.0)) * (g**3 + g**-3) / (g + 1.0 / g) - mean**2
        return mean, math.sqrt(variance)

    def cdf(self, standardised: FloatArray) -> FloatArray:
        mean, spread = self._moments()
        y = mean + spread * standardised
        g2 = self.gamma**2
        left = (2.0 / (g2 + 1.0)) * stdtr(self.nu, self.gamma * y)
        right = 1.0 / (g2 + 1.0) + (2.0 * g2 / (g2 + 1.0)) * (stdtr(self.nu, y / self.gamma) - 0.5)
        result: FloatArray = np.where(y < 0.0, left, right)
        return result


def microstructure_cdf(
    prices: FloatArray,
    spot: float,
    variance: float,
    horizon_seconds: float,
    horizon_bars: float,
    bar_excess_kurtosis: float,
    imbalance: float | None,
    funding_rate: float | None,
    *,
    flow_drift: float | None = None,
    toxicity_excess: float = 0.0,
    tail_kappa: float = 0.0,
) -> FloatArray:
    """CDF of the terminal price under the HAR + microstructure skewed-t model.

    ``flow_drift`` (integrated-OFI drift, log units) replaces the static depth
    imbalance drift when available; ``toxicity_excess`` fattens the tails via
    ``nu_tox = 4.2 + (nu - 4.2) exp(-tail_kappa * s)``.
    """
    sigma = math.sqrt(max(variance, 1.0e-16))
    horizon_kurtosis = float(
        np.clip(bar_excess_kurtosis / math.sqrt(max(horizon_bars, 0.05)), 0.03, 30.0)
    )
    nu = float(np.clip(4.0 + 6.0 / horizon_kurtosis, 4.2, 200.0))
    nu = toxic_tail_index(nu, toxicity_excess, tail_kappa)
    gamma = (
        math.exp(-FUNDING_SKEW * math.tanh(funding_rate / FUNDING_REFERENCE))
        if funding_rate is not None
        else 1.0
    )
    drift = 0.0
    if flow_drift is not None:
        drift = flow_drift
    elif imbalance is not None:
        decay = math.sqrt(min(1.0, IMBALANCE_HORIZON_SECONDS / max(horizon_seconds, 1.0)))
        drift = IMBALANCE_DRIFT * float(np.clip(imbalance, -1.0, 1.0)) * sigma * decay
    location = drift - 0.5 * variance
    standardised = (np.log(np.maximum(prices, 1.0e-12) / spot) - location) / sigma
    return SkewStudentT(nu=nu, gamma=gamma).cdf(standardised)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlowFit:
    """Per-asset model state that is expensive to compute and slow to change."""

    asset: AssetType
    interval_seconds: int
    fitted_at: float
    n_candles: int
    variance_path: FloatArray
    bar_excess_kurtosis: float
    garch: GjrGarchFit
    garch_paths: FloatArray
    har_fallback: bool
    timesfm_quantiles: FloatArray | None = None
    """Shape ``(steps, 9)``: TimesFM deciles of the price at each future bar."""

    def variance(self, horizon_bars: float) -> float:
        """Total log-return variance over a fractional number of bars."""
        path = self.variance_path
        cumulative = np.concatenate([[0.0], np.cumsum(path)])
        if horizon_bars >= path.size:
            return float(cumulative[-1] + (horizon_bars - path.size) * path[-1])
        return float(np.interp(horizon_bars, np.arange(path.size + 1), cumulative))


@dataclass(frozen=True, slots=True)
class FastInputs:
    """Per-asset inputs that change tick by tick."""

    spot: float
    imbalance: float | None
    funding_rate: float | None
    surface: VolSurface | None
    sources: tuple[str, ...]
    ts: float
    flow: FlowSignal | None = None


@dataclass(frozen=True, slots=True)
class EnsembleForecast:
    topic_id: int
    asset: AssetType
    ts: float
    spot: float
    horizon_seconds: float
    sigma: float
    """Standard deviation of the log return to expiry (HAR estimate)."""
    probabilities: Mapping[int, float]
    components: Mapping[str, Mapping[int, float]]
    weights: Mapping[str, float]
    disagreement: float
    """Weighted Jensen-Shannon divergence between components, normalised to [0, 1]."""
    kelly_multiplier: float
    spot_sources: tuple[str, ...] = field(default_factory=tuple)
    ambiguity: float = 0.0
    """KL radius ``eps`` for robust Kelly: ``floor + per_jsd * JSD`` (nats)."""
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    """Kernel, flow and calibration state behind this forecast."""

    def probability(self, option_id: int) -> float:
        return self.probabilities.get(option_id, 0.0)


def fit_slow(
    asset: AssetType,
    candles: Sequence[Candle],
    settings: Settings,
    now: float,
    timesfm_quantiles: FloatArray | None = None,
) -> SlowFit:
    """Fit the HAR and GJR-GARCH models from completed candles (CPU-bound)."""
    if len(candles) < 30:
        raise ForecastError(f"need at least 30 candles for {asset.value}, got {len(candles)}")
    interval = settings.candle_interval_seconds
    day = max(1, round(86400 / interval))
    steps = min(MAX_MODEL_STEPS, math.ceil(settings.max_horizon_hours * 3600.0 / interval) + 2)
    returns = close_returns(candles)
    har = fit_har(garman_klass_variance(candles), day=day, week=7 * day)
    garch = fit_gjr_garch(returns)
    return SlowFit(
        asset=asset,
        interval_seconds=interval,
        fitted_at=now,
        n_candles=len(candles),
        variance_path=har_variance_path(har, steps),
        bar_excess_kurtosis=float(np.clip(excess_kurtosis(returns), 0.1, 30.0)),
        garch=garch,
        garch_paths=simulate_gjr_paths(garch, steps, settings.garch_paths, settings.random_seed),
        har_fallback=har.fallback,
        timesfm_quantiles=timesfm_quantiles,
    )


def _empirical_cdf(sorted_prices: FloatArray) -> Cdf:
    positions = (np.arange(sorted_prices.size) + 0.5) / sorted_prices.size

    def cdf(prices: FloatArray) -> FloatArray:
        result: FloatArray = np.interp(prices, sorted_prices, positions, left=0.0, right=1.0)
        return result

    return cdf


def _decile_cdf(decile_prices: FloatArray) -> Cdf:
    levels = np.linspace(0.1, 0.9, 9)
    ordered = np.maximum.accumulate(decile_prices)
    low_slope = (levels[1] - levels[0]) / max(float(ordered[1] - ordered[0]), 1.0e-12)
    high_slope = (levels[-1] - levels[-2]) / max(float(ordered[-1] - ordered[-2]), 1.0e-12)

    def cdf(prices: FloatArray) -> FloatArray:
        body = np.interp(prices, ordered, levels)
        below = levels[0] + low_slope * (prices - ordered[0])
        above = levels[-1] + high_slope * (prices - ordered[-1])
        stitched = np.where(prices < ordered[0], below, np.where(prices > ordered[-1], above, body))
        result: FloatArray = np.clip(stitched, 0.0, 1.0)
        return result

    return cdf


def _sorted_eval(cdf: Cdf, prices: FloatArray) -> FloatArray:
    """Evaluate a CDF at unsorted prices and return monotone-rectified values."""
    order = np.argsort(prices, kind="stable")
    values = np.empty_like(prices)
    values[order] = rectify_cdf(cdf(prices[order]))
    return values


def _resolve_weights(settings: Settings, has_options: bool, ts_name: str) -> dict[str, float]:
    if has_options:
        raw = {
            "options_rnd": settings.weight_options,
            "har_microstructure": settings.weight_microstructure,
            ts_name: settings.weight_timeseries,
        }
    else:
        raw = {
            "har_microstructure": settings.fallback_weight_microstructure,
            ts_name: settings.fallback_weight_timeseries,
        }
    positive = {name: weight for name, weight in raw.items() if weight > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        raise ForecastError("no ensemble component has positive weight")
    return {name: weight / total for name, weight in positive.items()}


def jensen_shannon(distributions: Sequence[FloatArray], weights: Sequence[float]) -> float:
    """Weighted Jensen-Shannon divergence normalised by the entropy of the weights."""

    def entropy(p: FloatArray) -> float:
        positive = p[p > 0.0]
        return float(-(positive * np.log(positive)).sum())

    weight_array = np.asarray(weights, dtype=np.float64)
    ceiling = entropy(weight_array)
    if len(distributions) < 2 or ceiling <= 0.0:
        return 0.0
    mixture = np.zeros_like(distributions[0])
    for weight, dist in zip(weights, distributions, strict=True):
        mixture = mixture + weight * dist
    divergence = entropy(mixture) - sum(
        weight * entropy(dist) for weight, dist in zip(weights, distributions, strict=True)
    )
    return float(np.clip(divergence / ceiling, 0.0, 1.0))


def build_forecast(
    topic: TopicMeta,
    slow: SlowFit,
    fast: FastInputs,
    now: float,
    settings: Settings,
    calibration: CalibrationView | None = None,
) -> EnsembleForecast:
    """Pool the component densities and map them onto the topic's buckets.

    Pure and synchronous: every input is passed in, so the backtester can call
    it with historical data and get exactly the live behaviour. ``calibration``
    (learned pool weights and temperature) is optional; without it the
    configured weights and ``tau = 1`` are used.
    """
    horizon_seconds = topic.seconds_to_expiry(now)
    if horizon_seconds <= 0.0:
        raise ForecastError(f"topic {topic.topic_id} has expired")
    if fast.spot <= 0.0 or not topic.buckets:
        raise ForecastError(f"topic {topic.topic_id} has no usable spot or buckets")

    spot = fast.spot
    horizon_bars = horizon_seconds / slow.interval_seconds
    diagnostics: dict[str, float] = {}

    # Flow toxicity scales the physical variance before anything consumes it.
    toxicity_excess = 0.0
    flow_drift: float | None = None
    flow = fast.flow if settings.enable_flow_signals else None
    if flow is not None and flow.toxicity is not None:
        toxicity_excess = flow.toxicity.excess
        diagnostics["vpin"] = flow.toxicity.vpin
        diagnostics["vpin_percentile"] = flow.toxicity.percentile
    variance = slow.variance(horizon_bars) * toxic_variance_multiplier(
        toxicity_excess, settings.toxicity_variance_kappa
    )
    sigma = math.sqrt(max(variance, 1.0e-16))
    t_years = horizon_seconds / YEAR_SECONDS
    if flow is not None and flow.ofi is not None:
        flow_drift = ofi_drift(flow.ofi, horizon_seconds, sigma, settings.ofi_drift_cap_sigmas)
        diagnostics["ofi"] = flow.ofi.integrated
        diagnostics["ofi_beta"] = flow.ofi.beta
        diagnostics["ofi_drift_sigmas"] = flow_drift / sigma

    cdfs: dict[str, Cdf] = {}
    surface = fast.surface
    if topic.asset.has_listed_options and surface is not None:
        forward = surface.forward(spot, t_years)
        rnd = risk_neutral_distribution(forward, t_years, surface)
        if settings.enable_pricing_kernel:
            physical, kernel = physical_distribution(
                rnd,
                forward,
                variance,
                blend=settings.kernel_blend,
                prior_ratio=settings.kernel_prior_ratio,
                asymmetry=settings.kernel_asymmetry,
                log_drift=settings.physical_drift_annual * t_years,
                max_ratio=settings.kernel_max_ratio,
            )
            cdfs["options_rnd"] = physical.cdf_at
            diagnostics["kernel_xi"] = kernel.xi
            diagnostics["kernel_gamma"] = kernel.gamma
            diagnostics["kernel_variance_ratio"] = kernel.variance_ratio
        else:
            cdfs["options_rnd"] = rnd.cdf_at

    def micro_cdf(prices: FloatArray) -> FloatArray:
        return microstructure_cdf(
            prices,
            spot,
            variance,
            horizon_seconds,
            horizon_bars,
            slow.bar_excess_kurtosis,
            fast.imbalance,
            fast.funding_rate,
            flow_drift=flow_drift,
            toxicity_excess=toxicity_excess,
            tail_kappa=settings.toxicity_tail_kappa,
        )

    cdfs["har_microstructure"] = micro_cdf

    if slow.timesfm_quantiles is not None and slow.timesfm_quantiles.shape[0] > 0:
        ts_name = "timesfm"
        step = min(max(math.ceil(horizon_bars), 1), slow.timesfm_quantiles.shape[0]) - 1
        cdfs[ts_name] = _decile_cdf(slow.timesfm_quantiles[step])
    else:
        ts_name = "gjr_garch"
        simulated = terminal_log_returns(slow.garch_paths, horizon_bars)
        cdfs[ts_name] = _empirical_cdf(spot * np.exp(np.sort(simulated)))

    weights = _resolve_weights(settings, "options_rnd" in cdfs, ts_name)
    if calibration is not None and calibration.weights:
        learned = {name: max(calibration.weights.get(name, 0.0), 0.0) for name in weights}
        total_learned = sum(learned.values())
        if total_learned > 0.0:
            weights = {name: value / total_learned for name, value in learned.items()}
    names = [name for name in weights if name in cdfs and weights[name] > 0.0]

    lows = np.array([bucket.lo for bucket in topic.buckets])
    highs = np.array([bucket.hi for bucket in topic.buckets])
    component_arrays: dict[str, FloatArray] = {}
    for name in names:
        mass = np.maximum(_sorted_eval(cdfs[name], highs) - _sorted_eval(cdfs[name], lows), 0.0)
        total = float(mass.sum())
        if total <= 0.0:
            raise ForecastError(f"{name} puts no mass inside the buckets of topic {topic.topic_id}")
        component_arrays[name] = mass / total

    half_width = max(GRID_SIGMAS * sigma, 0.01)
    grid = spot * np.exp(np.linspace(-half_width, half_width, GRID_POINTS))
    mixture = np.zeros(GRID_POINTS)
    for name in names:
        mixture = mixture + weights[name] * rectify_cdf(cdfs[name](grid))
    mixture = rectify_cdf(mixture) + np.linspace(0.0, 1.0e-9, GRID_POINTS)
    quantile_values = np.interp(np.asarray(QUANTILE_LEVELS), mixture, grid)

    in_window = [bucket for bucket in topic.buckets if bucket.hi > grid[0] and bucket.lo < grid[-1]]
    outcomes = [
        QuoteOutcome(option_id=bucket.option_id, name=bucket.name, yes_price=0.0)
        for bucket in in_window
    ]
    mapped = sdk_bucket_probabilities(
        outcomes, [float(value) for value in quantile_values], QUANTILE_LEVELS
    )
    raw = {signal.option_id: max(signal.model_probability, 0.0) for signal in mapped}
    total = sum(raw.values())
    if total <= 0.0:
        raise ForecastError(f"ensemble puts no mass inside the buckets of topic {topic.topic_id}")
    pooled = np.array([raw.get(bucket.option_id, 0.0) / total for bucket in topic.buckets])
    tau = 1.0 if calibration is None else calibration.temperature
    if abs(tau - 1.0) > 1.0e-9:
        pooled = apply_temperature(pooled, tau)
        diagnostics["temperature"] = tau
    probabilities = {bucket.option_id: float(pooled[i]) for i, bucket in enumerate(topic.buckets)}

    pool_weights = [weights[name] for name in names]
    disagreement = jensen_shannon([component_arrays[name] for name in names], pool_weights)
    weight_entropy = -sum(w * math.log(w) for w in pool_weights if w > 0.0)
    jsd_nats = disagreement * weight_entropy
    ambiguity = settings.ambiguity_floor + settings.ambiguity_per_jsd * jsd_nats
    kelly_multiplier = float(
        np.clip(1.0 - 2.0 * disagreement, settings.kelly_disagreement_floor, 1.0)
    )
    components = {
        name: {
            bucket.option_id: float(component_arrays[name][i])
            for i, bucket in enumerate(topic.buckets)
            if component_arrays[name][i] > 1.0e-6
        }
        for name in names
    }
    return EnsembleForecast(
        topic_id=topic.topic_id,
        asset=topic.asset,
        ts=now,
        spot=spot,
        horizon_seconds=horizon_seconds,
        sigma=sigma,
        probabilities=probabilities,
        components=components,
        weights={name: weights[name] for name in names},
        disagreement=disagreement,
        kelly_multiplier=kelly_multiplier,
        spot_sources=fast.sources,
        ambiguity=ambiguity,
        diagnostics=diagnostics,
    )


def build_signals(
    topic: TopicMeta, book: LmsrBook, forecast: EnsembleForecast, min_probability: float = 1.0e-4
) -> list[BucketSignal]:
    """Model-versus-market comparison for every bucket, best edge first."""
    prices = book.prices()
    signals: list[BucketSignal] = []
    for bucket in topic.buckets:
        if bucket.option_id not in book:
            continue
        probability = forecast.probability(bucket.option_id)
        market = float(prices[book.index_of(bucket.option_id)])
        if probability < min_probability and market < 0.5:
            continue
        signals.append(
            BucketSignal(
                topic_id=topic.topic_id,
                option_id=bucket.option_id,
                name=bucket.name,
                lo=bucket.lo,
                hi=bucket.hi,
                market_price=PriceUnits(market),
                model_probability=probability,
                edge_points=PriceUnits(100.0 * probability - market),
                components={
                    name: values.get(bucket.option_id, 0.0)
                    for name, values in forecast.components.items()
                },
            )
        )
    signals.sort(key=lambda signal: float(signal.edge_points), reverse=True)
    return signals


class _TimesFmRunner:
    """Lazy wrapper around the SDK's ``TimesFMForecaster`` (Apache-2.0 TimesFM 2.5)."""

    def __init__(self, enabled: bool) -> None:
        self.available = enabled and importlib.util.find_spec("timesfm") is not None
        self._model: object | None = None

    def quantiles(self, closes: Sequence[float], steps: int) -> FloatArray | None:
        """Deciles of the price for each of the next ``steps`` bars (blocking)."""
        if not self.available:
            return None
        try:
            from glimpse_markets.forecasting.model import TimesFMForecaster

            if self._model is None:
                self._model = TimesFMForecaster()
            assert isinstance(self._model, TimesFMForecaster)
            horizon = min(steps, self._model.max_horizon)
            result = self._model.forecast(list(closes)[-self._model.max_context :], horizon)
            return np.asarray(result.raw_quantiles, dtype=np.float64)[:, 1:]
        except Exception as exc:
            log.warning("forecast.timesfm_unavailable", error=repr(exc))
            self.available = False
            return None


class EnsembleForecaster:
    """Gathers live inputs, caches the slow fits and produces per-topic forecasts."""

    def __init__(self, hub: MarketDataHub, settings: Settings) -> None:
        self._hub = hub
        self._settings = settings
        self._slow: dict[AssetType, SlowFit] = {}
        self._fast: dict[AssetType, FastInputs] = {}
        self._locks: dict[AssetType, asyncio.Lock] = {}
        self._timesfm = _TimesFmRunner(settings.enable_timesfm)
        self._ofi: dict[AssetType, OfiTracker] = {}

    @property
    def timeseries_model(self) -> str:
        return "timesfm" if self._timesfm.available else "gjr_garch"

    async def slow_fit(self, asset: AssetType, now: float) -> SlowFit:
        cached = self._slow.get(asset)
        if cached is not None and now - cached.fitted_at < self._settings.model_refit_seconds:
            return cached
        async with self._locks.setdefault(asset, asyncio.Lock()):
            cached = self._slow.get(asset)
            if cached is not None and now - cached.fitted_at < self._settings.model_refit_seconds:
                return cached
            candles = await self._hub.candles(
                asset, self._settings.candle_interval_seconds, self._settings.candle_history
            )
            steps = math.ceil(
                self._settings.max_horizon_hours * 3600.0 / self._settings.candle_interval_seconds
            )
            quantiles = await asyncio.to_thread(
                self._timesfm.quantiles, [bar.close for bar in candles], steps
            )
            fit = await asyncio.to_thread(fit_slow, asset, candles, self._settings, now, quantiles)
            self._slow[asset] = fit
            log.info(
                "forecast.refit",
                asset=asset.value,
                candles=fit.n_candles,
                garch_persistence=round(fit.garch.persistence, 4),
                har_fallback=fit.har_fallback,
                timeseries=self.timeseries_model,
            )
            return fit

    async def fast_inputs(self, asset: AssetType, now: float) -> FastInputs:
        spot = await self._hub.spot(asset)
        cached = self._fast.get(asset)
        if cached is not None and now - cached.ts < self._settings.fast_input_ttl_seconds:
            return FastInputs(
                spot=spot.price,
                imbalance=cached.imbalance,
                funding_rate=cached.funding_rate,
                surface=cached.surface,
                sources=spot.sources,
                ts=cached.ts,
                flow=cached.flow,
            )
        settings = self._settings
        book, funding, chain, minute_bars = await asyncio.gather(
            self._hub.orderbook(asset),
            self._hub.funding_rate(asset),
            self._hub.option_chain(asset),
            self._hub.flow_candles(asset)
            if settings.enable_flow_signals
            else asyncio.sleep(0.0, result=()),
        )
        imbalance = book.imbalance() if book is not None and book.bids and book.asks else None
        flow: FlowSignal | None = None
        if settings.enable_flow_signals:
            tracker = self._ofi.setdefault(asset, OfiTracker(levels=settings.ofi_levels))
            snapshot = book_levels(book, now, settings.ofi_levels) if book is not None else None
            if snapshot is not None:
                tracker.update(snapshot)
            flow = FlowSignal(
                ofi=tracker.state(settings.ofi_min_obs),
                toxicity=toxicity_from_candles(
                    minute_bars,
                    settings.vpin_bucket_minutes,
                    settings.vpin_window,
                    settings.toxicity_threshold,
                ),
            )
        surface = VolSurface.from_chain(chain, now) if chain else None
        fresh = FastInputs(
            spot=spot.price,
            imbalance=imbalance,
            funding_rate=funding,
            surface=surface,
            sources=spot.sources,
            ts=now,
            flow=flow,
        )
        self._fast[asset] = fresh
        return fresh

    async def forecast(
        self,
        topic: TopicMeta,
        now: float | None = None,
        calibration: CalibrationView | None = None,
    ) -> EnsembleForecast:
        moment = time.time() if now is None else now
        slow, fast = await asyncio.gather(
            self.slow_fit(topic.asset, moment), self.fast_inputs(topic.asset, moment)
        )
        return build_forecast(topic, slow, fast, moment, self._settings, calibration)
