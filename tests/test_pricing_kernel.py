"""P-vs-Q pricing kernel: closed-form Gaussian limit, martingale, tail shrinkage, bounds."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pytest

from glimpse_quant_bot.backtester import synthetic_dataset
from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.forecaster import FastInputs, build_forecast, fit_slow
from glimpse_quant_bot.options_density import (
    YEAR_SECONDS,
    RiskNeutralDistribution,
    risk_neutral_distribution,
)
from glimpse_quant_bot.pricing_kernel import (
    apply_kernel,
    fit_pricing_kernel,
    log_moneyness_variance,
    physical_distribution,
    target_physical_variance,
)

from .test_options_density import FORWARD, ONE_DAY, flat, surface_for


def lognormal(sigma: float, forward: float = FORWARD, n: int = 8001) -> RiskNeutralDistribution:
    """Exact log-normal Q density of ``S_T`` with ``E[S_T] = F`` on a log-strike grid."""
    k = np.linspace(-14.0 * sigma, 14.0 * sigma, n)
    strikes = forward * np.exp(k)
    mu = -0.5 * sigma**2
    density = np.exp(-0.5 * ((k - mu) / sigma) ** 2) / (strikes * sigma * math.sqrt(2.0 * math.pi))
    density = density / float(np.trapezoid(density, strikes))
    steps = 0.5 * (density[1:] + density[:-1]) * np.diff(strikes)
    cdf = np.concatenate([[0.0], np.cumsum(steps)])
    return RiskNeutralDistribution(strikes, density, cdf / cdf[-1])


def mean_price(dist: RiskNeutralDistribution) -> float:
    return float(np.trapezoid(dist.density * dist.strikes, dist.strikes))


def test_gaussian_limit_matches_the_closed_form() -> None:
    sigma = 0.02
    q = lognormal(sigma)
    ratio = 1.6
    q_var = log_moneyness_variance(q, FORWARD)
    fit = fit_pricing_kernel(q, FORWARD, q_var / ratio, asymmetry=0.0)
    assert fit.active
    assert fit.p_variance == pytest.approx(q_var / ratio, rel=1e-8)
    # xi = (rho - 1) / (2 s^2) exactly for a Gaussian, gamma = 0 (martingale already).
    assert fit.xi == pytest.approx((ratio - 1.0) / (2.0 * q_var), rel=1e-3)
    assert abs(fit.gamma) < 1e-3
    p = apply_kernel(q, FORWARD, fit)
    assert mean_price(p) == pytest.approx(FORWARD, rel=1e-9)


def test_martingale_holds_for_a_skewed_smile() -> None:
    surface = surface_for(lambda z: 0.55 - 0.06 * z + 0.02 * z * z)
    t = ONE_DAY / YEAR_SECONDS
    q = risk_neutral_distribution(FORWARD, t, surface)
    q_var = log_moneyness_variance(q, FORWARD)
    p, fit = physical_distribution(q, FORWARD, q_var / 2.5, blend=1.0, asymmetry=0.3)
    assert fit.active
    assert fit.xi > 0.0
    # The floored Breeden-Litzenberger Q is ~2e-4 off the forward; P is exactly F.
    assert mean_price(p) == pytest.approx(FORWARD, rel=1e-9)
    assert abs(mean_price(q) / FORWARD - 1.0) < 1e-3
    assert fit.variance_ratio == pytest.approx(2.5, rel=1e-6)
    assert np.all(np.diff(p.cdf) >= -1e-15)
    assert p.cdf[0] == 0.0
    assert p.cdf[-1] == 1.0


def test_fear_regime_shrinks_tails_and_the_left_tail_most() -> None:
    surface = surface_for(lambda z: 0.80 - 0.10 * z + 0.03 * z * z)  # panic: high IV, put skew
    t = ONE_DAY / YEAR_SECONDS
    q = risk_neutral_distribution(FORWARD, t, surface)
    q_var = log_moneyness_variance(q, FORWARD)
    p, _ = physical_distribution(q, FORWARD, q_var / 2.0, blend=1.0, asymmetry=0.4)
    sd = math.sqrt(q_var)
    lo, hi = FORWARD * math.exp(-2.5 * sd), FORWARD * math.exp(2.5 * sd)
    q_left, p_left = float(q.cdf_at(np.array([lo]))[0]), float(p.cdf_at(np.array([lo]))[0])
    q_right = 1.0 - float(q.cdf_at(np.array([hi]))[0])
    p_right = 1.0 - float(p.cdf_at(np.array([hi]))[0])
    assert p_left < 0.5 * q_left
    assert p_right < q_right
    assert p_left / q_left < p_right / q_right


def test_kernel_never_fattens_tails() -> None:
    q = lognormal(0.03)
    q_var = log_moneyness_variance(q, FORWARD)
    fit = fit_pricing_kernel(q, FORWARD, 4.0 * q_var)  # realised far above implied
    assert not fit.active
    assert fit.xi == 0.0
    p = apply_kernel(q, FORWARD, fit)
    np.testing.assert_allclose(p.cdf, q.cdf, atol=1e-9)


def test_variance_ratio_is_capped() -> None:
    q = lognormal(0.03)
    q_var = log_moneyness_variance(q, FORWARD)
    fit = fit_pricing_kernel(q, FORWARD, q_var / 100.0, max_ratio=3.0)
    assert fit.variance_ratio == pytest.approx(3.0, rel=1e-6)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_invalid_targets_leave_q_unchanged(bad: float) -> None:
    q = lognormal(0.02)
    fit = fit_pricing_kernel(q, FORWARD, bad)
    assert not fit.active
    assert fit.xi == 0.0


def test_target_is_a_geometric_pool() -> None:
    assert target_physical_variance(4.0, 1.0, 0.5, 1.0) == pytest.approx(2.0)
    assert target_physical_variance(4.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert target_physical_variance(4.0, 1.0, 0.0, 2.0) == pytest.approx(2.0)
    # Fear: implied variance jumps while realised does not -> larger ratio.
    calm = 1.0 / target_physical_variance(1.0, 0.8, 0.5, 1.25)
    fear = 3.0 / target_physical_variance(3.0, 0.8, 0.5, 1.25)
    assert fear > calm


def test_build_forecast_applies_the_kernel(make_settings: Callable[..., Settings]) -> None:
    data = synthetic_dataset(seed=5, n_topics=1)
    topic = next(iter(data.topics.values())).meta
    now = float(topic.end_time_utc - 2 * 3600)
    closed = [bar for bar in data.candles[topic.asset] if bar.ts + 3600 <= now]
    surface = surface_for(flat(1.2))  # implied far above the synthetic ~0.5%/h realised
    spot = closed[-1].close
    fast = FastInputs(spot, None, None, surface, ("test",), now)

    with_kernel = make_settings(garch_paths=400)
    slow = fit_slow(topic.asset, closed, with_kernel, now)
    forecast = build_forecast(topic, slow, fast, now, with_kernel)
    assert forecast.diagnostics["kernel_xi"] > 0.0
    assert forecast.diagnostics["kernel_variance_ratio"] > 1.5
    assert sum(forecast.probabilities.values()) == pytest.approx(1.0)

    raw = build_forecast(topic, slow, fast, now, make_settings(enable_pricing_kernel=False))
    # The kernel pulls the options component's mass toward the centre.
    mids = {b.option_id: 0.5 * (b.lo + b.hi) for b in topic.buckets}

    def spread(probabilities: Mapping[int, float]) -> float:
        mean = sum(mids[o] * p for o, p in probabilities.items())
        return math.sqrt(sum((mids[o] - mean) ** 2 * p for o, p in probabilities.items()))

    assert spread(forecast.components["options_rnd"]) < spread(raw.components["options_rnd"])
