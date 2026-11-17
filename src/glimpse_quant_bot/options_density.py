"""Risk-neutral density from an options chain via Breeden-Litzenberger.

Breeden and Litzenberger (1978): the risk-neutral density of the terminal
price is the discounted second strike-derivative of the call price,

    f(K) = e^{rT} * d^2 C / dK^2,

so the risk-neutral CDF is ``F(K) = 1 + e^{rT} * dC/dK`` and the probability
of a Glimpse bucket is ``P(a < S_T <= b) = F(b) - F(a)``.

Differentiating raw quotes twice amplifies noise, so the chain is smoothed
first. For every expiry the out-of-the-money mark volatilities are fitted
with a quadratic in *standardised* log-moneyness

    z = ln(K / F) / (sigma_atm * sqrt(T)),      sigma(z) = a + b z + c z^2.

Wings
-----
Beyond the quoted strikes the smile is *not* clipped. Total variance
``w = sigma^2 T`` is continued linearly in log-moneyness ``k = ln(K / F)`` with
the slope the fit has at the last quoted strike, so ``sigma`` and its first
derivative are continuous there and ``d^2C/dK^2`` has no kink (a hard clip puts
a jump in ``d sigma / dK`` and therefore a negative Dirac spike in the density).
The slope is capped at ``|dw/dk| < 2``, Roger Lee's moment bound (2004): a wing
steeper than that implies option prices that violate static arbitrage.

Between expiries, total variance is interpolated linearly at fixed ``z`` (no
calendar arbitrage when both slices are sane); below the first expiry the first
slice's shape is reused with flat forward volatility.

Density
-------
Call prices are regenerated from the smooth surface with Black-76 on a dense
log-strike grid and differenced twice. Two repairs then make the result a
proper distribution: the density is floored at zero (a quadratic smile can
still breach the butterfly condition near the wings), and it is renormalised
with ``np.trapezoid`` so that it integrates to exactly one. The CDF is the
running trapezoid integral of that density, hence non-decreasing on ``[0, 1]``
by construction; bucket probabilities are differences of it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from glimpse_quant_bot.external_feeds import OptionQuote

FloatArray = npt.NDArray[np.float64]

YEAR_SECONDS = 365.0 * 86400.0
VOL_FLOOR = 0.03
VOL_CAP = 5.0
MAX_ABS_Z = 4.0
LEE_SLOPE_LIMIT = 1.9
"""Cap on the wing slope ``|dw/dk|``, strictly inside Roger Lee's bound of 2."""
DEEP_ITM_FRACTION = 1.0e-3
"""The strike grid never extends below this fraction of the forward (or above
its reciprocal): no probability mass lives there."""
GRID_POINTS = 8001
GRID_START_SIGMAS = 12.0
GRID_MAX_SIGMAS = 96.0
TAIL_TOLERANCE = 1.0e-10
"""The grid is widened until less than this much mass is left beyond each end."""


def black76_call(
    forward: float, strikes: FloatArray, vol: FloatArray, t_years: float, rate: float = 0.0
) -> FloatArray:
    """Black-76 call price on a forward, vectorised over strikes."""
    if t_years <= 0.0:
        intrinsic: FloatArray = np.maximum(forward - strikes, 0.0)
        return intrinsic
    sigma_root_t = np.maximum(vol, 1e-9) * math.sqrt(t_years)
    d1 = (np.log(forward / strikes) + 0.5 * sigma_root_t**2) / sigma_root_t
    d2 = d1 - sigma_root_t
    price: FloatArray = math.exp(-rate * t_years) * (forward * ndtr(d1) - strikes * ndtr(d2))
    return price


@dataclass(frozen=True, slots=True)
class SmileSlice:
    """Fitted implied-volatility smile for one expiry."""

    expiry_ts: int
    t_years: float
    forward: float
    scale_vol: float
    """ATM volatility used to standardise moneyness for this slice."""
    coeffs: tuple[float, float, float]
    """``(a, b, c)`` of ``sigma(z) = a + b z + c z^2``."""
    z_lo: float
    z_hi: float
    n_quotes: int

    def _fitted(self, z: float) -> tuple[float, float]:
        """Fitted volatility and its ``z``-derivative at one point of the quoted range."""
        a, b, c = self.coeffs
        sigma = a + b * z + c * z * z
        if sigma <= VOL_FLOOR:
            return VOL_FLOOR, 0.0
        return sigma, b + 2.0 * c * z

    def variance_at_z(self, z: FloatArray, max_slope: float = math.inf) -> FloatArray:
        """Annualised variance ``sigma(z)^2``.

        Inside ``[z_lo, z_hi]`` this is the fitted quadratic. Outside, variance
        continues *linearly* in ``z`` (equivalently in log-moneyness) with the
        value and slope it has at the boundary, the slope being limited to
        ``max_slope`` -- Lee's bound expressed in ``z`` units by the caller.
        """
        a, b, c = self.coeffs
        inner = np.clip(z, self.z_lo, self.z_hi)
        variance = np.maximum(a + b * inner + c * inner**2, VOL_FLOOR) ** 2
        for edge, outside in ((self.z_lo, z < self.z_lo), (self.z_hi, z > self.z_hi)):
            if not bool(np.any(outside)):
                continue
            sigma, dsigma = self._fitted(edge)
            slope = float(np.clip(2.0 * sigma * dsigma, -max_slope, max_slope))
            variance = np.where(outside, sigma**2 + slope * (z - edge), variance)
        clipped: FloatArray = np.clip(variance, VOL_FLOOR**2, VOL_CAP**2)
        return clipped

    def vol_at_z(self, z: FloatArray, max_slope: float = math.inf) -> FloatArray:
        vol: FloatArray = np.sqrt(self.variance_at_z(z, max_slope))
        return vol

    @property
    def atm_vol(self) -> float:
        return float(self.vol_at_z(np.zeros(1))[0])


def fit_smile_slice(quotes: Sequence[OptionQuote], now: float) -> SmileSlice | None:
    """Fit one expiry's smile from its out-of-the-money mark volatilities."""
    if not quotes:
        return None
    expiry_ts = quotes[0].expiry_ts
    t_years = (expiry_ts - now) / YEAR_SECONDS
    if t_years <= 0.0:
        return None
    forward = float(np.median([quote.forward for quote in quotes]))
    otm = [
        quote
        for quote in quotes
        if quote.mark_iv > 0.0
        and (
            (quote.is_call and quote.strike >= forward)
            or (not quote.is_call and quote.strike <= forward)
        )
    ]
    if not otm:
        return None
    nearest = sorted(otm, key=lambda quote: abs(math.log(quote.strike / forward)))[:2]
    scale_vol = float(np.mean([quote.mark_iv for quote in nearest]))
    scale = max(scale_vol, VOL_FLOOR) * math.sqrt(t_years)
    z = np.array([math.log(quote.strike / forward) / scale for quote in otm])
    iv = np.array([quote.mark_iv for quote in otm])
    keep = np.abs(z) <= MAX_ABS_Z
    z, iv = z[keep], iv[keep]
    if z.size == 0:
        return None
    if z.size < 5 or float(np.ptp(z)) < 0.5:
        level = float(np.mean(iv))
        return SmileSlice(
            expiry_ts, t_years, forward, scale_vol, (level, 0.0, 0.0), 0.0, 0.0, int(z.size)
        )

    root_w = np.sqrt(np.exp(-0.5 * (z / 2.0) ** 2))
    design = np.column_stack([np.ones_like(z), z, z**2])
    solution = np.linalg.lstsq(design * root_w[:, None], iv * root_w, rcond=None)[0]
    a, b, c = (float(value) for value in solution)
    if c < 0.0:
        linear = np.linalg.lstsq(design[:, :2] * root_w[:, None], iv * root_w, rcond=None)[0]
        a, b, c = float(linear[0]), float(linear[1]), 0.0
    return SmileSlice(
        expiry_ts=expiry_ts,
        t_years=t_years,
        forward=forward,
        scale_vol=scale_vol,
        coeffs=(a, b, c),
        z_lo=float(np.min(z)),
        z_hi=float(np.max(z)),
        n_quotes=int(z.size),
    )


@dataclass(frozen=True, slots=True)
class VolSurface:
    """Smile slices ordered by time to expiry."""

    slices: tuple[SmileSlice, ...]

    @classmethod
    def from_chain(
        cls, chain: Sequence[OptionQuote], now: float, min_seconds_to_expiry: float = 1800.0
    ) -> VolSurface | None:
        by_expiry: dict[int, list[OptionQuote]] = defaultdict(list)
        for quote in chain:
            if quote.expiry_ts - now >= min_seconds_to_expiry:
                by_expiry[quote.expiry_ts].append(quote)
        fitted = [fit_smile_slice(group, now) for group in by_expiry.values()]
        slices = sorted((s for s in fitted if s is not None), key=lambda s: s.t_years)
        return cls(tuple(slices)) if slices else None

    def _bracket(self, t_years: float) -> tuple[SmileSlice, SmileSlice, float]:
        """Neighbouring slices and the interpolation weight of the later one."""
        first, last = self.slices[0], self.slices[-1]
        if t_years <= first.t_years:
            return first, first, 0.0
        if t_years >= last.t_years:
            return last, last, 0.0
        for near, far in zip(self.slices, self.slices[1:], strict=False):
            if near.t_years <= t_years <= far.t_years:
                return near, far, (t_years - near.t_years) / (far.t_years - near.t_years)
        return last, last, 0.0

    def scale_vol(self, t_years: float) -> float:
        """ATM volatility at ``t_years`` by total-variance interpolation."""
        near, far, weight = self._bracket(t_years)
        if near is far:
            return near.scale_vol
        total_variance = (1.0 - weight) * near.scale_vol**2 * near.t_years + (
            weight * far.scale_vol**2 * far.t_years
        )
        return math.sqrt(total_variance / t_years)

    def forward(self, spot: float, t_years: float) -> float:
        """Forward at ``t_years`` implied by the nearest slice's basis to spot."""
        near, _, _ = self._bracket(t_years)
        if near.t_years <= 0.0 or spot <= 0.0:
            return spot
        annual_basis = math.log(near.forward / spot) / near.t_years
        return spot * math.exp(float(np.clip(annual_basis, -1.0, 1.0)) * t_years)

    def vol(self, strikes: FloatArray, forward: float, t_years: float) -> FloatArray:
        """Implied volatility for arbitrary strikes at horizon ``t_years``."""
        near, far, weight = self._bracket(t_years)
        scale_vol = max(self.scale_vol(t_years), VOL_FLOOR)
        root_t = math.sqrt(t_years)
        z = np.log(strikes / forward) / (scale_vol * root_t)
        # dw/dk = (d sigma^2 / dz) * sqrt(T) / sigma_atm, so Lee's bound on dw/dk
        # becomes this bound on the slope of annualised variance in z.
        max_slope = LEE_SLOPE_LIMIT * scale_vol / root_t
        if near is far:
            return near.vol_at_z(z, max_slope)
        total_variance = (1.0 - weight) * near.variance_at_z(z, max_slope) * near.t_years + (
            weight * far.variance_at_z(z, max_slope) * far.t_years
        )
        vol: FloatArray = np.clip(np.sqrt(total_variance / t_years), VOL_FLOOR, VOL_CAP)
        return vol


def _call_prices(
    strikes: FloatArray, forward: float, t_years: float, surface: VolSurface, rate: float
) -> FloatArray:
    return black76_call(forward, strikes, surface.vol(strikes, forward, t_years), t_years, rate)


@dataclass(frozen=True, slots=True)
class RiskNeutralDistribution:
    """A proper probability distribution of the terminal price on a strike grid."""

    strikes: FloatArray
    density: FloatArray
    """Non-negative; integrates to one over ``strikes`` (trapezoid rule)."""
    cdf: FloatArray
    """Running integral of ``density``: non-decreasing from 0 to 1."""

    def cdf_at(self, strikes: FloatArray) -> FloatArray:
        values: FloatArray = np.interp(strikes, self.strikes, self.cdf, left=0.0, right=1.0)
        return values

    def density_at(self, strikes: FloatArray) -> FloatArray:
        values: FloatArray = np.interp(strikes, self.strikes, self.density, left=0.0, right=0.0)
        return values


def _second_derivative(values: FloatArray, points: FloatArray) -> FloatArray:
    """Three-point second derivative on a non-uniform grid; zero at both ends."""
    left = np.diff(points)[:-1]
    right = np.diff(points)[1:]
    result = np.zeros_like(values)
    result[1:-1] = 2.0 * (
        values[2:] / (right * (left + right))
        - values[1:-1] / (left * right)
        + values[:-2] / (left * (left + right))
    )
    return result


def risk_neutral_distribution(
    forward: float, t_years: float, surface: VolSurface, rate: float = 0.0
) -> RiskNeutralDistribution:
    """Breeden-Litzenberger density and CDF on a dense log-strike grid.

    ``f(K) = e^{rT} d^2C/dK^2`` is evaluated from Black-76 prices on the smooth
    surface (the smile is re-evaluated at every grid strike, so the derivative
    carries the ``vega * d sigma / dK`` terms a flat-volatility formula misses),
    floored at zero and renormalised to integrate to one. The grid starts at
    +/- ``GRID_START_SIGMAS`` at-the-money standard deviations and is doubled
    while the fat wings still hold more than ``TAIL_TOLERANCE`` of the mass.
    """
    if t_years <= 0.0 or forward <= 0.0:
        raise ValueError("forward and time to expiry must be positive")
    discount = math.exp(rate * t_years)
    sigma_root_t = max(surface.scale_vol(t_years), VOL_FLOOR) * math.sqrt(t_years)
    limit = -math.log(DEEP_ITM_FRACTION)
    half_width = GRID_START_SIGMAS
    while True:
        reach = min(half_width * sigma_root_t, limit)
        strikes = forward * np.exp(np.linspace(-reach, reach, GRID_POINTS))
        calls = _call_prices(strikes, forward, t_years, surface, rate)
        slope = np.gradient(calls, strikes)
        tail = max(1.0 + discount * float(slope[1]), -discount * float(slope[-2]))
        if tail <= TAIL_TOLERANCE or half_width >= GRID_MAX_SIGMAS or reach >= limit:
            break
        half_width *= 2.0

    density = np.maximum(discount * _second_derivative(calls, strikes), 0.0)
    mass = float(np.trapezoid(density, strikes))
    if mass <= 0.0:
        raise ValueError("risk-neutral density has no mass on the strike grid")
    density = density / mass
    steps = 0.5 * (density[1:] + density[:-1]) * np.diff(strikes)
    cdf = np.clip(np.concatenate([[0.0], np.cumsum(steps)]), 0.0, 1.0)
    cdf[-1] = 1.0
    return RiskNeutralDistribution(strikes=strikes, density=density, cdf=cdf)


def risk_neutral_cdf(
    strikes: FloatArray, forward: float, t_years: float, surface: VolSurface, rate: float = 0.0
) -> FloatArray:
    """Risk-neutral ``P(S_T <= K)``; non-decreasing in ``K`` and within ``[0, 1]``."""
    return risk_neutral_distribution(forward, t_years, surface, rate).cdf_at(strikes)


def risk_neutral_density(
    strikes: FloatArray, forward: float, t_years: float, surface: VolSurface, rate: float = 0.0
) -> FloatArray:
    """Risk-neutral density ``f(K) = max(e^{rT} d^2C/dK^2, 0)``, normalised to unit mass."""
    return risk_neutral_distribution(forward, t_years, surface, rate).density_at(strikes)


def rectify_cdf(cdf: FloatArray) -> FloatArray:
    """Force a CDF sampled on increasing strikes to be monotone within ``[0, 1]``."""
    rectified: FloatArray = np.clip(np.maximum.accumulate(cdf), 0.0, 1.0)
    return rectified


def bucket_probabilities_rnd(
    lows: FloatArray,
    highs: FloatArray,
    forward: float,
    t_years: float,
    surface: VolSurface,
    rate: float = 0.0,
) -> FloatArray:
    """Risk-neutral ``P(lo < S_T <= hi)`` for every bucket, normalised to sum to one.

    Buckets may be given in any order. The distribution is built once and its
    CDF differenced at the bucket edges.
    """
    distribution = risk_neutral_distribution(forward, t_years, surface, rate)
    mass = np.maximum(distribution.cdf_at(highs) - distribution.cdf_at(lows), 0.0)
    total = float(mass.sum())
    if total <= 0.0:
        raise ValueError("risk-neutral density has no mass inside the bucket range")
    normalised: FloatArray = mass / total
    return normalised
