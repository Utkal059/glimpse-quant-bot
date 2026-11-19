"""Physical (P) density from the Deribit risk-neutral (Q) density via a pricing kernel.

Glimpse buckets settle on the *real-world* price, but Breeden-Litzenberger gives
the risk-neutral density, which over-weights the states investors pay to insure
against. Bitcoin's short-dated pricing kernel is U/W-shaped and steepest for
negative returns (Winkel and Hardle 2023; Almeida, Grith, Miftachov and Wang
2024), so in fear regimes the raw Q density overprices both tails -- the left
one most.

Kernel
------
With ``k = ln(K / F)`` the log-moneyness, the kernel is exponential-quadratic
(Christoffersen, Heston and Jacobs 2013, whose variance-dependent kernel
projects to exactly this shape) with a steeper downside branch::

    ln M(k) = const - gamma k + xi_-(k) k^2,   xi_-(k) = xi (1 + a) for k < 0
                                                         xi (1 - a) for k >= 0

and ``f_P(k) = f_Q(k) / M(k)`` renormalised, i.e.::

    f_P(k) = f_Q(k) exp(gamma k - xi_a(k) k^2) / Z(gamma, xi).

* ``xi >= 0`` removes the variance risk premium: for a Gaussian Q with variance
  ``s^2`` the tilt gives a Gaussian P with variance ``s^2 / (1 + 2 xi s^2)``,
  so a target variance ratio ``rho = Var_Q / Var_P`` needs exactly
  ``xi = (rho - 1) / (2 s^2)`` (used as the solver's starting bracket).
* ``gamma`` keeps ``E_P[S_T] = F exp(mu T)`` (default ``mu = 0``: no physical
  drift at Glimpse horizons). For a Gaussian Q it is exactly zero; for a
  skewed Q it is the unique root of a monotone function.
* ``a in [0, 0.9]`` makes the downside branch steeper (the empirical W shape).

Target variance
---------------
The physical variance target pools the two forecasts of the same quantity in
log space (a geometric pool, the log-score-optimal combination of two
log-normal scale forecasts)::

    ln Var_P* = lambda ln Var_HAR + (1 - lambda) ln(Var_Q / rho_0)

``rho_0`` is the unconditional variance ratio (Bitcoin IV^2 / RV ~ 1.2-1.6).
When implied variance spikes relative to realised (fear), ``Var_Q / Var_P*``
rises and the kernel automatically strengthens. ``rho`` is clipped to
``[1, rho_max]``: the kernel never *fattens* Q tails (that would require
``xi < 0``, unbounded for fat-tailed Q) and never shrinks variance by more
than ``rho_max``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from glimpse_quant_bot.options_density import RiskNeutralDistribution

FloatArray = npt.NDArray[np.float64]

MAX_ASYMMETRY = 0.9
DEFAULT_MAX_RATIO = 3.0


@dataclass(frozen=True, slots=True)
class KernelFit:
    """Fitted kernel parameters and the moments they achieve."""

    gamma: float
    xi: float
    asymmetry: float
    q_variance: float
    """Variance of ``ln(S_T / F)`` under Q."""
    target_variance: float
    p_variance: float
    """Variance of ``ln(S_T / F)`` under the fitted P."""
    active: bool
    """``False`` when Q's variance is already at or below the target (no tilt)."""

    @property
    def variance_ratio(self) -> float:
        return self.q_variance / self.p_variance if self.p_variance > 0.0 else 1.0


def _node_weights(strikes: FloatArray) -> FloatArray:
    """Trapezoid quadrature weights for a (non-uniform) grid."""
    gaps = np.diff(strikes)
    weights = np.zeros_like(strikes)
    weights[:-1] += 0.5 * gaps
    weights[1:] += 0.5 * gaps
    return weights


def _log_tilt(k: FloatArray, gamma: float, xi: float, asymmetry: float) -> FloatArray:
    curvature = np.where(k < 0.0, xi * (1.0 + asymmetry), xi * (1.0 - asymmetry))
    tilt: FloatArray = gamma * k - curvature * k * k
    return tilt


def _tilted(log_mass: FloatArray, k: FloatArray, gamma: float, xi: float, a: float) -> FloatArray:
    z = log_mass + _log_tilt(k, gamma, xi, a)
    z = z - float(np.max(z))
    p = np.exp(z)
    normalised: FloatArray = p / float(p.sum())
    return normalised


def _moments(p: FloatArray, k: FloatArray) -> tuple[float, float]:
    mean = float(p @ k)
    return mean, max(float(p @ (k * k)) - mean * mean, 0.0)


def _solve_gamma(
    log_mass: FloatArray, k: FloatArray, ek: FloatArray, xi: float, a: float, log_target: float
) -> float:
    """Root of ``ln E_P[e^k] = log_target`` in ``gamma`` (strictly increasing).

    Safeguarded Newton: ``d/dgamma ln E[e^k] = E[k e^k] / E[e^k] - E[k] > 0``.
    """
    lo, hi = -math.inf, math.inf
    gamma = 0.0
    for _ in range(80):
        p = _tilted(log_mass, k, gamma, xi, a)
        mean_ek = float(p @ ek)
        value = math.log(mean_ek) - log_target
        if abs(value) < 1.0e-13:
            return gamma
        if value > 0.0:
            hi = gamma
        else:
            lo = gamma
        slope = float(p @ (k * ek)) / mean_ek - float(p @ k)
        step = gamma - value / slope if slope > 0.0 else math.nan
        if math.isfinite(step) and lo < step < hi:
            gamma = step
        elif math.isfinite(lo) and math.isfinite(hi):
            gamma = 0.5 * (lo + hi)
        else:
            gamma = (gamma + 1.0) * 4.0 if value < 0.0 else (gamma - 1.0) * 4.0
    return gamma


def log_moneyness_variance(distribution: RiskNeutralDistribution, forward: float) -> float:
    """``Var_Q[ln(S_T / F)]`` of a density on a strike grid."""
    k = np.log(distribution.strikes / forward)
    mass = distribution.density * _node_weights(distribution.strikes)
    mass = mass / float(mass.sum())
    return _moments(mass, k)[1]


def target_physical_variance(
    q_variance: float, physical_variance: float, blend: float, prior_ratio: float
) -> float:
    """Geometric pool ``Var_HAR^lambda * (Var_Q / rho_0)^(1 - lambda)``."""
    if q_variance <= 0.0 or physical_variance <= 0.0:
        return q_variance
    blend = float(np.clip(blend, 0.0, 1.0))
    return math.exp(
        blend * math.log(physical_variance)
        + (1.0 - blend) * math.log(q_variance / max(prior_ratio, 1.0))
    )


def fit_pricing_kernel(
    distribution: RiskNeutralDistribution,
    forward: float,
    target_variance: float,
    *,
    asymmetry: float = 0.3,
    log_drift: float = 0.0,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> KernelFit:
    """Find ``(gamma, xi)`` so that P has ``Var = target`` and ``E[S_T] = F e^{log_drift}``.

    Outer: ``brentq`` on ``xi`` (``Var_P`` falls as ``xi`` rises). Inner: Newton
    on ``gamma`` for the martingale condition. ``target`` is clipped to
    ``[Var_Q / max_ratio, Var_Q]``.
    """
    a = float(np.clip(asymmetry, 0.0, MAX_ASYMMETRY))
    k = np.log(distribution.strikes / forward)
    ek = np.exp(k)
    mass = distribution.density * _node_weights(distribution.strikes)
    log_mass = np.log(np.maximum(mass, 1.0e-300))
    q_mass = _tilted(log_mass, k, 0.0, 0.0, a)
    q_var = _moments(q_mass, k)[1]
    ratio_cap = max(max_ratio, 1.0)

    if not (math.isfinite(target_variance) and target_variance > 0.0) or q_var <= 0.0:
        return KernelFit(0.0, 0.0, a, q_var, q_var, q_var, active=False)
    target = float(np.clip(target_variance, q_var / ratio_cap, q_var))
    if target >= q_var * (1.0 - 1.0e-9):
        gamma = _solve_gamma(log_mass, k, ek, 0.0, a, log_drift)
        p_var = _moments(_tilted(log_mass, k, gamma, 0.0, a), k)[1]
        return KernelFit(gamma, 0.0, a, q_var, target, p_var, active=False)

    def excess(xi: float) -> float:
        gamma = _solve_gamma(log_mass, k, ek, xi, a, log_drift)
        return _moments(_tilted(log_mass, k, gamma, xi, a), k)[1] - target

    hi = (q_var / target - 1.0) / (2.0 * q_var)  # exact for a Gaussian Q
    for _ in range(60):
        if excess(hi) < 0.0:
            break
        hi *= 2.0
    xi = float(brentq(excess, 0.0, hi, xtol=1.0e-12 * max(hi, 1.0), rtol=1.0e-10, maxiter=200))
    gamma = _solve_gamma(log_mass, k, ek, xi, a, log_drift)
    p_var = _moments(_tilted(log_mass, k, gamma, xi, a), k)[1]
    return KernelFit(gamma, xi, a, q_var, target, p_var, active=True)


def apply_kernel(
    distribution: RiskNeutralDistribution, forward: float, fit: KernelFit
) -> RiskNeutralDistribution:
    """``f_P = f_Q exp(gamma k - xi_a k^2) / Z`` on the same strike grid."""
    k = np.log(distribution.strikes / forward)
    log_tilt = _log_tilt(k, fit.gamma, fit.xi, fit.asymmetry)
    support = distribution.density > 0.0
    shift = float(np.max(log_tilt[support])) if bool(np.any(support)) else 0.0
    density = distribution.density * np.exp(np.minimum(log_tilt - shift, 0.0))
    mass = float(np.trapezoid(density, distribution.strikes))
    if mass <= 0.0:
        return distribution
    density = density / mass
    steps = 0.5 * (density[1:] + density[:-1]) * np.diff(distribution.strikes)
    cdf = np.clip(np.concatenate([[0.0], np.cumsum(steps)]), 0.0, 1.0)
    cdf[-1] = 1.0
    return RiskNeutralDistribution(strikes=distribution.strikes, density=density, cdf=cdf)


def physical_distribution(
    distribution: RiskNeutralDistribution,
    forward: float,
    physical_variance: float,
    *,
    blend: float = 0.5,
    prior_ratio: float = 1.25,
    asymmetry: float = 0.3,
    log_drift: float = 0.0,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> tuple[RiskNeutralDistribution, KernelFit]:
    """Convenience: target variance, kernel fit and tilted density in one call."""
    q_var = log_moneyness_variance(distribution, forward)
    target = target_physical_variance(q_var, physical_variance, blend, prior_ratio)
    fit = fit_pricing_kernel(
        distribution,
        forward,
        target,
        asymmetry=asymmetry,
        log_drift=log_drift,
        max_ratio=max_ratio,
    )
    return apply_kernel(distribution, forward, fit), fit
