"""Distributionally robust Kelly sizing and the multi-leg corridor basket.

Ambiguity set: every distribution within KL radius ``eps`` of the model,
``U = {p' : KL(p' || p) <= eps}``, and the bettor maximises worst-case log growth.

Single binary contract (exact, closed form)
-------------------------------------------
A long contract bought at all-in price ``c`` pays ``pi`` with probability ``p'``.
For any stake ``f > 0`` the log growth ``p' ln(1 + f b) + (1 - p') ln(1 - f)`` is
increasing in ``p'``, so the worst case in ``U`` is the *same* for every ``f``:
the KL lower confidence bound

    p_lo(eps) = min { q <= p : q ln(q/p) + (1-q) ln((1-q)/(1-p)) <= eps },

and the robust Kelly fraction is just Kelly evaluated there:

    f_rob = max(0, (p_lo pi - c) / (pi - c)).

For small edges this is the Sharpe-shrunk fraction
``f_rob ~= f_Kelly (1 - sqrt(2 eps) / SR)`` with per-bet Sharpe
``SR = (p pi - c) / (pi sqrt(p (1 - p)))`` -- bets with a thin Sharpe ratio are
cut hardest, which a fixed fractional-Kelly multiplier cannot do.

Basket on mutually exclusive buckets (one topic)
------------------------------------------------
Buying ``delta`` contracts on a support ``S`` of buckets costs
``K(delta) = (1 + fee) [C(q + delta) - C(q)]`` on the LS-LMSR, and wealth in
outcome ``j`` (bucket ``j`` wins, or "none of S" with mass ``1 - sum_S p``) is
``W_j = W0 - K + pi delta_j``. By Lagrangian duality for the KL ball (Hu and
Hong 2013), the worst-case log growth is a one-dimensional concave sup:

    G_eps(delta) = sup_{lambda > 0}  -lambda ln sum_j p_j W_j^{-1/lambda}  -  lambda eps,

i.e. the best CRRA certainty equivalent (relative risk aversion ``1 + 1/lambda``)
minus a penalty -- robustness *is* endogenous risk aversion. ``(delta, lambda)``
is jointly concave, so L-BFGS-B over ``(delta, ln lambda)`` with the analytic
gradient converges to the global optimum:

    a_j = p_j W_j^{-1/lambda} / Z                    (worst-case distribution)
    dG/ddelta_i = pi a_i / W_i - K_i sum_j a_j / W_j,  K_i = (1 + fee) price_i(q + delta)
    dG/dlambda  = -ln Z - (sum_j a_j ln W_j) / lambda - eps

``W0`` is the *sub-bankroll* ``kappa * equity`` (fractional Kelly as a separate
log-utility account, exact rather than a post-hoc scaling of stakes).

Cost attribution (Aumann-Shapley)
---------------------------------
Because ``C`` is path-independent, ``C(q + delta) - C(q) = sum_i delta_i
int_0^1 p_i(q + t delta) dt`` exactly; Gauss-Legendre quadrature of that
integral splits the basket's cost across legs for the ledger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, minimize, minimize_scalar

from glimpse_quant_bot.lmsr import LmsrBook, cost, marginal_prices

FloatArray = npt.NDArray[np.float64]

LAMBDA_BOUNDS = (1.0e-4, 1.0e4)


# ---------------------------------------------------------------------------
# Binary contracts
# ---------------------------------------------------------------------------


def kl_bernoulli(q: float, p: float) -> float:
    """``KL(Bern(q) || Bern(p))`` in nats."""
    total = 0.0
    if q > 0.0:
        total += q * math.log(q / p)
    if q < 1.0:
        total += (1.0 - q) * math.log((1.0 - q) / (1.0 - p))
    return total


def kl_lower_probability(p: float, eps: float) -> float:
    """Smallest ``q <= p`` with ``KL(q || p) <= eps`` (the KL lower confidence bound)."""
    if eps <= 0.0 or p <= 0.0:
        return max(p, 0.0)
    if p >= 1.0:
        return math.exp(-eps)
    if kl_bernoulli(0.0, p) <= eps:
        return 0.0
    return float(brentq(lambda q: kl_bernoulli(q, p) - eps, 0.0, p, xtol=1.0e-14))


def robust_kelly_fraction(
    probability: float, all_in_price: float, payout: float, eps: float
) -> float:
    """Exact worst-case-over-KL-ball Kelly fraction of a binary contract."""
    if all_in_price <= 0.0 or all_in_price >= payout:
        return 0.0
    p_lo = kl_lower_probability(probability, eps)
    return max(0.0, (p_lo * payout - all_in_price) / (payout - all_in_price))


def sharpe_shrinkage(probability: float, all_in_price: float, payout: float, eps: float) -> float:
    """Small-edge approximation ``f_rob / f_Kelly ~= 1 - sqrt(2 eps) / SR``."""
    edge = probability * payout - all_in_price
    spread = payout * math.sqrt(max(probability * (1.0 - probability), 1.0e-18))
    if edge <= 0.0:
        return 0.0
    return max(0.0, 1.0 - math.sqrt(2.0 * max(eps, 0.0)) * spread / edge)


# ---------------------------------------------------------------------------
# Worst-case log growth over a KL ball
# ---------------------------------------------------------------------------


def _dual_terms(probs: FloatArray, log_w: FloatArray, lam: float) -> tuple[float, FloatArray]:
    """``ln Z`` and the worst-case weights ``a`` for ``Z = sum p W^{-1/lambda}``."""
    z = np.where(probs > 0.0, np.log(np.maximum(probs, 1.0e-300)) - log_w / lam, -np.inf)
    peak = float(np.max(z))
    e = np.exp(z - peak)
    total = float(e.sum())
    weights: FloatArray = e / total
    return peak + math.log(total), weights


def robust_log_growth(probs: FloatArray, wealth: FloatArray, eps: float) -> tuple[float, float]:
    """``(G_eps, lambda*)``: worst-case expected ``ln W`` over ``KL(p' || p) <= eps``.

    ``eps = 0`` returns the nominal ``E_p[ln W]`` and ``lambda = inf``.
    """
    if np.any(wealth <= 0.0):
        return -math.inf, 0.0
    log_w = np.log(wealth)
    nominal = float(probs @ log_w)
    if eps <= 0.0:
        return nominal, math.inf

    def negative(u: float) -> float:
        lam = math.exp(u)
        log_z, _ = _dual_terms(probs, log_w, lam)
        return lam * log_z + lam * eps

    lo, hi = (math.log(bound) for bound in LAMBDA_BOUNDS)
    result = minimize_scalar(negative, bounds=(lo, hi), method="bounded")
    lam = math.exp(float(result.x))
    value = -float(result.fun)
    worst = float(np.min(log_w[probs > 0.0]))
    return float(np.clip(value, worst, nominal)), lam


# ---------------------------------------------------------------------------
# Corridor basket on one topic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BasketLeg:
    option_id: int
    contracts: float
    model_probability: float
    market_price: float
    """Marginal price before the basket, in points."""
    cost_points: float
    """Aumann-Shapley share of the basket's LS-LMSR cost (excl. commission)."""


@dataclass(frozen=True, slots=True)
class BasketSolution:
    topic_id: int
    legs: tuple[BasketLeg, ...]
    cost_points: float
    """``C(q + delta) - C(q)``, excluding commission."""
    commission_points: float
    expected_profit_points: float
    robust_growth: float
    """Worst-case expected log growth of the sub-bankroll."""
    nominal_growth: float
    worst_case_lambda: float
    sub_bankroll_points: float

    @property
    def all_in_points(self) -> float:
        return self.cost_points + self.commission_points

    @property
    def leg_tuples(self) -> list[tuple[int, float]]:
        return [(leg.option_id, leg.contracts) for leg in self.legs]


def aumann_shapley_costs(
    book: LmsrBook, legs: Sequence[tuple[int, float]], nodes: int = 16
) -> FloatArray:
    """Per-leg cost ``delta_i int_0^1 p_i(q + t delta) dt`` (points); sums to the exact cost."""
    delta = np.zeros_like(book.shares)
    index = np.array([book.index_of(option_id) for option_id, _ in legs])
    delta[index] = [contracts for _, contracts in legs]
    t, w = np.polynomial.legendre.leggauss(nodes)
    t = 0.5 * (t + 1.0)
    w = 0.5 * w
    integral = np.zeros(index.size)
    for ti, wi in zip(t, w, strict=True):
        integral += wi * marginal_prices(book.shares + ti * delta, book.alpha)[index]
    shares: FloatArray = delta[index] * integral
    exact = cost(book.shares + delta, book.alpha) - cost(book.shares, book.alpha)
    total = float(shares.sum())
    if total > 0.0:
        shares = shares * (exact / total)
    return shares


def optimize_basket(
    book: LmsrBook,
    probabilities: Mapping[int, float],
    *,
    sub_bankroll_points: float,
    max_cost_points: float,
    fee_rate: float,
    ambiguity: float,
    max_legs: int = 12,
    min_leg_contracts: float = 1.0,
    max_leg_contracts: float = 400.0,
    payout_points: float = 100.0,
    min_commission_points: float = 1.0,
) -> BasketSolution | None:
    """Maximise worst-case log growth of a multi-leg buy on one LS-LMSR topic.

    Support: buckets whose nominal marginal edge ``pi p_i - (1 + fee) price_i``
    is positive, best ``max_legs`` first (contiguous in practice: a corridor).
    Returns ``None`` when no basket improves on holding cash.
    """
    pi = payout_points * (1.0 - fee_rate)
    w0 = sub_bankroll_points
    if w0 <= 0.0 or max_cost_points <= 0.0:
        return None
    prices = book.prices()
    ids = list(book.option_ids)
    p_all = np.array([probabilities.get(option_id, 0.0) for option_id in ids])
    edge = pi * p_all - (1.0 + fee_rate) * prices
    candidates = np.flatnonzero(edge > 0.0)
    if candidates.size == 0:
        return None
    support = candidates[np.argsort(edge[candidates])[::-1][:max_legs]]
    support.sort()
    p_s = p_all[support]
    p_rest = max(0.0, 1.0 - float(p_s.sum()))
    probs = np.append(p_s, p_rest)
    base_cost = cost(book.shares, book.alpha)
    n = support.size
    eps = max(ambiguity, 0.0)

    def wealth(delta: FloatArray) -> tuple[FloatArray, float, FloatArray]:
        shifted = book.shares.copy()
        shifted[support] += delta
        spend = (1.0 + fee_rate) * (cost(shifted, book.alpha) - base_cost)
        grad_k = (1.0 + fee_rate) * marginal_prices(shifted, book.alpha)[support]
        w = np.append(w0 - spend + pi * delta, w0 - spend)
        return w, spend, grad_k

    def objective(x: FloatArray) -> tuple[float, FloatArray]:
        # Work in ln(W / W0): the objective sits near zero, so L-BFGS-B's relative
        # tolerances resolve growth differences of 1e-9 instead of 1e-8 * ln W0.
        delta, u = x[:n], float(x[n])
        w_abs, _, grad_k = wealth(delta)
        if np.any(w_abs <= 0.0):
            return 1.0e6, np.concatenate([grad_k, [0.0]])
        w = w_abs / w0
        grad_k = grad_k / w0
        log_w = np.log(w)
        if eps <= 0.0:
            a = probs
            value = float(probs @ log_w)
            d_u = 0.0
        else:
            lam = math.exp(u)
            log_z, a = _dual_terms(probs, log_w, lam)
            value = -lam * log_z - lam * eps
            d_lam = -log_z - float(a @ log_w) / lam - eps
            d_u = d_lam * lam
        inv = a / w
        d_delta = (pi / w0) * inv[:n] - grad_k * float(inv.sum())
        return -value, -np.concatenate([d_delta, [d_u]])

    x0 = np.append(np.full(n, min_leg_contracts * 0.5), 0.0)
    bounds = [(0.0, max_leg_contracts)] * n + [tuple(math.log(b) for b in LAMBDA_BOUNDS)]
    result = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1.0e-15, "gtol": 1.0e-12, "maxiter": 2000},
    )
    delta = np.asarray(result.x[:n], dtype=np.float64)

    # Respect the hard risk budget by shrinking along the ray (objective is concave on it).
    _, spend, _ = wealth(delta)
    if spend > max_cost_points:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if wealth(delta * mid)[1] > max_cost_points:
                hi = mid
            else:
                lo = mid
        delta = delta * lo

    delta = np.floor(delta * 100.0) / 100.0
    delta[delta < min_leg_contracts] = 0.0
    keep = delta > 0.0
    if not bool(np.any(keep)):
        return None
    legs = [(ids[int(i)], float(d)) for i, d in zip(support[keep], delta[keep], strict=True)]
    _w, spend, _ = wealth(delta)
    raw_cost = spend / (1.0 + fee_rate)
    commission = max(min_commission_points, fee_rate * raw_cost)
    all_in = raw_cost + commission
    w_exact = np.append(w0 - all_in + pi * delta, w0 - all_in)
    robust, lam = robust_log_growth(probs, w_exact, eps)
    if not math.isfinite(robust) or robust <= math.log(w0):
        return None
    nominal = float(probs @ np.log(w_exact))
    shares = aumann_shapley_costs(book, legs)
    leg_objects = tuple(
        BasketLeg(
            option_id=option_id,
            contracts=contracts,
            model_probability=float(p_all[book.index_of(option_id)]),
            market_price=float(prices[book.index_of(option_id)]),
            cost_points=float(share),
        )
        for (option_id, contracts), share in zip(legs, shares, strict=True)
    )
    expected = float(pi * (p_s * delta).sum()) - all_in
    return BasketSolution(
        topic_id=book.topic_id,
        legs=leg_objects,
        cost_points=raw_cost,
        commission_points=commission,
        expected_profit_points=expected,
        robust_growth=robust - math.log(w0),
        nominal_growth=nominal - math.log(w0),
        worst_case_lambda=lam,
        sub_bankroll_points=w0,
    )
