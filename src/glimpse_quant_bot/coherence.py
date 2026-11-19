"""Cross-topic coherence: Bregman projection and buy-only cover arbitrage.

Glimpse lists several range markets on the same underlying that can settle at
the *same* instant -- e.g. the hourly and the daily BTC topic ending at 00:00.
Their bucket grids differ, so their prices must agree on every interval both
can express. Within one LS-LMSR topic there is no buy-side arbitrage (buying
every bucket costs ``100 (1 + n alpha H) > 98``), so coherence across topics is
where mispricing can be locked in.

Common refinement
-----------------
Union of all bucket edges -> cells ``c``. Topic ``t``'s bucket ``i`` pays in
cell ``c`` iff ``c`` lies inside it: a 0/1 matrix ``M_t``. A joint price vector
``mu = (mu_t)`` is coherent iff ``mu_t = M_t x`` for some ``x`` in the simplex
over cells -- the marginal polytope of this market.

Bregman projection (Dudik, Lahaie, Pennock 2012; Kroer et al. 2016)
-------------------------------------------------------------------
For an LMSR with depth ``b_t`` the Bregman divergence of the conjugate cost is
``b_t KL``. The arbitrage-free state nearest the books is

    x* = argmin_{x in simplex}  D(x) = sum_t 100 b_t KL_gen(M_t x || w_t)

with ``w_t`` the books' probability vectors. Frank-Wolfe solves it with a
linear oracle that here is just ``argmin_c grad_c`` (one cell), and the
duality gap ``<grad, x - e_c>`` certifies ``D(x) - D(x*)``. ``D(x*)`` (in sats)
is the first-order profit an unconstrained arbitrageur could lock in moving
both books to ``x*``; buckets with ``mu*_ti > w_ti`` are the cheap side.

Buy-only cover (exact, LP + concave line search)
------------------------------------------------
Entries on Glimpse are buy-only, so the executable arbitrage is a *cover*:
buy ``x_ti >= 0`` so that every cell pays at least one contract::

    min_x  sum (1 + fee) price_ti x_ti   s.t.  sum_t M_t' x_t >= 1,  x >= 0     (HiGHS LP)

If the optimum ``c* < pi = 100 (1 - fee)`` the cover earns at least
``pi - c*`` per unit at the margin. Scaling it by ``s`` costs
``(1 + fee) sum_t [C_t(q_t + s x_t) - C_t(q_t)]`` (convex), so the profit-maximising
size solves ``pi = (1 + fee) sum_t x_t' p_t(q_t + s x_t)`` -- one bisection. The
legs, grouped by topic, are exactly one ``enter_multi_topic_multi_leg`` request.
Residual risk: settlement outside the union of bucket ranges, reported from the
model density.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog, minimize_scalar

from glimpse_quant_bot.lmsr import LmsrBook, cost, marginal_prices
from glimpse_quant_bot.types import AssetType, TopicMeta

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

EDGE_TOLERANCE = 1.0e-9


def linked_groups(topics: Mapping[int, TopicMeta]) -> list[list[int]]:
    """Topic ids that share an asset and an end time (two or more per group)."""
    groups: dict[tuple[AssetType, int], list[int]] = defaultdict(list)
    for topic_id, topic in topics.items():
        groups[(topic.asset, topic.end_time_utc)].append(topic_id)
    return [sorted(ids) for ids in groups.values() if len(ids) >= 2]


@dataclass(frozen=True, slots=True)
class Refinement:
    edges: FloatArray
    membership: tuple[BoolArray, ...]
    """Per topic, ``(n_buckets, n_cells)``: bucket ``i`` pays in cell ``c``."""

    @property
    def n_cells(self) -> int:
        return int(self.edges.size - 1)


def refine(topics: Sequence[TopicMeta]) -> Refinement:
    edges = np.unique(
        np.array([edge for topic in topics for b in topic.buckets for edge in (b.lo, b.hi)])
    )
    lo, hi = edges[:-1], edges[1:]
    membership = tuple(
        np.array(
            [(lo >= b.lo - EDGE_TOLERANCE) & (hi <= b.hi + EDGE_TOLERANCE) for b in topic.buckets],
            dtype=np.bool_,
        )
        for topic in topics
    )
    return Refinement(edges=edges, membership=membership)


# ---------------------------------------------------------------------------
# Frank-Wolfe Bregman projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Projection:
    cells: FloatArray
    """Coherent probability of every refinement cell."""
    coherent: tuple[FloatArray, ...]
    """``mu*_t = M_t x*`` per topic, in bucket order."""
    divergence_points: float
    """``D(x*)``: first-order arbitrage value in points (sats per unit depth scale)."""
    gap: float
    iterations: int


def _divergence(
    x: FloatArray, mats: Sequence[FloatArray], ws: Sequence[FloatArray], bs: Sequence[float]
) -> float:
    total = 0.0
    for m, w, b in zip(mats, ws, bs, strict=True):
        mu = m @ x
        positive = mu > 0.0
        total += b * float(
            (mu[positive] * np.log(mu[positive] / w[positive])).sum() - mu.sum() + w.sum()
        )
    return 100.0 * total


def frank_wolfe_projection(
    topics: Sequence[TopicMeta],
    books: Sequence[LmsrBook],
    max_iter: int = 400,
    tol: float = 1.0e-6,
) -> Projection:
    """Project the books' joint state onto the coherent (marginal) polytope."""
    refinement = refine(topics)
    mats = [m.astype(np.float64) for m in refinement.membership]
    ws = [np.maximum(book.probabilities(), 1.0e-15) for book in books]
    bs = [book.liquidity for book in books]
    widths = np.diff(refinement.edges)

    # Start strictly inside: spread each book's mass over its cells by width.
    x = np.zeros(refinement.n_cells)
    for m, w in zip(mats, ws, strict=True):
        covered_width = m @ widths
        x += (m * (w / np.maximum(covered_width, 1.0e-300))[:, None]).sum(axis=0) * widths
    x = np.maximum(x, 1.0e-12)
    x /= float(x.sum())

    gap = math.inf
    iteration = 0
    for iteration in range(1, max_iter + 1):  # noqa: B007
        grad = np.zeros_like(x)
        for m, w, b in zip(mats, ws, bs, strict=True):
            mu = np.maximum(m @ x, 1.0e-300)
            grad += 100.0 * b * (m.T @ np.log(mu / w))
        vertex = int(np.argmin(grad))
        gap = float(grad @ x - grad[vertex])
        if gap <= tol * max(1.0, _divergence(x, mats, ws, bs)):
            break
        direction = -x.copy()
        direction[vertex] += 1.0

        def along(step: float, x0: FloatArray = x, d: FloatArray = direction) -> float:
            return _divergence(x0 + step * d, mats, ws, bs)

        step = float(minimize_scalar(along, bounds=(0.0, 1.0), method="bounded").x)
        x = x + step * direction
    coherent = tuple(m @ x for m in mats)
    return Projection(
        cells=x,
        coherent=coherent,
        divergence_points=_divergence(x, mats, ws, bs),
        gap=gap,
        iterations=iteration,
    )


# ---------------------------------------------------------------------------
# Buy-only cover arbitrage
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverArbitrage:
    legs: Mapping[int, tuple[tuple[int, float], ...]]
    """``topic_id -> ((option_id, contracts), ...)``: one request, one group per topic."""
    scale: float
    """Guaranteed minimum winning contracts in every covered cell."""
    unit_cost_points: float
    """LP optimum ``c*`` (marginal all-in cost of a unit cover)."""
    all_in_points: float
    guaranteed_profit_points: float
    """``pi * scale - all_in`` -- earned wherever the price settles inside the covered range."""
    uncovered_probability: float | None
    """Model probability of settling outside the covered range (residual risk)."""


def cover_arbitrage(
    topics: Sequence[TopicMeta],
    books: Sequence[LmsrBook],
    *,
    fee_rate: float,
    max_cost_points: float,
    payout_points: float = 100.0,
    min_commission_points: float = 1.0,
    min_profit_points: float = 1.0,
    outside_probability: float | None = None,
) -> CoverArbitrage | None:
    """Exact buy-only cover across linked topics, or ``None`` if none pays."""
    pi = payout_points * (1.0 - fee_rate)
    refinement = refine(topics)
    a_cover = np.hstack([m.T.astype(np.float64) for m in refinement.membership])
    if not bool(np.all(a_cover.sum(axis=1) > 0.0)):
        return None
    prices = np.concatenate([book.prices() for book in books]) * (1.0 + fee_rate)
    result = linprog(
        prices,
        A_ub=-a_cover,
        b_ub=-np.ones(refinement.n_cells),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or float(result.fun) >= pi:
        return None
    unit = np.asarray(result.x, dtype=np.float64)
    unit[unit < 1.0e-9] = 0.0
    sizes = [len(book.option_ids) for book in books]
    splits = np.cumsum(sizes)[:-1]
    per_topic = np.split(unit, splits)
    bases = [cost(book.shares, book.alpha) for book in books]

    def spend(scale: float) -> float:
        return (1.0 + fee_rate) * sum(
            cost(book.shares + scale * x, book.alpha) - base
            for book, x, base in zip(books, per_topic, bases, strict=True)
        )

    def marginal(scale: float) -> float:
        return pi - (1.0 + fee_rate) * sum(
            float(x @ marginal_prices(book.shares + scale * x, book.alpha))
            for book, x in zip(books, per_topic, strict=True)
        )

    hi = 1.0
    while marginal(hi) > 0.0 and spend(hi) < max_cost_points and hi < 1.0e5:
        hi *= 2.0
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if marginal(mid) > 0.0 and spend(mid) <= max_cost_points:
            lo = mid
        else:
            hi = mid
    scale = math.floor(lo * 100.0) / 100.0
    if scale <= 0.0:
        return None

    legs: dict[int, tuple[tuple[int, float], ...]] = {}
    all_in = 0.0
    for book, x, base in zip(books, per_topic, bases, strict=True):
        contracts = np.ceil(scale * x * 100.0) / 100.0
        chosen = tuple(
            (book.option_ids[i], float(contracts[i])) for i in np.flatnonzero(contracts > 0.0)
        )
        if not chosen:
            continue
        shifted = book.shares.copy()
        for option_id, k in chosen:
            shifted[book.index_of(option_id)] += k
        raw = cost(shifted, book.alpha) - base
        all_in += raw + max(min_commission_points, fee_rate * raw)
        legs[book.topic_id] = chosen
    profit = pi * scale - all_in
    if profit < min_profit_points:
        return None
    return CoverArbitrage(
        legs=legs,
        scale=scale,
        unit_cost_points=float(result.fun),
        all_in_points=all_in,
        guaranteed_profit_points=profit,
        uncovered_probability=outside_probability,
    )
