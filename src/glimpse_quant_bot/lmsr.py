"""Exact local replica of Glimpse's quantile LS-LMSR market maker.

The cost function is (Glimpse whitepaper eq. 1; Othman, Sandholm, Pennock and
Reeves 2010)::

    C(q) = 100 * b(q) * ln( sum_j exp(q_j / b(q)) ),   b(q) = alpha * sum_j q_j,
    alpha = v / (n ln n)

and the marginal price of bucket ``i`` is ``p_i = 100 * (w_i + alpha * H(w))``
with ``w = softmax(q / b)`` and ``H`` the Shannon entropy of ``w``.

``C`` is measured in price points, i.e. sats; ``C(q') - C(q)`` for a trade
therefore gives the cost in sats. Against the live ``/trades/estimate``
endpoint this replica agrees to the last reported millisat for single-leg,
multi-leg, fractional and sell trades (see ARCHITECTURE.md), which lets the
bot price an arbitrary number of candidate sizes with zero API requests and
spend its rate-limit budget only on confirming the size it intends to trade.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from glimpse_markets.money import Millisats, PriceUnits

FloatArray = npt.NDArray[np.float64]
Leg = tuple[int, float]
"""``(option_id, contracts)``."""

PAYOUT_POINTS = 100.0
MILLISATS_PER_POINT = 1000.0
DEFAULT_VIG = 2.0
PROBABILITY_FLOOR = 1e-9


def alpha_for(n_outcomes: int, vig: float = DEFAULT_VIG) -> float:
    """Liquidity-sensitivity parameter ``alpha = v / (n ln n)``."""
    if n_outcomes < 2:
        raise ValueError("an LS-LMSR market needs at least two outcomes")
    return vig / (n_outcomes * math.log(n_outcomes))


GLIMPSE_ALPHA = alpha_for(500)
"""The liquidity-sensitivity parameter Glimpse actually runs: ``2 / (500 ln 500)``.

It is the same constant for every market, *including* those with a different
number of buckets. Evidence: the WebSocket reports ``alpha = 0.00064364476``,
and the 650-bucket Solana markets (40 seeded contracts per bucket) carry a
``subsidy_amount`` of 14,839.06 sats, which is ``C(q0)`` under this alpha and
not under ``2 / (650 ln 650)`` (that would give exactly 12,000)."""


def liquidity(shares: FloatArray, alpha: float) -> float:
    """Depth parameter ``b(q) = alpha * sum(q)``, in contracts."""
    total = float(shares.sum())
    if total <= 0.0:
        raise ValueError("total outstanding shares must be positive")
    return alpha * total


def cost(shares: FloatArray, alpha: float) -> float:
    """Cost function ``C(q)`` in price points (sats), evaluated stably."""
    b = liquidity(shares, alpha)
    z = shares / b
    peak = float(np.max(z))
    return PAYOUT_POINTS * b * (peak + math.log(float(np.exp(z - peak).sum())))


def weights(shares: FloatArray, alpha: float) -> FloatArray:
    """Probability forecast ``w = softmax(q / b)``; non-negative, sums to one."""
    b = liquidity(shares, alpha)
    z = shares / b
    expz = np.exp(z - float(np.max(z)))
    result: FloatArray = expz / float(expz.sum())
    return result


def entropy(probabilities: FloatArray) -> float:
    """Shannon entropy in nats, treating ``0 * ln 0`` as zero."""
    positive = probabilities[probabilities > 0.0]
    return float(-(positive * np.log(positive)).sum())


def marginal_prices(shares: FloatArray, alpha: float) -> FloatArray:
    """Marginal buy price of every bucket: ``100 * (w_i + alpha * H(w))``."""
    w = weights(shares, alpha)
    result: FloatArray = PAYOUT_POINTS * (w + alpha * entropy(w))
    return result


def buy_commission(
    cost_millisats: Millisats, fee_rate: float, min_commission: Millisats
) -> Millisats:
    """Commission on a purchase: ``fee_rate`` of the cost with a per-trade minimum."""
    return Millisats(max(int(min_commission), round(int(cost_millisats) * fee_rate)))


def sell_commission(proceeds_millisats: Millisats, fee_rate: float) -> Millisats:
    """Commission on a sale: ``fee_rate`` of the proceeds (no minimum observed)."""
    return Millisats(round(int(proceeds_millisats) * fee_rate))


@dataclass(slots=True)
class LmsrBook:
    """The full market-maker state of one topic."""

    topic_id: int
    option_ids: tuple[int, ...]
    shares: FloatArray
    alpha: float
    ts: float = 0.0
    _index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.option_ids) != self.shares.shape[0]:
            raise ValueError("option_ids and shares must have the same length")
        self._index = {option_id: i for i, option_id in enumerate(self.option_ids)}
        if len(self._index) != len(self.option_ids):
            raise ValueError("option_ids must be unique")

    @classmethod
    def from_shares(
        cls,
        topic_id: int,
        option_ids: Sequence[int],
        shares: Sequence[float],
        alpha: float | None = None,
        ts: float = 0.0,
    ) -> LmsrBook:
        resolved_alpha = alpha if alpha is not None and alpha > 0 else GLIMPSE_ALPHA
        return cls(
            topic_id=topic_id,
            option_ids=tuple(option_ids),
            shares=np.asarray(shares, dtype=np.float64),
            alpha=resolved_alpha,
            ts=ts,
        )

    def __contains__(self, option_id: int) -> bool:
        return option_id in self._index

    def index_of(self, option_id: int) -> int:
        return self._index[option_id]

    @property
    def liquidity(self) -> float:
        return liquidity(self.shares, self.alpha)

    def probabilities(self) -> FloatArray:
        """Market-implied probability of every bucket, in ``option_ids`` order."""
        return weights(self.shares, self.alpha)

    def prices(self) -> FloatArray:
        """Marginal buy price of every bucket, in ``option_ids`` order."""
        return marginal_prices(self.shares, self.alpha)

    def price(self, option_id: int) -> PriceUnits:
        return PriceUnits(float(self.prices()[self.index_of(option_id)]))

    def _shifted(self, legs: Sequence[Leg], sign: float) -> FloatArray:
        shifted = self.shares.copy()
        for option_id, contracts in legs:
            if contracts <= 0:
                raise ValueError("leg contracts must be positive")
            shifted[self.index_of(option_id)] += sign * contracts
        if float(np.min(shifted)) < 0.0:
            raise ValueError("cannot sell more contracts than are outstanding")
        return shifted

    def buy_cost(self, legs: Sequence[Leg]) -> Millisats:
        """``C(q + delta) - C(q)`` in millisats, excluding commission."""
        delta = cost(self._shifted(legs, 1.0), self.alpha) - cost(self.shares, self.alpha)
        return Millisats(round(delta * MILLISATS_PER_POINT))

    def sell_proceeds(self, legs: Sequence[Leg]) -> Millisats:
        """``C(q) - C(q - delta)`` in millisats, excluding commission."""
        delta = cost(self.shares, self.alpha) - cost(self._shifted(legs, -1.0), self.alpha)
        return Millisats(round(delta * MILLISATS_PER_POINT))

    def price_after_buy(self, option_id: int, contracts: float) -> PriceUnits:
        """Marginal price of ``option_id`` after buying ``contracts`` of it."""
        shifted = self._shifted([(option_id, contracts)], 1.0)
        return PriceUnits(float(marginal_prices(shifted, self.alpha)[self.index_of(option_id)]))

    def applied(self, legs: Sequence[Leg], sign: float = 1.0) -> LmsrBook:
        """A new book with ``legs`` bought (``sign=1``) or sold (``sign=-1``)."""
        return LmsrBook(
            topic_id=self.topic_id,
            option_ids=self.option_ids,
            shares=self._shifted(legs, sign),
            alpha=self.alpha,
            ts=self.ts,
        )


def max_contracts_below_price(
    book: LmsrBook, option_id: int, limit_price: PriceUnits, upper: float = 1.0e5
) -> float:
    """Largest ``k`` such that the marginal price after buying ``k`` is ``<= limit_price``.

    The marginal price is strictly increasing in ``k`` (the cost function is
    convex), so the answer is found by bisection. Returns ``0.0`` when the
    current price already exceeds the limit.
    """
    if float(book.price(option_id)) >= float(limit_price):
        return 0.0
    lo, hi = 0.0, 1.0
    while hi < upper and float(book.price_after_buy(option_id, hi)) < float(limit_price):
        lo, hi = hi, hi * 2.0
    if hi >= upper:
        return upper
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(book.price_after_buy(option_id, mid)) < float(limit_price):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return lo


def book_from_prices(
    topic_id: int,
    option_ids: Sequence[int],
    prices: Sequence[float],
    liquidity_b: float,
    alpha: float | None = None,
    ts: float = 0.0,
) -> LmsrBook:
    """Reconstruct a book from quoted prices and an assumed depth ``b``.

    Price recordings (``MarketRecorder`` NDJSON) hold only the 0-100 quote of
    every bucket, not the share vector. Because ``w = softmax(q / b)``, the
    shares are recovered up to an additive constant as ``q_i = b ln w_i + c``;
    the constant is pinned by ``sum(q) = b / alpha``. Used by the backtester.
    """
    resolved_alpha = alpha if alpha is not None and alpha > 0 else alpha_for(len(option_ids))
    raw = np.maximum(np.asarray(prices, dtype=np.float64), 0.0)
    total = float(raw.sum())
    if total <= 0.0:
        raise ValueError("at least one bucket must have a positive price")
    n = len(option_ids)
    floor = PROBABILITY_FLOOR
    shares = np.zeros(n)
    # Buckets quoted at zero get a tiny probability. If that floor implies
    # negative shares for the assumed depth, the floor is raised until the
    # share vector is feasible (it always is once the floor reaches 1/n).
    while True:
        w = np.maximum(raw / total, floor)
        w = w / float(w.sum())
        log_w = np.log(w)
        constant = (liquidity_b / resolved_alpha - liquidity_b * float(log_w.sum())) / n
        shares = liquidity_b * log_w + constant
        if float(np.min(shares)) >= 0.0 or floor >= 1.0 / n:
            break
        floor = min(floor * 4.0, 1.0 / n)
    shares = np.maximum(shares, 0.0)
    return LmsrBook(
        topic_id=topic_id,
        option_ids=tuple(option_ids),
        shares=shares,
        alpha=resolved_alpha,
        ts=ts,
    )
