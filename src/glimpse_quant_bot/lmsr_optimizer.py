"""LS-LMSR nonlinear slippage optimiser and fractional-Kelly sizer.

LS-LMSR prices are convex in the quantity bought: each extra contract costs
more than the last, sharply so on thin markets (the depth ``b`` of live Glimpse
topics is only a few dozen contracts). Quoted edge therefore says little about
tradeable edge. For every candidate bucket the optimiser prices a ladder of
sizes and picks the one that maximises expected net profit in millisats,

    max_k   k * 100 * p_model * (1 - fee)  -  [ C(q + k e_i) - C(q) + commission(k) ]

subject to a minimum net edge per contract, a minimum absolute expected gain,
the fractional-Kelly budget and the risk manager's caps.

Two pricing sources are combined. The local replica in
:mod:`glimpse_quant_bot.lmsr` is exact and free, so it screens the whole ladder
(plus the analytically optimal size) without touching the rate limit; the free
``/trades/estimate`` endpoint then confirms the best few sizes against the
server's own book, and the final decision is taken on server numbers only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from glimpse_markets import AsyncClient, TradeLeg, TradeType
from glimpse_markets.money import Millisats, PriceUnits

from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.lmsr import (
    Leg,
    LmsrBook,
    buy_commission,
    max_contracts_below_price,
    sell_commission,
)
from glimpse_quant_bot.types import (
    PAYOUT_PRICE,
    BucketSignal,
    LadderPoint,
    Position,
    Side,
    TradePlan,
    millisats_to_price_points,
    price_points_to_millisats,
)

MAX_SERVER_LADDER_POINTS = 8


class EstimateError(RuntimeError):
    """The pricing source could not quote the requested trade."""


@dataclass(frozen=True, slots=True)
class CostQuote:
    """A priced trade. ``amount_millisats`` is the cost of a buy or the gross
    proceeds of a sell; commission is reported separately in both cases."""

    amount_millisats: Millisats
    commission_millisats: Millisats
    source: Literal["server", "local"]


class CostEstimator(Protocol):
    async def quote_buy(self, topic_id: int, legs: Sequence[Leg]) -> CostQuote: ...

    async def quote_sell(self, topic_id: int, legs: Sequence[Leg]) -> CostQuote: ...


class ServerEstimator:
    """Prices trades through Glimpse's side-effect-free ``/trades/estimate``."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def _estimate(
        self, topic_id: int, trade_type: TradeType, legs: Sequence[Leg]
    ) -> CostQuote:
        response = await self._client.estimate_trade(
            topic_id,
            trade_type,
            [TradeLeg(option_id=option_id, contracts=contracts) for option_id, contracts in legs],
        )
        if response.total_cost_millisats is None:
            raise EstimateError(f"estimate for topic {topic_id} returned no cost")
        return CostQuote(
            amount_millisats=Millisats(abs(int(response.total_cost_millisats))),
            commission_millisats=Millisats(int(response.commission_millisats or 0)),
            source="server",
        )

    async def quote_buy(self, topic_id: int, legs: Sequence[Leg]) -> CostQuote:
        return await self._estimate(topic_id, TradeType.BUY, legs)

    async def quote_sell(self, topic_id: int, legs: Sequence[Leg]) -> CostQuote:
        return await self._estimate(topic_id, TradeType.SELL, legs)


class LocalEstimator:
    """Prices trades on the local LS-LMSR replica (backtests and pre-screening)."""

    def __init__(
        self,
        book_for: Callable[[int], LmsrBook | None],
        fee_rate: float,
        min_buy_commission: Millisats,
    ) -> None:
        self._book_for = book_for
        self._fee_rate = fee_rate
        self._min_buy_commission = min_buy_commission

    def _book(self, topic_id: int) -> LmsrBook:
        book = self._book_for(topic_id)
        if book is None:
            raise EstimateError(f"no local book for topic {topic_id}")
        return book

    def buy(self, book: LmsrBook, legs: Sequence[Leg]) -> CostQuote:
        cost = book.buy_cost(legs)
        return CostQuote(
            amount_millisats=cost,
            commission_millisats=buy_commission(cost, self._fee_rate, self._min_buy_commission),
            source="local",
        )

    def sell(self, book: LmsrBook, legs: Sequence[Leg]) -> CostQuote:
        proceeds = book.sell_proceeds(legs)
        return CostQuote(
            amount_millisats=proceeds,
            commission_millisats=sell_commission(proceeds, self._fee_rate),
            source="local",
        )

    async def quote_buy(self, topic_id: int, legs: Sequence[Leg]) -> CostQuote:
        return self.buy(self._book(topic_id), legs)

    async def quote_sell(self, topic_id: int, legs: Sequence[Leg]) -> CostQuote:
        return self.sell(self._book(topic_id), legs)


def settlement_value(probability: float, fee_rate: float) -> PriceUnits:
    """Expected settlement value of one contract in price points, net of the
    settlement fee: ``100 * p * (1 - fee)``."""
    return PriceUnits(float(PAYOUT_PRICE) * probability * (1.0 - fee_rate))


def kelly_fraction(probability: float, all_in_price: PriceUnits, fee_rate: float) -> float:
    """Full-Kelly bankroll fraction for a binary contract.

    A contract bought at ``c`` pays ``pi = 100 (1 - fee)`` with probability ``p``
    and nothing otherwise, so the net odds are ``(pi - c) / c`` and

        f* = (p * pi - c) / (pi - c),   floored at zero.
    """
    payout = float(PAYOUT_PRICE) * (1.0 - fee_rate)
    price = float(all_in_price)
    if price <= 0.0 or price >= payout:
        return 0.0
    return max(0.0, (probability * payout - price) / (payout - price))


def ladder_point(
    contracts: float, quote: CostQuote, probability: float, fee_rate: float
) -> LadderPoint:
    """Economics of buying ``contracts`` at a quoted cost."""
    all_in = Millisats(int(quote.amount_millisats) + int(quote.commission_millisats))
    exec_price = millisats_to_price_points(all_in, contracts)
    fair = settlement_value(probability, fee_rate)
    expected_value = price_points_to_millisats(fair, contracts)
    return LadderPoint(
        contracts=contracts,
        cost_millisats=quote.amount_millisats,
        commission_millisats=quote.commission_millisats,
        exec_price=exec_price,
        net_edge_points=PriceUnits(float(fair) - float(exec_price)),
        expected_profit_millisats=Millisats(int(expected_value) - int(all_in)),
        source=quote.source,
    )


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    plan: TradePlan | None
    ladder: tuple[LadderPoint, ...]
    """Every server-confirmed rung (or local rungs when the screen rejected early)."""
    local_ladder: tuple[LadderPoint, ...]
    rejection: str | None


class SlippageOptimizer:
    """Chooses the contract count ``k*`` that maximises expected net profit."""

    def __init__(self, settings: Settings, estimator: CostEstimator) -> None:
        self._settings = settings
        self._estimator = estimator
        self._local = LocalEstimator(
            lambda _topic_id: None,
            settings.fee_rate,
            Millisats(settings.min_buy_commission_millisats),
        )

    def candidate_sizes(self, signal: BucketSignal, book: LmsrBook | None) -> list[float]:
        """The configured ladder plus, when the book is known, sizes around the
        analytic optimum (where the marginal all-in price reaches fair value)."""
        settings = self._settings
        sizes = {size for size in settings.ladder if size <= settings.ladder_max_contracts}
        if book is not None and signal.option_id in book:
            fair = settlement_value(signal.model_probability, settings.fee_rate)
            limit = PriceUnits(float(fair) / (1.0 + settings.fee_rate))
            optimum = max_contracts_below_price(
                book, signal.option_id, limit, upper=settings.ladder_max_contracts
            )
            for scale in (0.5, 0.75, 1.0):
                size = float(int(optimum * scale))
                if 1.0 <= size <= settings.ladder_max_contracts:
                    sizes.add(size)
        return sorted(sizes)

    def _passes(self, point: LadderPoint, budget: Millisats) -> str | None:
        settings = self._settings
        if int(point.all_in_millisats) > int(budget):
            return "cost exceeds the Kelly / risk budget"
        if float(point.net_edge_points) < settings.min_net_edge_points:
            return (
                f"net edge {float(point.net_edge_points):.2f} below "
                f"{settings.min_net_edge_points:.2f} points"
            )
        if int(point.expected_profit_millisats) < int(settings.min_expected_gain_millisats):
            return (
                f"expected gain {int(point.expected_profit_millisats) / 1000:.1f} sats below "
                f"{settings.min_expected_gain_sats:.0f} sats"
            )
        return None

    async def optimize(
        self,
        signal: BucketSignal,
        *,
        book: LmsrBook | None,
        bankroll: Millisats,
        max_cost: Millisats,
        kelly_multiplier: float = 1.0,
        full_server_ladder: bool = False,
    ) -> OptimizationResult:
        settings = self._settings
        probability = signal.model_probability
        if float(signal.edge_points) < settings.min_raw_edge_points:
            return OptimizationResult(None, (), (), "raw edge below threshold")

        marginal_all_in = PriceUnits(float(signal.market_price) * (1.0 + settings.fee_rate))
        full_kelly = kelly_fraction(probability, marginal_all_in, settings.fee_rate)
        fraction = settings.kelly_fraction * kelly_multiplier * full_kelly
        budget = Millisats(min(int(max_cost), int(int(bankroll) * fraction)))
        if int(budget) <= 0:
            return OptimizationResult(None, (), (), "no Kelly or risk budget for this bucket")

        sizes = self.candidate_sizes(signal, book)
        local_points: list[LadderPoint] = []
        if book is not None and signal.option_id in book:
            local_points = [
                ladder_point(
                    size,
                    self._local.buy(book, [(signal.option_id, size)]),
                    probability,
                    settings.fee_rate,
                )
                for size in sizes
            ]

        if full_server_ladder or not local_points:
            confirm = sizes[:MAX_SERVER_LADDER_POINTS]
        else:
            viable = [point for point in local_points if self._passes(point, budget) is None]
            if not viable:
                best = max(local_points, key=lambda point: int(point.expected_profit_millisats))
                reason = self._passes(best, budget) or "no viable size"
                return OptimizationResult(None, (), tuple(local_points), f"local screen: {reason}")
            viable.sort(key=lambda point: int(point.expected_profit_millisats), reverse=True)
            confirm = sorted(point.contracts for point in viable[: settings.server_confirm_points])

        server_points: list[LadderPoint] = []
        for size in confirm:
            quote = await self._estimator.quote_buy(signal.topic_id, [(signal.option_id, size)])
            point = ladder_point(size, quote, probability, settings.fee_rate)
            server_points.append(point)
            if int(point.all_in_millisats) > int(budget) and not full_server_ladder:
                break

        feasible = [point for point in server_points if self._passes(point, budget) is None]
        if not feasible:
            best = max(server_points, key=lambda point: int(point.expected_profit_millisats))
            return OptimizationResult(
                None,
                tuple(server_points),
                tuple(local_points),
                self._passes(best, budget) or "no viable size",
            )

        chosen = max(feasible, key=lambda point: int(point.expected_profit_millisats))
        plan = TradePlan(
            topic_id=signal.topic_id,
            option_id=signal.option_id,
            bucket_name=signal.name,
            side=Side.BUY,
            contracts=chosen.contracts,
            cost_millisats=chosen.cost_millisats,
            commission_millisats=chosen.commission_millisats,
            exec_price=chosen.exec_price,
            market_price=signal.market_price,
            model_probability=probability,
            net_edge_points=chosen.net_edge_points,
            expected_profit_millisats=chosen.expected_profit_millisats,
            kelly_fraction=fraction,
            budget_millisats=budget,
            ladder=tuple(server_points),
            rationale=(
                f"model {100.0 * probability:.1f} vs market {float(signal.market_price):.1f}; "
                f"k*={chosen.contracts:g} of {len(server_points)} confirmed sizes maximises "
                f"expected profit at {float(chosen.exec_price):.2f} all-in"
            ),
        )
        return OptimizationResult(plan, tuple(server_points), tuple(local_points), None)


@dataclass(frozen=True, slots=True)
class ExitDecision:
    should_exit: bool
    reason: Literal["take_profit", "invalidation", "hold"]
    detail: str


def evaluate_exit(
    position: Position,
    model_probability: float,
    market_price: PriceUnits,
    net_proceeds: Millisats,
    settings: Settings,
) -> ExitDecision:
    """Active convergence exit rules.

    * **Invalidation** -- the model's probability has fallen more than
      ``invalidation_drop_points`` below the average entry price: the thesis is
      gone, cut the position.
    * **Take profit** -- the market has converged to within
      ``take_profit_gap_points`` of the model *and* selling now nets more than
      was paid: the remaining edge no longer pays for the variance of holding
      to a binary settlement, so bank it and recycle the bankroll.
    """
    model_points = 100.0 * model_probability
    entry = float(position.entry_price)
    if model_points < entry - settings.invalidation_drop_points:
        return ExitDecision(
            True,
            "invalidation",
            f"model {model_points:.1f} is more than {settings.invalidation_drop_points:.1f} "
            f"points below entry {entry:.1f}",
        )
    converged = float(market_price) >= model_points - settings.take_profit_gap_points
    if converged and int(net_proceeds) > int(position.cost_basis_millisats):
        return ExitDecision(
            True,
            "take_profit",
            f"market {float(market_price):.1f} converged to model {model_points:.1f}; "
            f"net proceeds {int(net_proceeds) / 1000:.1f} sats exceed cost "
            f"{int(position.cost_basis_millisats) / 1000:.1f} sats",
        )
    return ExitDecision(False, "hold", "edge remains")
