"""``GlimpseAmbiguousTradeStateError`` handling, reconciliation and the drawdown breaker.

The exchange is replaced by :class:`FakeExchange`, which counts every
trade-mutating call so the tests can prove that an ambiguous order is never
submitted twice.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from glimpse_markets import (
    AsyncClient,
    DryRunTradeResult,
    EnterMultiTopicLegGroup,
    GlimpseAmbiguousTradeStateError,
    GlimpseAuthenticationError,
    GlimpseServerError,
    TradeType,
)
from glimpse_markets.models import (
    EnterMultiTopicMultiLegResponse,
    EnterMultiTopicTopicResult,
    EstimateTradeLegsResponse,
    PortfolioItem,
    PortfolioListResponse,
)
from glimpse_markets.money import Millisats, PriceUnits

from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.execution import ExecutionRouter
from glimpse_quant_bot.ledger import Ledger
from glimpse_quant_bot.lmsr_optimizer import CostQuote
from glimpse_quant_bot.risk_manager import RiskManager
from glimpse_quant_bot.types import Mode, RecordKind, Side, TopicMeta, TradePlan

from .conftest import NOW

TOPIC_ID = 5398
OPTION_ID = 200
BANKROLL = Millisats(100_000_000)


class FakeExchange:
    """Just enough of ``AsyncClient`` for the execution router and the reconciler."""

    def __init__(self) -> None:
        self.holdings: dict[tuple[int, int], float] = {}
        self.enter_calls: list[tuple[list[EnterMultiTopicLegGroup], bool | None]] = []
        self.exit_calls: list[tuple[int, int, float | None, bool | None]] = []
        self.portfolio_calls = 0
        self.enter_error: Exception | None = None
        self.exit_error: Exception | None = None
        self.portfolio_error: Exception | None = None
        self.fill_before_failing = False

    async def enter_multi_topic_multi_leg(
        self, topics: list[EnterMultiTopicLegGroup], *, dry_run: bool | None = None
    ) -> EnterMultiTopicMultiLegResponse | DryRunTradeResult:
        self.enter_calls.append((topics, dry_run))
        group = topics[0]
        leg = group.legs[0]
        if self.enter_error is not None:
            if self.fill_before_failing:
                key = (group.topic_id, leg.option_id)
                self.holdings[key] = self.holdings.get(key, 0.0) + leg.contracts
            raise self.enter_error
        estimate = EstimateTradeLegsResponse(
            topic_id=group.topic_id,
            trade_type="buy",
            total_cost_millisats=Millisats(200_000),
            commission_millisats=Millisats(4_000),
        )
        if dry_run:
            return DryRunTradeResult(trade_type=TradeType.BUY, estimates=[estimate])
        key = (group.topic_id, leg.option_id)
        self.holdings[key] = self.holdings.get(key, 0.0) + leg.contracts
        return EnterMultiTopicMultiLegResponse(
            results=[
                EnterMultiTopicTopicResult(
                    topic_id=group.topic_id,
                    trade_id="trade-1",
                    total_cost_millisats=Millisats(200_000),
                    commission_millisats=Millisats(4_000),
                )
            ],
            topics_requested=1,
            topics_entered=1,
            topics_failed=0,
        )

    async def exit_consolidated(
        self,
        topic_id: int,
        option_id: int,
        shares: float | None = None,
        *,
        dry_run: bool | None = None,
    ) -> dict[str, Any] | DryRunTradeResult:
        self.exit_calls.append((topic_id, option_id, shares, dry_run))
        if self.exit_error is not None:
            raise self.exit_error
        if dry_run:
            return DryRunTradeResult(
                trade_type=TradeType.SELL,
                estimates=[
                    EstimateTradeLegsResponse(
                        total_cost_millisats=Millisats(-260_000),
                        commission_millisats=Millisats(5_200),
                    )
                ],
            )
        self.holdings.pop((topic_id, option_id), None)
        return {"success": True, "message": {"total_proceeds_millisats": 260_000}}

    async def portfolio_active(self) -> PortfolioListResponse:
        self.portfolio_calls += 1
        if self.portfolio_error is not None:
            raise self.portfolio_error
        return PortfolioListResponse(
            success=True,
            message=[
                PortfolioItem(topic_id=t, option_id=o, option_name="64800-65000", shares=shares)
                for (t, o), shares in self.holdings.items()
            ],
        )


def ambiguous() -> GlimpseAmbiguousTradeStateError:
    return GlimpseAmbiguousTradeStateError(
        "/api/v1/nmarket/enter-multi-topic-multi-leg", httpx.ReadTimeout("timed out")
    )


def make_plan(contracts: float = 10.0) -> TradePlan:
    return TradePlan(
        topic_id=TOPIC_ID,
        option_id=OPTION_ID,
        bucket_name="64800-65000",
        side=Side.BUY,
        contracts=contracts,
        cost_millisats=Millisats(200_000),
        commission_millisats=Millisats(4_000),
        exec_price=PriceUnits(20.4),
        market_price=PriceUnits(18.0),
        model_probability=0.30,
        net_edge_points=PriceUnits(9.0),
        expected_profit_millisats=Millisats(300_000),
        kelly_fraction=0.03,
        budget_millisats=Millisats(3_000_000),
        ladder=(),
        rationale="test",
    )


def build(
    settings: Settings, ledger: Ledger, mode: Mode = "live"
) -> tuple[FakeExchange, RiskManager, ExecutionRouter, list[int]]:
    exchange = FakeExchange()
    invalidations: list[int] = []
    risk = RiskManager(settings, ledger, mode, clock=lambda: NOW)
    router = ExecutionRouter(
        cast(AsyncClient, exchange),
        ledger,
        risk,
        mode,
        on_trade=lambda: invalidations.append(1),
    )
    return exchange, risk, router, invalidations


def contracts_held(ledger: Ledger, mode: Mode = "live") -> float:
    position = ledger.positions(mode).get((TOPIC_ID, OPTION_ID))
    return 0.0 if position is None else position.contracts


# --- ambiguous trade state ---------------------------------------------------


async def test_ambiguous_entry_that_executed_is_adopted_without_reordering(
    settings: Settings, ledger: Ledger
) -> None:
    exchange, risk, router, invalidations = build(settings, ledger)
    exchange.enter_error = ambiguous()
    exchange.fill_before_failing = True  # the order went through; only the reply was lost

    assert await router.enter(make_plan()) is None

    assert len(exchange.enter_calls) == 1, "an ambiguous order must never be resubmitted"
    assert exchange.portfolio_calls == 1
    assert contracts_held(ledger) == pytest.approx(10.0)
    assert exchange.holdings[(TOPIC_ID, OPTION_ID)] == pytest.approx(10.0), "no double order"
    # The adopted position carries the expected cost so P&L and caps stay right.
    position = ledger.positions("live")[(TOPIC_ID, OPTION_ID)]
    assert position.cost_basis_millisats == 204_000
    assert ledger.cash_millisats("live", BANKROLL) == int(BANKROLL) - 204_000
    assert not risk.locked, "trading resumes only after a successful reconciliation"
    assert invalidations, "PositionTracker must be invalidated after the trade attempt"

    notes = [record.note for record in ledger.records("live", kinds=[RecordKind.RISK_EVENT])]
    assert any("NOT retried" in note for note in notes)
    assert any(note.startswith("trading locked") for note in notes)
    assert any(note.startswith("trading unlocked") for note in notes)
    reconcile = ledger.records("live", kinds=[RecordKind.RECONCILE])[-1]
    assert ledger.payload(reconcile.record_id)["landed"] is True


async def test_ambiguous_entry_that_did_not_execute_leaves_the_ledger_flat(
    settings: Settings, ledger: Ledger
) -> None:
    exchange, risk, router, _ = build(settings, ledger)
    exchange.enter_error = ambiguous()  # the request never reached the exchange

    assert await router.enter(make_plan()) is None

    assert len(exchange.enter_calls) == 1
    assert exchange.portfolio_calls == 1
    assert contracts_held(ledger) == 0.0
    assert ledger.cash_millisats("live", BANKROLL) == BANKROLL
    assert ledger.records("live", kinds=[RecordKind.ADJUSTMENT]) == []
    assert not risk.locked
    reconcile = ledger.records("live", kinds=[RecordKind.RECONCILE])[-1]
    assert ledger.payload(reconcile.record_id)["landed"] is False


async def test_trading_stays_locked_while_the_portfolio_is_unreachable(
    settings: Settings, ledger: Ledger
) -> None:
    exchange, risk, router, _ = build(settings, ledger)
    exchange.enter_error = ambiguous()
    exchange.fill_before_failing = True
    exchange.portfolio_error = GlimpseServerError(503, "UNAVAILABLE")

    assert await router.enter(make_plan()) is None
    assert risk.locked
    assert not risk.entries_allowed

    # While locked, nothing reaches the exchange: no entry, no exit.
    exchange.enter_error = None
    assert await router.enter(make_plan()) is None
    assert len(exchange.enter_calls) == 1
    decision = risk.check_entry(
        make_plan(), _topic(), BANKROLL, BANKROLL, ledger.positions("live"), NOW
    )
    assert not decision.approved
    assert any("trading locked" in reason for reason in decision.reasons)

    # The next timer tick retries the reconciliation; once it succeeds trading resumes.
    exchange.portfolio_error = None
    report = await risk.reconcile(cast(AsyncClient, exchange))
    assert report.ok
    assert risk.lock_reason is None
    assert contracts_held(ledger) == pytest.approx(10.0)
    assert len(exchange.enter_calls) == 1, "reconciliation itself never trades"


async def test_ambiguous_exit_is_reconciled_from_the_portfolio(
    settings: Settings, ledger: Ledger
) -> None:
    exchange, risk, router, _ = build(settings, ledger)
    assert await router.enter(make_plan()) is not None
    position = ledger.positions("live")[(TOPIC_ID, OPTION_ID)]

    exchange.exit_error = GlimpseAmbiguousTradeStateError(
        "/api/v1/nmarket/exit-consolidated", httpx.ConnectError("reset")
    )
    exchange.holdings.clear()  # the sale executed before the connection dropped
    quote = CostQuote(Millisats(260_000), Millisats(5_200), "local")
    record = await router.exit(position, 10.0, quote, PriceUnits(29.0), 0.30, "take_profit")

    assert record is None
    assert len(exchange.exit_calls) == 1, "an ambiguous exit must never be resubmitted"
    assert contracts_held(ledger) == 0.0
    assert not risk.locked
    # Cash reflects the expected net proceeds of the sale that did execute.
    assert ledger.cash_millisats("live", BANKROLL) == int(BANKROLL) - 204_000 + 254_800


async def test_reconcile_detects_drift_in_both_directions(
    settings: Settings, ledger: Ledger
) -> None:
    exchange, risk, router, _ = build(settings, ledger)
    assert await router.enter(make_plan()) is not None
    exchange.holdings[(TOPIC_ID, OPTION_ID)] = 4.0  # six contracts sold elsewhere
    exchange.holdings[(7000, 12)] = 3.0  # a position the bot never opened

    report = await risk.reconcile(cast(AsyncClient, exchange))

    assert report.ok
    assert report.landed is None
    assert {(d.topic_id, d.option_id, d.delta) for d in report.discrepancies} == {
        (TOPIC_ID, OPTION_ID, -6.0),
        (7000, 12, 3.0),
    }
    positions = ledger.positions("live")
    assert positions[(TOPIC_ID, OPTION_ID)].contracts == pytest.approx(4.0)
    assert positions[(7000, 12)].contracts == pytest.approx(3.0)
    # Ended markets waiting for settlement are not treated as drift.
    exchange.holdings.clear()
    skipped = await risk.reconcile(cast(AsyncClient, exchange), skip_topics={TOPIC_ID, 7000})
    assert skipped.discrepancies == ()


# --- dry-run safety ----------------------------------------------------------


async def test_dry_run_router_never_sends_a_real_order(settings: Settings, ledger: Ledger) -> None:
    exchange, _, router, _ = build(settings, ledger, mode="dry_run")
    record = await router.enter(make_plan())
    assert record is not None
    assert [dry_run for _, dry_run in exchange.enter_calls] == [True]
    assert exchange.holdings == {}, "a paper fill must not touch the exchange"
    assert record.cash_flow_millisats == -204_000
    assert record.mode == "dry_run"

    position = ledger.positions("dry_run")[(TOPIC_ID, OPTION_ID)]
    quote = CostQuote(Millisats(260_000), Millisats(5_200), "local")
    sold = await router.exit(position, 10.0, quote, PriceUnits(29.0), 0.30, "take_profit")
    assert sold is not None
    assert exchange.exit_calls == [(TOPIC_ID, OPTION_ID, 10.0, True)]
    assert sold.realized_pnl_millisats == 254_800 - 204_000
    assert ledger.positions("dry_run") == {}


async def test_authentication_failure_locks_trading(settings: Settings, ledger: Ledger) -> None:
    exchange, risk, router, _ = build(settings, ledger)
    exchange.enter_error = GlimpseAuthenticationError(401, "UNAUTHORIZED")
    assert await router.enter(make_plan()) is None
    assert risk.locked
    assert exchange.portfolio_calls == 0  # a clean rejection is not ambiguous


async def test_per_topic_error_in_a_200_response_is_not_a_fill(
    settings: Settings, ledger: Ledger
) -> None:
    exchange, _, router, _ = build(settings, ledger)

    async def failing(
        topics: list[EnterMultiTopicLegGroup], *, dry_run: bool | None = None
    ) -> EnterMultiTopicMultiLegResponse:
        return EnterMultiTopicMultiLegResponse(
            results=[EnterMultiTopicTopicResult(topic_id=TOPIC_ID, error="INSUFFICIENT_BALANCE")],
            topics_requested=1,
            topics_entered=0,
            topics_failed=1,
        )

    exchange.enter_multi_topic_multi_leg = failing  # type: ignore[method-assign]
    assert await router.enter(make_plan()) is None
    assert ledger.positions("live") == {}


# --- drawdown kill-switch and flow control -----------------------------------


def _topic() -> TopicMeta:
    from glimpse_quant_bot.types import AssetType, Bucket

    return TopicMeta(
        TOPIC_ID,
        "batch",
        "Hourly Bitcoin",
        AssetType.BTC,
        int(NOW) + 7200,
        (Bucket(OPTION_ID, "64800-65000", 64_800.0, 65_000.0),),
    )


def test_daily_drawdown_breaker_halts_entries_until_the_next_utc_day(
    settings: Settings, ledger: Ledger
) -> None:
    risk = RiskManager(settings, ledger, "dry_run", clock=lambda: NOW)
    assert not risk.update_equity(Millisats(100_000_000), NOW)
    assert not risk.update_equity(Millisats(102_000_000), NOW + 60)  # new intraday peak
    assert not risk.update_equity(Millisats(96_500_000), NOW + 120)  # -5.4% from the peak
    assert risk.update_equity(Millisats(95_800_000), NOW + 180)  # -6.08% -> tripped
    assert risk.breaker_tripped
    assert not risk.entries_allowed

    decision = risk.check_entry(make_plan(), _topic(), BANKROLL, BANKROLL, {}, NOW + 200)
    assert not decision.approved
    assert any("drawdown breaker" in reason for reason in decision.reasons)

    # Recovering intraday does not re-arm it ...
    assert risk.update_equity(Millisats(101_000_000), NOW + 600)
    # ... the next UTC day does.
    assert not risk.update_equity(Millisats(95_800_000), NOW + 86_400)
    assert risk.breaker_reason is None
    events = [record.note for record in ledger.records("dry_run", kinds=[RecordKind.RISK_EVENT])]
    assert any("drawdown breaker tripped" in note for note in events)


def test_risk_gate_reports_every_violated_rule(settings: Settings, ledger: Ledger) -> None:
    risk = RiskManager(settings, ledger, "dry_run", clock=lambda: NOW)
    approved = risk.check_entry(make_plan(), _topic(), BANKROLL, BANKROLL, {}, NOW)
    assert approved.approved

    # Cost above the 15% bucket cap.
    oversized = make_plan()
    oversized = TradePlan(
        **{
            **{name: getattr(oversized, name) for name in oversized.__slots__},
            "cost_millisats": Millisats(16_000_000),
        }
    )
    rejected = risk.check_entry(oversized, _topic(), BANKROLL, BANKROLL, {}, NOW)
    assert not rejected.approved
    assert any("exceeds the remaining" in reason for reason in rejected.reasons)

    # Too close to expiry.
    late = risk.check_entry(make_plan(), _topic(), BANKROLL, BANKROLL, {}, NOW + 7200 - 60)
    assert any("to expiry" in reason for reason in late.reasons)

    # Re-entry cooldown and order-rate limit.
    risk.note_order(TOPIC_ID, OPTION_ID, NOW)
    cooling = risk.check_entry(make_plan(), _topic(), BANKROLL, BANKROLL, {}, NOW + 5)
    assert any("cooldown" in reason for reason in cooling.reasons)
    for i in range(settings.max_orders_per_minute):
        risk.note_order(9000 + i, 1, NOW + 10)
    throttled = risk.check_entry(make_plan(), _topic(), BANKROLL, BANKROLL, {}, NOW + 20)
    assert any("order rate limit" in reason for reason in throttled.reasons)


def test_ledger_is_append_only(ledger: Ledger) -> None:
    import sqlite3

    ledger.record_event("dry_run", RecordKind.RISK_EVENT, "hello")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger._db.execute("UPDATE records SET note = 'tampered'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger._db.execute("DELETE FROM records")
    assert [record.note for record in ledger.records("dry_run")] == ["hello"]


def test_settlement_books_the_binary_payout(settings: Settings, ledger: Ledger) -> None:
    ledger.record_entry("dry_run", make_plan(), Millisats(200_000), Millisats(4_000))
    position = ledger.positions("dry_run")[(TOPIC_ID, OPTION_ID)]
    won = ledger.record_settlement("dry_run", position, won=True, fee_rate=settings.fee_rate)
    # 10 contracts x 100 points x (1 - 2%) = 980 sats.
    assert won.cash_flow_millisats == 980_000
    assert won.realized_pnl_millisats == 980_000 - 204_000
    assert ledger.positions("dry_run") == {}
    summary = ledger.summary("dry_run", BANKROLL)
    assert (summary.entries, summary.settlements, summary.wins) == (1, 1, 1)
    assert summary.roi == pytest.approx(776_000 / 100_000_000)
