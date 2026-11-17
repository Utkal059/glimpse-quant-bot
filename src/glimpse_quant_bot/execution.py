"""Execution router: the only module that calls a trade-mutating SDK method.

Invariants enforced here:

* **Dry run by default.** Every call passes ``dry_run`` explicitly from the
  router's mode, so a client that was accidentally built with
  ``dry_run=False`` still cannot place an order unless the mode is ``"live"``.
* **Buy-only entries** through ``enter_multi_topic_multi_leg``; exits through
  ``exit_consolidated`` with an explicit share count.
* **Zero blind retries.** ``GlimpseAmbiguousTradeStateError`` locks trading and
  hands over to :meth:`RiskManager.reconcile`. The order is never resubmitted
  by this module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from glimpse_markets import (
    AsyncClient,
    DryRunTradeResult,
    EnterMultiTopicLegGroup,
    GlimpseAmbiguousTradeStateError,
    GlimpseAPIError,
    GlimpseAuthenticationError,
    GlimpseTradingNotEligibleError,
    TradeLeg,
)
from glimpse_markets.money import Millisats, PriceUnits

from glimpse_quant_bot.ledger import Ledger
from glimpse_quant_bot.lmsr_optimizer import CostQuote
from glimpse_quant_bot.risk_manager import RiskManager, TradeIntent
from glimpse_quant_bot.types import LedgerRecord, Mode, Position, RecordKind, Side, TradePlan

log = structlog.get_logger(__name__)

_PROCEEDS_KEYS = ("total_proceeds_millisats", "proceeds_millisats", "total_cost_millisats")
_COMMISSION_KEYS = ("total_commission_millisats", "commission_millisats")
_TRADE_ID_KEYS = ("trade_id", "bet_slip_uuid", "uuid")


def _find(payload: Any, keys: tuple[str, ...]) -> Any:
    """Depth-first search of a loosely specified JSON response for the first of ``keys``."""
    if isinstance(payload, dict):
        for key in keys:
            if payload.get(key) is not None:
                return payload[key]
        for value in payload.values():
            found = _find(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find(value, keys)
            if found is not None:
                return found
    return None


class ExecutionRouter:
    def __init__(
        self,
        client: AsyncClient,
        ledger: Ledger,
        risk: RiskManager,
        mode: Mode,
        on_trade: Callable[[], None] | None = None,
    ) -> None:
        self._client = client
        self._ledger = ledger
        self._risk = risk
        self._mode: Mode = mode
        self._on_trade = on_trade

    @property
    def dry_run(self) -> bool:
        return self._mode != "live"

    def _after_trade(self) -> None:
        if self._on_trade is not None:
            self._on_trade()

    async def _ambiguous(self, intent: TradeIntent, exc: GlimpseAmbiguousTradeStateError) -> None:
        """Lock, reconcile against the exchange, and never resubmit."""
        self._risk.lock(f"ambiguous trade state on {exc.path}")
        self._ledger.record_event(
            self._mode,
            RecordKind.RISK_EVENT,
            f"GlimpseAmbiguousTradeStateError on {exc.path}; order NOT retried",
            payload={
                "side": intent.side.value,
                "contracts": intent.contracts,
                "error": repr(exc.original),
            },
            topic_id=intent.topic_id,
            option_id=intent.option_id,
        )
        self._after_trade()
        report = await self._risk.reconcile(self._client, intent)
        self._after_trade()
        log.warning(
            "execution.ambiguous_resolved", ok=report.ok, landed=report.landed, note=report.note
        )

    def _api_failure(
        self, action: str, exc: GlimpseAPIError, topic_id: int, option_id: int
    ) -> None:
        self._ledger.record_event(
            self._mode,
            RecordKind.RISK_EVENT,
            f"{action} rejected by API: {exc}",
            topic_id=topic_id,
            option_id=option_id,
        )
        log.error("execution.api_error", action=action, error=str(exc))
        if isinstance(exc, GlimpseAuthenticationError | GlimpseTradingNotEligibleError):
            self._risk.lock(f"account cannot trade: {exc}")

    async def enter(self, plan: TradePlan) -> LedgerRecord | None:
        """Buy ``plan.contracts`` of one bucket. Returns the fill, or ``None``."""
        if self._risk.locked:
            return None
        if plan.side is not Side.BUY:
            raise ValueError("entries are buy-only")
        intent = TradeIntent(
            side=Side.BUY,
            topic_id=plan.topic_id,
            option_id=plan.option_id,
            bucket_name=plan.bucket_name,
            contracts=plan.contracts,
            expected_cash_flow_millisats=Millisats(-int(plan.all_in_millisats)),
            model_probability=plan.model_probability,
        )
        group = EnterMultiTopicLegGroup(
            topic_id=plan.topic_id,
            legs=[TradeLeg(option_id=plan.option_id, contracts=plan.contracts)],
        )
        self._risk.note_order(plan.topic_id, plan.option_id)
        try:
            result = await self._client.enter_multi_topic_multi_leg([group], dry_run=self.dry_run)
        except GlimpseAmbiguousTradeStateError as exc:
            await self._ambiguous(intent, exc)
            return None
        except GlimpseAPIError as exc:
            self._api_failure("entry", exc, plan.topic_id, plan.option_id)
            return None

        trade_id: str | None = None
        if isinstance(result, DryRunTradeResult):
            estimate = result.estimates[0]
            cost = Millisats(abs(int(estimate.total_cost_millisats or plan.cost_millisats)))
            commission = Millisats(int(estimate.commission_millisats or 0))
            note = "paper fill priced by /trades/estimate"
        else:
            outcomes = result.results or []
            outcome = outcomes[0] if outcomes else None
            if outcome is None or outcome.error:
                error = "no result returned" if outcome is None else str(outcome.error)
                self._ledger.record_event(
                    self._mode,
                    RecordKind.RISK_EVENT,
                    f"entry failed at the exchange: {error}",
                    topic_id=plan.topic_id,
                    option_id=plan.option_id,
                )
                log.error("execution.entry_failed", topic_id=plan.topic_id, error=error)
                return None
            cost = Millisats(abs(int(outcome.total_cost_millisats or plan.cost_millisats)))
            commission = Millisats(int(outcome.commission_millisats or plan.commission_millisats))
            trade_id = outcome.trade_id
            note = "live fill"

        record = self._ledger.record_entry(self._mode, plan, cost, commission, trade_id, note)
        self._after_trade()
        log.info(
            "execution.entry",
            mode=self._mode,
            topic_id=plan.topic_id,
            bucket=plan.bucket_name,
            contracts=plan.contracts,
            cost_sats=int(cost) / 1000,
            commission_sats=int(commission) / 1000,
        )
        return record

    async def exit(
        self,
        position: Position,
        contracts: float,
        expected: CostQuote,
        market_price: PriceUnits,
        model_probability: float,
        reason: str,
    ) -> LedgerRecord | None:
        """Sell ``contracts`` of an open position. Returns the fill, or ``None``."""
        if self._risk.locked:
            return None
        expected_net = int(expected.amount_millisats) - int(expected.commission_millisats)
        intent = TradeIntent(
            side=Side.SELL,
            topic_id=position.topic_id,
            option_id=position.option_id,
            bucket_name=position.bucket_name,
            contracts=contracts,
            expected_cash_flow_millisats=Millisats(expected_net),
            model_probability=model_probability,
        )
        self._risk.note_order(position.topic_id, position.option_id)
        try:
            result = await self._client.exit_consolidated(
                position.topic_id, position.option_id, shares=contracts, dry_run=self.dry_run
            )
        except GlimpseAmbiguousTradeStateError as exc:
            await self._ambiguous(intent, exc)
            return None
        except GlimpseAPIError as exc:
            self._api_failure("exit", exc, position.topic_id, position.option_id)
            return None

        trade_id: str | None = None
        if isinstance(result, DryRunTradeResult):
            estimate = result.estimates[0]
            proceeds = Millisats(
                abs(int(estimate.total_cost_millisats or expected.amount_millisats))
            )
            commission = Millisats(int(estimate.commission_millisats or 0))
            note = f"{reason}; paper fill priced by /trades/estimate"
        else:
            reported = _find(result, _PROCEEDS_KEYS)
            reported_commission = _find(result, _COMMISSION_KEYS)
            found_id = _find(result, _TRADE_ID_KEYS)
            trade_id = None if found_id is None else str(found_id)
            if reported is None:
                proceeds = expected.amount_millisats
                commission = expected.commission_millisats
                note = f"{reason}; live fill, proceeds taken from the pre-trade estimate"
            else:
                proceeds = Millisats(abs(int(reported)))
                commission = Millisats(
                    int(expected.commission_millisats)
                    if reported_commission is None
                    else int(reported_commission)
                )
                note = f"{reason}; live fill"

        record = self._ledger.record_exit(
            self._mode,
            position,
            contracts,
            proceeds,
            commission,
            market_price=market_price,
            model_probability=model_probability,
            trade_id=trade_id,
            note=note,
        )
        self._after_trade()
        log.info(
            "execution.exit",
            mode=self._mode,
            topic_id=position.topic_id,
            bucket=position.bucket_name,
            contracts=contracts,
            reason=reason,
            realized_sats=int(record.realized_pnl_millisats) / 1000,
        )
        return record
