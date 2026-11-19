"""Pre-trade risk gatekeeper, drawdown circuit breaker and state reconciler.

Nothing reaches the execution router without passing :meth:`RiskManager.check_entry`.
The gatekeeper enforces, in order:

* the **trading lock** -- set whenever exchange state is uncertain;
* the **daily drawdown kill-switch** -- a 6% fall from the UTC day's equity
  peak halts new entries until the next UTC day or a manual reset;
* **exposure caps** -- 15% of equity per bucket, 35% per topic (correlated
  buckets of one expiry);
* **flow control** -- orders per minute, per-bucket re-entry cooldown, maximum
  number of open positions, time-to-expiry window.

Exits are never blocked by the breaker or the caps: reducing risk is always
allowed. Only the trading lock stops them, because while it is set the bot does
not know what it holds.

Reconciliation
--------------
Glimpse has no idempotency key. If an enter/exit call dies at the network
layer the SDK raises ``GlimpseAmbiguousTradeStateError`` and the order may or
may not have executed. The bot never retries. It locks, fetches
``portfolio_active()``, diffs it against the positions the append-only ledger
implies, writes one adjustment row per difference, and only then unlocks. If
the portfolio cannot be fetched the lock stays on and reconciliation is retried
on the next timer tick.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import httpx
import structlog
from glimpse_markets import GlimpseError
from glimpse_markets.models import PortfolioListResponse
from glimpse_markets.money import Millisats

from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.ledger import Ledger, PositionKey
from glimpse_quant_bot.types import (
    BasketPlan,
    Mode,
    Position,
    RecordKind,
    Side,
    TopicMeta,
    TradePlan,
)

log = structlog.get_logger(__name__)

SHARE_TOLERANCE = 1.0e-6


class PortfolioSource(Protocol):
    async def portfolio_active(self) -> PortfolioListResponse: ...


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """What the bot was trying to do when a call became ambiguous."""

    side: Side
    topic_id: int
    option_id: int
    bucket_name: str
    contracts: float
    expected_cash_flow_millisats: Millisats
    """Signed: negative for an entry, positive for an exit."""
    model_probability: float


@dataclass(frozen=True, slots=True)
class Discrepancy:
    topic_id: int
    option_id: int
    ledger_contracts: float
    actual_contracts: float

    @property
    def delta(self) -> float:
        return self.actual_contracts - self.ledger_contracts


@dataclass(frozen=True, slots=True)
class BreakerState:
    """The drawdown breaker's memory for one UTC day (see :meth:`RiskManager.breaker_state`)."""

    day: str | None
    day_peak_millisats: Millisats
    reason: str | None


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    ok: bool
    """Whether the exchange state could be fetched and the ledger aligned."""
    landed: bool | None
    """For an ambiguous trade: did it execute? ``None`` when unknown or not applicable."""
    discrepancies: tuple[Discrepancy, ...]
    note: str


class RiskManager:
    def __init__(
        self,
        settings: Settings,
        ledger: Ledger,
        mode: Mode,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._ledger = ledger
        self._mode: Mode = mode
        self._clock = clock
        self._lock_reason: str | None = None
        self._breaker_reason: str | None = None
        self._day: str | None = None
        self._day_peak = 0
        self._order_times: deque[float] = deque()
        self._last_order: dict[PositionKey, float] = {}

    # --- trading lock -------------------------------------------------------

    @property
    def locked(self) -> bool:
        return self._lock_reason is not None

    @property
    def lock_reason(self) -> str | None:
        return self._lock_reason

    def lock(self, reason: str) -> None:
        if self._lock_reason is None:
            log.error("risk.locked", reason=reason)
            self._ledger.record_event(
                self._mode, RecordKind.RISK_EVENT, f"trading locked: {reason}"
            )
        self._lock_reason = reason

    def unlock(self, note: str) -> None:
        if self._lock_reason is not None:
            log.warning("risk.unlocked", note=note)
            self._ledger.record_event(
                self._mode, RecordKind.RISK_EVENT, f"trading unlocked: {note}"
            )
        self._lock_reason = None

    # --- daily drawdown kill-switch -----------------------------------------

    @property
    def breaker_tripped(self) -> bool:
        return self._breaker_reason is not None

    @property
    def breaker_reason(self) -> str | None:
        return self._breaker_reason

    @property
    def day_peak_millisats(self) -> Millisats:
        return Millisats(self._day_peak)

    def update_equity(self, equity: Millisats, now: float | None = None) -> bool:
        """Feed the latest equity mark; returns ``True`` if the breaker is tripped."""
        moment = self._clock() if now is None else now
        day = datetime.fromtimestamp(moment, tz=UTC).strftime("%Y-%m-%d")
        if day != self._day:
            if self._breaker_reason is not None:
                self._ledger.record_event(
                    self._mode, RecordKind.RISK_EVENT, f"drawdown breaker reset for {day}"
                )
            self._day = day
            self._day_peak = int(equity)
            self._breaker_reason = None
        self._day_peak = max(self._day_peak, int(equity))
        if self._breaker_reason is None and self._day_peak > 0:
            drawdown = 1.0 - int(equity) / self._day_peak
            if drawdown >= self._settings.daily_drawdown_limit:
                self._breaker_reason = (
                    f"intraday drawdown {100.0 * drawdown:.2f}% reached the "
                    f"{100.0 * self._settings.daily_drawdown_limit:.1f}% limit"
                )
                log.error("risk.breaker_tripped", reason=self._breaker_reason)
                self._ledger.record_event(
                    self._mode,
                    RecordKind.RISK_EVENT,
                    f"drawdown breaker tripped: {self._breaker_reason}",
                    payload={"equity_millisats": int(equity), "peak_millisats": self._day_peak},
                )
        return self.breaker_tripped

    def reset_breaker(self, note: str = "manual reset") -> None:
        if self._breaker_reason is not None:
            self._ledger.record_event(
                self._mode, RecordKind.RISK_EVENT, f"drawdown breaker reset: {note}"
            )
        self._breaker_reason = None
        self._day_peak = 0
        self._day = None

    def breaker_state(self) -> BreakerState:
        """Snapshot of the day's peak and breaker, to carry across a restart."""
        return BreakerState(self._day, Millisats(self._day_peak), self._breaker_reason)

    def restore_breaker(self, state: BreakerState, now: float | None = None) -> bool:
        """Re-apply a snapshot taken earlier the same UTC day.

        A process that stops and starts again within one day must not get a
        fresh peak (and an untripped breaker) for free. Snapshots from an
        earlier day are ignored, exactly as :meth:`update_equity` would reset
        them. Returns ``True`` if the snapshot was applied.
        """
        moment = self._clock() if now is None else now
        today = datetime.fromtimestamp(moment, tz=UTC).strftime("%Y-%m-%d")
        if state.day != today:
            return False
        self._day = state.day
        self._day_peak = max(self._day_peak, int(state.day_peak_millisats))
        if state.reason is not None and self._breaker_reason is None:
            self._breaker_reason = state.reason
            log.warning("risk.breaker_restored", reason=state.reason)
        return True

    @property
    def entries_allowed(self) -> bool:
        return not self.locked and not self.breaker_tripped

    # --- exposure caps ------------------------------------------------------

    def entry_budget(
        self,
        topic_id: int,
        option_id: int,
        equity: Millisats,
        cash: Millisats,
        positions: Mapping[PositionKey, Position],
    ) -> Millisats:
        """Most that may still be spent on this bucket under every cap."""
        bucket_exposure = 0
        topic_exposure = 0
        for (held_topic, held_option), position in positions.items():
            if held_topic != topic_id:
                continue
            topic_exposure += int(position.cost_basis_millisats)
            if held_option == option_id:
                bucket_exposure += int(position.cost_basis_millisats)
        bucket_room = int(int(equity) * self._settings.max_bucket_fraction) - bucket_exposure
        topic_room = int(int(equity) * self._settings.max_topic_fraction) - topic_exposure
        return Millisats(max(0, min(bucket_room, topic_room, int(cash))))

    def check_entry(
        self,
        plan: TradePlan,
        topic: TopicMeta,
        equity: Millisats,
        cash: Millisats,
        positions: Mapping[PositionKey, Position],
        now: float | None = None,
        *,
        per_trade_thresholds: bool = True,
    ) -> RiskDecision:
        """Final go/no-go for an entry. Every failed rule is reported.

        ``per_trade_thresholds=False`` skips the net-edge and expected-gain rules
        (a basket leg is judged by the basket's total, see :meth:`check_basket`).
        """
        settings = self._settings
        moment = self._clock() if now is None else now
        reasons: list[str] = []
        key = (plan.topic_id, plan.option_id)

        if self.locked:
            reasons.append(f"trading locked: {self._lock_reason}")
        if self.breaker_tripped:
            reasons.append(f"drawdown breaker: {self._breaker_reason}")
        if plan.side is not Side.BUY:
            reasons.append("entries are buy-only")
        if plan.contracts <= 0 or plan.contracts > settings.ladder_max_contracts:
            reasons.append(
                f"size {plan.contracts:g} outside (0, {settings.ladder_max_contracts:g}]"
            )

        remaining = topic.seconds_to_expiry(moment)
        if remaining < settings.min_seconds_to_expiry:
            reasons.append(f"only {remaining:.0f}s to expiry")
        if remaining > settings.max_horizon_hours * 3600.0:
            reasons.append("expiry beyond the forecasting horizon")

        budget = self.entry_budget(plan.topic_id, plan.option_id, equity, cash, positions)
        if int(plan.all_in_millisats) > int(budget):
            reasons.append(
                f"cost {int(plan.all_in_millisats) / 1000:.0f} sats exceeds the remaining "
                f"bucket/topic/cash budget of {int(budget) / 1000:.0f} sats"
            )
        if per_trade_thresholds:
            if float(plan.net_edge_points) < settings.min_net_edge_points:
                reasons.append("net edge below threshold")
            if int(plan.expected_profit_millisats) < int(settings.min_expected_gain_millisats):
                reasons.append("expected gain below threshold")
        if key not in positions and len(positions) >= settings.max_open_positions:
            reasons.append(f"{len(positions)} open positions is the maximum")

        while self._order_times and moment - self._order_times[0] >= 60.0:
            self._order_times.popleft()
        if len(self._order_times) >= settings.max_orders_per_minute:
            reasons.append("order rate limit reached")
        last = self._last_order.get(key)
        if last is not None and moment - last < settings.reentry_cooldown_seconds:
            reasons.append("re-entry cooldown active for this bucket")

        decision = RiskDecision(approved=not reasons, reasons=tuple(reasons))
        if not decision.approved:
            log.info("risk.entry_rejected", topic_id=plan.topic_id, reasons=reasons)
        return decision

    def check_basket(
        self,
        basket: BasketPlan,
        topic: TopicMeta,
        equity: Millisats,
        cash: Millisats,
        positions: Mapping[PositionKey, Position],
        now: float | None = None,
        *,
        require_gain: bool = True,
    ) -> RiskDecision:
        """Every leg against every cap, legs accumulated as they would be held;
        the gain threshold applies to the basket as a whole (``require_gain=False``
        for one topic's group of a cross-topic cover, whose profit is joint)."""
        held: dict[PositionKey, Position] = dict(positions)
        reasons: list[str] = []
        for leg in basket.legs:
            decision = self.check_entry(
                leg, topic, equity, cash, held, now, per_trade_thresholds=False
            )
            reasons.extend(f"{leg.bucket_name}: {reason}" for reason in decision.reasons)
            key = (leg.topic_id, leg.option_id)
            previous = held.get(key)
            held[key] = Position(
                topic_id=leg.topic_id,
                option_id=leg.option_id,
                bucket_name=leg.bucket_name,
                contracts=leg.contracts + (previous.contracts if previous else 0.0),
                cost_basis_millisats=Millisats(
                    int(leg.all_in_millisats)
                    + (int(previous.cost_basis_millisats) if previous else 0)
                ),
                entry_model_probability=leg.model_probability,
            )
            cash = Millisats(int(cash) - int(leg.all_in_millisats))
        if require_gain and int(basket.expected_profit_millisats) < int(
            self._settings.min_expected_gain_millisats
        ):
            reasons.append("basket expected gain below threshold")
        unique = tuple(dict.fromkeys(reasons))
        return RiskDecision(approved=not unique, reasons=unique)

    def note_order(self, topic_id: int, option_id: int, now: float | None = None) -> None:
        """Record that an order was sent (feeds the flow-control rules)."""
        moment = self._clock() if now is None else now
        self._order_times.append(moment)
        self._last_order[(topic_id, option_id)] = moment

    def note_basket(self, keys: Collection[PositionKey], now: float | None = None) -> None:
        """One order (one request) that touched several buckets."""
        moment = self._clock() if now is None else now
        self._order_times.append(moment)
        for key in keys:
            self._last_order[key] = moment

    # --- reconciliation -----------------------------------------------------

    async def reconcile(
        self,
        source: PortfolioSource,
        intent: TradeIntent | Sequence[TradeIntent] | None = None,
        skip_topics: Collection[int] = (),
    ) -> ReconcileReport:
        """Align the ledger with the exchange's open positions.

        Safe to call at any time. Trading is locked for the duration and stays
        locked if the exchange state cannot be read. ``skip_topics`` excludes
        markets that have ended and are waiting for settlement: they leave the
        *active* portfolio before the ledger books their payout, which is not a
        discrepancy.
        """
        if not self.locked:
            self.lock("reconciling against portfolio_active()")
        try:
            portfolio = await source.portfolio_active()
        except (GlimpseError, httpx.HTTPError) as exc:
            note = f"portfolio_active() failed, trading stays locked: {exc!r}"
            log.error("risk.reconcile_failed", error=repr(exc))
            self._ledger.record_event(self._mode, RecordKind.RECONCILE, note)
            return ReconcileReport(ok=False, landed=None, discrepancies=(), note=note)

        actual: dict[PositionKey, float] = {}
        names: dict[PositionKey, str] = {}
        for item in portfolio.message or []:
            if item.topic_id is None or item.option_id is None or not item.shares:
                continue
            key = (item.topic_id, item.option_id)
            actual[key] = actual.get(key, 0.0) + float(item.shares)
            names[key] = item.option_name or names.get(key, "")

        expected = self._ledger.positions(self._mode)
        if intent is None:
            intents: list[TradeIntent] = []
        elif isinstance(intent, TradeIntent):
            intents = [intent]
        else:
            intents = list(intent)
        by_key = {(i.topic_id, i.option_id): i for i in intents}
        discrepancies: list[Discrepancy] = []
        for key in sorted(set(actual) | set(expected)):
            if key[0] in skip_topics:
                continue
            ledger_contracts = expected[key].contracts if key in expected else 0.0
            actual_contracts = actual.get(key, 0.0)
            delta = actual_contracts - ledger_contracts
            if abs(delta) <= SHARE_TOLERANCE:
                continue
            discrepancies.append(Discrepancy(key[0], key[1], ledger_contracts, actual_contracts))
            cash_flow = Millisats(0)
            matched = by_key.get(key)
            if matched is not None and matched.contracts > 0:
                signed = matched.contracts if matched.side is Side.BUY else -matched.contracts
                if delta * signed > 0:
                    share = min(1.0, abs(delta) / matched.contracts)
                    cash_flow = Millisats(round(int(matched.expected_cash_flow_millisats) * share))
            bucket_name = names.get(key) or (expected[key].bucket_name if key in expected else "")
            self._ledger.record_adjustment(
                self._mode,
                key[0],
                key[1],
                bucket_name,
                delta,
                cash_flow,
                note=(
                    f"reconciled to exchange: ledger {ledger_contracts:g} -> actual "
                    f"{actual_contracts:g} contracts"
                ),
            )

        landed: bool | None = None
        if intents:
            outcomes = []
            for planned in intents:
                signed = planned.contracts if planned.side is Side.BUY else -planned.contracts
                match = next(
                    (
                        d
                        for d in discrepancies
                        if (d.topic_id, d.option_id) == (planned.topic_id, planned.option_id)
                    ),
                    None,
                )
                outcomes.append(match is not None and match.delta * signed > 0)
            landed = all(outcomes)

        first = intents[0] if intents else None
        note = f"{len(discrepancies)} discrepancy(ies) adjusted" + (
            ""
            if first is None
            else f"; ambiguous {first.side.value} "
            f"{'executed' if landed else 'did not (fully) execute'}"
            + (f" ({len(intents)} legs)" if len(intents) > 1 else "")
        )
        log.warning(
            "risk.reconciled",
            discrepancies=[
                {
                    "topic_id": d.topic_id,
                    "option_id": d.option_id,
                    "ledger": d.ledger_contracts,
                    "actual": d.actual_contracts,
                }
                for d in discrepancies
            ],
            landed=landed,
        )
        self._ledger.record_event(
            self._mode,
            RecordKind.RECONCILE,
            note,
            payload={
                "landed": landed,
                "discrepancies": [
                    {
                        "topic_id": d.topic_id,
                        "option_id": d.option_id,
                        "ledger_contracts": d.ledger_contracts,
                        "actual_contracts": d.actual_contracts,
                    }
                    for d in discrepancies
                ],
            },
            topic_id=None if first is None else first.topic_id,
            option_id=None if first is None else first.option_id,
        )
        self.unlock(f"reconciliation complete: {note}")
        return ReconcileReport(
            ok=True, landed=landed, discrepancies=tuple(discrepancies), note=note
        )
