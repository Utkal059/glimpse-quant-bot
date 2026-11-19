"""The trading strategy, wired into the SDK's ``StrategyRunner``.

Pipeline (one direction, no back-references)::

    MarketDataHub -> EnsembleForecaster -> SlippageOptimizer -> RiskManager -> ExecutionRouter
       (feeds)          (alpha)              (sizing)          (gatekeeper)     (orders)

Event flow:

* ``on_quote`` -- every ``market_update`` from the Glimpse WebSocket replaces
  the topic's LS-LMSR book and, at most once per ``eval_interval_seconds`` per
  topic, triggers an entry evaluation. No REST request is made unless a
  candidate survives the local, zero-cost slippage screen.
* ``on_tick`` -- marks the portfolio, feeds the drawdown breaker, runs the
  convergence / invalidation exits, re-evaluates a few topics round-robin (the
  model moves with spot even when nobody trades on Glimpse) and, every
  ``resolution_poll_seconds``, refreshes the universe and settles resolved
  markets.

Rate-limit budget per minute at the defaults: ~8 universe pages, 1-3 confirming
estimates per candidate that passes the local screen, 1 call per fill, and one
``market_quotes`` per expired position awaiting resolution.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from glimpse_markets import (
    AsyncClient,
    GlimpseAPIError,
    GlimpseRateLimitError,
    MarketUpdate,
    QuoteMode,
    Strategy,
)
from glimpse_markets.forecasting.signal import parse_bucket_name
from glimpse_markets.models import BatchMarketItem, BatchMarketItemV2, MarketUpdateData
from glimpse_markets.money import Millisats

from glimpse_quant_bot.calibration import (
    CalibrationView,
    OnlineCalibrator,
    PendingForecast,
    calibration_prior,
)
from glimpse_quant_bot.coherence import cover_arbitrage, frank_wolfe_projection, linked_groups
from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.execution import ExecutionRouter
from glimpse_quant_bot.external_feeds import FeedError, MarketDataHub
from glimpse_quant_bot.forecaster import (
    EnsembleForecast,
    EnsembleForecaster,
    ForecastError,
    build_signals,
)
from glimpse_quant_bot.ledger import Ledger, PositionKey
from glimpse_quant_bot.lmsr import LmsrBook
from glimpse_quant_bot.lmsr_optimizer import (
    EstimateError,
    LocalEstimator,
    ServerEstimator,
    SlippageOptimizer,
    basket_plan,
    basket_plan_from_legs,
    evaluate_exit,
)
from glimpse_quant_bot.risk_manager import SHARE_TOLERANCE, RiskManager
from glimpse_quant_bot.robust_kelly import optimize_basket
from glimpse_quant_bot.types import (
    AssetType,
    BasketPlan,
    Bucket,
    Mode,
    Position,
    RecordKind,
    TopicMeta,
    classify_asset,
)

log = structlog.get_logger(__name__)

FORECAST_REUSE_SECONDS = 30.0
EQUITY_SNAPSHOT_SECONDS = 60.0
_WALLET_KEYS = (
    "available_balance_millisats",
    "balance_millisats",
    "wallet_balance_millisats",
    "available_balance",
    "balance",
)


@dataclass(slots=True)
class Universe:
    """Tradable topics and their latest books."""

    topics: dict[int, TopicMeta] = field(default_factory=dict)
    books: dict[int, LmsrBook] = field(default_factory=dict)
    batches: dict[str, AssetType] = field(default_factory=dict)


def build_topic(
    market: BatchMarketItem | BatchMarketItemV2, asset: AssetType, now: float
) -> tuple[TopicMeta, LmsrBook] | None:
    """Turn one API market into a ``TopicMeta`` and its LS-LMSR book.

    Returns ``None`` unless *every* outcome is a parseable ``"LO-HI"`` bucket
    with a share count: a partial book would misprice every trade.
    """
    outcomes = market.outcomes or []
    if market.topic_id is None or market.end_time_utc is None or len(outcomes) < 2:
        return None
    buckets: list[Bucket] = []
    shares: list[float] = []
    for outcome in outcomes:
        lo, hi = parse_bucket_name(outcome.name)
        if outcome.option_id is None or lo is None or hi is None or outcome.shares is None:
            return None
        buckets.append(Bucket(outcome.option_id, outcome.name or "", lo, hi))
        shares.append(float(outcome.shares))
    topic = TopicMeta(
        topic_id=market.topic_id,
        batch_id=market.batch_id or "",
        title=market.title or "",
        asset=asset,
        end_time_utc=int(market.end_time_utc),
        buckets=tuple(buckets),
    )
    book = LmsrBook.from_shares(
        topic.topic_id,
        [bucket.option_id for bucket in buckets],
        shares,
        alpha=None,  # the venue-wide GLIMPSE_ALPHA; it does not depend on bucket count
        ts=now,
    )
    return topic, book


async def discover_batches(client: AsyncClient, settings: Settings) -> dict[str, AssetType]:
    """Map every batch whose underlying the engine can price to its asset."""
    listing = await client.batches()
    batches: dict[str, AssetType] = {}
    for batch in listing.batches or []:
        asset = classify_asset(batch.main_topic_title)
        if batch.batch_id and asset is not None and asset in settings.assets:
            batches[batch.batch_id] = asset
    return batches


async def load_batch_topics(
    client: AsyncClient, batch_id: str, asset: AssetType, settings: Settings, now: float
) -> list[tuple[TopicMeta, LmsrBook]]:
    """Live topics of one batch inside the tradable time-to-expiry window.

    One request returns the 24 nearest-expiry markets *including every bucket's
    share count*, i.e. 24 complete LS-LMSR books.
    """
    page = await client.batch_active_markets_page(batch_id)
    loaded: list[tuple[TopicMeta, LmsrBook]] = []
    for market in page.markets or []:
        if market.quote_mode != QuoteMode.LIVE.value or market.is_resolved:
            continue
        if classify_asset(market.title, market.description, market.category) is not asset:
            continue
        built = build_topic(market, asset, now)
        if built is None:
            continue
        remaining = built[0].seconds_to_expiry(now)
        if settings.min_seconds_to_expiry <= remaining <= settings.max_horizon_hours * 3600.0:
            loaded.append(built)
    return loaded


def book_from_update(topic: TopicMeta, data: MarketUpdateData, now: float) -> LmsrBook | None:
    """Rebuild a topic's book from one WebSocket ``market_update``."""
    shares_by_option = {
        quote.option_id: quote.shares
        for quote in data.quotes or []
        if quote.option_id is not None and quote.shares is not None
    }
    shares: list[float] = []
    for bucket in topic.buckets:
        held = shares_by_option.get(bucket.option_id)
        if held is None:
            return None
        shares.append(float(held))
    return LmsrBook.from_shares(
        topic.topic_id,
        [bucket.option_id for bucket in topic.buckets],
        shares,
        alpha=data.alpha,
        ts=now,
    )


def parse_wallet_millisats(payload: Any) -> Millisats | None:
    """Best-effort read of the wallet balance (the endpoint has no published schema)."""
    if isinstance(payload, dict):
        for key in _WALLET_KEYS:
            value = payload.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return Millisats(int(value))
        for value in payload.values():
            found = parse_wallet_millisats(value)
            if found is not None:
                return found
    return None


class GlimpseQuantStrategy(Strategy):
    """Probability-calibration strategy over Glimpse BTC/ETH/SOL/Gold buckets."""

    def __init__(
        self,
        settings: Settings,
        mode: Mode = "dry_run",
        ledger: Ledger | None = None,
        hub: MarketDataHub | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.mode: Mode = mode
        self._clock = clock
        self._ledger = ledger or Ledger(settings.ledger_path, settings.audit_path)
        self._hub = hub or MarketDataHub(
            settings.assets, enable_websocket=settings.enable_spot_websocket
        )
        self._forecaster = EnsembleForecaster(self._hub, settings)
        self._risk = RiskManager(settings, self._ledger, mode, clock)
        self._local = LocalEstimator(
            lambda topic_id: self.universe.books.get(topic_id),
            settings.fee_rate,
            Millisats(settings.min_buy_commission_millisats),
        )
        self.universe = Universe()
        self._forecasts: dict[int, EnsembleForecast] = {}
        self._last_eval: dict[int, float] = {}
        self._router: ExecutionRouter | None = None
        self._optimizer: SlippageOptimizer | None = None
        self._wallet: Millisats | None = None
        self._paused_until = 0.0
        self._last_slow = 0.0
        self._last_equity_snapshot = 0.0
        self._round_robin = 0
        self._started = False
        self.fills = 0
        self._calibrator = (
            OnlineCalibrator(
                prior=calibration_prior(settings),
                eta=settings.calibration_eta,
                share=settings.calibration_share,
                min_resolved=settings.calibration_min_resolved,
                path=settings.calibration_path,
            )
            if settings.enable_calibration
            else None
        )
        self._resolution_polled: dict[str, float] = {}

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    @property
    def risk(self) -> RiskManager:
        return self._risk

    # --- lifecycle ----------------------------------------------------------

    async def on_start(self) -> None:
        now = self._clock()
        self._router = ExecutionRouter(
            self.client, self._ledger, self._risk, self.mode, on_trade=self.positions.invalidate
        )
        self._optimizer = SlippageOptimizer(self.settings, ServerEstimator(self.client))
        if not self._started:
            await self._hub.start()
            self._started = True
        self.universe.batches = await discover_batches(self.client, self.settings)
        await self._refresh_universe(now)
        if self.mode == "live":
            await self._risk.reconcile(self.client, skip_topics=self._expired_topics(now))
            await self._poll_wallet()
        self._last_slow = now
        log.info(
            "strategy.started",
            mode=self.mode,
            batches=len(self.universe.batches),
            topics=len(self.universe.topics),
            timeseries_model=self._forecaster.timeseries_model,
        )

    async def on_stop(self) -> None:
        log.info("strategy.stopped", fills=self.fills)

    async def close(self) -> None:
        """Release the external feeds and the ledger (call once, after the runner)."""
        await self._hub.close()
        self._ledger.close()

    # --- hooks --------------------------------------------------------------

    async def on_quote(self, update: MarketUpdate) -> None:
        data = update.data
        topic_id = (
            update.topic_id if update.topic_id is not None else (data.topic_id if data else None)
        )
        if topic_id is None or data is None:
            return
        topic = self.universe.topics.get(topic_id)
        if topic is None:
            return
        now = self._clock()
        book = book_from_update(topic, data, now)
        if book is None:
            return
        self.universe.books[topic_id] = book
        if now - self._last_eval.get(topic_id, 0.0) >= self.settings.eval_interval_seconds:
            await self._guarded("evaluate", self._evaluate(topic, now))

    async def on_tick(self) -> None:
        now = self._clock()
        if self._risk.locked and self.mode == "live":
            await self._guarded(
                "reconcile",
                self._risk.reconcile(self.client, skip_topics=self._expired_topics(now)),
            )
            if self._risk.locked:
                return
        await self._guarded("mark", self._mark_and_exit(now))
        await self._guarded("round_robin", self._evaluate_round_robin(now))
        if now - self._last_slow >= self.settings.resolution_poll_seconds:
            self._last_slow = now
            await self._guarded("slow_poll", self._slow_poll(now))

    async def _guarded(self, what: str, work: Awaitable[Any]) -> None:
        """Run one unit of work, absorbing operational failures.

        Feed outages, estimate hiccups, 5xx responses and rate-limit pushback
        are part of normal operation and must not stop the runner. Anything
        else -- a programming error -- propagates and fails loudly.
        """
        try:
            await work
        except GlimpseRateLimitError as exc:
            pause = exc.retry_after or 15.0
            self._paused_until = self._clock() + pause
            log.warning("strategy.rate_limited", what=what, pause_seconds=pause)
        except (FeedError, ForecastError, EstimateError, httpx.HTTPError) as exc:
            log.warning("strategy.operational_error", what=what, error=repr(exc))
        except GlimpseAPIError as exc:
            log.error("strategy.api_error", what=what, error=str(exc))

    # --- portfolio state ----------------------------------------------------

    def _expired_topics(self, now: float) -> frozenset[int]:
        return frozenset(
            topic_id
            for topic_id, topic in self.universe.topics.items()
            if topic.seconds_to_expiry(now) <= 0.0
        )

    async def _open_positions(self, now: float) -> dict[PositionKey, Position]:
        """Open positions. In live mode the cached ``PositionTracker`` is the
        source of truth and any drift from the ledger triggers a reconciliation."""
        held = self._ledger.positions(self.mode)
        if self.mode != "live":
            return held
        expired = self._expired_topics(now)
        actual: dict[PositionKey, float] = {}
        for item in await self.positions.get():
            if item.topic_id is None or item.option_id is None or not item.shares:
                continue
            key = (item.topic_id, item.option_id)
            actual[key] = actual.get(key, 0.0) + float(item.shares)
        drift = any(
            abs(actual.get(key, 0.0) - (held[key].contracts if key in held else 0.0))
            > SHARE_TOLERANCE
            for key in set(actual) | set(held)
            if key[0] not in expired
        )
        if drift:
            await self._risk.reconcile(self.client, skip_topics=expired)
            held = self._ledger.positions(self.mode)
        return held

    def _cash(self) -> Millisats:
        if self.mode == "live" and self._wallet is not None:
            return self._wallet
        return self._ledger.cash_millisats(self.mode, self.settings.bankroll_millisats)

    def _positions_value(self, positions: dict[PositionKey, Position]) -> Millisats:
        """Liquidation value: what selling every position into its book would net."""
        total = 0
        for (topic_id, option_id), position in positions.items():
            book = self.universe.books.get(topic_id)
            if book is None or option_id not in book:
                total += int(position.cost_basis_millisats)
                continue
            quote = self._local.sell(book, [(option_id, position.contracts)])
            total += int(quote.amount_millisats) - int(quote.commission_millisats)
        return Millisats(total)

    async def _poll_wallet(self) -> None:
        balance = parse_wallet_millisats(await self.client.wallet_balance())
        if balance is None:
            log.warning("strategy.wallet_unparsed")
        else:
            self._wallet = balance

    # --- entries ------------------------------------------------------------

    def _calibration_view(self, topic: TopicMeta) -> CalibrationView | None:
        if self._calibrator is None:
            return None
        return self._calibrator.view(OnlineCalibrator.key_for(topic.asset.value, topic.batch_id))

    def _remember(self, topic: TopicMeta, forecast: EnsembleForecast, now: float) -> None:
        """Keep the latest forecast made at least ``calibration_min_lead_seconds``
        before expiry; it is scored when the topic resolves."""
        if self._calibrator is None:
            return
        if topic.seconds_to_expiry(now) < self.settings.calibration_min_lead_seconds:
            return
        self._calibrator.record(
            topic.topic_id,
            PendingForecast(
                key=OnlineCalibrator.key_for(topic.asset.value, topic.batch_id),
                components=forecast.components,
                pooled=forecast.probabilities,
                option_ids=tuple(bucket.option_id for bucket in topic.buckets),
                end_time_utc=topic.end_time_utc,
                batch_id=topic.batch_id,
            ),
        )

    async def _forecast(self, topic: TopicMeta, now: float) -> EnsembleForecast:
        cached = self._forecasts.get(topic.topic_id)
        if cached is not None and now - cached.ts < FORECAST_REUSE_SECONDS:
            return cached
        forecast = await self._forecaster.forecast(topic, now, self._calibration_view(topic))
        self._forecasts[topic.topic_id] = forecast
        self._remember(topic, forecast, now)
        return forecast

    async def _evaluate(self, topic: TopicMeta, now: float) -> None:
        self._last_eval[topic.topic_id] = now
        settings = self.settings
        if now < self._paused_until or self._optimizer is None or self._router is None:
            return
        remaining = topic.seconds_to_expiry(now)
        if not settings.min_seconds_to_expiry <= remaining <= settings.max_horizon_hours * 3600.0:
            return
        book = self.universe.books.get(topic.topic_id)
        if book is None:
            return
        forecast = await self._forecaster.forecast(topic, now, self._calibration_view(topic))
        self._forecasts[topic.topic_id] = forecast
        self._remember(topic, forecast, now)
        if not self._risk.entries_allowed:
            return

        signals = build_signals(topic, book, forecast)
        candidates = [
            signal
            for signal in signals
            if float(signal.edge_points) >= settings.min_raw_edge_points
        ][: settings.max_candidates_per_topic]
        if not candidates:
            return

        positions = await self._open_positions(now)
        cash = self._cash()
        equity = Millisats(int(cash) + int(self._positions_value(positions)))
        if settings.enable_basket_entries and await self._try_basket(
            topic, book, forecast, equity, cash, positions, now
        ):
            return
        for signal in candidates:
            budget = self._risk.entry_budget(
                signal.topic_id, signal.option_id, equity, cash, positions
            )
            if int(budget) <= 0:
                continue
            result = await self._optimizer.optimize(
                signal,
                book=book,
                bankroll=equity,
                max_cost=budget,
                kelly_multiplier=forecast.kelly_multiplier,
                ambiguity=forecast.ambiguity,
            )
            if result.plan is None:
                log.debug(
                    "strategy.candidate_rejected",
                    topic_id=topic.topic_id,
                    bucket=signal.name,
                    reason=result.rejection,
                )
                continue
            plan = result.plan
            self._ledger.record_signal(self.mode, signal, note=f"{topic.asset.value} {topic.title}")
            self._ledger.record_plan(self.mode, plan)
            decision = self._risk.check_entry(plan, topic, equity, cash, positions, now)
            if not decision.approved:
                self._ledger.record_event(
                    self.mode,
                    RecordKind.RISK_EVENT,
                    "entry blocked: " + "; ".join(decision.reasons),
                    topic_id=plan.topic_id,
                    option_id=plan.option_id,
                )
                continue
            if await self._router.enter(plan) is not None:
                self.fills += 1
                break

    async def _try_basket(
        self,
        topic: TopicMeta,
        book: LmsrBook,
        forecast: EnsembleForecast,
        equity: Millisats,
        cash: Millisats,
        positions: dict[PositionKey, Position],
        now: float,
    ) -> bool:
        """Robust-Kelly corridor basket on one topic; ``True`` if one was sent."""
        settings = self.settings
        if self._router is None:
            return False
        held_in_topic = sum(
            int(p.cost_basis_millisats)
            for (tid, _), p in positions.items()
            if tid == topic.topic_id
        )
        room = min(int(int(equity) * settings.max_topic_fraction) - held_in_topic, int(cash))
        solution = optimize_basket(
            book,
            forecast.probabilities,
            sub_bankroll_points=int(equity)
            / 1000.0
            * settings.kelly_fraction
            * forecast.kelly_multiplier,
            max_cost_points=room / 1000.0 / (1.0 + settings.fee_rate),
            fee_rate=settings.fee_rate,
            ambiguity=forecast.ambiguity,
            max_legs=settings.basket_max_legs,
            min_leg_contracts=settings.basket_min_leg_contracts,
            max_leg_contracts=settings.ladder_max_contracts,
        )
        if solution is None or len(solution.legs) < 2:
            return False
        names = {bucket.option_id: bucket.name for bucket in topic.buckets}
        quote = await ServerEstimator(self.client).quote_buy(topic.topic_id, solution.leg_tuples)
        plan = basket_plan(
            solution,
            names,
            settings,
            cost_millisats=quote.amount_millisats,
            commission_millisats=quote.commission_millisats,
        )
        self._ledger.record_event(
            self.mode,
            RecordKind.ESTIMATE,
            plan.rationale,
            payload={
                "legs": [[leg.option_id, leg.contracts] for leg in plan.legs],
                "cost_millisats": int(plan.cost_millisats),
                "commission_millisats": int(plan.commission_millisats),
                "expected_profit_millisats": int(plan.expected_profit_millisats),
                "ambiguity": forecast.ambiguity,
            },
            topic_id=topic.topic_id,
        )
        decision = self._risk.check_basket(plan, topic, equity, cash, positions, now)
        if not decision.approved:
            self._ledger.record_event(
                self.mode,
                RecordKind.RISK_EVENT,
                "basket blocked: " + "; ".join(decision.reasons),
                topic_id=topic.topic_id,
            )
            return False
        records = await self._router.enter_baskets([plan])
        self.fills += len(records)
        return bool(records)

    async def _evaluate_round_robin(self, now: float) -> None:
        topics = sorted(self.universe.topics.values(), key=lambda topic: topic.end_time_utc)
        if not topics:
            return
        evaluated = 0
        for offset in range(len(topics)):
            if evaluated >= self.settings.topics_per_tick:
                break
            topic = topics[(self._round_robin + offset) % len(topics)]
            if now - self._last_eval.get(topic.topic_id, 0.0) < self.settings.eval_interval_seconds:
                continue
            await self._evaluate(topic, now)
            evaluated += 1
        self._round_robin = (self._round_robin + max(evaluated, 1)) % len(topics)

    # --- exits --------------------------------------------------------------

    async def _mark_and_exit(self, now: float) -> None:
        positions = await self._open_positions(now)
        cash = self._cash()
        value = self._positions_value(positions)
        self._risk.update_equity(Millisats(int(cash) + int(value)), now)
        if now - self._last_equity_snapshot >= EQUITY_SNAPSHOT_SECONDS:
            self._last_equity_snapshot = now
            self._ledger.record_equity(self.mode, cash, value)
        if self._router is None:
            return
        for (topic_id, option_id), position in positions.items():
            topic = self.universe.topics.get(topic_id)
            book = self.universe.books.get(topic_id)
            if topic is None or book is None or option_id not in book:
                continue
            if topic.seconds_to_expiry(now) <= 0.0:
                continue
            forecast = await self._forecast(topic, now)
            quote = self._local.sell(book, [(option_id, position.contracts)])
            net = Millisats(int(quote.amount_millisats) - int(quote.commission_millisats))
            probability = forecast.probability(option_id)
            market_price = book.price(option_id)
            decision = evaluate_exit(position, probability, market_price, net, self.settings)
            if not decision.should_exit:
                continue
            record = await self._router.exit(
                position,
                position.contracts,
                quote,
                market_price,
                probability,
                f"{decision.reason}: {decision.detail}",
            )
            if record is not None:
                self.fills += 1

    # --- slow timer: universe refresh and resolution ------------------------

    async def _refresh_universe(self, now: float) -> None:
        held_topics = {key[0] for key in self._ledger.positions(self.mode)}
        for topic_id in held_topics - set(self.universe.topics):
            await self._adopt_topic(topic_id, now)
        fresh_topics: dict[int, TopicMeta] = {}
        fresh_books: dict[int, LmsrBook] = {}
        for batch_id, asset in self.universe.batches.items():
            for topic, book in await load_batch_topics(
                self.client, batch_id, asset, self.settings, now
            ):
                fresh_topics[topic.topic_id] = topic
                fresh_books[topic.topic_id] = book
        for topic_id in held_topics:
            if topic_id not in fresh_topics and topic_id in self.universe.topics:
                fresh_topics[topic_id] = self.universe.topics[topic_id]
                if topic_id in self.universe.books:
                    fresh_books[topic_id] = self.universe.books[topic_id]
        self.universe.topics = fresh_topics
        self.universe.books = fresh_books
        for stale in [topic_id for topic_id in self._forecasts if topic_id not in fresh_topics]:
            del self._forecasts[stale]
            self._last_eval.pop(stale, None)

    async def _adopt_topic(self, topic_id: int, now: float) -> None:
        """Load metadata for a held position's topic that is not in the universe
        (for example after a restart, once the market has left the active page)."""
        quotes = await self.client.market_quotes(topic_id)
        asset = classify_asset(quotes.title, quotes.description, quotes.category)
        if asset is None:
            return
        built = build_topic(
            BatchMarketItem(
                topic_id=topic_id,
                batch_id=quotes.batch_id,
                title=quotes.title,
                outcomes=quotes.outcomes,
                num_outcomes=len(quotes.outcomes or []),
                end_time_utc=quotes.market_end_time_utc,
            ),
            asset,
            now,
        )
        if built is not None:
            self.universe.topics[topic_id], self.universe.books[topic_id] = built

    async def _slow_poll(self, now: float) -> None:
        await self._refresh_universe(now)
        if self.mode == "live":
            await self._poll_wallet()
        await self._settle_resolved(now)
        await self._guarded("calibration", self._resolve_calibration(now))
        await self._guarded("coherence", self._scan_coherence(now))

    async def _resolve_calibration(self, now: float) -> None:
        """Score pending forecasts of resolved topics (one page per batch, at
        most every ``calibration_poll_seconds``)."""
        calibrator = self._calibrator
        if calibrator is None:
            return
        calibrator.expire(now)
        due: dict[str, set[int]] = {}
        for topic_id, pending in calibrator.pending.items():
            if pending.end_time_utc <= now:
                due.setdefault(pending.batch_id, set()).add(topic_id)
        updated = False
        poll = self.settings.calibration_poll_seconds
        for batch_id, topic_ids in due.items():
            if now - self._resolution_polled.get(batch_id, 0.0) < poll:
                continue
            self._resolution_polled[batch_id] = now
            page = await self.client.batch_resolved_markets_page(batch_id)
            for market in page.markets or []:
                if market.topic_id not in topic_ids or market.resolved_option_id is None:
                    continue
                scores = calibrator.resolve(market.topic_id, market.resolved_option_id)
                if scores is not None:
                    updated = True
                    log.info("calibration.scored", topic_id=market.topic_id, brier=scores)
        if updated:
            calibrator.save()

    async def _scan_coherence(self, now: float) -> None:
        """Bregman-project linked books; log and (optionally) take a buy-only cover."""
        for group in linked_groups(self.universe.topics):
            topics = [self.universe.topics[t] for t in group if t in self.universe.books]
            if len(topics) < 2:
                continue
            if topics[0].seconds_to_expiry(now) < self.settings.min_seconds_to_expiry:
                continue
            books = [self.universe.books[topic.topic_id] for topic in topics]
            projection = frank_wolfe_projection(topics, books)
            arbitrage = cover_arbitrage(
                topics,
                books,
                fee_rate=self.settings.fee_rate,
                max_cost_points=int(self._cash()) / 1000.0 * self.settings.max_topic_fraction,
            )
            log.info(
                "coherence.scan",
                topics=group,
                divergence_sats=round(projection.divergence_points, 3),
                fw_gap=projection.gap,
                cover_profit_sats=None
                if arbitrage is None
                else round(arbitrage.guaranteed_profit_points, 2),
            )
            if arbitrage is None:
                continue
            self._ledger.record_event(
                self.mode,
                RecordKind.SIGNAL,
                f"cross-topic cover: guaranteed {arbitrage.guaranteed_profit_points:.1f} sats "
                f"for {arbitrage.all_in_points:.1f} sats all-in",
                payload={
                    "legs": {str(k): [list(leg) for leg in v] for k, v in arbitrage.legs.items()},
                    "unit_cost_points": arbitrage.unit_cost_points,
                    "divergence_points": projection.divergence_points,
                },
            )
            if self.settings.enable_cross_topic_arbitrage and self._risk.entries_allowed:
                await self._execute_cover(arbitrage.legs, now)

    async def _execute_cover(
        self, legs: Mapping[int, tuple[tuple[int, float], ...]], now: float
    ) -> None:
        if self._router is None:
            return
        positions = await self._open_positions(now)
        cash = self._cash()
        equity = Millisats(int(cash) + int(self._positions_value(positions)))
        baskets: list[BasketPlan] = []
        for topic_id, topic_legs in legs.items():
            topic = self.universe.topics[topic_id]
            book = self.universe.books[topic_id]
            forecast = self._forecasts.get(topic_id)
            probabilities = (
                forecast.probabilities
                if forecast is not None
                else dict(zip(book.option_ids, book.probabilities().tolist(), strict=True))
            )
            names = {bucket.option_id: bucket.name for bucket in topic.buckets}
            plan = basket_plan_from_legs(book, topic_legs, probabilities, names, self.settings)
            decision = self._risk.check_basket(
                plan, topic, equity, cash, positions, now, require_gain=False
            )
            if not decision.approved:
                self._ledger.record_event(
                    self.mode,
                    RecordKind.RISK_EVENT,
                    "cover blocked: " + "; ".join(decision.reasons),
                    topic_id=topic_id,
                )
                return
            baskets.append(plan)
        self.fills += len(await self._router.enter_baskets(baskets))

    async def _settle_resolved(self, now: float) -> None:
        """Detect resolution via ``quote_mode`` and book the binary settlement."""
        positions = self._ledger.positions(self.mode)
        expired = {
            topic_id
            for topic_id, _ in positions
            if topic_id in self.universe.topics
            and self.universe.topics[topic_id].seconds_to_expiry(now) <= 0.0
        }
        for topic_id in sorted(expired):
            topic = self.universe.topics[topic_id]
            quotes = await self.client.market_quotes(topic_id)
            if not (quotes.is_resolved or quotes.quote_mode == QuoteMode.RESOLVED.value):
                continue
            resolved = await self.client.batch_resolved_markets_page(topic.batch_id)
            winner = next(
                (
                    market.resolved_option_id
                    for market in resolved.markets or []
                    if market.topic_id == topic_id and market.resolved_option_id is not None
                ),
                None,
            )
            if winner is None:
                continue
            for (held_topic, option_id), position in positions.items():
                if held_topic != topic_id:
                    continue
                record = self._ledger.record_settlement(
                    self.mode,
                    position,
                    won=option_id == winner,
                    fee_rate=self.settings.fee_rate,
                    note=f"resolved to option {winner}",
                )
                log.info(
                    "strategy.settled",
                    topic_id=topic_id,
                    bucket=position.bucket_name,
                    won=option_id == winner,
                    realized_sats=int(record.realized_pnl_millisats) / 1000,
                )
            self.positions.invalidate()


__all__ = [
    "GlimpseQuantStrategy",
    "Universe",
    "book_from_update",
    "build_topic",
    "discover_batches",
    "load_batch_topics",
    "parse_wallet_millisats",
]
