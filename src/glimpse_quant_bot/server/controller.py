"""Engine controller: runs the real strategy inside the API server's event loop.

The controller owns exactly one :class:`~glimpse_quant_bot.ledger.Ledger` and
hands it to every strategy instance it starts, so the server process is the
ledger's single writer. Operator actions (start, stop, reconcile, manual
orders) are serialised by one ``asyncio.Lock``. The kill switch deliberately
does *not* wait for that lock: it gates the router first and tears down after.

Safety rules enforced here, on top of the engine's own:

* Paper (``dry_run``) is the default for every start.
* Live needs the environment flag and API key (checked by the engine's own
  ``Settings.resolve_mode``), ``serve --allow-live``, a real venue, and a
  single-use arming token obtained by typing a confirmation phrase.
* Only three tunables can be overridden over HTTP; risk caps, the drawdown
  limit and the Kelly fraction come from the environment.
* The drawdown breaker's state survives a stop/start within the UTC day.
* Every order -- the engine's and the operator's -- passes
  ``RiskManager.check_entry`` and is sent by ``ExecutionRouter``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import secrets
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx
import structlog
import websockets
from dotenv import load_dotenv
from glimpse_markets import (
    AsyncClient,
    GlimpseRateLimitError,
    GlimpseServerError,
    MarketUpdate,
    StrategyRunner,
)
from glimpse_markets.money import Millisats, PriceUnits
from glimpse_markets.ratelimit import AsyncRateLimiter
from pydantic import ValidationError

from glimpse_quant_bot import __version__
from glimpse_quant_bot.backtester import Backtester, BacktestResult, synthetic_dataset
from glimpse_quant_bot.config import LiveTradingNotArmedError, Settings
from glimpse_quant_bot.execution import ExecutionRouter
from glimpse_quant_bot.external_feeds import MarketDataHub
from glimpse_quant_bot.forecaster import EnsembleForecast, ForecastError
from glimpse_quant_bot.ledger import Ledger, PositionKey
from glimpse_quant_bot.lmsr_optimizer import (
    CostQuote,
    EstimateError,
    ServerEstimator,
    kelly_fraction,
    ladder_point,
)
from glimpse_quant_bot.risk_manager import BreakerState, RiskDecision, RiskManager
from glimpse_quant_bot.server import views
from glimpse_quant_bot.server.schemas import (
    ARM_PHRASE,
    LIVE_ORDER_PHRASE,
    RELEASE_PHRASE,
    ActionResult,
    ArmingStatus,
    ArmResponse,
    BacktestResultsResponse,
    BacktestRunRequest,
    BacktestRunView,
    DiscrepancyView,
    DistributionResponse,
    EngineState,
    ExposureRow,
    HealthResponse,
    LadderResponse,
    LedgerPage,
    LiveEligibility,
    MarketsResponse,
    MarketSummary,
    PositionView,
    RateLimitBudget,
    ReconcileResult,
    RiskMetrics,
    SafetyMode,
    SpotTicker,
    StartRequest,
    TestOrderRequest,
    TestOrderResult,
    VenueName,
)
from glimpse_quant_bot.server.simulation import SimulatedGlimpse, SimulatedHub, SimulatedWorld
from glimpse_quant_bot.server.telemetry import Subscriber, TelemetryHub
from glimpse_quant_bot.strategy import GlimpseQuantStrategy
from glimpse_quant_bot.types import (
    AssetType,
    BasketPlan,
    BucketSignal,
    LedgerRecord,
    Mode,
    Position,
    RecordKind,
    Side,
    TopicMeta,
    TradePlan,
)

log = structlog.get_logger(__name__)

ARM_TTL_SECONDS = 60.0
KILL_GRACE_SECONDS = 1.0
PUBLISH_INTERVAL_SECONDS = 1.0
FORECAST_STALE_SECONDS = 30.0
FILL_KINDS = (RecordKind.ENTRY, RecordKind.EXIT, RecordKind.SETTLEMENT, RecordKind.ADJUSTMENT)
_RETRYABLE = (
    websockets.WebSocketException,
    OSError,
    httpx.HTTPError,
    GlimpseServerError,
    GlimpseRateLimitError,
)


# --- errors -------------------------------------------------------------------


class ControlError(Exception):
    """An operator request that cannot be honoured; mapped to an HTTP status."""

    status_code = 400

    def __init__(self, detail: str, reasons: Sequence[str] = ()) -> None:
        super().__init__(detail)
        self.detail = detail
        self.reasons = list(reasons)


class Conflict(ControlError):
    status_code = 409


class Forbidden(ControlError):
    status_code = 403


class NotFound(ControlError):
    status_code = 404


class Invalid(ControlError):
    status_code = 422


class Unavailable(ControlError):
    status_code = 503


# --- plumbing -----------------------------------------------------------------


class CountingRateLimiter(AsyncRateLimiter):
    """The SDK's sliding-window limiter, plus a count of what was actually sent."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        super().__init__(max_requests=max_requests, window_seconds=window_seconds)
        self._sent: deque[float] = deque()
        self.total = 0

    async def acquire(self) -> None:
        await super().acquire()
        self._sent.append(time.monotonic())
        self.total += 1

    def used(self) -> int:
        horizon = time.monotonic() - self.window_seconds
        while self._sent and self._sent[0] < horizon:
            self._sent.popleft()
        return len(self._sent)


@dataclass(frozen=True, slots=True)
class Venue:
    """Where orders and market data come from."""

    name: VenueName
    supports_live: bool
    make_client: Callable[[Settings, Mode, AsyncRateLimiter], AsyncClient]
    make_hub: Callable[[Settings], MarketDataHub | None]
    world: SimulatedWorld | None = None


def glimpse_venue() -> Venue:
    def client(settings: Settings, mode: Mode, limiter: AsyncRateLimiter) -> AsyncClient:
        load_dotenv()
        return AsyncClient.from_env(dry_run=mode != "live", rate_limiter=limiter)

    return Venue("glimpse", True, client, lambda _settings: None)


def simulated_venue(world: SimulatedWorld, *, supports_live: bool = False) -> Venue:
    """The simulator. ``supports_live=True`` exists only for tests of the arming path."""

    def client(settings: Settings, mode: Mode, limiter: AsyncRateLimiter) -> AsyncClient:
        return SimulatedGlimpse(world, rate_limiter=limiter, dry_run=mode != "live")

    def hub(settings: Settings) -> MarketDataHub:
        return SimulatedHub(world, settings.assets)

    return Venue("simulated", supports_live, client, hub, world)


def configure_logging(settings: Settings, telemetry: TelemetryHub) -> None:
    """The CLI's structlog setup plus a tap that feeds the terminal's log stream."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            telemetry.processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


class GuardedRouter(ExecutionRouter):
    """``ExecutionRouter`` that refuses to send anything once the kill switch fires."""

    def __init__(
        self,
        client: AsyncClient,
        ledger: Ledger,
        risk: RiskManager,
        mode: Mode,
        on_trade: Callable[[], None] | None,
        halted: Callable[[], bool],
    ) -> None:
        super().__init__(client, ledger, risk, mode, on_trade=on_trade)
        self._halted = halted

    async def enter(self, plan: TradePlan) -> LedgerRecord | None:
        if self._halted():
            log.warning("execution.halted", action="entry", topic_id=plan.topic_id)
            return None
        return await super().enter(plan)

    async def enter_baskets(self, baskets: Sequence[BasketPlan]) -> list[LedgerRecord]:
        if self._halted():
            log.warning("execution.halted", action="basket")
            return []
        return await super().enter_baskets(baskets)

    async def exit(
        self,
        position: Position,
        contracts: float,
        expected: CostQuote,
        market_price: PriceUnits,
        model_probability: float,
        reason: str,
    ) -> LedgerRecord | None:
        if self._halted():
            log.warning("execution.halted", action="exit", topic_id=position.topic_id)
            return None
        return await super().exit(
            position, contracts, expected, market_price, model_probability, reason
        )


class EngineStrategy(GlimpseQuantStrategy):
    """The production strategy plus read-only accessors and a halt gate.

    No pricing, sizing or risk behaviour is overridden. ``halted`` stops new
    hook invocations and is checked again by :class:`GuardedRouter` at the
    single point where orders leave the process.
    """

    def __init__(
        self,
        settings: Settings,
        mode: Mode,
        ledger: Ledger,
        hub: MarketDataHub | None,
        on_ready: Callable[[], None],
    ) -> None:
        super().__init__(settings, mode, ledger=ledger, hub=hub)
        self.halted = False
        self.ready = False
        self._on_ready = on_ready

    async def on_start(self) -> None:
        await super().on_start()
        self._router = GuardedRouter(
            self.client,
            self.ledger,
            self.risk,
            self.mode,
            on_trade=self.positions.invalidate,
            halted=lambda: self.halted,
        )
        self.ready = True
        self._on_ready()

    async def on_quote(self, update: MarketUpdate) -> None:
        if not self.halted:
            await super().on_quote(update)

    async def on_tick(self) -> None:
        if not self.halted:
            await super().on_tick()

    # --- read-only accessors ----------------------------------------------------

    @property
    def router(self) -> ExecutionRouter | None:
        return self._router

    @property
    def forecasts(self) -> dict[int, EnsembleForecast]:
        return self._forecasts

    @property
    def timeseries_model(self) -> str:
        return self._forecaster.timeseries_model

    @property
    def has_client(self) -> bool:
        return hasattr(self, "client")

    def cash(self) -> Millisats:
        return self._cash()

    def positions_value(self, positions: dict[PositionKey, Position]) -> Millisats:
        try:
            return self._positions_value(positions)
        except ValueError:
            # A paper position larger than the bucket's outstanding shares cannot
            # be priced as a sale; fall back to cost basis for display only.
            return Millisats(sum(int(p.cost_basis_millisats) for p in positions.values()))

    async def open_positions(self, now: float) -> dict[PositionKey, Position]:
        return await self._open_positions(now)

    def expired_topics(self, now: float) -> frozenset[int]:
        return self._expired_topics(now)

    async def forecast_for(self, topic: TopicMeta, now: float) -> EnsembleForecast:
        return await self._forecast(topic, now)

    async def release_feeds(self) -> None:
        await self._hub.close()


# --- controller -------------------------------------------------------------------


class EngineController:
    def __init__(
        self,
        *,
        settings_factory: Callable[..., Settings],
        venue: Venue,
        allow_live: bool = False,
        autostart: bool = False,
        telemetry: TelemetryHub | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings_factory = settings_factory
        self.base_settings = settings_factory()
        self.settings = self.base_settings
        self.venue = venue
        self.allow_live = allow_live
        self.autostart = autostart
        self.telemetry = telemetry or TelemetryHub(clock)
        self._clock = clock
        self.ledger = Ledger(self.base_settings.ledger_path, self.base_settings.audit_path)
        self.limiter = CountingRateLimiter(self.base_settings.glimpse_requests_per_minute)
        self._lock = asyncio.Lock()
        self.state: EngineState = "stopped"
        self.mode: Mode = "dry_run"
        self.strategy: EngineStrategy | None = None
        self.client: AsyncClient | None = None
        self._runner: asyncio.Task[None] | None = None
        self._background: list[asyncio.Task[None]] = []
        self.kill_switch_engaged = False
        self._arming: tuple[str, float] | None = None
        self._breaker_memory: dict[Mode, BreakerState] = {}
        self.started_at: float | None = None
        self.last_error: str | None = None
        self._booted_at = clock()
        self._session_open: dict[AssetType, float] = {}
        self._last_fill_id = self.ledger.last_record_id()
        self._forecast_jobs: dict[int, asyncio.Task[None]] = {}
        self._backtest: BacktestRunView | None = None
        self._backtest_job: asyncio.Task[None] | None = None

    # --- lifecycle --------------------------------------------------------------

    async def startup(self) -> None:
        self.telemetry.bind_loop(asyncio.get_running_loop())
        if self.venue.world is not None:
            self._background.append(asyncio.create_task(self.venue.world.run()))
        self._background.append(asyncio.create_task(self._publish_loop()))
        log.info(
            "server.started",
            venue=self.venue.name,
            allow_live=self.allow_live,
            ledger=str(self.base_settings.ledger_path),
        )
        if self.autostart:
            await self.start(StartRequest())

    async def shutdown(self) -> None:
        with contextlib.suppress(ControlError):
            await self.stop()
        for task in [*self._background, *self._forecast_jobs.values()]:
            task.cancel()
        if self._backtest_job is not None:
            self._backtest_job.cancel()
        for task in self._background:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._background.clear()
        self.ledger.close()

    # --- safety state -------------------------------------------------------------

    @property
    def safety_mode(self) -> SafetyMode:
        live = self.mode == "live" and self.state in ("starting", "running")
        return "LIVE_ARMED" if live else "PAPER"

    def live_eligibility(self, settings: Settings | None = None) -> LiveEligibility:
        active = settings or self.base_settings
        key = active.glimpse_api_key
        api_key_present = key is not None and bool(key.get_secret_value())
        blockers: list[str] = []
        if not active.glimpse_live_trading:
            blockers.append("GLIMPSE_LIVE_TRADING is not 'true' in the server environment")
        if not api_key_present:
            blockers.append("GLIMPSE_API_KEY is empty")
        if not self.allow_live:
            blockers.append("server was not started with `serve --allow-live`")
        if not self.venue.supports_live:
            blockers.append("the simulated venue can never trade live")
        if self.kill_switch_engaged:
            blockers.append("kill switch is engaged")
        return LiveEligibility(
            server_flag=active.glimpse_live_trading,
            api_key_present=api_key_present,
            serve_allow_live=self.allow_live,
            real_venue=self.venue.supports_live,
            kill_switch_clear=not self.kill_switch_engaged,
            eligible=not blockers,
            blockers=blockers,
        )

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def arm(self, phrase: str) -> ArmResponse:
        if phrase.strip() != ARM_PHRASE:
            raise Forbidden(f"type the phrase '{ARM_PHRASE}' exactly to arm live trading")
        if self.safety_mode == "LIVE_ARMED":
            raise Conflict("the engine is already running live")
        eligibility = self.live_eligibility()
        if not eligibility.eligible:
            raise Forbidden("live trading is not available", eligibility.blockers)
        token = secrets.token_urlsafe(32)
        expires = self._clock() + ARM_TTL_SECONDS
        self._arming = (self._digest(token), expires)
        log.warning("operator.armed", expires_in_seconds=ARM_TTL_SECONDS)
        return ArmResponse(arming_token=token, expires_at=expires, eligibility=eligibility)

    def disarm(self) -> ActionResult:
        if self._arming is not None:
            log.info("operator.disarmed")
        self._arming = None
        return self._result("live arming cancelled")

    def _consume_arming(self, token: str | None) -> bool:
        """Single use: any attempt, right or wrong, burns the pending token."""
        pending, self._arming = self._arming, None
        if not token or pending is None:
            return False
        digest, expires = pending
        return self._clock() <= expires and hmac.compare_digest(digest, self._digest(token))

    # --- start / stop -------------------------------------------------------------

    def _overrides(self, request: StartRequest) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if request.min_expected_gain_sats is not None:
            overrides["min_expected_gain_sats"] = request.min_expected_gain_sats
        if request.max_horizon_hours is not None:
            overrides["max_horizon_hours"] = request.max_horizon_hours
        if request.assets is not None:
            overrides["assets"] = tuple(dict.fromkeys(request.assets))
        return overrides

    async def start(self, request: StartRequest) -> ActionResult:
        async with self._lock:
            if self.state in ("starting", "running", "stopping"):
                raise Conflict(f"engine is {self.state}; stop it first")
            if self.kill_switch_engaged:
                raise Conflict("kill switch is engaged; release it before starting")
            try:
                settings = self._settings_factory(**self._overrides(request))
            except ValidationError as exc:
                raise Invalid(
                    "invalid runtime parameters", [e["msg"] for e in exc.errors()]
                ) from exc
            mode: Mode = "dry_run"
            if not request.dry_run:
                reasons = list(self.live_eligibility(settings).blockers)
                if not self._consume_arming(request.arming_token):
                    reasons.append("missing, expired or already-used arming token")
                if reasons:
                    raise Forbidden("live trading refused", reasons)
                try:
                    mode = settings.resolve_mode(cli_live_flag=True)
                except LiveTradingNotArmedError as exc:
                    raise Forbidden("live trading refused", [str(exc)]) from exc
            self._launch(settings, mode)
            if mode == "live":
                self.ledger.record_event(
                    mode, RecordKind.RISK_EVENT, "live trading armed and started by operator"
                )
            return self._result(f"engine starting in {'LIVE' if mode == 'live' else 'paper'} mode")

    def _launch(self, settings: Settings, mode: Mode) -> None:
        self.settings = settings
        self.mode = mode
        client = self.venue.make_client(settings, mode, self.limiter)
        strategy = EngineStrategy(
            settings, mode, self.ledger, self.venue.make_hub(settings), self._on_ready
        )
        memory = self._breaker_memory.get(mode)
        if memory is not None:
            strategy.risk.restore_breaker(memory)
        self.client = client
        self.strategy = strategy
        self.state = "starting"
        self.started_at = self._clock()
        self.last_error = None
        self._runner = asyncio.create_task(self._run(strategy, client, settings))
        log.info(
            "operator.start",
            mode=mode,
            venue=self.venue.name,
            assets=",".join(asset.value for asset in settings.assets),
            min_expected_gain_sats=settings.min_expected_gain_sats,
            max_horizon_hours=settings.max_horizon_hours,
        )

    def _on_ready(self) -> None:
        if self.state == "starting":
            self.state = "running"

    async def _run(self, strategy: EngineStrategy, client: AsyncClient, settings: Settings) -> None:
        """``cmd_run``'s reconnect loop, inside the server's event loop."""
        backoff = 2.0
        try:
            while True:
                runner = StrategyRunner(
                    strategy, client=client, tick_interval=settings.tick_interval_seconds
                )
                try:
                    await runner.run()
                except _RETRYABLE as exc:
                    log.warning("run.stream_dropped", error=repr(exc), retry_in=backoff)
                else:
                    log.warning("run.stream_ended", retry_in=backoff)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a programming error: stop loudly, keep the server up
            self.last_error = repr(exc)
            self.state = "error"
            log.error("engine.crashed", error=repr(exc))
            self._remember_breaker(strategy)
            with contextlib.suppress(Exception):
                await strategy.release_feeds()
            with contextlib.suppress(Exception):
                await client.close()

    def _remember_breaker(self, strategy: EngineStrategy) -> None:
        self._breaker_memory[strategy.mode] = strategy.risk.breaker_state()

    async def _teardown(self) -> None:
        task, self._runner = self._runner, None
        strategy, client = self.strategy, self.client
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if strategy is not None:
            self._remember_breaker(strategy)
            with contextlib.suppress(Exception):
                await strategy.release_feeds()
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()
        self.client = None

    async def stop(self) -> ActionResult:
        async with self._lock:
            if self._runner is None and self.state not in ("error",):
                return self._result("engine already stopped")
            self.state = "stopping"
            await self._teardown()
            self.state = "halted" if self.kill_switch_engaged else "stopped"
            log.info("operator.stop")
            return self._result("engine stopped")

    async def kill(self, reason: str) -> ActionResult:
        """Emergency halt. Gates the router before awaiting anything."""
        self.kill_switch_engaged = True
        self._arming = None
        strategy = self.strategy
        note = f"kill switch: {reason}"
        if strategy is not None:
            strategy.halted = True
            strategy.risk.lock(note)
        else:
            self.ledger.record_event(self.mode, RecordKind.RISK_EVENT, f"trading locked: {note}")
        log.error("operator.kill_switch", reason=reason, mode=self.mode)
        running_live = self.mode == "live" and self._runner is not None
        if running_live:
            # Let an order already on the wire come back and be booked.
            await asyncio.sleep(KILL_GRACE_SECONDS)
        async with self._lock:
            await self._teardown()
            self.state = "halted"
        return self._result(
            "kill switch engaged: runner cancelled, trading locked, arming revoked. Glimpse "
            "fills immediately, so there are no resting orders to cancel; positions are "
            "reconciled against portfolio_active() at the next live start."
        )

    async def release(self, phrase: str) -> ActionResult:
        if phrase.strip() != RELEASE_PHRASE:
            raise Forbidden(
                f"type the phrase '{RELEASE_PHRASE}' exactly to release the kill switch"
            )
        async with self._lock:
            if not self.kill_switch_engaged:
                return self._result("kill switch was not engaged")
            self.kill_switch_engaged = False
            self.state = "stopped"
            self.ledger.record_event(
                self.mode, RecordKind.RISK_EVENT, "kill switch released by operator"
            )
            log.warning("operator.kill_switch_released")
            return self._result("kill switch released; the engine stays stopped until started")

    def _result(self, message: str, ok: bool = True) -> ActionResult:
        return ActionResult(ok=ok, message=message, engine_state=self.state, mode=self.safety_mode)

    def _running(self) -> EngineStrategy:
        strategy = self.strategy
        if (
            strategy is None
            or self.state != "running"
            or not strategy.ready
            or strategy.router is None
            or self.client is None
        ):
            raise Conflict("the engine is not running")
        return strategy

    # --- reconcile ----------------------------------------------------------------

    async def reconcile(self) -> ReconcileResult:
        async with self._lock:
            strategy = self.strategy
            now = self._clock()
            running = strategy is not None and self.state == "running" and self.client is not None
            if running and strategy is not None and self.client is not None and self.mode == "live":
                report = await strategy.risk.reconcile(
                    self.client, skip_topics=strategy.expired_topics(now)
                )
                return self._reconcile_result(
                    "live_engine",
                    report.ok,
                    report.landed,
                    report.discrepancies,
                    report.note,
                    strategy.risk.locked,
                )
            if running and strategy is not None:
                if strategy.risk.locked and not strategy.halted:
                    strategy.risk.unlock("paper reconcile: ledger replayed, no exchange to diff")
                held = self.ledger.positions(self.mode)
                note = (
                    f"paper mode: the ledger is the source of truth; {len(held)} open "
                    "position(s) replayed, nothing to diff against an exchange"
                )
                log.info("operator.reconcile", scope="paper", positions=len(held))
                return self._reconcile_result("paper", True, None, (), note, strategy.risk.locked)
            eligibility = self.live_eligibility()
            if self.venue.supports_live and eligibility.api_key_present:
                risk = RiskManager(self.base_settings, self.ledger, "live", self._clock)
                client = self.venue.make_client(self.base_settings, "dry_run", self.limiter)
                try:
                    report = await risk.reconcile(client)
                finally:
                    await client.close()
                return self._reconcile_result(
                    "live_ledger",
                    report.ok,
                    report.landed,
                    report.discrepancies,
                    report.note,
                    risk.locked,
                )
            return self._reconcile_result(
                "paper",
                True,
                None,
                (),
                "engine stopped and no live account is configured: nothing to reconcile",
                False,
            )

    @staticmethod
    def _reconcile_result(
        scope: Literal["live_engine", "live_ledger", "paper"],
        ok: bool,
        landed: bool | None,
        discrepancies: Sequence[Any],
        note: str,
        locked: bool,
    ) -> ReconcileResult:
        return ReconcileResult(
            ok=ok,
            scope=scope,
            landed=landed,
            discrepancies=[
                DiscrepancyView(
                    topic_id=d.topic_id,
                    option_id=d.option_id,
                    ledger_contracts=d.ledger_contracts,
                    actual_contracts=d.actual_contracts,
                )
                for d in discrepancies
            ],
            note=note,
            trading_locked=locked,
        )

    # --- manual ticket ------------------------------------------------------------

    async def test_order(self, request: TestOrderRequest) -> TestOrderResult:
        """An operator order, through the same gate and router as the engine's own."""
        async with self._lock:
            if self.kill_switch_engaged:
                raise Conflict("kill switch is engaged")
            strategy = self._running()
            client = self.client
            router = strategy.router
            if client is None or router is None:
                raise Conflict("the engine is not running")
            mode = self.mode
            if mode == "live" and (request.confirmation_phrase or "").strip() != LIVE_ORDER_PHRASE:
                raise Forbidden(f"live orders need the phrase '{LIVE_ORDER_PHRASE}'")
            now = self._clock()
            topic = strategy.universe.topics.get(request.topic_id)
            book = strategy.universe.books.get(request.topic_id)
            bucket = topic.bucket(request.option_id) if topic is not None else None
            if topic is None or book is None or bucket is None or request.option_id not in book:
                raise NotFound("unknown topic or bucket")
            try:
                forecast = await strategy.forecast_for(topic, now)
            except ForecastError as exc:
                raise Unavailable("no forecast available for this topic", [str(exc)]) from exc
            settings = self.settings
            probability = forecast.probability(request.option_id)
            market_price = book.price(request.option_id)
            signal = BucketSignal(
                topic_id=topic.topic_id,
                option_id=bucket.option_id,
                name=bucket.name,
                lo=bucket.lo,
                hi=bucket.hi,
                market_price=market_price,
                model_probability=probability,
                edge_points=PriceUnits(100.0 * probability - float(market_price)),
                components={
                    name: values.get(bucket.option_id, 0.0)
                    for name, values in forecast.components.items()
                },
            )
            marginal_all_in = PriceUnits(float(market_price) * (1.0 + settings.fee_rate))
            legs = [(bucket.option_id, request.contracts)]

            async def checked(quote: CostQuote) -> tuple[TradePlan, RiskDecision]:
                """Plan the ticket on ``quote`` and put it to the RiskManager against
                the book as it stands at this moment."""
                moment = self._clock()
                positions = await strategy.open_positions(moment)
                cash = strategy.cash()
                equity = Millisats(int(cash) + int(strategy.positions_value(positions)))
                budget = strategy.risk.entry_budget(
                    topic.topic_id, bucket.option_id, equity, cash, positions
                )
                point = ladder_point(request.contracts, quote, probability, settings.fee_rate)
                plan = TradePlan(
                    topic_id=topic.topic_id,
                    option_id=bucket.option_id,
                    bucket_name=bucket.name,
                    side=Side.BUY,
                    contracts=request.contracts,
                    cost_millisats=point.cost_millisats,
                    commission_millisats=point.commission_millisats,
                    exec_price=point.exec_price,
                    market_price=market_price,
                    model_probability=probability,
                    net_edge_points=point.net_edge_points,
                    expected_profit_millisats=point.expected_profit_millisats,
                    kelly_fraction=settings.kelly_fraction
                    * forecast.kelly_multiplier
                    * kelly_fraction(probability, marginal_all_in, settings.fee_rate),
                    budget_millisats=budget,
                    ladder=(point,),
                    rationale=(
                        f"operator test order: {request.contracts:g} x {bucket.name} at "
                        f"{float(point.exec_price):.2f} all-in vs model "
                        f"{100.0 * probability:.1f}"
                    ),
                )
                # Paper tickets exercise the plumbing, so only the strategy's profitability
                # filters are relaxed; every cap, lock, breaker and flow rule still applies.
                # Live tickets must also clear the engine's own net-edge and gain thresholds.
                decision = strategy.risk.check_entry(
                    plan,
                    topic,
                    equity,
                    cash,
                    positions,
                    moment,
                    per_trade_thresholds=mode == "live",
                )
                return plan, decision

            # Pre-screen on the local LS-LMSR replica (zero REST calls), so a ticket the
            # RiskManager refuses never queues for a slot in the shared Glimpse budget.
            plan, decision = await checked(views.local_estimator(book, settings).buy(book, legs))
            if decision.approved:
                try:
                    quote = await ServerEstimator(client).quote_buy(topic.topic_id, legs)
                except EstimateError as exc:
                    raise Unavailable("the venue could not price this order", [str(exc)]) from exc
                # The venue's quote is authoritative, and the engine kept trading while the
                # request waited for a rate-limit slot: re-plan and re-check on fresh state.
                plan, decision = await checked(quote)
            self.ledger.record_signal(mode, signal, note=f"operator ticket {topic.title}")
            self.ledger.record_plan(mode, plan)
            view = views.ladder_point_view(plan.ladder[0])
            if not decision.approved:
                self.ledger.record_event(
                    mode,
                    RecordKind.RISK_EVENT,
                    "operator entry blocked: " + "; ".join(decision.reasons),
                    topic_id=topic.topic_id,
                    option_id=bucket.option_id,
                )
                log.warning(
                    "operator.order_blocked",
                    topic_id=topic.topic_id,
                    bucket=bucket.name,
                    reasons="; ".join(decision.reasons),
                )
                return TestOrderResult(
                    ok=False,
                    mode=mode,
                    approved=False,
                    reasons=list(decision.reasons),
                    plan=view,
                    fill=None,
                    message="blocked by the risk manager",
                )
            record = await router.enter(plan)
            if record is not None:
                strategy.fills += 1
            log.info(
                "operator.order",
                mode=mode,
                topic_id=topic.topic_id,
                bucket=bucket.name,
                contracts=request.contracts,
                filled=record is not None,
            )
            return TestOrderResult(
                ok=record is not None,
                mode=mode,
                approved=True,
                reasons=[],
                plan=view,
                fill=views.record_view(record) if record is not None else None,
                message="filled" if record is not None else "the router did not fill the order",
            )

    # --- read models ----------------------------------------------------------------

    def _portfolio(self) -> tuple[dict[PositionKey, Position], Millisats, Millisats]:
        positions = self.ledger.positions(self.mode)
        strategy = self.strategy
        if strategy is not None:
            cash = strategy.cash()
            value = strategy.positions_value(positions)
        else:
            cash = self.ledger.cash_millisats(self.mode, self.settings.bankroll_millisats)
            value = Millisats(sum(int(p.cost_basis_millisats) for p in positions.values()))
        return positions, cash, value

    def health(self) -> HealthResponse:
        strategy = self.strategy
        risk = strategy.risk if strategy is not None else None
        now = self._clock()
        used = self.limiter.used()
        limit = self.limiter.max_requests
        pending = self._arming is not None and self._arming[1] >= now
        return HealthResponse(
            version=__version__,
            engine_state=self.state,
            mode=self.safety_mode,
            venue=self.venue.name,
            dry_run=self.safety_mode != "LIVE_ARMED",
            trading_locked=bool(risk and risk.locked),
            lock_reason=risk.lock_reason if risk is not None else None,
            breaker_tripped=bool(risk and risk.breaker_tripped),
            breaker_reason=risk.breaker_reason if risk is not None else None,
            kill_switch_engaged=self.kill_switch_engaged,
            entries_allowed=bool(risk and risk.entries_allowed and self.state == "running"),
            rate_limit=RateLimitBudget(
                limit_per_minute=limit,
                used_last_minute=used,
                remaining=max(0, limit - used),
                total_requests=self.limiter.total,
            ),
            live_eligibility=self.live_eligibility(),
            arming=ArmingStatus(
                pending=pending, expires_at=self._arming[1] if pending and self._arming else None
            ),
            started_at=self.started_at,
            uptime_seconds=now - self._booted_at,
            server_time=now,
            ws_clients=self.telemetry.client_count,
            fills=strategy.fills if strategy is not None else 0,
            topics=len(strategy.universe.topics) if strategy is not None else 0,
            timeseries_model=strategy.timeseries_model if strategy is not None else None,
            last_error=self.last_error,
            min_expected_gain_sats=self.settings.min_expected_gain_sats,
            max_horizon_hours=self.settings.max_horizon_hours,
            assets=list(self.settings.assets),
        )

    def _held_by_topic(self, positions: dict[PositionKey, Position]) -> dict[int, float]:
        held: dict[int, float] = {}
        for (topic_id, _), position in positions.items():
            held[topic_id] = held.get(topic_id, 0.0) + position.contracts
        return held

    def tickers(self) -> list[SpotTicker]:
        strategy = self.strategy
        latest: dict[AssetType, EnsembleForecast] = {}
        counts: dict[AssetType, int] = {}
        if strategy is not None:
            for topic in strategy.universe.topics.values():
                counts[topic.asset] = counts.get(topic.asset, 0) + 1
            for candidate in strategy.forecasts.values():
                best = latest.get(candidate.asset)
                if best is None or candidate.ts > best.ts:
                    latest[candidate.asset] = candidate
        rows: list[SpotTicker] = []
        for asset in AssetType:
            forecast = latest.get(asset)
            price = forecast.spot if forecast is not None else None
            if price is not None:
                self._session_open.setdefault(asset, price)
            opened = self._session_open.get(asset)
            change = 100.0 * (price / opened - 1.0) if price is not None and opened else None
            rows.append(
                SpotTicker(
                    asset=asset,
                    price=price,
                    sources=list(forecast.spot_sources) if forecast is not None else [],
                    ts=forecast.ts if forecast is not None else None,
                    change_pct=change,
                    active_topics=counts.get(asset, 0),
                )
            )
        return rows

    def markets(self) -> MarketsResponse:
        strategy = self.strategy
        topics: list[MarketSummary] = []
        if strategy is not None:
            now = self._clock()
            held = self._held_by_topic(self.ledger.positions(self.mode))
            spots = {row.asset: row.price for row in self.tickers()}
            for topic in sorted(
                strategy.universe.topics.values(), key=lambda t: (t.asset.value, t.end_time_utc)
            ):
                book = strategy.universe.books.get(topic.topic_id)
                if book is None:
                    continue
                topics.append(
                    views.market_summary(
                        topic,
                        book,
                        strategy.forecasts.get(topic.topic_id),
                        held.get(topic.topic_id, 0.0),
                        spots.get(topic.asset),
                        self.settings,
                        now,
                    )
                )
        return MarketsResponse(tickers=self.tickers(), topics=topics)

    def _topic(self, topic_id: int) -> tuple[EngineStrategy, TopicMeta, Any]:
        strategy = self.strategy
        if strategy is None:
            raise NotFound("the engine has not loaded any markets")
        topic = strategy.universe.topics.get(topic_id)
        book = strategy.universe.books.get(topic_id)
        if topic is None or book is None:
            raise NotFound(f"topic {topic_id} is not in the tradable universe")
        return strategy, topic, book

    async def _fresh_forecast(
        self, strategy: EngineStrategy, topic: TopicMeta, wait: bool
    ) -> EnsembleForecast | None:
        """Cached forecast; refreshed from external feeds (never Glimpse) when stale."""
        now = self._clock()
        cached = strategy.forecasts.get(topic.topic_id)
        fresh = cached is not None and now - cached.ts < FORECAST_STALE_SECONDS
        if fresh or not strategy.ready or self.state != "running":
            return cached
        if wait:
            try:
                return await strategy.forecast_for(topic, now)
            except (ForecastError, httpx.HTTPError, ValueError) as exc:
                log.warning("forecast.on_demand_failed", topic_id=topic.topic_id, error=repr(exc))
                return cached
        job = self._forecast_jobs.get(topic.topic_id)
        if job is None or job.done():

            async def refresh() -> None:
                with contextlib.suppress(Exception):
                    await strategy.forecast_for(topic, self._clock())

            self._forecast_jobs[topic.topic_id] = asyncio.create_task(refresh())
        return cached

    async def distribution(self, topic_id: int, *, wait: bool = True) -> DistributionResponse:
        strategy, topic, book = self._topic(topic_id)
        forecast = await self._fresh_forecast(strategy, topic, wait)
        positions, cash, value = self._portfolio()
        equity = Millisats(int(cash) + int(value))
        held = {
            option_id: position.contracts
            for (held_topic, option_id), position in positions.items()
            if held_topic == topic_id
        }
        spot = next((t.price for t in self.tickers() if t.asset is topic.asset), None)
        risk = strategy.risk

        def budget_for(option_id: int) -> Millisats:
            return risk.entry_budget(topic_id, option_id, equity, cash, positions)

        return await views.distribution_view(
            topic, book, forecast, self.settings, held, spot, equity, budget_for, self._clock()
        )

    async def ladder(self, topic_id: int, option_id: int, max_contracts: int) -> LadderResponse:
        strategy, topic, book = self._topic(topic_id)
        forecast = await self._fresh_forecast(strategy, topic, wait=True)
        positions, cash, value = self._portfolio()
        equity = Millisats(int(cash) + int(value))
        budget = strategy.risk.entry_budget(topic_id, option_id, equity, cash, positions)
        try:
            return await views.ladder_view(
                topic, book, option_id, forecast, self.settings, equity, budget, max_contracts
            )
        except KeyError as exc:
            raise NotFound(f"bucket {option_id} is not in topic {topic_id}") from exc

    def positions(self) -> list[PositionView]:
        strategy = self.strategy
        positions = self.ledger.positions(self.mode)
        if strategy is None:
            return views.position_views(positions, {}, {}, {}, self.settings, self._clock())
        return views.position_views(
            positions,
            strategy.universe.topics,
            strategy.universe.books,
            strategy.forecasts,
            self.settings,
            self._clock(),
        )

    def risk_metrics(self) -> RiskMetrics:
        settings = self.settings
        strategy = self.strategy
        positions, cash, value = self._portfolio()
        equity = int(cash) + int(value)
        risk = strategy.risk if strategy is not None else None
        peak = int(risk.day_peak_millisats) if risk is not None else 0
        peak = max(peak, equity) if peak > 0 else equity
        drawdown = max(0.0, 1.0 - equity / peak) if peak > 0 else 0.0
        topics = strategy.universe.topics if strategy is not None else {}
        bucket_rows: list[ExposureRow] = []
        topic_cost: dict[int, int] = {}
        for (topic_id, option_id), position in positions.items():
            cost = int(position.cost_basis_millisats)
            topic_cost[topic_id] = topic_cost.get(topic_id, 0) + cost
            fraction = cost / equity if equity > 0 else 0.0
            topic = topics.get(topic_id)
            bucket_rows.append(
                ExposureRow(
                    topic_id=topic_id,
                    option_id=option_id,
                    label=position.bucket_name or str(option_id),
                    asset=topic.asset if topic is not None else None,
                    cost_basis_millisats=cost,
                    fraction_of_equity=fraction,
                    cap_fraction=settings.max_bucket_fraction,
                    utilization=fraction / settings.max_bucket_fraction,
                )
            )
        topic_rows = []
        for topic_id, cost in topic_cost.items():
            fraction = cost / equity if equity > 0 else 0.0
            topic = topics.get(topic_id)
            topic_rows.append(
                ExposureRow(
                    topic_id=topic_id,
                    option_id=None,
                    label=topic.title if topic is not None else f"topic {topic_id}",
                    asset=topic.asset if topic is not None else None,
                    cost_basis_millisats=cost,
                    fraction_of_equity=fraction,
                    cap_fraction=settings.max_topic_fraction,
                    utilization=fraction / settings.max_topic_fraction,
                )
            )
        now = self._clock()
        recent = self.ledger.page(self.mode, [RecordKind.ENTRY, RecordKind.EXIT], limit=50)
        open_cost = sum(int(p.cost_basis_millisats) for p in positions.values())
        return RiskMetrics(
            mode=self.mode,
            bankroll_millisats=int(settings.bankroll_millisats),
            equity_millisats=equity,
            cash_millisats=int(cash),
            positions_value_millisats=int(value),
            realized_pnl_millisats=int(self.ledger.realized_pnl_millisats(self.mode)),
            unrealized_pnl_millisats=int(value) - open_cost,
            day_peak_millisats=peak,
            intraday_drawdown=drawdown,
            drawdown_limit=settings.daily_drawdown_limit,
            breaker_tripped=bool(risk and risk.breaker_tripped),
            breaker_reason=risk.breaker_reason if risk is not None else None,
            trading_locked=bool(risk and risk.locked),
            lock_reason=risk.lock_reason if risk is not None else None,
            entries_allowed=bool(risk and risk.entries_allowed and self.state == "running"),
            max_bucket_fraction=settings.max_bucket_fraction,
            max_topic_fraction=settings.max_topic_fraction,
            bucket_exposures=sorted(bucket_rows, key=lambda r: -r.utilization),
            topic_exposures=sorted(topic_rows, key=lambda r: -r.utilization),
            open_positions=len(positions),
            max_open_positions=settings.max_open_positions,
            fills_last_minute=sum(1 for r in recent if now - r.ts < 60.0),
            max_orders_per_minute=settings.max_orders_per_minute,
            kelly_fraction=settings.kelly_fraction,
        )

    def ledger_page(
        self,
        mode: Mode | None,
        kinds: Sequence[RecordKind] | None,
        limit: int,
        before_id: int | None,
    ) -> LedgerPage:
        active: Mode = mode or self.mode
        selected = list(kinds) if kinds else list(FILL_KINDS)
        rows = self.ledger.page(active, selected, limit=limit, before_id=before_id)
        summary = self.ledger.summary(active, self.settings.bankroll_millisats)
        return LedgerPage(
            mode=active,
            total=self.ledger.count(active, selected),
            records=[views.record_view(row) for row in rows],
            next_before_id=rows[-1].record_id if len(rows) == limit and rows else None,
            summary=views.summary_view(summary),
        )

    def backtest_results(self, mode: Mode | None) -> BacktestResultsResponse:
        active: Mode = mode or self.mode
        curve = self.ledger.equity_curve(active)
        return BacktestResultsResponse(
            mode=active,
            equity_curve=views.equity_points(curve),
            metrics=views.performance(curve, self.settings.bankroll_millisats),
            latest_backtest=self._backtest,
            backtest_running=self._backtest_job is not None and not self._backtest_job.done(),
        )

    async def run_backtest(self, request: BacktestRunRequest) -> ActionResult:
        if self._backtest_job is not None and not self._backtest_job.done():
            raise Conflict("a backtest is already running")
        settings = self.base_settings

        def work() -> BacktestResult:
            ident = threading.get_ident()
            self.telemetry.muted_threads.add(ident)
            try:
                data = synthetic_dataset(seed=request.seed, mispriced=not request.control)
                return asyncio.run(
                    Backtester(settings, data, step_seconds=request.step_seconds).run()
                )
            finally:
                self.telemetry.muted_threads.discard(ident)

        async def job() -> None:
            try:
                result = await asyncio.to_thread(work)
            except Exception as exc:
                log.error("backtest.failed", error=repr(exc))
                return
            curve = [(float(ts), Millisats(value)) for ts, value in result.equity_curve]
            metrics = views.performance(curve, result.starting_millisats)
            if metrics is None:
                return
            self._backtest = BacktestRunView(
                finished_at=self._clock(),
                synthetic=True,
                control=request.control,
                seed=request.seed,
                step_seconds=request.step_seconds,
                metrics=metrics.model_copy(
                    update={
                        "sharpe": result.sharpe,
                        "sortino": result.sortino,
                        "max_drawdown": result.max_drawdown,
                        "roi": result.roi,
                    }
                ),
                win_rate=result.win_rate,
                entries=result.entries,
                exits=result.exits,
                settlements=result.settlements,
                commission_millisats=int(result.commission_millisats),
                equity_curve=views.equity_points(curve),
            )
            log.info(
                "backtest.finished",
                roi=round(result.roi, 6),
                sharpe=round(result.sharpe, 3),
                max_drawdown=round(result.max_drawdown, 6),
            )

        self._backtest_job = asyncio.create_task(job())
        log.info("backtest.started", control=request.control, seed=request.seed)
        return self._result("synthetic backtest started; results appear when it finishes")

    # --- telemetry --------------------------------------------------------------------

    def recent_fills(self, limit: int = 50) -> list[LedgerRecord]:
        return self.ledger.page(self.mode, list(FILL_KINDS), limit=limit)

    def hello(self) -> dict[str, Any]:
        return {
            "engine_status": self.health().model_dump(mode="json"),
            "market_quotes": self.markets().model_dump(mode="json"),
            "risk_metrics": self.risk_metrics().model_dump(mode="json"),
            "active_positions": [p.model_dump(mode="json") for p in self.positions()],
            "ledger_fills": [
                views.record_view(r).model_dump(mode="json") for r in self.recent_fills()
            ],
            "system_logs": [e.model_dump(mode="json") for e in self.telemetry.recent_logs()],
        }

    async def push_distribution(self, subscriber: Subscriber, *, force: bool = False) -> None:
        topic_id = subscriber.focus_topic
        strategy = self.strategy
        if topic_id is None or strategy is None:
            return
        book = strategy.universe.books.get(topic_id)
        if book is None:
            return
        forecast = strategy.forecasts.get(topic_id)
        key = (book.ts, forecast.ts if forecast is not None else 0.0)
        if not force and subscriber.last_distribution_key == key:
            await self._fresh_forecast(strategy, strategy.universe.topics[topic_id], wait=False)
            return
        try:
            view = await self.distribution(topic_id, wait=False)
        except ControlError:
            return
        subscriber.last_distribution_key = key
        self.telemetry.send(subscriber, "ensemble_distribution", view.model_dump(mode="json"))

    async def _publish_loop(self) -> None:
        tick = 0
        while True:
            await asyncio.sleep(PUBLISH_INTERVAL_SECONDS)
            tick += 1
            try:
                await self._publish(tick)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # telemetry must never take the engine down
                log.warning("telemetry.publish_failed", error=repr(exc))

    async def _publish(self, tick: int) -> None:
        hub = self.telemetry
        fills = self.ledger.records_after(self._last_fill_id, self.mode, list(FILL_KINDS))
        self._last_fill_id = max([self._last_fill_id, *(r.record_id for r in fills)])
        if not hub.client_count:
            return
        hub.publish("engine_status", self.health().model_dump(mode="json"))
        hub.publish("risk_metrics", self.risk_metrics().model_dump(mode="json"))
        hub.publish("active_positions", [p.model_dump(mode="json") for p in self.positions()])
        if fills:
            hub.publish(
                "ledger_fills", [views.record_view(r).model_dump(mode="json") for r in fills]
            )
        if tick % 2 == 0:
            hub.publish("market_quotes", self.markets().model_dump(mode="json"))
            for subscriber in hub.subscribers():
                await self.push_distribution(subscriber)
