"""HTTP + WebSocket surface of the Quant Terminal.

Every route is a thin adapter over :class:`EngineController`; nothing here
talks to Glimpse, builds orders or touches the ledger directly. That keeps the
safety argument in one place (the controller), and lets this module focus on
the transport-level guards:

* **Loopback by default.** ``TrustedHostMiddleware`` rejects requests whose
  ``Host`` is not an allowed name, which defeats DNS-rebinding pages.
* **Operator intent.** Every state-changing request (anything but GET/HEAD/
  OPTIONS) must carry ``X-GQB-Intent: operator`` *and* either no ``Origin`` or
  an allowed one. A cross-site page cannot set a custom header without a CORS
  preflight, and the preflight only succeeds for the terminal's own origins,
  so a malicious tab cannot stop, start or arm the engine.
* **Strict bodies.** Request models forbid unknown fields, so a client cannot
  smuggle risk overrides (``bankroll``, caps, ``dry_run=false`` without a
  token, ...) through extra JSON keys.
* **Reads are free.** GET handlers serve in-memory snapshots; a page refresh
  never costs a Glimpse request.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import structlog
from fastapi import Body, FastAPI, Path, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

from glimpse_quant_bot import __version__
from glimpse_quant_bot.server.controller import ControlError, EngineController
from glimpse_quant_bot.server.schemas import (
    ActionResult,
    ArmRequest,
    ArmResponse,
    BacktestResultsResponse,
    BacktestRunRequest,
    DistributionResponse,
    ErrorDetail,
    HealthResponse,
    KillSwitchRequest,
    LadderResponse,
    LedgerPage,
    LogCategory,
    LogEntry,
    MarketsResponse,
    ModeName,
    PositionView,
    ReconcileResult,
    RecordKindName,
    ReleaseRequest,
    RiskMetrics,
    StartRequest,
    TestOrderRequest,
    TestOrderResult,
)
from glimpse_quant_bot.server.telemetry import LOG_BUFFER, Subscriber
from glimpse_quant_bot.types import RecordKind

log = structlog.get_logger(__name__)

INTENT_HEADER = "X-GQB-Intent"
INTENT_VALUE = "operator"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
DEFAULT_ORIGINS: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")
DEFAULT_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1", "[::1]")
WS_POLICY_VIOLATION = 1008
MAX_CLIENT_MESSAGE = 4096


# --- client -> server WebSocket messages --------------------------------------


class _ClientMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class FocusMessage(_ClientMessage):
    type: Literal["focus"]
    topic_id: int | None = Field(default=None, gt=0)


class PingMessage(_ClientMessage):
    type: Literal["ping"]
    t: float


ClientMessage = Annotated[FocusMessage | PingMessage, Field(discriminator="type")]
_CLIENT_MESSAGE: TypeAdapter[FocusMessage | PingMessage] = TypeAdapter(ClientMessage)


def _error(status_code: int, detail: str, reasons: Sequence[str] = ()) -> JSONResponse:
    body = ErrorDetail(detail=detail, reasons=list(reasons)).model_dump(mode="json")
    return JSONResponse(status_code=status_code, content=body)


def create_app(
    controller: EngineController,
    *,
    allowed_origins: Sequence[str] = DEFAULT_ORIGINS,
    allowed_hosts: Sequence[str] = DEFAULT_HOSTS,
) -> FastAPI:
    """Build the FastAPI app around one controller (and therefore one ledger)."""
    origins = frozenset(origin.rstrip("/") for origin in allowed_origins)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await controller.startup()
        try:
            yield
        finally:
            await controller.shutdown()

    app = FastAPI(
        title="Glimpse Quant Terminal",
        version=__version__,
        summary="Paper-first telemetry and control plane for glimpse-quant-bot.",
        lifespan=lifespan,
    )
    app.state.controller = controller

    @app.exception_handler(ControlError)
    async def control_error(_: Request, exc: ControlError) -> JSONResponse:
        return _error(exc.status_code, exc.detail, exc.reasons)

    @app.middleware("http")
    async def operator_guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method not in SAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and origin.rstrip("/") not in origins:
                log.warning("operator.refused", reason="origin", origin=origin)
                return _error(403, "cross-origin control requests are refused")
            if request.headers.get(INTENT_HEADER) != INTENT_VALUE:
                log.warning("operator.refused", reason="intent", path=request.url.path)
                return _error(403, f"control requests must send '{INTENT_HEADER}: operator'")
        return await call_next(request)

    # Added last == outermost: host check, then CORS (so even refusals carry
    # CORS headers the terminal can read), then the operator guard above.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", INTENT_HEADER],
        allow_credentials=False,
        max_age=600,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    # --- reads (in-memory only) -----------------------------------------------

    @app.get("/api/health")
    async def health() -> HealthResponse:
        return controller.health()

    @app.get("/api/markets")
    async def markets() -> MarketsResponse:
        return controller.markets()

    @app.get("/api/distribution/{topic_id}")
    async def distribution(topic_id: Annotated[int, Path(gt=0)]) -> DistributionResponse:
        return await controller.distribution(topic_id)

    @app.get("/api/ladder/{topic_id}/{option_id}")
    async def ladder(
        topic_id: Annotated[int, Path(gt=0)],
        option_id: Annotated[int, Path(gt=0)],
        max_contracts: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> LadderResponse:
        return await controller.ladder(topic_id, option_id, max_contracts)

    @app.get("/api/positions")
    async def positions() -> list[PositionView]:
        return controller.positions()

    @app.get("/api/risk")
    async def risk() -> RiskMetrics:
        return controller.risk_metrics()

    @app.get("/api/ledger")
    async def ledger(
        mode: Annotated[ModeName | None, Query()] = None,
        kinds: Annotated[list[RecordKindName] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        before_id: Annotated[int | None, Query(gt=0)] = None,
    ) -> LedgerPage:
        selected = [RecordKind(kind) for kind in kinds] if kinds else None
        return controller.ledger_page(mode, selected, limit, before_id)

    @app.get("/api/backtest/results")
    async def backtest_results(
        mode: Annotated[ModeName | None, Query()] = None,
    ) -> BacktestResultsResponse:
        return controller.backtest_results(mode)

    @app.get("/api/logs")
    async def logs(
        limit: Annotated[int, Query(ge=1, le=LOG_BUFFER)] = 200,
        category: Annotated[LogCategory | None, Query()] = None,
    ) -> list[LogEntry]:
        entries = controller.telemetry.recent_logs(LOG_BUFFER)
        if category is not None:
            entries = [entry for entry in entries if entry.category == category]
        return entries[-limit:]

    # --- control (operator intent required) ------------------------------------

    @app.post("/api/bot/start")
    async def start(request: StartRequest) -> ActionResult:
        return await controller.start(request)

    @app.post("/api/bot/stop")
    async def stop() -> ActionResult:
        return await controller.stop()

    @app.post("/api/bot/arm")
    async def arm(request: ArmRequest) -> ArmResponse:
        return controller.arm(request.confirmation_phrase)

    @app.post("/api/bot/disarm")
    async def disarm() -> ActionResult:
        return controller.disarm()

    @app.post("/api/bot/reconcile")
    async def reconcile() -> ReconcileResult:
        return await controller.reconcile()

    @app.post("/api/bot/kill-switch")
    async def kill_switch(
        request: Annotated[KillSwitchRequest | None, Body()] = None,
    ) -> ActionResult:
        return await controller.kill((request or KillSwitchRequest()).reason)

    @app.post("/api/bot/kill-switch/release")
    async def release(request: ReleaseRequest) -> ActionResult:
        return await controller.release(request.confirmation_phrase)

    @app.post("/api/orders/test")
    async def test_order(request: TestOrderRequest) -> TestOrderResult:
        return await controller.test_order(request)

    @app.post("/api/backtest/run")
    async def run_backtest(
        request: Annotated[BacktestRunRequest | None, Body()] = None,
    ) -> ActionResult:
        return await controller.run_backtest(request or BacktestRunRequest())

    # --- telemetry stream ---------------------------------------------------------

    async def pump(websocket: WebSocket, subscriber: Subscriber) -> None:
        while True:
            frame = await subscriber.queue.get()
            await websocket.send_text(frame)

    async def handle(raw: str, subscriber: Subscriber) -> None:
        if len(raw) > MAX_CLIENT_MESSAGE:
            return
        try:
            message = _CLIENT_MESSAGE.validate_json(raw)
        except ValidationError:
            return
        if isinstance(message, PingMessage):
            controller.telemetry.send(subscriber, "pong", {"t": message.t})
            return
        subscriber.focus_topic = message.topic_id
        subscriber.last_distribution_key = None
        await controller.push_distribution(subscriber, force=True)

    @app.websocket("/ws/telemetry")
    async def telemetry(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin is not None and origin.rstrip("/") not in origins:
            log.warning("operator.refused", reason="ws-origin", origin=origin)
            await websocket.close(code=WS_POLICY_VIOLATION)
            return
        await websocket.accept()
        hub = controller.telemetry
        subscriber = hub.subscribe()
        hub.send(subscriber, "hello", controller.hello())
        sender = asyncio.create_task(pump(websocket, subscriber))
        try:
            while True:
                receive = asyncio.ensure_future(websocket.receive_text())
                done, _ = await asyncio.wait({receive, sender}, return_when=asyncio.FIRST_COMPLETED)
                if sender in done:
                    receive.cancel()
                    break
                await handle(receive.result(), subscriber)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(subscriber)
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender

    return app
