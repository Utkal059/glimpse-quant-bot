"""Quant Terminal server: the HTTP surface cannot weaken the engine's safety rules.

Every test drives the real FastAPI app, the real ``EngineController`` and the
unmodified strategy against the in-process simulated venue, so nothing touches
the network or a real ``.env``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from glimpse_markets.money import Millisats

from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.ledger import Ledger
from glimpse_quant_bot.lmsr import Leg
from glimpse_quant_bot.lmsr_optimizer import CostQuote, ServerEstimator
from glimpse_quant_bot.risk_manager import BreakerState, RiskManager
from glimpse_quant_bot.server import EngineController, create_app, simulated_venue, typescript
from glimpse_quant_bot.server import controller as controller_module
from glimpse_quant_bot.server.api import DEFAULT_HOSTS, INTENT_HEADER
from glimpse_quant_bot.server.schemas import (
    ARM_PHRASE,
    LIVE_ORDER_PHRASE,
    RELEASE_PHRASE,
    StartRequest,
)
from glimpse_quant_bot.server.simulation import SimulatedWorld
from glimpse_quant_bot.types import AssetType, RecordKind

OPERATOR = {INTENT_HEADER: "operator"}
TERMINAL_ORIGIN = "http://localhost:3000"
EVIL_ORIGIN = "https://evil.example"
WAIT_SECONDS = 90.0


class Terminal:
    """One app + controller + simulated world, driven through ``TestClient``."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        live_env: bool = False,
        allow_live: bool = False,
        venue_live: bool = False,
        **overrides: Any,
    ) -> None:
        base: dict[str, Any] = {
            "data_dir": tmp_path,
            "garch_paths": 200,
            "enable_timesfm": False,
            "tick_interval_seconds": 1.0,
            "resolution_poll_seconds": 600.0,
            "calibration_poll_seconds": 600.0,
            # The engine's own entries need a huge gain, so only operator
            # tickets trade and every assertion below is deterministic.
            "min_expected_gain_sats": 1_000_000.0,
            "glimpse_live_trading": live_env,
            "glimpse_api_key": "test-key-not-real" if live_env else None,
            **overrides,
        }

        def factory(**runtime: Any) -> Settings:
            return Settings(_env_file=None, **{**base, **runtime})

        self.world = SimulatedWorld(
            (AssetType.BTC, AssetType.ETH), seed=3, allow_fake_live=venue_live
        )
        self.controller = EngineController(
            settings_factory=factory,
            venue=simulated_venue(self.world, supports_live=venue_live),
            allow_live=allow_live,
        )
        self.app = create_app(self.controller, allowed_hosts=[*DEFAULT_HOSTS, "testserver"])

    def client(self) -> TestClient:
        return TestClient(self.app, base_url="http://localhost")


def wait_until(
    client: TestClient, predicate: Callable[[dict[str, Any]], bool], what: str
) -> dict[str, Any]:
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        health: dict[str, Any] = client.get("/api/health").json()
        if predicate(health):
            return health
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for {what}")


def start_paper(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/api/bot/start", json=body, headers=OPERATOR)
    assert response.status_code == 200, response.text
    return wait_until(client, lambda h: h["engine_state"] == "running", "a running engine")


def tradable_topic(client: TestClient) -> tuple[int, int]:
    """The longest-dated topic (comfortably inside the expiry window) and its mode bucket."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        topics = client.get("/api/markets").json()["topics"]
        live = [t for t in topics if t["seconds_to_expiry"] > 600]
        if live:
            topic = max(live, key=lambda t: t["seconds_to_expiry"])
            columns = client.get(f"/api/distribution/{topic['topic_id']}").json()["buckets"]
            prices = columns["market_price"]
            best = max(range(len(prices)), key=prices.__getitem__)
            return int(topic["topic_id"]), int(columns["option_id"][best])
        time.sleep(0.25)
    raise AssertionError("no tradable topic appeared")


@pytest.fixture
def terminal(tmp_path: Path) -> Terminal:
    return Terminal(tmp_path)


@pytest.fixture
def client(terminal: Terminal) -> Iterator[TestClient]:
    with terminal.client() as test_client:
        yield test_client


# --- paper-first defaults ---------------------------------------------------------


def test_defaults_are_paper(terminal: Terminal, client: TestClient) -> None:
    assert StartRequest().dry_run is True
    health = client.get("/api/health").json()
    assert health["mode"] == "PAPER"
    assert health["dry_run"] is True
    assert health["engine_state"] == "stopped"
    assert health["live_eligibility"]["eligible"] is False

    health = start_paper(client)
    assert health["mode"] == "PAPER"
    assert health["dry_run"] is True
    assert terminal.controller.mode == "dry_run"
    sdk = terminal.controller.client
    assert sdk is not None
    assert sdk.dry_run is True


def test_live_requires_server_flag(tmp_path: Path) -> None:
    terminal = Terminal(tmp_path, live_env=False, allow_live=True, venue_live=True)
    with terminal.client() as client:
        armed = client.post(
            "/api/bot/arm", json={"confirmation_phrase": ARM_PHRASE}, headers=OPERATOR
        )
        assert armed.status_code == 403
        reasons = " ".join(armed.json()["reasons"])
        assert "GLIMPSE_LIVE_TRADING" in reasons
        assert "GLIMPSE_API_KEY" in reasons
        started = client.post(
            "/api/bot/start", json={"dry_run": False, "arming_token": "x"}, headers=OPERATOR
        )
        assert started.status_code == 403
        assert client.get("/api/health").json()["mode"] == "PAPER"


def test_live_requires_allow_live_flag(tmp_path: Path) -> None:
    terminal = Terminal(tmp_path, live_env=True, allow_live=False, venue_live=True)
    with terminal.client() as client:
        armed = client.post(
            "/api/bot/arm", json={"confirmation_phrase": ARM_PHRASE}, headers=OPERATOR
        )
        assert armed.status_code == 403
        assert any("--allow-live" in reason for reason in armed.json()["reasons"])


def test_simulated_venue_can_never_go_live(tmp_path: Path) -> None:
    terminal = Terminal(tmp_path, live_env=True, allow_live=True, venue_live=False)
    with terminal.client() as client:
        armed = client.post(
            "/api/bot/arm", json={"confirmation_phrase": ARM_PHRASE}, headers=OPERATOR
        )
        assert armed.status_code == 403
        assert any("simulated venue" in reason for reason in armed.json()["reasons"])


def test_live_requires_typed_arming_and_a_single_use_token(tmp_path: Path) -> None:
    terminal = Terminal(tmp_path, live_env=True, allow_live=True, venue_live=True)
    with terminal.client() as client:
        assert client.get("/api/health").json()["live_eligibility"]["eligible"] is True

        # 1. dry_run=false without arming is refused.
        refused = client.post("/api/bot/start", json={"dry_run": False}, headers=OPERATOR)
        assert refused.status_code == 403
        # 2. a sloppy phrase does not arm.
        sloppy = client.post("/api/bot/arm", json={"confirmation_phrase": "arm"}, headers=OPERATOR)
        assert sloppy.status_code == 403
        # 3. a wrong token burns the pending one.
        token = client.post(
            "/api/bot/arm", json={"confirmation_phrase": ARM_PHRASE}, headers=OPERATOR
        ).json()["arming_token"]
        wrong = client.post(
            "/api/bot/start",
            json={"dry_run": False, "arming_token": token + "x"},
            headers=OPERATOR,
        )
        assert wrong.status_code == 403
        burned = client.post(
            "/api/bot/start", json={"dry_run": False, "arming_token": token}, headers=OPERATOR
        )
        assert burned.status_code == 403
        assert client.get("/api/health").json()["mode"] == "PAPER"

        # 4. phrase + fresh token + env + --allow-live: live.
        token = client.post(
            "/api/bot/arm", json={"confirmation_phrase": ARM_PHRASE}, headers=OPERATOR
        ).json()["arming_token"]
        started = client.post(
            "/api/bot/start", json={"dry_run": False, "arming_token": token}, headers=OPERATOR
        )
        assert started.status_code == 200, started.text
        assert started.json()["mode"] == "LIVE_ARMED"
        notes = [r.note for r in terminal.controller.ledger.records("live")]
        assert any("live trading armed" in note for note in notes)

        # 5. live operator tickets need their own typed phrase.
        ticket = client.post(
            "/api/orders/test",
            json={"topic_id": 1, "option_id": 1, "contracts": 1},
            headers=OPERATOR,
        )
        assert ticket.status_code == 403
        assert LIVE_ORDER_PHRASE in ticket.json()["detail"]

        # 6. the token was single use: stop, then reuse fails.
        assert client.post("/api/bot/stop", headers=OPERATOR).status_code == 200
        reused = client.post(
            "/api/bot/start", json={"dry_run": False, "arming_token": token}, headers=OPERATOR
        )
        assert reused.status_code == 403


def test_arming_token_expires(tmp_path: Path) -> None:
    terminal = Terminal(tmp_path, live_env=True, allow_live=True, venue_live=True)
    now = [time.time()]
    terminal.controller._clock = lambda: now[0]
    with terminal.client() as client:
        token = client.post(
            "/api/bot/arm", json={"confirmation_phrase": ARM_PHRASE}, headers=OPERATOR
        ).json()["arming_token"]
        now[0] += 61.0
        late = client.post(
            "/api/bot/start", json={"dry_run": False, "arming_token": token}, headers=OPERATOR
        )
        assert late.status_code == 403


# --- request hygiene ------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"bankroll_sats": 1_000_000_000},
        {"max_bucket_fraction": 1.0},
        {"daily_drawdown_limit": 0.99},
        {"glimpse_live_trading": True},
        {"dry_run": True, "max_topic_fraction": 1.0},
    ],
)
def test_unknown_fields_rejected(client: TestClient, body: dict[str, Any]) -> None:
    response = client.post("/api/bot/start", json=body, headers=OPERATOR)
    assert response.status_code == 422
    assert client.get("/api/health").json()["engine_state"] == "stopped"


@pytest.mark.parametrize(
    "body",
    [
        {"min_expected_gain_sats": -1},
        {"max_horizon_hours": 0},
        {"max_horizon_hours": 10_000},
        {"assets": []},
        {"assets": ["DOGE"]},
    ],
)
def test_start_rejects_risk_overrides(client: TestClient, body: dict[str, Any]) -> None:
    response = client.post("/api/bot/start", json=body, headers=OPERATOR)
    assert response.status_code == 422


def test_allowed_overrides_never_touch_risk_caps(terminal: Terminal, client: TestClient) -> None:
    start_paper(client, min_expected_gain_sats=5.0, max_horizon_hours=2.0, assets=["BTC"])
    settings = terminal.controller.settings
    assert settings.min_expected_gain_sats == 5.0
    assert settings.max_horizon_hours == 2.0
    assert settings.assets == (AssetType.BTC,)
    assert settings.max_bucket_fraction == 0.15
    assert settings.max_topic_fraction == 0.35
    assert settings.daily_drawdown_limit == 0.06


def test_cross_origin_post_rejected(client: TestClient) -> None:
    evil = client.post("/api/bot/kill-switch", headers={**OPERATOR, "Origin": EVIL_ORIGIN}, json={})
    assert evil.status_code == 403
    missing_intent = client.post("/api/bot/start", json={})
    assert missing_intent.status_code == 403
    preflight = client.options(
        "/api/bot/start",
        headers={
            "Origin": EVIL_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"content-type,{INTENT_HEADER.lower()}",
        },
    )
    assert preflight.headers.get("access-control-allow-origin") != EVIL_ORIGIN
    ok = client.options(
        "/api/bot/start",
        headers={
            "Origin": TERMINAL_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"content-type,{INTENT_HEADER.lower()}",
        },
    )
    assert ok.headers.get("access-control-allow-origin") == TERMINAL_ORIGIN
    assert client.get("/api/health").json()["kill_switch_engaged"] is False


def test_foreign_host_header_rejected(client: TestClient) -> None:
    response = client.get("/api/health", headers={"Host": "attacker.example"})
    assert response.status_code == 400


def test_cross_origin_websocket_rejected(client: TestClient) -> None:
    from starlette.testclient import WebSocketDenialResponse
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises((WebSocketDisconnect, WebSocketDenialResponse)),
        client.websocket_connect("/ws/telemetry", headers={"Origin": EVIL_ORIGIN}) as ws,
    ):
        ws.receive_text()


# --- risk locks survive the HTTP surface ------------------------------------------------


def test_manual_order_respects_caps(tmp_path: Path) -> None:
    terminal = Terminal(tmp_path, bankroll_sats=2_000)  # 15% bucket cap = 300 sats
    with terminal.client() as client:
        start_paper(client)
        topic_id, option_id = tradable_topic(client)
        big = client.post(
            "/api/orders/test",
            json={"topic_id": topic_id, "option_id": option_id, "contracts": 200},
            headers=OPERATOR,
        )
        assert big.status_code == 200, big.text
        result = big.json()
        assert result["approved"] is False
        assert result["fill"] is None
        assert any("budget" in reason for reason in result["reasons"])
        events = terminal.controller.ledger.records("dry_run", kinds=[RecordKind.RISK_EVENT])
        assert any("operator entry blocked" in record.note for record in events)
        entries = terminal.controller.ledger.records("dry_run", kinds=[RecordKind.ENTRY])
        assert entries == []

        small = client.post(
            "/api/orders/test",
            json={"topic_id": topic_id, "option_id": option_id, "contracts": 1},
            headers=OPERATOR,
        )
        assert small.status_code == 200, small.text
        assert small.json()["approved"] is True
        assert small.json()["fill"]["kind"] == "entry"
        page = client.get("/api/ledger").json()
        assert page["total"] == 1
        assert page["records"][0]["option_id"] == option_id


def test_blocked_ticket_never_spends_a_venue_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ticket the RiskManager refuses is answered from the local replica at once:
    it never queues behind the shared Glimpse REST budget."""
    quoted: list[int] = []

    class SpyEstimator(ServerEstimator):
        async def quote_buy(self, topic_id: int, legs: Sequence[Leg]) -> CostQuote:
            quoted.append(topic_id)
            return await super().quote_buy(topic_id, legs)

    monkeypatch.setattr(controller_module, "ServerEstimator", SpyEstimator)
    terminal = Terminal(tmp_path, bankroll_sats=2_000)
    with terminal.client() as client:
        start_paper(client)
        topic_id, option_id = tradable_topic(client)
        blocked = client.post(
            "/api/orders/test",
            json={"topic_id": topic_id, "option_id": option_id, "contracts": 200},
            headers=OPERATOR,
        )
        assert blocked.status_code == 200, blocked.text
        assert blocked.json()["approved"] is False
        assert quoted == []

        approved = client.post(
            "/api/orders/test",
            json={"topic_id": topic_id, "option_id": option_id, "contracts": 1},
            headers=OPERATOR,
        )
        assert approved.json()["approved"] is True
        assert quoted == [topic_id]  # the venue's quote is still what gets routed


def test_manual_order_validates_size(client: TestClient) -> None:
    for contracts in (0, -5, 401):
        response = client.post(
            "/api/orders/test",
            json={"topic_id": 1, "option_id": 1, "contracts": contracts},
            headers=OPERATOR,
        )
        assert response.status_code == 422


def test_order_refused_while_stopped(client: TestClient) -> None:
    response = client.post(
        "/api/orders/test",
        json={"topic_id": 1, "option_id": 1, "contracts": 1},
        headers=OPERATOR,
    )
    assert response.status_code == 409


def test_breaker_survives_restart(terminal: Terminal, client: TestClient) -> None:
    start_paper(client)
    strategy = terminal.controller.strategy
    assert strategy is not None
    now = time.time()
    strategy.risk.update_equity(Millisats(100_000_000), now)
    assert strategy.risk.update_equity(Millisats(90_000_000), now + 1)
    assert client.get("/api/health").json()["breaker_tripped"] is True

    assert client.post("/api/bot/stop", headers=OPERATOR).status_code == 200
    health = start_paper(client)
    assert health["breaker_tripped"] is True
    assert health["entries_allowed"] is False
    restarted = terminal.controller.strategy
    assert restarted is not None
    assert restarted is not strategy
    assert restarted.risk.breaker_tripped


def test_kill_switch_blocks_start(terminal: Terminal, client: TestClient) -> None:
    start_paper(client)
    killed = client.post("/api/bot/kill-switch", json={"reason": "test"}, headers=OPERATOR)
    assert killed.status_code == 200
    assert killed.json()["engine_state"] == "halted"
    health = client.get("/api/health").json()
    assert health["kill_switch_engaged"] is True
    assert health["trading_locked"] is True

    assert client.post("/api/bot/start", json={}, headers=OPERATOR).status_code == 409
    ticket = client.post(
        "/api/orders/test",
        json={"topic_id": 1, "option_id": 1, "contracts": 1},
        headers=OPERATOR,
    )
    assert ticket.status_code == 409
    arm = client.post("/api/bot/arm", json={"confirmation_phrase": ARM_PHRASE}, headers=OPERATOR)
    assert arm.status_code == 403

    sloppy = client.post(
        "/api/bot/kill-switch/release", json={"confirmation_phrase": "ok"}, headers=OPERATOR
    )
    assert sloppy.status_code == 403
    released = client.post(
        "/api/bot/kill-switch/release",
        json={"confirmation_phrase": RELEASE_PHRASE},
        headers=OPERATOR,
    )
    assert released.status_code == 200
    assert released.json()["engine_state"] == "stopped"
    start_paper(client)


def test_single_ledger_instance(terminal: Terminal, client: TestClient) -> None:
    owner = terminal.controller.ledger
    start_paper(client)
    first = terminal.controller.strategy
    assert first is not None
    assert first._ledger is owner
    client.post("/api/bot/stop", headers=OPERATOR)
    start_paper(client)
    second = terminal.controller.strategy
    assert second is not None
    assert second._ledger is owner


# --- rate-limit discipline and telemetry ----------------------------------------------


def test_reads_make_no_glimpse_requests(terminal: Terminal, client: TestClient) -> None:
    start_paper(client)
    topic_id, option_id = tradable_topic(client)
    time.sleep(1.5)  # let the startup burst finish
    before_world = terminal.world.requests
    before_limiter = terminal.controller.limiter.used()
    for _ in range(5):
        for path in (
            "/api/health",
            "/api/markets",
            f"/api/distribution/{topic_id}",
            f"/api/ladder/{topic_id}/{option_id}?max_contracts=50",
            "/api/positions",
            "/api/risk",
            "/api/ledger",
            "/api/backtest/results",
            "/api/logs",
        ):
            assert client.get(path).status_code == 200, path
    assert terminal.world.requests == before_world
    assert terminal.controller.limiter.used() == before_limiter


def test_websocket_hello_focus_and_pong(client: TestClient) -> None:
    start_paper(client)
    topic_id, _ = tradable_topic(client)
    with client.websocket_connect("/ws/telemetry") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello"
        assert set(hello["data"]) >= {
            "engine_status",
            "market_quotes",
            "risk_metrics",
            "active_positions",
            "ledger_fills",
            "system_logs",
        }
        ws.send_text(json.dumps({"type": "focus", "topic_id": topic_id}))
        ws.send_text(json.dumps({"type": "ping", "t": 42.5}))
        ws.send_text("not json")  # ignored, never fatal
        seen: dict[str, Any] = {}
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not {"pong", "ensemble_distribution"} <= set(seen):
            frame = json.loads(ws.receive_text())
            seen[frame["type"]] = frame["data"]
        assert seen["pong"] == {"t": 42.5}
        assert seen["ensemble_distribution"]["topic_id"] == topic_id
        assert len(seen["ensemble_distribution"]["buckets"]["option_id"]) == 500


def test_ledger_endpoint_validates_query(client: TestClient) -> None:
    assert client.get("/api/ledger?limit=0").status_code == 422
    assert client.get("/api/ledger?limit=501").status_code == 422
    assert client.get("/api/ledger?mode=yolo").status_code == 422
    assert client.get("/api/ledger?kinds=entry&kinds=exit").status_code == 200


# --- engine-side helpers the server relies on --------------------------------------------


def test_ledger_paging_helpers() -> None:
    ledger = Ledger(":memory:", clock=lambda: 1_800_000_000.0)
    try:
        for i in range(7):
            ledger.record_event("dry_run", RecordKind.RISK_EVENT, f"event {i}")
        ledger.record_event("live", RecordKind.RISK_EVENT, "live event")
        first = ledger.page("dry_run", [RecordKind.RISK_EVENT], limit=3)
        assert [r.note for r in first] == ["event 6", "event 5", "event 4"]
        second = ledger.page("dry_run", [RecordKind.RISK_EVENT], 3, first[-1].record_id)
        assert [r.note for r in second] == ["event 3", "event 2", "event 1"]
        assert ledger.count("dry_run", [RecordKind.RISK_EVENT]) == 7
        assert ledger.count("live") == 1
        newer = ledger.records_after(first[0].record_id - 2, "dry_run")
        assert [r.note for r in newer] == ["event 5", "event 6"]
        assert ledger.last_record_id() == 8

        ledger.record_equity("dry_run", Millisats(90_000_000), Millisats(5_000_000))
        ledger.record_equity("dry_run", Millisats(91_000_000), Millisats(6_000_000))
        curve = ledger.equity_curve("dry_run")
        assert [int(equity) for _, equity in curve] == [95_000_000, 97_000_000]
    finally:
        ledger.close()


def test_restore_breaker_only_applies_the_same_utc_day(settings: Settings) -> None:
    now = 1_800_000_000.0
    ledger = Ledger(":memory:", clock=lambda: now)
    try:
        risk = RiskManager(settings, ledger, "dry_run", clock=lambda: now)
        risk.update_equity(Millisats(100_000_000), now)
        assert risk.update_equity(Millisats(90_000_000), now + 1)
        state = risk.breaker_state()

        same_day = RiskManager(settings, ledger, "dry_run", clock=lambda: now)
        assert same_day.restore_breaker(state, now + 60)
        assert same_day.breaker_tripped
        assert not same_day.entries_allowed

        next_day = RiskManager(settings, ledger, "dry_run", clock=lambda: now)
        assert not next_day.restore_breaker(state, now + 86_400)
        assert not next_day.breaker_tripped

        empty = BreakerState(day=None, day_peak_millisats=Millisats(0), reason=None)
        assert not RiskManager(settings, ledger, "dry_run").restore_breaker(empty)
    finally:
        ledger.close()


def test_typescript_contract_is_current() -> None:
    """``frontend/src/lib/contract.ts`` is generated from the Pydantic models; a
    schema change without regenerating it would let the UI drift from the API."""
    contract = Path(__file__).resolve().parents[1] / "frontend" / "src" / "lib" / "contract.ts"
    if not contract.exists():
        pytest.skip("frontend/ is not part of this checkout")
    on_disk = contract.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert on_disk == typescript.render().replace("\r\n", "\n"), (
        "contract.ts is stale: run "
        "`python -m glimpse_quant_bot.server.typescript > frontend/src/lib/contract.ts`"
    )
