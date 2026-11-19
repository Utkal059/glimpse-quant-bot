# Quant Terminal: integration blueprint

The terminal is a thin layer over the engine. It adds no pricing, sizing or risk
logic of its own: every number it shows is computed by the same objects the
headless `run` command uses, and every order it can send goes through
`RiskManager.check_entry` and `ExecutionRouter`.

```text
                    one process, one event loop, one Ledger connection
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ EngineController (server/controller.py)                                                │
 │   ├─ ServedStrategy(GlimpseQuantStrategy)  ← the real strategy, unmodified logic        │
 │   │     MarketDataHub → EnsembleForecaster → SlippageOptimizer → RiskManager → Router    │
 │   ├─ StrategyRunner task (reconnect + backoff, as in `cmd_run`)                         │
 │   ├─ Ledger (owned by the controller, handed to every strategy instance)               │
 │   ├─ CountingRateLimiter (shared by every client; feeds the budget gauge)              │
 │   └─ operator lock (asyncio.Lock): start / stop / kill / reconcile / manual orders     │
 │                                                                                        │
 │ TelemetryHub (server/telemetry.py)  ← structlog tap + 1 s publisher over in-memory state │
 │ FastAPI app (server/api.py)  REST + ws://localhost:8000/ws/telemetry                    │
 └──────────────────────────────────────────────────────────────────────────────────────┘
                                          ▲ JSON (Pydantic v2 → strict TS interfaces)
                         Next.js terminal (frontend/), localhost:3000
```

## 1. Class → endpoint map

| Engine object (existing) | Read through | Exposed as |
|---|---|---|
| `Universe.topics` / `Universe.books` (`LmsrBook`) | in memory, updated by the WebSocket `on_quote` and the 45 s universe refresh | `GET /api/markets`, WS `market_quotes` |
| `LmsrBook.prices()` / `.probabilities()` / `.liquidity` | pure functions of the book | bucket `market_price`, `market_probability`, `liquidity_b` |
| `EnsembleForecast.probabilities` / `.components` / `.weights` | the strategy's 30 s forecast cache (`_forecast`) | `GET /api/distribution/{topic_id}`, WS `ensemble_distribution` (`component_densities`) |
| `EnsembleForecast.disagreement` | | `js_divergence` |
| `EnsembleForecast.kelly_multiplier` (`clip(1 − 2·JSD, 0.25, 1)`) | | `kelly_multiplier_kappa` |
| `build_signals(topic, book, forecast)` → `BucketSignal` | pure | `edge_points` per bucket |
| `SlippageOptimizer.optimize` with a `LocalEstimator` as the pricing source | exact replica, **zero REST calls** | positive-EV buckets: recommended size `k*`, Kelly fraction, budget, ladder |
| `LmsrBook.buy_cost` for `k = 1 … 50`, `ladder_point()` | pure | `GET /api/ladder/{topic_id}/{option_id}` (inspector drawer) |
| `RiskManager.locked` / `.lock_reason` / `.breaker_*` / `.day_peak_millisats` / `.entry_budget` | in memory | `GET /api/health`, WS `risk_metrics` (drawdown gauge, 15 % / 35 % utilisation) |
| `Ledger.positions(mode)` + `LocalEstimator.sell` + `evaluate_exit` | ledger replay + local book | WS `active_positions` (mark, liquidation value, unrealised P&L, exit rule state) |
| `Ledger.page` / `.summary` / `.equity_curve` | SQLite reads | `GET /api/ledger`, WS `ledger_fills` |
| `backtester.performance_metrics` | pure | `GET /api/backtest/results` (live/paper curve; latest synthetic backtest) |
| `RiskManager.reconcile(client)` | one `portfolio_active()` call, operator-triggered | `POST /api/bot/reconcile` |
| `SlippageOptimizer` → `RiskManager.check_entry` → `ExecutionRouter.enter` | identical path to the engine | `POST /api/orders/test` |
| structlog events (`execution.*`, `risk.*`, `strategy.*`, …) | logging processor | WS `system_logs` (`INFO` / `SIGNAL` / `EXECUTION` / `RISK_LOCK`) |

Field names follow the engine's dataclasses (`market_price`, `model_probability`,
`edge_points`, `cost_millisats`, `exec_price`, `expected_profit_millisats`, …). The
two telemetry names the brief asked for — `js_divergence` and
`kelly_multiplier_kappa` — are the engine's `disagreement` and `kelly_multiplier`.

## 2. Units across the boundary

| Python | JSON | TypeScript |
|---|---|---|
| `Millisats` (int) | integer, field name ends in `_millisats` | `Millisats` — branded `number` |
| `PriceUnits` (0–100) | float, fields `*_price`, `*_points` | `PriceUnits` — branded `number` |
| probability (0–1) | float, `*_probability` | `Probability` — branded `number` |

The only crossings are explicit helpers on both sides (`types.price_points_to_millisats`
in Python, `units.ts` in the frontend). A `Millisats` cannot be passed where a
`PriceUnits` is expected without a compile error.

`max_extractable_edge_millisats = round(1000 · 100 · b · KL(p_model ‖ p_market))` —
the gross amount that moving the whole book to the model distribution would earn
before fees; ARCHITECTURE.md §3.3 gives the per-bucket form.

## 3. Safety invariants and how the server keeps them

| Invariant | Mechanism | Test |
|---|---|---|
| Paper by default | `dry_run=True` is the default of every request model and the default engine mode; `GET /api/health` reports `PAPER` | `test_server.py::test_defaults_are_paper` |
| Live needs several independent keys | (1) `GLIMPSE_LIVE_TRADING=true`, (2) `GLIMPSE_API_KEY`, (3) `serve --allow-live`, (4) a real venue (never the simulator), (5) typed phrase → single-use arming token valid for 60 s, (6) `POST /api/bot/start` with `dry_run=false` and that token. Keys 1–2 are checked by the engine's own `Settings.resolve_mode` | `test_live_requires_*` |
| No endpoint bypasses risk | the manual ticket builds a `TradePlan`, runs the full `check_entry` (caps, breaker, lock, flow control, thresholds) and only then calls `ExecutionRouter.enter`; request models forbid extra fields. The ticket is pre-screened on the local LS-LMSR replica, so a refused order answers at once without a REST call; an approved one is re-planned on the venue's `/trades/estimate` quote and re-checked against fresh positions before routing | `test_manual_order_respects_caps`, `test_blocked_ticket_never_spends_a_venue_quote`, `test_unknown_fields_rejected` |
| Safety parameters are not tunable over HTTP | `start` accepts only `dry_run`, `min_expected_gain_sats`, `max_horizon_hours`, `assets`; caps, drawdown limit and Kelly fraction are read from the environment only | `test_start_rejects_risk_overrides` |
| Breaker survives restarts | the controller snapshots the day's peak and breaker state on stop and restores it into the next `RiskManager` | `test_breaker_survives_restart` |
| Kill switch | cancels the runner, locks the risk manager, revokes any arming token, writes a ledger `risk_event`; `start` refuses until a typed release | `test_kill_switch_blocks_start` |
| Zero blind retries | unchanged: only `ExecutionRouter` sends orders; ambiguous state locks and reconciles | existing `test_risk_reconciler.py` |
| Single writer | one `Ledger` instance per server process; operator actions serialised by an `asyncio.Lock`; the headless `bot` compose service moves to a profile so it cannot run beside `api` on the same volume | `test_single_ledger_instance` |
| No REST on page refresh | every GET and the WebSocket read in-memory state; a counting rate limiter proves it | `test_reads_make_no_glimpse_requests` |
| Cross-site requests | binds to 127.0.0.1; mutating requests need an allowed `Origin` (or none) **and** the `X-GQB-Intent: operator` header, which forces a CORS pre-flight; the WebSocket checks `Origin` | `test_cross_origin_post_rejected` |

## 4. WebSocket protocol (`/ws/telemetry`)

Server → client, one JSON object per frame: `{"type": <channel>, "ts": <unix>, "data": …}`

| Channel | Cadence | Payload |
|---|---|---|
| `hello` | on connect | full snapshot + last 200 log lines |
| `engine_status` | 1 s | `HealthResponse` |
| `market_quotes` | 2 s | `{tickers: SpotTicker[], topics: MarketSummary[]}` |
| `ensemble_distribution` | 2 s, focused topic only, when book or forecast changed | `DistributionResponse` (includes `component_densities`, `js_divergence`, `kelly_multiplier_kappa`) |
| `active_positions` | 1 s | `PositionView[]` |
| `risk_metrics` | 1 s | `RiskMetrics` |
| `ledger_fills` | on new rows | `LedgerRecordView[]` (delta) |
| `system_logs` | as emitted | `LogEntry` |
| `pong` | reply | `{t}` — round-trip latency for the ribbon |

Client → server: `{"type":"focus","topic_id":123}`, `{"type":"ping","t":<ms>}`.

Each client has a bounded queue (256 frames). A slow browser loses its oldest
frames; it never blocks the trading loop.

## 5. Simulated venue (`serve --demo`)

`server/simulation.py` provides `SimulatedGlimpse(AsyncClient)` and
`SimulatedHub(MarketDataHub)`: a geometric-Brownian world with 500-bucket
LS-LMSR books built by `lmsr.book_from_prices`, a stale and over-wide crowd that
slowly converges, synthetic candles, order books, funding and a Deribit-like
smile for BTC and ETH. The **unmodified** strategy runs against it, so the
terminal, the tests and a first look at the UI need no network and no key. Each
run writes a fresh ledger under `data/terminal-demo/sim-<UTC timestamp>/`, apart
from the real one (a new world numbers its topics from scratch, so an old paper
book would point at markets that no longer exist), and it can never be armed for
live trading.
