"""Pydantic v2 models: the API contract between the engine and the terminal.

Unit discipline mirrors :mod:`glimpse_quant_bot.types`:

* every money field is an ``int`` whose name ends in ``_millisats``;
* every quote field is a ``float`` on the 0-100 LS-LMSR scale (``*_price``,
  ``*_points``);
* every probability is a ``float`` in ``[0, 1]`` (``*_probability``).

Field names follow the engine's dataclasses (``market_price``,
``model_probability``, ``edge_points``, ``exec_price`` ...). The frontend's
``frontend/src/lib/contract.ts`` is generated from it by
:mod:`glimpse_quant_bot.server.typescript`.

Request models forbid unknown fields, so a client cannot smuggle a parameter
(a risk cap, a "skip checks" flag) past validation.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from glimpse_quant_bot.types import AssetType

# ``x-unit`` travels through the JSON schema into the generated TypeScript
# contract (``server/typescript.py``), where it becomes a branded type.
MillisatsInt = Annotated[int, Field(strict=True, json_schema_extra={"x-unit": "millisats"})]
"""Integer millisats (1 sat = 1000). Never a float."""
PricePoints = Annotated[float, Field(allow_inf_nan=False, json_schema_extra={"x-unit": "points"})]
"""LS-LMSR quote scale, 0-100 points."""
SignedPoints = Annotated[float, Field(allow_inf_nan=False, json_schema_extra={"x-unit": "points"})]
"""Signed difference on the 0-100 scale."""
Probability = Annotated[
    float, Field(ge=0.0, le=1.0, allow_inf_nan=False, json_schema_extra={"x-unit": "probability"})
]
ProbabilityValue = Annotated[
    float, Field(allow_inf_nan=False, json_schema_extra={"x-unit": "probability"})
]
"""A probability from a numerical model (no bounds check, so round-off never 500s)."""

EngineState = Literal["stopped", "starting", "running", "stopping", "halted", "error"]
SafetyMode = Literal["PAPER", "LIVE_ARMED"]
VenueName = Literal["glimpse", "simulated"]
LogCategory = Literal["INFO", "SIGNAL", "EXECUTION", "RISK_LOCK"]
LogLevel = Literal["debug", "info", "warning", "error", "critical"]
ExitRule = Literal["take_profit", "invalidation", "hold"]
RecordKindName = Literal[
    "signal",
    "estimate",
    "entry",
    "exit",
    "settlement",
    "adjustment",
    "reconcile",
    "risk_event",
    "equity",
]
ModeName = Literal["dry_run", "live"]

ARM_PHRASE = "ARM LIVE TRADING"
RELEASE_PHRASE = "RESUME TRADING"
LIVE_ORDER_PHRASE = "SEND LIVE ORDER"


class ApiModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)


# --- health ---------------------------------------------------------------


class RateLimitBudget(ApiModel):
    limit_per_minute: int
    used_last_minute: int
    remaining: int
    total_requests: int


class LiveEligibility(ApiModel):
    """Every independent key that must be turned before live trading is possible."""

    server_flag: bool
    """``GLIMPSE_LIVE_TRADING=true`` in the server's environment."""
    api_key_present: bool
    serve_allow_live: bool
    """``glimpse-quant-bot serve --allow-live`` was passed."""
    real_venue: bool
    """False for the simulated venue, which can never trade live."""
    kill_switch_clear: bool
    eligible: bool
    blockers: list[str]


class ArmingStatus(ApiModel):
    pending: bool
    expires_at: float | None


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"
    version: str
    engine_state: EngineState
    mode: SafetyMode
    venue: VenueName
    dry_run: bool
    trading_locked: bool
    lock_reason: str | None
    breaker_tripped: bool
    breaker_reason: str | None
    kill_switch_engaged: bool
    entries_allowed: bool
    rate_limit: RateLimitBudget
    live_eligibility: LiveEligibility
    arming: ArmingStatus
    started_at: float | None
    uptime_seconds: float
    server_time: float
    ws_clients: int
    fills: int
    topics: int
    timeseries_model: str | None
    last_error: str | None
    min_expected_gain_sats: float
    max_horizon_hours: float
    assets: list[AssetType]


# --- markets --------------------------------------------------------------


class SpotTicker(ApiModel):
    asset: AssetType
    price: float | None
    sources: list[str]
    ts: float | None
    change_pct: float | None
    """Change since the server first saw this asset (session change)."""
    active_topics: int


class MarketSummary(ApiModel):
    topic_id: int
    batch_id: str
    title: str
    asset: AssetType
    end_time_utc: int
    seconds_to_expiry: float
    n_buckets: int
    liquidity_b: float
    """LS-LMSR depth ``b = alpha * sum(q)``, in contracts."""
    alpha: float
    book_ts: float
    spot_price: float | None
    market_mode_bucket: str | None
    """Bucket with the highest market-implied probability."""
    forecast_available: bool
    forecast_ts: float | None
    js_divergence: float | None
    kelly_multiplier_kappa: float | None
    kl_divergence: float | None
    """KL(p_model || p_market) in nats."""
    max_extractable_edge_millisats: MillisatsInt | None
    """``100 * b * KL(p_model || p_market)`` sats, gross of fees."""
    best_bucket: str | None
    best_edge_points: SignedPoints | None
    positive_ev_buckets: int
    held_contracts: float


class MarketsResponse(ApiModel):
    tickers: list[SpotTicker]
    topics: list[MarketSummary]


# --- distribution ---------------------------------------------------------


class ForecastMeta(ApiModel):
    ts: float
    horizon_seconds: float
    spot: float
    sigma: float
    js_divergence: float
    kelly_multiplier_kappa: float
    ambiguity: float
    weights: dict[str, float]
    spot_sources: list[str]
    diagnostics: dict[str, float]


class BucketColumns(ApiModel):
    """Column-oriented bucket table (500 rows; columns share one index)."""

    option_id: list[int]
    name: list[str]
    lo: list[float]
    hi: list[float]
    market_price: list[PricePoints]
    """Marginal LS-LMSR buy price, 0-100 points."""
    market_probability: list[ProbabilityValue]
    """``w = softmax(q / b)``."""
    model_probability: list[ProbabilityValue] | None
    edge_points: list[SignedPoints] | None
    """``100 * p_model - market_price``."""
    net_edge_points: list[SignedPoints] | None
    """``100 * p_model * (1 - fee) - market_price * (1 + fee)``: the marginal
    contract after the 2% commission and the 2% settlement fee."""
    held_contracts: list[float]


class LadderPointView(ApiModel):
    contracts: float
    cost_millisats: MillisatsInt
    commission_millisats: MillisatsInt
    exec_price: PricePoints
    """All-in average price per contract (cost plus commission)."""
    net_edge_points: SignedPoints
    expected_profit_millisats: MillisatsInt
    source: Literal["server", "local"]


class PositiveEvBucket(ApiModel):
    option_id: int
    name: str
    lo: float
    hi: float
    market_price: PricePoints
    model_probability: Probability
    edge_points: SignedPoints
    net_edge_points: SignedPoints
    recommended_contracts: float | None
    """``k*`` the engine's optimiser would choose under the current budget."""
    kelly_fraction: float | None
    budget_millisats: MillisatsInt | None
    expected_profit_millisats: MillisatsInt | None
    slippage_cost_millisats: MillisatsInt | None
    """All-in cost of ``k*`` minus ``k* * market_price`` (price impact plus fees)."""
    rejection: str | None
    ladder: list[LadderPointView]


class DistributionResponse(ApiModel):
    topic_id: int
    title: str
    asset: AssetType
    end_time_utc: int
    seconds_to_expiry: float
    liquidity_b: float
    alpha: float
    book_ts: float
    fee_rate: float
    min_raw_edge_points: float
    min_net_edge_points: float
    min_expected_gain_sats: float
    spot_price: float | None
    forecast: ForecastMeta | None
    js_divergence: float | None
    kelly_multiplier_kappa: float | None
    kl_divergence: float | None
    max_extractable_edge_millisats: MillisatsInt | None
    buckets: BucketColumns
    component_densities: dict[str, list[ProbabilityValue]]
    """Per-model bucket probabilities (``options_rnd``, ``har_microstructure``,
    ``gjr_garch`` or ``timesfm``), aligned with ``buckets``."""
    positive_ev: list[PositiveEvBucket]


class LadderRow(ApiModel):
    contracts: float
    cost_millisats: MillisatsInt
    """``C(q + k e_i) - C(q)``."""
    commission_millisats: MillisatsInt
    exec_price: PricePoints
    marginal_price: PricePoints
    """Marginal price of the bucket after buying ``k``."""
    net_edge_points: SignedPoints
    expected_profit_millisats: MillisatsInt
    marginal_profit_millisats: MillisatsInt
    """Expected profit of the k-th contract alone."""
    within_budget: bool


class LadderResponse(ApiModel):
    topic_id: int
    option_id: int
    bucket_name: str
    market_price: PricePoints
    model_probability: Probability | None
    fair_value_points: PricePoints | None
    """``100 * p_model * (1 - fee)``."""
    budget_millisats: MillisatsInt
    max_contracts: int
    optimal_contracts: float | None
    """Size with the highest expected profit on this ladder."""
    marginal_breakeven_contracts: float | None
    """First size at which the next contract's expected profit turns negative."""
    rows: list[LadderRow]
    engine_choice: PositiveEvBucket | None


# --- portfolio and risk ---------------------------------------------------


class PositionView(ApiModel):
    topic_id: int
    option_id: int
    bucket_name: str
    asset: AssetType | None
    title: str | None
    contracts: float
    cost_basis_millisats: MillisatsInt
    entry_price: PricePoints
    entry_model_probability: Probability
    mark_price: PricePoints | None
    """Current marginal LS-LMSR quote of the bucket."""
    liquidation_value_millisats: MillisatsInt | None
    """What selling into the book now would net after commission."""
    unrealized_pnl_millisats: MillisatsInt | None
    model_probability: Probability | None
    take_profit_trigger_price: PricePoints | None
    """Take profit when ``market_price >= 100 * p_model - take_profit_gap_points``."""
    invalidation_model_points: PricePoints
    """Invalidate when ``100 * p_model < entry_price - invalidation_drop_points``."""
    exit_rule: ExitRule | None
    exit_detail: str | None
    seconds_to_expiry: float | None


class ExposureRow(ApiModel):
    topic_id: int
    option_id: int | None
    label: str
    asset: AssetType | None
    cost_basis_millisats: MillisatsInt
    fraction_of_equity: float
    cap_fraction: float
    utilization: float
    """``fraction_of_equity / cap_fraction``; 1.0 means the cap binds."""


class RiskMetrics(ApiModel):
    mode: ModeName
    bankroll_millisats: MillisatsInt
    equity_millisats: MillisatsInt
    cash_millisats: MillisatsInt
    positions_value_millisats: MillisatsInt
    realized_pnl_millisats: MillisatsInt
    unrealized_pnl_millisats: MillisatsInt
    day_peak_millisats: MillisatsInt
    intraday_drawdown: float
    drawdown_limit: float
    breaker_tripped: bool
    breaker_reason: str | None
    trading_locked: bool
    lock_reason: str | None
    entries_allowed: bool
    max_bucket_fraction: float
    max_topic_fraction: float
    bucket_exposures: list[ExposureRow]
    topic_exposures: list[ExposureRow]
    open_positions: int
    max_open_positions: int
    fills_last_minute: int
    max_orders_per_minute: int
    kelly_fraction: float


# --- ledger ---------------------------------------------------------------


class LedgerRecordView(ApiModel):
    record_id: int
    ts: float
    mode: ModeName
    kind: RecordKindName
    topic_id: int | None
    option_id: int | None
    bucket_name: str | None
    contracts: float
    cash_flow_millisats: MillisatsInt
    commission_millisats: MillisatsInt
    exec_price: PricePoints | None
    model_probability: ProbabilityValue | None
    market_price: PricePoints | None
    realized_pnl_millisats: MillisatsInt
    trade_id: str | None
    note: str


class LedgerSummaryView(ApiModel):
    mode: ModeName
    starting_millisats: MillisatsInt
    cash_millisats: MillisatsInt
    open_cost_basis_millisats: MillisatsInt
    realized_pnl_millisats: MillisatsInt
    commission_millisats: MillisatsInt
    entries: int
    exits: int
    settlements: int
    wins: int
    losses: int
    open_positions: int
    roi: float
    win_rate: float


class LedgerPage(ApiModel):
    mode: ModeName
    total: int
    records: list[LedgerRecordView]
    next_before_id: int | None
    summary: LedgerSummaryView


class EquityPoint(ApiModel):
    ts: float
    equity_millisats: MillisatsInt


class PerformanceMetrics(ApiModel):
    points: int
    starting_millisats: MillisatsInt
    ending_millisats: MillisatsInt
    pnl_millisats: MillisatsInt
    roi: float
    sharpe: float
    sortino: float
    max_drawdown: float


class BacktestRunView(ApiModel):
    finished_at: float
    synthetic: bool
    control: bool
    seed: int
    step_seconds: int
    metrics: PerformanceMetrics
    win_rate: float
    entries: int
    exits: int
    settlements: int
    commission_millisats: MillisatsInt
    equity_curve: list[EquityPoint]


class BacktestResultsResponse(ApiModel):
    mode: ModeName
    equity_curve: list[EquityPoint]
    metrics: PerformanceMetrics | None
    latest_backtest: BacktestRunView | None
    backtest_running: bool


# --- telemetry ------------------------------------------------------------


class LogEntry(ApiModel):
    seq: int
    ts: float
    level: LogLevel
    category: LogCategory
    event: str
    message: str
    fields: dict[str, str]


# --- control requests and responses --------------------------------------


class StartRequest(RequestModel):
    dry_run: bool = True
    arming_token: str | None = Field(default=None, max_length=128)
    min_expected_gain_sats: float | None = Field(default=None, ge=0.0, le=1_000_000.0)
    max_horizon_hours: float | None = Field(default=None, gt=0.0, le=720.0)
    assets: list[AssetType] | None = Field(default=None, min_length=1, max_length=4)


class ArmRequest(RequestModel):
    confirmation_phrase: str = Field(max_length=64)


class ArmResponse(ApiModel):
    arming_token: str
    expires_at: float
    eligibility: LiveEligibility


class KillSwitchRequest(RequestModel):
    reason: str = Field(default="operator kill switch", max_length=200)


class ReleaseRequest(RequestModel):
    confirmation_phrase: str = Field(max_length=64)


class TestOrderRequest(RequestModel):
    __test__ = False  # not a pytest class

    topic_id: int = Field(gt=0)
    option_id: int = Field(gt=0)
    contracts: float = Field(gt=0.0, le=400.0, allow_inf_nan=False)
    confirmation_phrase: str | None = Field(default=None, max_length=64)
    """Required (``SEND LIVE ORDER``) only when the engine is running live."""


class BacktestRunRequest(RequestModel):
    control: bool = False
    seed: int = Field(default=11, ge=0, le=1_000_000)
    step_seconds: int = Field(default=300, ge=60, le=3600)


class ActionResult(ApiModel):
    ok: bool
    message: str
    engine_state: EngineState
    mode: SafetyMode


class TestOrderResult(ApiModel):
    __test__ = False

    ok: bool
    mode: ModeName
    approved: bool
    reasons: list[str]
    plan: LadderPointView | None
    fill: LedgerRecordView | None
    message: str


class DiscrepancyView(ApiModel):
    topic_id: int
    option_id: int
    ledger_contracts: float
    actual_contracts: float


class ReconcileResult(ApiModel):
    ok: bool
    scope: Literal["live_engine", "live_ledger", "paper"]
    landed: bool | None
    discrepancies: list[DiscrepancyView]
    note: str
    trading_locked: bool


class ErrorDetail(ApiModel):
    detail: str
    reasons: list[str] = Field(default_factory=list)
