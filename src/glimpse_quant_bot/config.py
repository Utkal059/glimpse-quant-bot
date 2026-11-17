"""Validated runtime configuration with safe dry-run defaults.

Glimpse has no sandbox environment, so the configuration is built around one
rule: real orders require two independent, explicit opt-ins -- the
``GLIMPSE_LIVE_TRADING=true`` environment flag *and* the ``--live`` CLI flag.
Either one alone leaves the bot paper-trading through ``/trades/estimate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from glimpse_markets.money import Millisats, sats_to_millisats
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from glimpse_quant_bot.types import AssetType, Mode


class LiveTradingNotArmedError(RuntimeError):
    """``--live`` was requested without the matching environment opt-in."""


class Settings(BaseSettings):
    """All tunables. Glimpse credentials use the SDK's own ``GLIMPSE_*`` names;
    everything else is read from ``GQB_*`` variables or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="GQB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Glimpse connection -------------------------------------------------
    glimpse_api_key: SecretStr | None = Field(default=None, validation_alias="GLIMPSE_API_KEY")
    glimpse_base_url: str = Field(
        default="https://main.bpmapi.io", validation_alias="GLIMPSE_BASE_URL"
    )
    glimpse_live_trading: bool = Field(default=False, validation_alias="GLIMPSE_LIVE_TRADING")
    glimpse_requests_per_minute: int = Field(default=50, ge=1, le=60)

    # --- Bankroll -----------------------------------------------------------
    bankroll_sats: int = Field(default=100_000, gt=0)

    # --- Universe -----------------------------------------------------------
    # USD asset-unit guard. Every model in the ensemble forecasts a *USD* price:
    # the Deribit density is in USD per coin, and the HAR / GARCH fits run on
    # USD(T) candles. A batch is tradable only if its buckets are USD prices of
    # one of these assets. Ratio-denominated batches -- "Tether Gold in sats per
    # troy ounce" (XAUT/BTC) and "WTI/BTC" -- settle on the quotient of two
    # prices. Their bucket edges are in satoshis, so mapping a USD density onto
    # them would be a unit error, and pricing them properly needs the joint
    # distribution of both legs (a gold/BTC or oil/BTC correlation model), which
    # the engine does not have; WTI also has no free, commercially usable feed.
    # `types.classify_asset` therefore returns None for any title, description
    # or category mentioning sats, a "/BTC" quote or crude oil, and such
    # batches never enter the universe, whatever `assets` contains.
    assets: Annotated[tuple[AssetType, ...], NoDecode] = (
        AssetType.BTC,
        AssetType.ETH,
        AssetType.SOL,
        AssetType.GOLD,
    )
    max_horizon_hours: float = Field(default=30.0, gt=0)
    min_seconds_to_expiry: float = Field(default=240.0, ge=0)

    # --- Ensemble weights ---------------------------------------------------
    weight_options: float = Field(default=0.50, ge=0)
    weight_microstructure: float = Field(default=0.30, ge=0)
    weight_timeseries: float = Field(default=0.20, ge=0)
    fallback_weight_microstructure: float = Field(default=0.60, ge=0)
    fallback_weight_timeseries: float = Field(default=0.40, ge=0)
    candle_interval_seconds: int = Field(default=3600, gt=0)
    candle_history: int = Field(default=1000, ge=120, le=1000)
    garch_paths: int = Field(default=3000, ge=200, le=20_000)
    random_seed: int = 7
    model_refit_seconds: float = Field(default=300.0, gt=0)
    fast_input_ttl_seconds: float = Field(default=15.0, gt=0)
    enable_timesfm: bool = True
    enable_spot_websocket: bool = True

    # --- Signal & execution thresholds (0-100 price points) -----------------
    min_raw_edge_points: float = Field(default=3.0, ge=0)
    min_net_edge_points: float = Field(default=2.5, ge=0)
    # Calibrated to live depth. Glimpse topics carry b = alpha * sum(q) of only
    # 13-40 contracts, and the most a trade can extract from one bucket is about
    # 100 * b * KL(p_model || p_market) sats -- typically 10-150 sats. A 250-sat
    # floor is therefore unreachable in practice (zero fills in replay), while
    # 25 sats is still 25x the 1-sat minimum commission, so fees cannot erode
    # the trade. The per-contract guard `min_net_edge_points` stays at 2.5.
    min_expected_gain_sats: float = Field(default=25.0, ge=0)
    ladder: Annotated[tuple[float, ...], NoDecode] = (1.0, 2.0, 5.0, 10.0, 20.0)
    ladder_max_contracts: float = Field(default=400.0, gt=0)
    server_confirm_points: int = Field(default=3, ge=1, le=8)
    max_candidates_per_topic: int = Field(default=2, ge=1)
    fee_rate: float = Field(default=0.02, ge=0, lt=0.5)
    min_buy_commission_millisats: int = Field(default=1000, ge=0)

    # --- Sizing & portfolio guardrails --------------------------------------
    kelly_fraction: float = Field(default=0.35, gt=0, le=1)
    kelly_disagreement_floor: float = Field(default=0.25, gt=0, le=1)
    max_bucket_fraction: float = Field(default=0.15, gt=0, le=1)
    max_topic_fraction: float = Field(default=0.35, gt=0, le=1)
    daily_drawdown_limit: float = Field(default=0.06, gt=0, lt=1)
    max_open_positions: int = Field(default=12, ge=1)
    max_orders_per_minute: int = Field(default=6, ge=1)
    reentry_cooldown_seconds: float = Field(default=120.0, ge=0)

    # --- Exit rules ---------------------------------------------------------
    take_profit_gap_points: float = Field(default=1.0, ge=0)
    invalidation_drop_points: float = Field(default=4.0, ge=0)

    # --- Timers -------------------------------------------------------------
    tick_interval_seconds: float = Field(default=5.0, gt=0)
    resolution_poll_seconds: float = Field(default=45.0, ge=10)
    eval_interval_seconds: float = Field(default=10.0, ge=1)
    topics_per_tick: int = Field(default=4, ge=1)

    # --- Backtest -----------------------------------------------------------
    backtest_liquidity_b: float = Field(default=35.0, gt=0)

    # --- Storage & logging --------------------------------------------------
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("assets", mode="before")
    @classmethod
    def _split_assets(cls, value: Any) -> Any:
        if isinstance(value, str):
            return tuple(
                AssetType(part.strip().upper()) for part in value.split(",") if part.strip()
            )
        return value

    @field_validator("ladder", mode="before")
    @classmethod
    def _split_ladder(cls, value: Any) -> Any:
        if isinstance(value, str):
            return tuple(float(part) for part in value.split(",") if part.strip())
        return value

    @field_validator("ladder")
    @classmethod
    def _check_ladder(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(size <= 0 for size in value):
            raise ValueError("ladder must contain positive contract sizes")
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if not self.assets:
            raise ValueError("at least one asset must be enabled")
        if self.max_bucket_fraction > self.max_topic_fraction:
            raise ValueError("max_bucket_fraction cannot exceed max_topic_fraction")
        if self.weight_options + self.weight_microstructure + self.weight_timeseries <= 0:
            raise ValueError("ensemble weights must not all be zero")
        if self.fallback_weight_microstructure + self.fallback_weight_timeseries <= 0:
            raise ValueError("fallback ensemble weights must not all be zero")
        if self.min_seconds_to_expiry >= self.max_horizon_hours * 3600:
            raise ValueError("min_seconds_to_expiry must be below max_horizon_hours")
        return self

    # --- Derived values -----------------------------------------------------
    @property
    def bankroll_millisats(self) -> Millisats:
        return Millisats(sats_to_millisats(self.bankroll_sats))

    @property
    def min_expected_gain_millisats(self) -> Millisats:
        return Millisats(sats_to_millisats(self.min_expected_gain_sats))

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "ledger.sqlite"

    @property
    def audit_path(self) -> Path:
        return self.data_dir / "audit.ndjson"

    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    def resolve_mode(self, cli_live_flag: bool) -> Mode:
        """Decide between paper and real trading.

        Real orders need the CLI flag, the environment flag and an API key.
        Asking for ``--live`` without the other two is treated as an operator
        error and raises rather than silently paper-trading or, worse,
        silently trading.
        """
        if not cli_live_flag:
            return "dry_run"
        if not self.glimpse_live_trading:
            raise LiveTradingNotArmedError(
                "--live was passed but GLIMPSE_LIVE_TRADING is not 'true'. "
                "Refusing to place real orders."
            )
        if self.glimpse_api_key is None or not self.glimpse_api_key.get_secret_value():
            raise LiveTradingNotArmedError(
                "--live was passed but GLIMPSE_API_KEY is empty. Refusing to place real orders."
            )
        return "live"


def load_settings(**overrides: Any) -> Settings:
    """Build settings from the environment / ``.env`` with optional overrides."""
    return Settings(**overrides)
