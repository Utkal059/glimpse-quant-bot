"""Typed domain models.

Two numeric scales exist on Glimpse and they must never be mixed:

* ``Millisats`` -- real money (wallet balances, order costs, commissions).
* ``PriceUnits`` -- the 0-100 LS-LMSR quote scale. One price point on one
  contract is worth exactly one satoshi (1,000 millisats); a winning contract
  pays 100 points before the settlement fee.

Every monetary field in this module is a ``Millisats`` and every quote field a
``PriceUnits``; the only crossings between the two are the explicit helpers
:func:`price_points_to_millisats` and :func:`millisats_to_price_points`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Literal

from glimpse_markets.money import Millisats, PriceUnits

Mode = Literal["dry_run", "live"]

PAYOUT_PRICE: Final[PriceUnits] = PriceUnits(100.0)
"""A winning contract pays 100 price points (100 sats) before the settlement fee."""

MILLISATS_PER_PRICE_POINT: Final[int] = 1000
"""One price point on one contract = 1 sat = 1,000 millisats."""


def price_points_to_millisats(price: PriceUnits, contracts: float) -> Millisats:
    """Money value of ``contracts`` contracts each worth ``price`` points."""
    return Millisats(round(float(price) * contracts * MILLISATS_PER_PRICE_POINT))


def millisats_to_price_points(amount: Millisats, contracts: float) -> PriceUnits:
    """Average price per contract implied by a money amount."""
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    return PriceUnits(int(amount) / (contracts * MILLISATS_PER_PRICE_POINT))


class AssetType(StrEnum):
    """Underlying assets the engine can forecast."""

    BTC = "BTC"
    ETH = "ETH"
    SOL = "SOL"
    GOLD = "GOLD"

    @property
    def has_listed_options(self) -> bool:
        """Whether Deribit lists an options chain usable for a risk-neutral density."""
        return self in (AssetType.BTC, AssetType.ETH)


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class RecordKind(StrEnum):
    SIGNAL = "signal"
    ESTIMATE = "estimate"
    ENTRY = "entry"
    EXIT = "exit"
    SETTLEMENT = "settlement"
    ADJUSTMENT = "adjustment"
    RECONCILE = "reconcile"
    RISK_EVENT = "risk_event"
    EQUITY = "equity"


_CROSS_QUOTED = re.compile(r"\bsats?\b|satoshi|/\s*btc\b|\bwti\b|crude", re.IGNORECASE)
_ASSET_PATTERNS: tuple[tuple[AssetType, re.Pattern[str]], ...] = (
    (AssetType.GOLD, re.compile(r"\b(gold|paxg|paxgusd|xau|xaut)\b", re.IGNORECASE)),
    (AssetType.ETH, re.compile(r"\b(ethereum|eth)\b", re.IGNORECASE)),
    (AssetType.SOL, re.compile(r"\b(solana|sol)\b", re.IGNORECASE)),
    (AssetType.BTC, re.compile(r"\b(bitcoin|btc)\b", re.IGNORECASE)),
)


def classify_asset(*texts: str | None) -> AssetType | None:
    """Infer the USD-quoted underlying of a topic from its metadata.

    Returns ``None`` for anything the engine cannot price: unknown assets and
    cross-quoted markets (for example "Tether Gold in sats per troy ounce" or
    "WTI/BTC"), whose settlement value is a ratio of two assets rather than a
    USD price.
    """
    blob = " ".join(text for text in texts if text)
    if not blob or _CROSS_QUOTED.search(blob):
        return None
    for asset, pattern in _ASSET_PATTERNS:
        if pattern.search(blob):
            return asset
    return None


@dataclass(frozen=True, slots=True)
class Bucket:
    """One price-range outcome of a topic: pays if ``lo < S_T <= hi``."""

    option_id: int
    name: str
    lo: float
    hi: float


@dataclass(frozen=True, slots=True)
class TopicMeta:
    """Static description of one Glimpse market."""

    topic_id: int
    batch_id: str
    title: str
    asset: AssetType
    end_time_utc: int
    buckets: tuple[Bucket, ...]

    def seconds_to_expiry(self, now: float) -> float:
        return float(self.end_time_utc) - now

    def bucket(self, option_id: int) -> Bucket | None:
        for bucket in self.buckets:
            if bucket.option_id == option_id:
                return bucket
        return None


@dataclass(frozen=True, slots=True)
class BucketSignal:
    """Model-versus-market comparison for one bucket."""

    topic_id: int
    option_id: int
    name: str
    lo: float
    hi: float
    market_price: PriceUnits
    """Marginal LS-LMSR buy price for the next contract."""
    model_probability: float
    """Ensemble probability in [0, 1]."""
    edge_points: PriceUnits
    """``100 * model_probability - market_price``."""
    components: Mapping[str, float] = field(default_factory=dict)
    """Per-model probabilities that were pooled into ``model_probability``."""


@dataclass(frozen=True, slots=True)
class LadderPoint:
    """One rung of the slippage ladder: the economics of buying ``contracts``."""

    contracts: float
    cost_millisats: Millisats
    """LS-LMSR cost ``C(q') - C(q)``, excluding commission."""
    commission_millisats: Millisats
    exec_price: PriceUnits
    """All-in average price per contract (cost plus commission)."""
    net_edge_points: PriceUnits
    """Expected settlement value per contract minus ``exec_price``."""
    expected_profit_millisats: Millisats
    source: Literal["server", "local"]

    @property
    def all_in_millisats(self) -> Millisats:
        return Millisats(int(self.cost_millisats) + int(self.commission_millisats))


@dataclass(frozen=True, slots=True)
class TradePlan:
    """A fully priced, size-optimised entry ready for the risk gatekeeper."""

    topic_id: int
    option_id: int
    bucket_name: str
    side: Side
    contracts: float
    cost_millisats: Millisats
    commission_millisats: Millisats
    exec_price: PriceUnits
    market_price: PriceUnits
    model_probability: float
    net_edge_points: PriceUnits
    expected_profit_millisats: Millisats
    kelly_fraction: float
    """Fraction of the bankroll the fractional-Kelly rule allowed for this bet."""
    budget_millisats: Millisats
    ladder: tuple[LadderPoint, ...]
    rationale: str

    @property
    def all_in_millisats(self) -> Millisats:
        return Millisats(int(self.cost_millisats) + int(self.commission_millisats))


@dataclass(frozen=True, slots=True)
class Position:
    """An open long position in one bucket."""

    topic_id: int
    option_id: int
    bucket_name: str
    contracts: float
    cost_basis_millisats: Millisats
    """All-in money paid for the contracts still held."""
    entry_model_probability: float

    @property
    def entry_price(self) -> PriceUnits:
        return millisats_to_price_points(self.cost_basis_millisats, self.contracts)


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    """One immutable row of the append-only ledger."""

    record_id: int
    ts: float
    mode: Mode
    kind: RecordKind
    topic_id: int | None
    option_id: int | None
    bucket_name: str | None
    contracts: float
    cash_flow_millisats: Millisats
    """Signed wallet impact: negative for money paid, positive for money received."""
    commission_millisats: Millisats
    exec_price: PriceUnits | None
    model_probability: float | None
    market_price: PriceUnits | None
    realized_pnl_millisats: Millisats
    trade_id: str | None
    note: str
