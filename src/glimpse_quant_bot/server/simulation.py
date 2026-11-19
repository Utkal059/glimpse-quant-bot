"""A self-contained simulated Glimpse venue and market-data hub.

``serve --demo`` and the server tests run the **unmodified**
:class:`~glimpse_quant_bot.strategy.GlimpseQuantStrategy` against this module:

* :class:`SimulatedWorld` -- fat-tailed geometric Brownian spot paths for BTC,
  ETH, SOL and gold; rolling 500-bucket topics whose LS-LMSR books are built
  with :func:`~glimpse_quant_bot.lmsr.book_from_prices` from a *stale and
  over-wide* crowd distribution; a crowd that trades and slowly converges on
  the true distribution; synthetic candles, L2 books, funding and a
  Deribit-like smile; resolution at expiry.
* :class:`SimulatedGlimpse` -- an ``AsyncClient`` whose REST methods and
  WebSocket stream are served from the world. Every call still passes through
  the client's rate limiter, so the budget gauge is honest.
* :class:`SimulatedHub` -- a ``MarketDataHub`` served from the world.

Nothing here touches the network. The simulated venue refuses live orders
unless it was explicitly built with ``allow_fake_live=True`` (tests only), and
the controller never arms live trading against it.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
from glimpse_markets import (
    AsyncClient,
    DryRunTradeResult,
    EnterMultiTopicLegGroup,
    MarketUpdate,
    QuoteMode,
    TradeLeg,
    TradeType,
)
from glimpse_markets.exceptions import (
    GlimpseNotFoundError,
    GlimpseTradingNotEligibleError,
    GlimpseValidationError,
)
from glimpse_markets.models import (
    BatchMarketItemV2,
    BatchMarketsPageResponse,
    BatchOutcomeSlim,
    BatchSummaryItem,
    EnterMultiTopicMultiLegResponse,
    EnterMultiTopicTopicResult,
    EstimateTradeLegsResponse,
    ListBatchesResponse,
    MarketUpdateData,
    MarketUpdateQuote,
    PortfolioItem,
    PortfolioListResponse,
    QuoteOutcome,
    QuotesResponse,
)
from glimpse_markets.money import Millisats
from glimpse_markets.ratelimit import AsyncRateLimiter
from glimpse_markets.streaming import MarketStream
from scipy.special import erf

from glimpse_quant_bot.external_feeds import (
    Candle,
    MarketDataHub,
    OptionQuote,
    OrderBook,
    SpotQuote,
)
from glimpse_quant_bot.lmsr import (
    GLIMPSE_ALPHA,
    LmsrBook,
    book_from_prices,
    buy_commission,
    sell_commission,
)
from glimpse_quant_bot.types import AssetType, Bucket, TopicMeta

FloatArray = npt.NDArray[np.float64]

SIM_BASE_URL = "https://simulated.glimpse.invalid"
FEE_RATE = 0.02
MIN_BUY_COMMISSION = Millisats(1000)
N_BUCKETS = 500
HOURLY_HISTORY = 1000
MINUTE_HISTORY = 360
RESOLVE_DELAY_SECONDS = 20.0
RESOLVED_RETENTION_SECONDS = 7200.0
STREAM_QUEUE = 2000
SEED_CONTRACTS = 40.0
YEAR_HOURS = 24.0 * 365.0


@dataclass(frozen=True, slots=True)
class SimAssetSpec:
    asset: AssetType
    spot: float
    hourly_vol: float
    bucket_width: float
    decimals: int
    batch_title: str
    market_title: str
    category: str
    liquidity_b: float


SPECS: dict[AssetType, SimAssetSpec] = {
    AssetType.BTC: SimAssetSpec(
        AssetType.BTC,
        97_250.0,
        0.0055,
        100.0,
        0,
        "Hourly Bitcoin price",
        "Bitcoin price at {when}",
        "Crypto",
        37.0,
    ),
    AssetType.ETH: SimAssetSpec(
        AssetType.ETH,
        3_420.0,
        0.0075,
        4.0,
        0,
        "Hourly Ethereum price",
        "Ethereum price at {when}",
        "Crypto",
        31.0,
    ),
    AssetType.SOL: SimAssetSpec(
        AssetType.SOL,
        186.0,
        0.0095,
        0.25,
        2,
        "Hourly Solana price",
        "Solana price at {when}",
        "Crypto",
        24.0,
    ),
    AssetType.GOLD: SimAssetSpec(
        AssetType.GOLD,
        2_680.0,
        0.0018,
        1.0,
        0,
        "Hourly Gold (PAXG) price",
        "Gold (PAXG) price at {when}",
        "Commodities",
        35.0,
    ),
}


def _student_t(rng: np.random.Generator, nu: float = 5.0) -> float:
    """Unit-variance Student-t draw (fat tails, like hourly crypto returns)."""
    return float(rng.standard_t(nu)) / math.sqrt(nu / (nu - 2.0))


def lognormal_bucket_probabilities(
    buckets: Iterable[Bucket], center: float, sigma: float, skew: float = 0.0
) -> FloatArray:
    """Bucket masses of a (slightly skewed) lognormal around ``center``."""
    rows = list(buckets)
    lows = np.array([bucket.lo for bucket in rows], dtype=np.float64)
    highs = np.array([bucket.hi for bucket in rows], dtype=np.float64)
    width = max(sigma, 1e-6)

    def cdf(prices: FloatArray) -> FloatArray:
        z = (np.log(np.maximum(prices, 1e-12) / center) + 0.5 * width**2) / width
        z = z - skew * np.tanh(z)
        return np.asarray(0.5 * (1.0 + _erf(z / math.sqrt(2.0))), dtype=np.float64)

    mass = np.maximum(cdf(highs) - cdf(lows), 0.0)
    total = float(mass.sum())
    if total <= 0.0:
        return np.full(len(rows), 1.0 / max(len(rows), 1))
    return np.asarray(mass / total, dtype=np.float64)


def _erf(values: FloatArray) -> FloatArray:
    return np.asarray(erf(values), dtype=np.float64)


class _BarBuilder:
    """Accumulates spot samples into fixed-interval OHLCV bars."""

    def __init__(self, interval: int, maxlen: int, seed: Iterable[Candle]) -> None:
        self.interval = interval
        self.bars: deque[Candle] = deque(seed, maxlen=maxlen)
        self._open_ts: int | None = None
        self._ohlc: list[float] = []
        self._volume = 0.0
        self._buy = 0.0

    def update(self, ts: float, price: float, volume: float, buy_volume: float) -> None:
        bucket = int(ts // self.interval) * self.interval
        if self._open_ts is not None and bucket != self._open_ts:
            o, h, low, c = self._ohlc
            self.bars.append(
                Candle(self._open_ts, o, h, low, c, max(self._volume, 1e-9), self._buy)
            )
            self._open_ts = None
        if self._open_ts is None:
            self._open_ts = bucket
            self._ohlc = [price, price, price, price]
            self._volume = 0.0
            self._buy = 0.0
        self._ohlc[1] = max(self._ohlc[1], price)
        self._ohlc[2] = min(self._ohlc[2], price)
        self._ohlc[3] = price
        self._volume += volume
        self._buy += buy_volume


@dataclass(slots=True)
class _AssetState:
    spec: SimAssetSpec
    spot: float
    session_open: float
    variance_scale: float
    imbalance: float
    funding: float
    hourly: _BarBuilder
    minutes: _BarBuilder
    last_ts: float
    toxic_until: float = 0.0


@dataclass(slots=True)
class SimTopic:
    meta: TopicMeta
    book: LmsrBook
    created_at: float
    final_price: float | None = None
    resolved_option_id: int | None = None
    resolved_at: float | None = None

    @property
    def batch_id(self) -> str:
        return self.meta.batch_id

    def quote_mode(self, now: float) -> str:
        if self.resolved_option_id is not None:
            return QuoteMode.RESOLVED.value
        if now >= self.meta.end_time_utc:
            return QuoteMode.CLOSED.value
        return QuoteMode.LIVE.value


@dataclass(slots=True, eq=False)
class _Stream:
    queue: asyncio.Queue[MarketUpdate] = field(default_factory=lambda: asyncio.Queue(STREAM_QUEUE))
    topic_id: int | None = None
    batch_id: str | None = None

    def offer(self, update: MarketUpdate) -> None:
        if self.topic_id is not None and update.topic_id != self.topic_id:
            return
        if self.batch_id is not None and update.batch_id != self.batch_id:
            return
        while True:
            try:
                self.queue.put_nowait(update)
                return
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.queue.get_nowait()


class SimulatedWorld:
    """The whole simulated market: spot paths, topics, books, crowd and resolution."""

    def __init__(
        self,
        assets: Iterable[AssetType] = tuple(SPECS),
        *,
        seed: int = 7,
        clock: Callable[[], float] = time.time,
        topics_per_asset: int = 5,
        cadence_seconds: int = 900,
        crowd_trades_per_minute: float = 8.0,
        allow_fake_live: bool = False,
    ) -> None:
        self._clock = clock
        self._rng = np.random.default_rng(seed)
        self._cadence = cadence_seconds
        self._topics_per_asset = topics_per_asset
        self._crowd_rate = crowd_trades_per_minute / 60.0
        self.allow_fake_live = allow_fake_live
        self.requests = 0
        """REST calls served (the WebSocket stream is not counted)."""
        self.live_orders = 0
        self.holdings: dict[tuple[int, int], float] = {}
        self.topics: dict[int, SimTopic] = {}
        self._streams: set[_Stream] = set()
        self._next_topic_id = 90_001
        now = clock()
        self._assets: dict[AssetType, _AssetState] = {
            asset: self._init_asset(SPECS[asset], now) for asset in assets
        }
        for asset in self._assets:
            self._ensure_topics(asset, now)

    # --- construction -----------------------------------------------------------

    def _history(
        self, spec: SimAssetSpec, now: float, interval: int, count: int, vol: float
    ) -> tuple[list[Candle], float]:
        """GARCH-flavoured bars ending at ``now``; returns them and the last close."""
        rng = self._rng
        closes = np.empty(count + 1)
        closes[0] = 0.0
        variance = vol**2
        omega, a, b = vol**2 * 0.04, 0.08, 0.88
        for i in range(count):
            shock = _student_t(rng) * math.sqrt(variance)
            closes[i + 1] = closes[i] + shock
            variance = omega + a * shock**2 + b * variance
        prices = spec.spot * np.exp(closes - closes[-1])
        start = (int(now // interval) - count) * interval
        bars: list[Candle] = []
        for i in range(count):
            o, c = float(prices[i]), float(prices[i + 1])
            span = abs(float(rng.normal(0.0, vol))) * 0.6
            volume = float(rng.lognormal(3.0, 0.4))
            bars.append(
                Candle(
                    start + i * interval,
                    o,
                    max(o, c) * math.exp(span),
                    min(o, c) * math.exp(-span),
                    c,
                    volume,
                    volume * float(rng.uniform(0.4, 0.6)),
                )
            )
        return bars, float(prices[-1])

    def _init_asset(self, spec: SimAssetSpec, now: float) -> _AssetState:
        hourly, _ = self._history(spec, now, 3600, HOURLY_HISTORY, spec.hourly_vol)
        minutes, _ = self._history(spec, now, 60, MINUTE_HISTORY, spec.hourly_vol / math.sqrt(60.0))
        return _AssetState(
            spec=spec,
            spot=spec.spot,
            session_open=spec.spot,
            variance_scale=1.0,
            imbalance=0.0,
            funding=1.0e-4,
            hourly=_BarBuilder(3600, HOURLY_HISTORY + 48, hourly),
            minutes=_BarBuilder(60, 1440, minutes),
            last_ts=now,
        )

    def _buckets(self, spec: SimAssetSpec, topic_id: int, center: float) -> tuple[Bucket, ...]:
        width = spec.bucket_width
        lo0 = round(center / width) * width - (N_BUCKETS // 2) * width
        base = topic_id * 1000
        rows: list[Bucket] = []
        for i in range(N_BUCKETS):
            lo = lo0 + i * width
            hi = lo + width
            name = f"{lo:.{spec.decimals}f}-{hi:.{spec.decimals}f}"
            rows.append(Bucket(base + i + 1, name, round(lo, 6), round(hi, 6)))
        return tuple(rows)

    def _new_topic(self, asset: AssetType, end_ts: int, now: float) -> SimTopic:
        state = self._assets[asset]
        spec = state.spec
        topic_id = self._next_topic_id
        self._next_topic_id += 1
        when = datetime.fromtimestamp(end_ts, tz=UTC).strftime("%H:%M UTC")
        buckets = self._buckets(spec, topic_id, state.spot)
        meta = TopicMeta(
            topic_id=topic_id,
            batch_id=self.batch_id(asset),
            title=spec.market_title.format(when=when),
            asset=asset,
            end_time_utc=end_ts,
            buckets=buckets,
        )
        hours = max((end_ts - now) / 3600.0, 0.05)
        # The crowd: anchored to where spot was, too wide, a little skewed.
        crowd_sigma = spec.hourly_vol * math.sqrt(hours) * float(self._rng.uniform(1.25, 1.7))
        crowd_center = state.spot * math.exp(float(self._rng.normal(0.0, 0.35)) * crowd_sigma)
        probabilities = lognormal_bucket_probabilities(
            buckets, crowd_center, crowd_sigma, skew=float(self._rng.uniform(-0.15, 0.15))
        )
        shaped = book_from_prices(
            topic_id,
            [bucket.option_id for bucket in buckets],
            (100.0 * probabilities).tolist(),
            liquidity_b=spec.liquidity_b * float(self._rng.uniform(0.85, 1.15)),
            alpha=GLIMPSE_ALPHA,
            ts=now,
        )
        # Glimpse seeds every bucket with subsidised contracts; keep a floor too.
        seed = max(0.0, SEED_CONTRACTS - float(np.min(shaped.shares)))
        book = LmsrBook.from_shares(
            topic_id, shaped.option_ids, (shaped.shares + seed).tolist(), GLIMPSE_ALPHA, now
        )
        topic = SimTopic(meta, book, now)
        self.topics[topic_id] = topic
        return topic

    def _ensure_topics(self, asset: AssetType, now: float) -> None:
        live = [
            topic
            for topic in self.topics.values()
            if topic.meta.asset is asset and topic.meta.end_time_utc > now + 300
        ]
        ends = {topic.meta.end_time_utc for topic in live}
        first = (int((now + 300) // self._cadence) + 1) * self._cadence
        for offset in range(1, self._topics_per_asset + 1):
            end_ts = first + (offset - 1) * self._cadence
            if end_ts not in ends:
                self._new_topic(asset, end_ts, now)

    # --- public views -------------------------------------------------------------

    @staticmethod
    def batch_id(asset: AssetType) -> str:
        return f"sim-{asset.value.lower()}-hourly"

    @property
    def assets(self) -> tuple[AssetType, ...]:
        return tuple(self._assets)

    def asset_for_batch(self, batch_id: str) -> AssetType | None:
        return next((a for a in self._assets if self.batch_id(a) == batch_id), None)

    def spot(self, asset: AssetType) -> float:
        return self._assets[asset].spot

    def topic(self, topic_id: int) -> SimTopic:
        topic = self.topics.get(topic_id)
        if topic is None:
            raise GlimpseNotFoundError(404, "topic not found", message=str(topic_id))
        return topic

    def true_probabilities(self, topic: SimTopic, now: float) -> FloatArray:
        state = self._assets[topic.meta.asset]
        hours = max((topic.meta.end_time_utc - now) / 3600.0, 1e-4)
        sigma = state.spec.hourly_vol * math.sqrt(hours * state.variance_scale)
        return lognormal_bucket_probabilities(topic.meta.buckets, state.spot, sigma)

    # --- streams ----------------------------------------------------------------

    def open_stream(self, topic_id: int | None, batch_id: str | None) -> _Stream:
        stream = _Stream(topic_id=topic_id, batch_id=batch_id)
        self._streams.add(stream)
        return stream

    def close_stream(self, stream: _Stream) -> None:
        self._streams.discard(stream)

    def _broadcast(self, topic: SimTopic, now: float) -> None:
        if not self._streams:
            return
        book = topic.book
        prices = book.prices()
        quotes = [
            MarketUpdateQuote(
                option_id=bucket.option_id,
                option_name=bucket.name,
                yes_price=round(float(prices[i]), 4),
                no_price=round(100.0 - float(prices[i]), 4),
                shares=float(book.shares[i]),
            )
            for i, bucket in enumerate(topic.meta.buckets)
        ]
        update = MarketUpdate(
            type="market_update",
            topic_id=topic.meta.topic_id,
            batch_id=topic.batch_id,
            data=MarketUpdateData(
                topic_id=topic.meta.topic_id,
                topic_type="range",
                batch_id=topic.batch_id,
                alpha=book.alpha,
                timestamp=int(now * 1000),
                quotes=quotes,
            ),
        )
        for stream in list(self._streams):
            stream.offer(update)

    # --- dynamics ---------------------------------------------------------------

    def step(self, now: float | None = None) -> None:
        """Advance spot, crowd flow, resolution and topic rollover to ``now``."""
        moment = self._clock() if now is None else now
        for asset, state in self._assets.items():
            self._advance_asset(state, moment)
            self._ensure_topics(asset, moment)
        self._crowd(moment)
        self._resolve(moment)

    def _advance_asset(self, state: _AssetState, now: float) -> None:
        dt = now - state.last_ts
        if dt <= 0.0:
            return
        rng = self._rng
        hours = dt / 3600.0
        # Slow stochastic volatility and mean-reverting order-book imbalance.
        state.variance_scale = float(
            np.clip(
                state.variance_scale * math.exp(0.15 * math.sqrt(hours) * float(rng.normal()))
                + 0.5 * hours * (1.0 - state.variance_scale),
                0.4,
                2.5,
            )
        )
        state.imbalance = float(
            np.clip(
                state.imbalance * math.exp(-dt / 120.0)
                + 0.25 * math.sqrt(min(dt, 60.0) / 60.0) * float(rng.normal()),
                -0.9,
                0.9,
            )
        )
        state.funding = float(
            np.clip(state.funding + 2.0e-5 * math.sqrt(hours) * float(rng.normal()), -4e-4, 6e-4)
        )
        sigma = state.spec.hourly_vol * math.sqrt(hours * state.variance_scale)
        drift = 0.04 * state.imbalance * sigma
        state.spot *= math.exp(drift + sigma * _student_t(rng) - 0.5 * sigma**2)
        state.last_ts = now
        if now > state.toxic_until and rng.random() < hours * 0.6:
            state.toxic_until = now + float(rng.uniform(300.0, 900.0))
        volume = float(rng.lognormal(0.0, 0.5)) * dt / 60.0
        toxic = now < state.toxic_until
        buy_share = 0.5 + 0.5 * state.imbalance if not toxic else float(rng.choice([0.08, 0.92]))
        buy_volume = volume * float(np.clip(buy_share + rng.normal(0.0, 0.05), 0.0, 1.0))
        state.minutes.update(now, state.spot, volume, buy_volume)
        state.hourly.update(now, state.spot, volume, buy_volume)

    def _crowd(self, now: float) -> None:
        rng = self._rng
        for topic in list(self.topics.values()):
            if topic.quote_mode(now) != QuoteMode.LIVE.value:
                continue
            elapsed = max(now - topic.book.ts, 0.0)
            if rng.random() > 1.0 - math.exp(-self._crowd_rate * elapsed):
                continue
            remaining = topic.meta.end_time_utc - now
            life = max(topic.meta.end_time_utc - topic.created_at, 1.0)
            informed = 0.35 + 0.55 * (1.0 - remaining / life)
            if rng.random() < informed:
                weights = self.true_probabilities(topic, now)
            else:
                weights = topic.book.probabilities()
            index = int(rng.choice(len(weights), p=weights / weights.sum()))
            contracts = float(rng.integers(1, 9))
            option_id = topic.meta.buckets[index].option_id
            topic.book = topic.book.applied([(option_id, contracts)])
            topic.book.ts = now
            self._broadcast(topic, now)

    def _resolve(self, now: float) -> None:
        for topic_id, topic in list(self.topics.items()):
            end = topic.meta.end_time_utc
            if now < end:
                continue
            if topic.final_price is None:
                topic.final_price = self._assets[topic.meta.asset].spot
            if topic.resolved_option_id is None and now >= end + RESOLVE_DELAY_SECONDS:
                final = topic.final_price
                buckets = topic.meta.buckets
                winner = next((b.option_id for b in buckets if b.lo < final <= b.hi), None)
                if winner is None:
                    winner = (
                        buckets[0].option_id if final <= buckets[0].lo else buckets[-1].option_id
                    )
                topic.resolved_option_id = winner
                topic.resolved_at = now
                for key in [key for key in self.holdings if key[0] == topic_id]:
                    del self.holdings[key]
            resolved_at = topic.resolved_at
            if resolved_at is not None and now - resolved_at > RESOLVED_RETENTION_SECONDS:
                del self.topics[topic_id]

    async def run(self, interval: float = 0.5) -> None:
        """Drive the world in real time until cancelled."""
        while True:
            self.step()
            await asyncio.sleep(interval)

    # --- market data ------------------------------------------------------------

    def spot_quote(self, asset: AssetType, now: float) -> SpotQuote:
        spot = self._assets[asset].spot
        return SpotQuote(price=spot, sources=("binance", "coinbase", "kraken"), ts=now)

    def candles(self, asset: AssetType, interval: int, limit: int) -> tuple[Candle, ...]:
        builder = self._assets[asset].minutes if interval <= 60 else self._assets[asset].hourly
        bars = list(builder.bars)
        return tuple(bars[-limit:])

    def orderbook(self, asset: AssetType) -> OrderBook:
        state = self._assets[asset]
        rng = self._rng
        mid = state.spot
        tick = mid * 1.0e-4
        tilt = state.imbalance
        bids = tuple(
            (mid - tick * (i + 0.5), float(rng.exponential(1.0)) * (1.0 + tilt)) for i in range(20)
        )
        asks = tuple(
            (mid + tick * (i + 0.5), float(rng.exponential(1.0)) * (1.0 - tilt)) for i in range(20)
        )
        return OrderBook(bids=bids, asks=asks)

    def funding(self, asset: AssetType) -> float:
        return self._assets[asset].funding

    def option_chain(self, asset: AssetType, now: float) -> tuple[OptionQuote, ...]:
        if not asset.has_listed_options:
            return ()
        state = self._assets[asset]
        spot = state.spot
        atm_base = state.spec.hourly_vol * math.sqrt(YEAR_HOURS) * 1.08
        today = datetime.fromtimestamp(now, tz=UTC).replace(
            hour=8, minute=0, second=0, microsecond=0
        )
        first = today if today.timestamp() > now + 3600 else today + timedelta(days=1)
        expiries = [first + timedelta(days=days) for days in (0, 1, 2, 6, 27)]
        chain: list[OptionQuote] = []
        for expiry in expiries:
            expiry_ts = int(expiry.timestamp())
            t_years = (expiry_ts - now) / (YEAR_HOURS * 3600.0)
            atm = atm_base * (1.0 + 0.05 * math.log1p(t_years * 12.0))
            forward = spot * math.exp(0.05 * t_years)
            scale = atm * math.sqrt(t_years)
            for z in np.linspace(-3.0, 3.0, 25):
                strike = round(forward * math.exp(float(z) * scale), 2)
                moneyness = math.log(strike / forward) / scale
                iv = atm * (1.0 - 0.035 * moneyness + 0.025 * moneyness**2)
                chain.extend(
                    OptionQuote(
                        instrument=f"{asset.value}-{expiry:%d%b%y}-{strike:g}-"
                        + ("C" if is_call else "P"),
                        expiry_ts=expiry_ts,
                        strike=strike,
                        is_call=is_call,
                        mark_iv=max(iv, 0.05),
                        forward=forward,
                        open_interest=float(self._rng.uniform(5.0, 400.0)),
                    )
                    for is_call in (True, False)
                )
        return tuple(chain)

    # --- venue responses ----------------------------------------------------------

    def market_item(self, topic: SimTopic, now: float) -> BatchMarketItemV2:
        mode = topic.quote_mode(now)
        return BatchMarketItemV2(
            topic_id=topic.meta.topic_id,
            batch_id=topic.batch_id,
            title=topic.meta.title,
            description=f"Where will {topic.meta.asset.value} settle?",
            category=self._assets[topic.meta.asset].spec.category,
            topic_type="range",
            quote_mode=mode,
            is_active=mode == QuoteMode.LIVE.value,
            is_resolved=topic.resolved_option_id is not None,
            resolved_option_id=topic.resolved_option_id,
            market_subsidised=True,
            num_outcomes=len(topic.meta.buckets),
            outcomes=[
                BatchOutcomeSlim(option_id=bucket.option_id, name=bucket.name, shares=float(q))
                for bucket, q in zip(topic.meta.buckets, topic.book.shares, strict=True)
            ],
            end_time_utc=topic.meta.end_time_utc,
            date_created=int(topic.created_at),
        )

    def estimate(
        self, topic_id: int, trade_type: TradeType | str, legs: list[TradeLeg]
    ) -> EstimateTradeLegsResponse:
        topic = self.topic(topic_id)
        pairs = [(leg.option_id, leg.contracts) for leg in legs]
        side = trade_type.value if isinstance(trade_type, TradeType) else str(trade_type)
        try:
            if side == TradeType.BUY.value:
                amount = topic.book.buy_cost(pairs)
                commission = buy_commission(amount, FEE_RATE, MIN_BUY_COMMISSION)
            else:
                amount = topic.book.sell_proceeds(pairs)
                commission = sell_commission(amount, FEE_RATE)
        except (ValueError, KeyError) as exc:
            raise GlimpseValidationError(400, "invalid trade", message=str(exc)) from exc
        return EstimateTradeLegsResponse(
            topic_id=topic_id,
            trade_type=side,
            leg_count=len(pairs),
            total_cost=int(amount) / 1000.0,
            total_cost_millisats=amount,
            commission_millisats=commission,
        )


class SimulatedStream(MarketStream):
    """``MarketStream`` fed by the simulated world instead of a WebSocket."""

    def __init__(
        self, world: SimulatedWorld, topic_id: int | None = None, batch_id: str | None = None
    ) -> None:
        super().__init__(base_url=SIM_BASE_URL, topic_id=topic_id, batch_id=batch_id)
        self._world = world
        self._filter = (topic_id, batch_id)
        self._stream: _Stream | None = None

    async def connect(self) -> None:
        self._stream = self._world.open_stream(*self._filter)

    async def subscribe(self, topic_id: int | None = None, batch_id: str | None = None) -> None:
        if self._stream is None:
            raise RuntimeError("not connected")
        self._stream.topic_id = topic_id
        self._stream.batch_id = batch_id

    async def close(self) -> None:
        if self._stream is not None:
            self._world.close_stream(self._stream)
            self._stream = None

    def __aiter__(self) -> AsyncIterator[MarketUpdate]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MarketUpdate]:
        if self._stream is None:
            raise RuntimeError("not connected")
        while True:
            yield await self._stream.queue.get()


class SimulatedGlimpse(AsyncClient):
    """An ``AsyncClient`` served from a :class:`SimulatedWorld` (no network)."""

    def __init__(
        self,
        world: SimulatedWorld,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
        dry_run: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(
            api_key="simulated", base_url=SIM_BASE_URL, rate_limiter=rate_limiter, dry_run=dry_run
        )
        self._world = world
        self._sim_clock = clock

    async def _request(self) -> float:
        await self._rate_limiter.acquire()
        self._world.requests += 1
        return self._sim_clock()

    async def wallet_balance(self) -> dict[str, Any]:
        await self._request()
        return {"available_balance_millisats": 100_000_000}

    async def batches(self) -> ListBatchesResponse:
        await self._request()
        world = self._world
        items = [
            BatchSummaryItem(
                batch_id=world.batch_id(asset),
                main_topic_title=SPECS[asset].batch_title,
                topic_count=sum(1 for t in world.topics.values() if t.meta.asset is asset),
            )
            for asset in world.assets
        ]
        return ListBatchesResponse(batches=items, count=len(items))

    async def batch_active_markets_page(
        self, batch_id: str, page: int | None = None, page_size: int | None = None
    ) -> BatchMarketsPageResponse:
        now = await self._request()
        world = self._world
        live = sorted(
            (
                topic
                for topic in world.topics.values()
                if topic.batch_id == batch_id and topic.quote_mode(now) == QuoteMode.LIVE.value
            ),
            key=lambda topic: topic.meta.end_time_utc,
        )[:24]
        asset = world.asset_for_batch(batch_id)
        return BatchMarketsPageResponse(
            batch_id=batch_id,
            main_topic_title=SPECS[asset].batch_title if asset is not None else None,
            markets=[world.market_item(topic, now) for topic in live],
            count=len(live),
            total=len(live),
            has_more=False,
        )

    async def batch_resolved_markets_page(
        self, batch_id: str, page: int | None = None, page_size: int | None = None
    ) -> BatchMarketsPageResponse:
        now = await self._request()
        world = self._world
        resolved = [
            topic
            for topic in world.topics.values()
            if topic.batch_id == batch_id and topic.resolved_option_id is not None
        ]
        return BatchMarketsPageResponse(
            batch_id=batch_id,
            markets=[world.market_item(topic, now) for topic in resolved],
            count=len(resolved),
            total=len(resolved),
            has_more=False,
        )

    async def market_quotes(self, topic_id: int) -> QuotesResponse:
        now = await self._request()
        topic = self._world.topic(topic_id)
        prices = topic.book.prices()
        mode = topic.quote_mode(now)
        return QuotesResponse(
            topic_id=topic_id,
            batch_id=topic.batch_id,
            title=topic.meta.title,
            description=f"Where will {topic.meta.asset.value} settle?",
            category=SPECS[topic.meta.asset].category,
            quote_mode=mode,
            is_resolved=topic.resolved_option_id is not None,
            market_end_time_utc=topic.meta.end_time_utc,
            outcomes=[
                QuoteOutcome(
                    option_id=bucket.option_id,
                    name=bucket.name,
                    yes_price=float(prices[i]),
                    shares=float(topic.book.shares[i]),
                )
                for i, bucket in enumerate(topic.meta.buckets)
            ],
        )

    async def portfolio_active(self) -> PortfolioListResponse:
        await self._request()
        items = [
            PortfolioItem(topic_id=topic_id, option_id=option_id, shares=shares, is_active=True)
            for (topic_id, option_id), shares in self._world.holdings.items()
            if shares > 0
        ]
        return PortfolioListResponse(success=True, message=items, batch_id="")

    async def estimate_trade(
        self, topic_id: int, trade_type: TradeType | str, legs: list[TradeLeg]
    ) -> EstimateTradeLegsResponse:
        await self._request()
        return self._world.estimate(topic_id, trade_type, legs)

    def _refuse_live(self) -> None:
        if not self._world.allow_fake_live:
            raise GlimpseTradingNotEligibleError(
                403, "simulated venue", reason="SIMULATED", message="live orders are impossible"
            )

    async def enter_multi_topic_multi_leg(
        self, topics: list[EnterMultiTopicLegGroup], *, dry_run: bool | None = None
    ) -> EnterMultiTopicMultiLegResponse | DryRunTradeResult:
        if self._effective_dry_run(dry_run):
            return await super().enter_multi_topic_multi_leg(topics, dry_run=True)
        self._refuse_live()
        await self._request()
        world = self._world
        results: list[EnterMultiTopicTopicResult] = []
        for group in topics:
            estimate = world.estimate(group.topic_id, TradeType.BUY, group.legs)
            topic = world.topic(group.topic_id)
            pairs = [(leg.option_id, leg.contracts) for leg in group.legs]
            topic.book = topic.book.applied(pairs)
            for option_id, contracts in pairs:
                key = (group.topic_id, option_id)
                world.holdings[key] = world.holdings.get(key, 0.0) + contracts
            world.live_orders += 1
            results.append(
                EnterMultiTopicTopicResult(
                    topic_id=group.topic_id,
                    trade_id=f"sim-{world.live_orders}",
                    total_cost_millisats=estimate.total_cost_millisats,
                    commission_millisats=estimate.commission_millisats,
                )
            )
        return EnterMultiTopicMultiLegResponse(
            results=results, topics_requested=len(topics), topics_entered=len(results)
        )

    async def exit_consolidated(
        self,
        topic_id: int,
        option_id: int,
        shares: float | None = None,
        *,
        dry_run: bool | None = None,
    ) -> dict[str, Any] | DryRunTradeResult:
        if self._effective_dry_run(dry_run):
            return await super().exit_consolidated(topic_id, option_id, shares, dry_run=True)
        self._refuse_live()
        await self._request()
        world = self._world
        held = world.holdings.get((topic_id, option_id), 0.0)
        amount = min(shares or held, held)
        estimate = world.estimate(
            topic_id, TradeType.SELL, [TradeLeg(option_id=option_id, contracts=amount)]
        )
        topic = world.topic(topic_id)
        topic.book = topic.book.applied([(option_id, amount)], sign=-1.0)
        world.holdings[(topic_id, option_id)] = held - amount
        world.live_orders += 1
        return {
            "trade_id": f"sim-{world.live_orders}",
            "total_proceeds_millisats": int(estimate.total_cost_millisats or 0),
            "commission_millisats": int(estimate.commission_millisats or 0),
        }

    def stream_market_updates(
        self, topic_id: int | None = None, batch_id: str | None = None
    ) -> MarketStream:
        return SimulatedStream(self._world, topic_id, batch_id)


class SimulatedHub(MarketDataHub):
    """``MarketDataHub`` served from the simulated world (no network)."""

    def __init__(
        self,
        world: SimulatedWorld,
        assets: Iterable[AssetType],
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(tuple(assets), enable_websocket=False)
        self._world = world
        self._sim_clock = clock

    async def start(self) -> None:
        return None

    async def spot(self, asset: AssetType) -> SpotQuote:
        return self._world.spot_quote(asset, self._sim_clock())

    async def candles(
        self, asset: AssetType, interval_seconds: int = 3600, limit: int = 1000
    ) -> tuple[Candle, ...]:
        return self._world.candles(asset, interval_seconds, limit)

    async def orderbook(self, asset: AssetType) -> OrderBook | None:
        return self._world.orderbook(asset)

    async def flow_candles(self, asset: AssetType, limit: int = 1000) -> tuple[Candle, ...]:
        return self._world.candles(asset, 60, limit)

    async def funding_rate(self, asset: AssetType) -> float | None:
        return self._world.funding(asset)

    async def option_chain(self, asset: AssetType) -> tuple[OptionQuote, ...]:
        return self._world.option_chain(asset, self._sim_clock())


__all__ = [
    "SPECS",
    "SimTopic",
    "SimulatedGlimpse",
    "SimulatedHub",
    "SimulatedStream",
    "SimulatedWorld",
    "lognormal_bucket_probabilities",
]
