"""Async external market-data feeds with TTL caching and venue failover.

Only free, public, commercially usable endpoints are used -- no API keys and
no ``yfinance``:

* **Deribit** ``public/get_book_summary_by_currency`` -- BTC and ETH option
  chains (mark implied volatilities and per-expiry forwards).
* **Binance**, **Coinbase Exchange** and **Kraken** public REST -- spot
  tickers, OHLCV candles and L2 order books for BTC, ETH, SOL and PAXG (gold).
* **Binance USD-M futures** ``premiumIndex`` -- perpetual funding rates.
* **Binance** ``bookTicker`` WebSocket -- push updates of the best bid/ask so
  the spot reference stays fresh between REST polls.

:class:`MarketDataHub` is the single entry point. Spot is the median across
all healthy venues; candles, books and funding fall through a priority list.
A per-venue circuit breaker stops a failing venue from being retried on every
call, and the TTL cache serves a slightly stale value rather than failing when
every venue is down.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, TypeVar

import httpx
import structlog
import websockets

from glimpse_quant_bot.types import AssetType

log = structlog.get_logger(__name__)

T = TypeVar("T")

USER_AGENT: Final[str] = "glimpse-quant-bot/0.1"
HTTP_TIMEOUT_SECONDS: Final[float] = 8.0
DERIBIT_EXPIRY_HOUR_UTC: Final[int] = 8


class FeedError(RuntimeError):
    """No venue could serve the requested data."""


@dataclass(frozen=True, slots=True)
class Candle:
    ts: int
    """Bar open time, unix seconds."""
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class OrderBook:
    bids: tuple[tuple[float, float], ...]
    """``(price, size)`` best first."""
    asks: tuple[tuple[float, float], ...]

    @property
    def mid(self) -> float:
        return 0.5 * (self.bids[0][0] + self.asks[0][0])

    def imbalance(self, band_bps: float = 15.0) -> float:
        """Depth imbalance ``(B - A) / (B + A)`` within ``band_bps`` of the mid.

        ``+1`` means all resting size is on the bid (buy pressure), ``-1`` all
        on the ask.
        """
        mid = self.mid
        band = mid * band_bps / 1.0e4
        bid_size = sum(size for price, size in self.bids if price >= mid - band)
        ask_size = sum(size for price, size in self.asks if price <= mid + band)
        total = bid_size + ask_size
        return 0.0 if total <= 0.0 else (bid_size - ask_size) / total


@dataclass(frozen=True, slots=True)
class OptionQuote:
    instrument: str
    expiry_ts: int
    strike: float
    is_call: bool
    mark_iv: float
    """Annualised implied volatility as a decimal (0.55 = 55%)."""
    forward: float
    """Underlying forward for this expiry, in USD."""
    open_interest: float


@dataclass(frozen=True, slots=True)
class SpotQuote:
    price: float
    sources: tuple[str, ...]
    ts: float


class TTLCache:
    """Async memoisation with a time-to-live and stale-on-error fallback."""

    def __init__(self, max_stale_seconds: float = 300.0) -> None:
        self._values: dict[str, tuple[float, float, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._max_stale = max_stale_seconds

    async def get(self, key: str, ttl: float, fetch: Callable[[], Awaitable[T]]) -> T:
        now = time.monotonic()
        cached = self._values.get(key)
        if cached is not None and now < cached[0]:
            value: T = cached[2]
            return value
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            cached = self._values.get(key)
            if cached is not None and now < cached[0]:
                value = cached[2]
                return value
            try:
                fresh = await fetch()
            except (FeedError, httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
                if cached is not None and now - cached[1] <= self._max_stale:
                    log.warning("feed.serving_stale", key=key, age=round(now - cached[1], 1))
                    value = cached[2]
                    return value
                raise
            self._values[key] = (now + ttl, now, fresh)
            return fresh


class _Venue(ABC):
    """A spot venue: tickers, candles and order books for a set of assets."""

    name: str
    symbols: Mapping[AssetType, str]

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http
        self._failures = 0
        self._open_until = 0.0

    def supports(self, asset: AssetType) -> bool:
        return asset in self.symbols

    @property
    def healthy(self) -> bool:
        return time.monotonic() >= self._open_until

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= 3:
            self._open_until = time.monotonic() + min(120.0, 5.0 * 2 ** (self._failures - 3))

    async def _get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        response = await self._http.get(url, params=dict(params) if params else None)
        response.raise_for_status()
        return response.json()

    @abstractmethod
    async def ticker(self, asset: AssetType) -> float:
        """Best bid/ask mid price in USD."""

    @abstractmethod
    async def candles(self, asset: AssetType, interval_seconds: int, limit: int) -> list[Candle]:
        """Completed OHLCV bars, oldest first."""

    @abstractmethod
    async def orderbook(self, asset: AssetType) -> OrderBook:
        """Top-of-book L2 snapshot."""


class BinanceSpot(_Venue):
    name = "binance"
    symbols: Mapping[AssetType, str] = {
        AssetType.BTC: "BTCUSDT",
        AssetType.ETH: "ETHUSDT",
        AssetType.SOL: "SOLUSDT",
        AssetType.GOLD: "PAXGUSDT",
    }
    _HOSTS: tuple[str, ...] = ("https://data-api.binance.vision", "https://api.binance.com")
    _INTERVALS: Mapping[int, str] = {
        60: "1m",
        300: "5m",
        900: "15m",
        3600: "1h",
        14400: "4h",
        86400: "1d",
    }

    async def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        last_error: Exception | None = None
        for host in self._HOSTS:
            try:
                return await self._get_json(f"{host}{path}", params)
            except httpx.HTTPError as exc:
                last_error = exc
        raise FeedError(f"binance unreachable: {last_error!r}")

    async def ticker(self, asset: AssetType) -> float:
        data = await self._get("/api/v3/ticker/bookTicker", {"symbol": self.symbols[asset]})
        return 0.5 * (float(data["bidPrice"]) + float(data["askPrice"]))

    async def candles(self, asset: AssetType, interval_seconds: int, limit: int) -> list[Candle]:
        interval = self._INTERVALS.get(interval_seconds)
        if interval is None:
            raise FeedError(f"binance has no {interval_seconds}s candles")
        rows = await self._get(
            "/api/v3/klines",
            {"symbol": self.symbols[asset], "interval": interval, "limit": min(limit + 1, 1000)},
        )
        now_ms = time.time() * 1000.0
        return [
            Candle(
                ts=int(row[0]) // 1000,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in rows
            if float(row[6]) < now_ms
        ][-limit:]

    async def orderbook(self, asset: AssetType) -> OrderBook:
        data = await self._get("/api/v3/depth", {"symbol": self.symbols[asset], "limit": 100})
        return OrderBook(
            bids=tuple((float(p), float(s)) for p, s in data["bids"]),
            asks=tuple((float(p), float(s)) for p, s in data["asks"]),
        )


class CoinbaseSpot(_Venue):
    name = "coinbase"
    symbols: Mapping[AssetType, str] = {
        AssetType.BTC: "BTC-USD",
        AssetType.ETH: "ETH-USD",
        AssetType.SOL: "SOL-USD",
        AssetType.GOLD: "PAXG-USD",
    }
    _BASE = "https://api.exchange.coinbase.com"
    _GRANULARITIES = frozenset({60, 300, 900, 3600, 21600, 86400})

    async def ticker(self, asset: AssetType) -> float:
        data = await self._get_json(f"{self._BASE}/products/{self.symbols[asset]}/ticker")
        return 0.5 * (float(data["bid"]) + float(data["ask"]))

    async def candles(self, asset: AssetType, interval_seconds: int, limit: int) -> list[Candle]:
        if interval_seconds not in self._GRANULARITIES:
            raise FeedError(f"coinbase has no {interval_seconds}s candles")
        rows = await self._get_json(
            f"{self._BASE}/products/{self.symbols[asset]}/candles",
            {"granularity": interval_seconds},
        )
        cutoff = time.time() - interval_seconds
        bars = [
            Candle(
                ts=int(row[0]),
                low=float(row[1]),
                high=float(row[2]),
                open=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in rows
            if float(row[0]) <= cutoff
        ]
        bars.sort(key=lambda bar: bar.ts)
        return bars[-limit:]

    async def orderbook(self, asset: AssetType) -> OrderBook:
        data = await self._get_json(
            f"{self._BASE}/products/{self.symbols[asset]}/book", {"level": 2}
        )
        return OrderBook(
            bids=tuple((float(row[0]), float(row[1])) for row in data["bids"][:100]),
            asks=tuple((float(row[0]), float(row[1])) for row in data["asks"][:100]),
        )


class KrakenSpot(_Venue):
    name = "kraken"
    symbols: Mapping[AssetType, str] = {
        AssetType.BTC: "XBTUSD",
        AssetType.ETH: "ETHUSD",
        AssetType.SOL: "SOLUSD",
        AssetType.GOLD: "PAXGUSD",
    }
    _BASE = "https://api.kraken.com/0/public"

    async def _result(self, path: str, params: Mapping[str, Any]) -> Any:
        data = await self._get_json(f"{self._BASE}/{path}", params)
        if data.get("error"):
            raise FeedError(f"kraken error: {data['error']}")
        payload = {key: value for key, value in data["result"].items() if key != "last"}
        return next(iter(payload.values()))

    async def ticker(self, asset: AssetType) -> float:
        data = await self._result("Ticker", {"pair": self.symbols[asset]})
        return 0.5 * (float(data["b"][0]) + float(data["a"][0]))

    async def candles(self, asset: AssetType, interval_seconds: int, limit: int) -> list[Candle]:
        if interval_seconds % 60 != 0:
            raise FeedError(f"kraken has no {interval_seconds}s candles")
        rows = await self._result(
            "OHLC", {"pair": self.symbols[asset], "interval": interval_seconds // 60}
        )
        cutoff = time.time() - interval_seconds
        return [
            Candle(
                ts=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[6]),
            )
            for row in rows
            if float(row[0]) <= cutoff
        ][-limit:]

    async def orderbook(self, asset: AssetType) -> OrderBook:
        data = await self._result("Depth", {"pair": self.symbols[asset], "count": 100})
        return OrderBook(
            bids=tuple((float(row[0]), float(row[1])) for row in data["bids"]),
            asks=tuple((float(row[0]), float(row[1])) for row in data["asks"]),
        )


class BinancePerpFunding:
    """Most recent funding rate of the USD-M perpetual (per funding interval)."""

    _URL = "https://fapi.binance.com/fapi/v1/premiumIndex"

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def funding_rate(self, asset: AssetType) -> float:
        response = await self._http.get(self._URL, params={"symbol": BinanceSpot.symbols[asset]})
        response.raise_for_status()
        return float(response.json()["lastFundingRate"])


class DeribitOptions:
    """Deribit option chains for the assets that have listed options."""

    _URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    _CURRENCIES: Mapping[AssetType, str] = {AssetType.BTC: "BTC", AssetType.ETH: "ETH"}

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    def supports(self, asset: AssetType) -> bool:
        return asset in self._CURRENCIES

    async def chain(self, asset: AssetType) -> tuple[OptionQuote, ...]:
        response = await self._http.get(
            self._URL, params={"currency": self._CURRENCIES[asset], "kind": "option"}
        )
        response.raise_for_status()
        return parse_deribit_chain(response.json().get("result", []))


def parse_deribit_instrument(name: str) -> tuple[int, float, bool]:
    """``"BTC-20NOV26-64000-P"`` -> ``(expiry_ts, strike, is_call)``.

    Deribit options expire at 08:00 UTC on the expiry date.
    """
    _, date_part, strike_part, kind = name.split("-")
    expiry = datetime.strptime(date_part, "%d%b%y").replace(
        hour=DERIBIT_EXPIRY_HOUR_UTC, tzinfo=UTC
    )
    return int(expiry.timestamp()), float(strike_part), kind.upper() == "C"


def parse_deribit_chain(rows: Sequence[Mapping[str, Any]]) -> tuple[OptionQuote, ...]:
    quotes: list[OptionQuote] = []
    for row in rows:
        mark_iv = row.get("mark_iv")
        forward = row.get("underlying_price")
        name = row.get("instrument_name")
        if not name or not mark_iv or not forward:
            continue
        try:
            expiry_ts, strike, is_call = parse_deribit_instrument(str(name))
        except ValueError:
            continue
        quotes.append(
            OptionQuote(
                instrument=str(name),
                expiry_ts=expiry_ts,
                strike=strike,
                is_call=is_call,
                mark_iv=float(mark_iv) / 100.0,
                forward=float(forward),
                open_interest=float(row.get("open_interest") or 0.0),
            )
        )
    return tuple(quotes)


class BinanceTickerStream:
    """Background WebSocket keeping the latest Binance best bid/ask mid per asset."""

    _URL = "wss://data-stream.binance.vision/stream?streams="

    def __init__(self, assets: Sequence[AssetType]) -> None:
        self._symbol_to_asset = {BinanceSpot.symbols[asset].lower(): asset for asset in assets}
        self._mids: dict[AssetType, tuple[float, float]] = {}
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None and self._symbol_to_asset:
            self._task = asyncio.create_task(self._run(), name="binance-book-ticker")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def mid(self, asset: AssetType, max_age_seconds: float = 3.0) -> float | None:
        entry = self._mids.get(asset)
        if entry is None or time.monotonic() - entry[1] > max_age_seconds:
            return None
        return entry[0]

    async def _run(self) -> None:
        streams = "/".join(f"{symbol}@bookTicker" for symbol in self._symbol_to_asset)
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self._URL + streams, ping_interval=20) as conn:
                    backoff = 1.0
                    async for raw in conn:
                        self._ingest(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except (OSError, websockets.WebSocketException, ValueError) as exc:
                log.warning("feed.ws_reconnect", error=repr(exc), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)

    def _ingest(self, message: Mapping[str, Any]) -> None:
        data = message.get("data", message)
        symbol = str(data.get("s", "")).lower()
        asset = self._symbol_to_asset.get(symbol)
        if asset is None or "b" not in data or "a" not in data:
            return
        self._mids[asset] = (0.5 * (float(data["b"]) + float(data["a"])), time.monotonic())


class MarketDataHub:
    """Failover-aware facade over every external feed."""

    SPOT_TTL = 2.0
    BOOK_TTL = 5.0
    CANDLE_TTL = 60.0
    FUNDING_TTL = 120.0
    OPTIONS_TTL = 30.0
    OUTLIER_TOLERANCE = 0.01

    def __init__(
        self,
        assets: Sequence[AssetType],
        http: httpx.AsyncClient | None = None,
        enable_websocket: bool = True,
    ) -> None:
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
        )
        self._venues: tuple[_Venue, ...] = (
            BinanceSpot(self._http),
            CoinbaseSpot(self._http),
            KrakenSpot(self._http),
        )
        self._funding = BinancePerpFunding(self._http)
        self._options = DeribitOptions(self._http)
        self._cache = TTLCache()
        self._stream = BinanceTickerStream(assets) if enable_websocket else None

    async def start(self) -> None:
        if self._stream is not None:
            self._stream.start()

    async def close(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
        if self._owns_http:
            await self._http.aclose()

    def _candidates(self, asset: AssetType) -> list[_Venue]:
        return [venue for venue in self._venues if venue.supports(asset) and venue.healthy]

    async def _call(self, venue: _Venue, coro: Awaitable[T]) -> T:
        try:
            result = await coro
        except (httpx.HTTPError, FeedError, ValueError, KeyError, IndexError, TypeError) as exc:
            venue.record_failure()
            raise FeedError(f"{venue.name}: {exc!r}") from exc
        venue.record_success()
        return result

    async def _first(
        self, asset: AssetType, what: str, call: Callable[[_Venue], Awaitable[T]]
    ) -> T:
        errors: list[str] = []
        for venue in self._candidates(asset):
            try:
                return await self._call(venue, call(venue))
            except FeedError as exc:
                errors.append(str(exc))
        raise FeedError(f"no venue could serve {what} for {asset.value}: {errors}")

    async def spot(self, asset: AssetType) -> SpotQuote:
        """Median mid price across every healthy venue, outliers removed."""
        return await self._cache.get(
            f"spot:{asset.value}", self.SPOT_TTL, lambda: self._spot(asset)
        )

    async def _spot(self, asset: AssetType) -> SpotQuote:
        async def one(venue: _Venue) -> tuple[str, float]:
            if venue.name == "binance" and self._stream is not None:
                pushed = self._stream.mid(asset)
                if pushed is not None:
                    return venue.name, pushed
            return venue.name, await self._call(venue, venue.ticker(asset))

        results = await asyncio.gather(
            *(one(venue) for venue in self._candidates(asset)), return_exceptions=True
        )
        quotes = [item for item in results if not isinstance(item, BaseException)]
        if not quotes:
            raise FeedError(f"no venue returned a spot price for {asset.value}")
        median = statistics.median(price for _, price in quotes)
        kept = [(n, p) for n, p in quotes if abs(p / median - 1.0) <= self.OUTLIER_TOLERANCE]
        return SpotQuote(
            price=statistics.median(price for _, price in kept),
            sources=tuple(name for name, _ in kept),
            ts=time.time(),
        )

    async def candles(
        self, asset: AssetType, interval_seconds: int = 3600, limit: int = 1000
    ) -> tuple[Candle, ...]:
        async def fetch() -> tuple[Candle, ...]:
            bars = await self._first(
                asset, "candles", lambda venue: venue.candles(asset, interval_seconds, limit)
            )
            if len(bars) < 30:
                raise FeedError(f"only {len(bars)} candles available for {asset.value}")
            return tuple(bars)

        return await self._cache.get(
            f"candles:{asset.value}:{interval_seconds}:{limit}", self.CANDLE_TTL, fetch
        )

    async def orderbook_imbalance(self, asset: AssetType) -> float | None:
        """L2 depth imbalance in ``[-1, 1]``, or ``None`` when no book is available."""

        async def fetch() -> float:
            book = await self._first(asset, "order book", lambda venue: venue.orderbook(asset))
            return book.imbalance()

        try:
            return await self._cache.get(f"book:{asset.value}", self.BOOK_TTL, fetch)
        except FeedError:
            return None

    async def funding_rate(self, asset: AssetType) -> float | None:
        """Latest perpetual funding rate, or ``None`` when unavailable."""
        try:
            return await self._cache.get(
                f"funding:{asset.value}",
                self.FUNDING_TTL,
                lambda: self._funding.funding_rate(asset),
            )
        except (FeedError, httpx.HTTPError, ValueError, KeyError):
            return None

    async def option_chain(self, asset: AssetType) -> tuple[OptionQuote, ...]:
        """Deribit chain for ``asset``; empty when the asset has no listed options
        or Deribit is unreachable (the ensemble then re-weights)."""
        if not self._options.supports(asset):
            return ()
        try:
            return await self._cache.get(
                f"options:{asset.value}", self.OPTIONS_TTL, lambda: self._options.chain(asset)
            )
        except (FeedError, httpx.HTTPError, ValueError, KeyError):
            log.warning("feed.options_unavailable", asset=asset.value)
            return ()
