"""Walk-forward replay engine over ``MarketRecorder`` NDJSON recordings.

The replay drives the *same* components as live trading -- ``build_forecast``,
``SlippageOptimizer``, ``RiskManager``, ``evaluate_exit`` and the append-only
``Ledger`` -- so a backtest exercises the production decision path, not a
re-implementation of it. Only two things are substituted:

* prices come from a recording instead of the WebSocket, and
* fills are priced by the exact local LS-LMSR replica instead of the server.

Walk-forward discipline: at replay time ``t`` the models are fitted on candles
that had *closed* before ``t`` and see the spot price at ``t``; nothing from the
future is visible. The bot's own fills are added to the reconstructed book, so
buying the same mispricing twice gets progressively more expensive, as it would
live.

Inputs (all plain files, see :func:`save_dataset`):

``prices.ndjson``
    Rows ``{"ts", "topic_id", "option_id", "price"}`` exactly as written by the
    SDK's ``MarketRecorder(persist_path=...)``; loaded with ``MarketRecorder.load``.
``meta.json``
    Topic metadata the price feed does not carry: asset, expiry, bucket ranges
    and, when known, the resolved option.
``candles.ndjson`` / ``spot.ndjson``
    History of the underlying asset.

A price recording stores only the 0-100 quote of each bucket. The share vector
is reconstructed with :func:`glimpse_quant_bot.lmsr.book_from_prices` under an
assumed depth ``b`` (``backtest_liquidity_b``).
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from glimpse_markets.forecasting.recorder import MarketRecorder
from glimpse_markets.money import Millisats

from glimpse_quant_bot.calibration import OnlineCalibrator, PendingForecast, calibration_prior
from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.external_feeds import Candle
from glimpse_quant_bot.forecaster import (
    FastInputs,
    ForecastError,
    SlowFit,
    build_forecast,
    build_signals,
    fit_slow,
)
from glimpse_quant_bot.ledger import Ledger, PositionKey
from glimpse_quant_bot.lmsr import LmsrBook, book_from_prices
from glimpse_quant_bot.lmsr_optimizer import LocalEstimator, SlippageOptimizer, evaluate_exit
from glimpse_quant_bot.risk_manager import RiskManager
from glimpse_quant_bot.types import (
    AssetType,
    Bucket,
    LedgerRecord,
    Mode,
    Position,
    RecordKind,
    TopicMeta,
)

YEAR_SECONDS = 365.0 * 86400.0
MIN_RECORDED_PRICE = 0.01
PriceEvent = tuple[int, int, float]
"""``(ts, option_id, price)``."""


@dataclass(frozen=True, slots=True)
class BacktestTopic:
    meta: TopicMeta
    resolved_option_id: int | None = None


@dataclass(slots=True)
class BacktestData:
    topics: dict[int, BacktestTopic] = field(default_factory=dict)
    price_events: dict[int, list[PriceEvent]] = field(default_factory=dict)
    candles: dict[AssetType, list[Candle]] = field(default_factory=dict)
    spot: dict[AssetType, list[tuple[int, float]]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    starting_millisats: Millisats
    ending_equity_millisats: Millisats
    pnl_millisats: Millisats
    roi: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    entries: int
    exits: int
    settlements: int
    commission_millisats: Millisats
    equity_curve: tuple[tuple[int, int], ...]
    records: tuple[LedgerRecord, ...]


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------


def save_dataset(
    data: BacktestData, directory: Path | str, include_prices: bool = True
) -> dict[str, Path]:
    """Write a dataset in the on-disk format :func:`load_dataset` reads.

    ``include_prices=False`` leaves ``prices.ndjson`` alone -- used by the
    ``record`` command, where the SDK's ``MarketRecorder`` has already written it.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "prices": root / "prices.ndjson",
        "meta": root / "meta.json",
        "candles": root / "candles.ndjson",
        "spot": root / "spot.ndjson",
    }
    if include_prices:
        with paths["prices"].open("w", encoding="utf-8") as handle:
            rows = [
                (ts, topic_id, option_id, price)
                for topic_id, events in data.price_events.items()
                for ts, option_id, price in events
            ]
            for ts, topic_id, option_id, price in sorted(rows):
                handle.write(
                    json.dumps(
                        {"ts": ts, "topic_id": topic_id, "option_id": option_id, "price": price}
                    )
                    + "\n"
                )
    meta = {
        "topics": [
            {
                "topic_id": topic.meta.topic_id,
                "batch_id": topic.meta.batch_id,
                "title": topic.meta.title,
                "asset": topic.meta.asset.value,
                "end_time_utc": topic.meta.end_time_utc,
                "resolved_option_id": topic.resolved_option_id,
                "buckets": [
                    {"option_id": b.option_id, "name": b.name, "lo": b.lo, "hi": b.hi}
                    for b in topic.meta.buckets
                ],
            }
            for topic in data.topics.values()
        ]
    }
    paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    with paths["candles"].open("w", encoding="utf-8") as handle:
        for asset, bars in data.candles.items():
            for bar in bars:
                handle.write(
                    json.dumps(
                        {
                            "asset": asset.value,
                            "ts": bar.ts,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume,
                        }
                    )
                    + "\n"
                )
    with paths["spot"].open("w", encoding="utf-8") as handle:
        for asset, points in data.spot.items():
            for ts, price in points:
                handle.write(json.dumps({"asset": asset.value, "ts": ts, "price": price}) + "\n")
    return paths


def _read_ndjson(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_dataset(
    prices_path: Path | str,
    meta_path: Path | str,
    candles_path: Path | str,
    spot_path: Path | str | None = None,
) -> BacktestData:
    """Load a recording. Prices go through the SDK's ``MarketRecorder.load``."""
    data = BacktestData()
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    for raw in meta["topics"]:
        buckets = tuple(
            Bucket(int(b["option_id"]), str(b["name"]), float(b["lo"]), float(b["hi"]))
            for b in raw["buckets"]
        )
        topic = TopicMeta(
            topic_id=int(raw["topic_id"]),
            batch_id=str(raw.get("batch_id") or ""),
            title=str(raw.get("title") or ""),
            asset=AssetType(raw["asset"]),
            end_time_utc=int(raw["end_time_utc"]),
            buckets=buckets,
        )
        resolved = raw.get("resolved_option_id")
        data.topics[topic.topic_id] = BacktestTopic(
            topic, None if resolved is None else int(resolved)
        )

    recorder = MarketRecorder(max_points_per_series=10_000_000)
    recorder.load(prices_path)
    for topic_id, loaded in data.topics.items():
        events: list[PriceEvent] = []
        for bucket in loaded.meta.buckets:
            events.extend(
                (int(ts), bucket.option_id, float(price))
                for ts, price in recorder.series_for(topic_id, bucket.option_id)
            )
        events.sort()
        data.price_events[topic_id] = events

    for row in _read_ndjson(Path(candles_path)):
        data.candles.setdefault(AssetType(str(row["asset"])), []).append(
            Candle(
                ts=int(str(row["ts"])),
                open=float(str(row["open"])),
                high=float(str(row["high"])),
                low=float(str(row["low"])),
                close=float(str(row["close"])),
                volume=float(str(row["volume"])),
            )
        )
    for bars in data.candles.values():
        bars.sort(key=lambda bar: bar.ts)
    if spot_path is not None and Path(spot_path).exists():
        for row in _read_ndjson(Path(spot_path)):
            data.spot.setdefault(AssetType(str(row["asset"])), []).append(
                (int(str(row["ts"])), float(str(row["price"])))
            )
        for points in data.spot.values():
            points.sort()
    return data


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def performance_metrics(
    equity_curve: Sequence[tuple[int, int]], step_seconds: int
) -> tuple[float, float, float]:
    """``(sharpe, sortino, max_drawdown)`` of an equity curve.

    Sharpe and Sortino are computed on per-step returns and annualised with
    ``sqrt(steps per year)`` (zero risk-free rate).
    """
    equity = np.array([value for _, value in equity_curve], dtype=np.float64)
    if equity.size == 0:
        return 0.0, 0.0, 0.0
    peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.max((peak - equity) / peak))
    if equity.size < 3:
        # Too few returns for a meaningful Sharpe/Sortino, but a drawdown is
        # already real after two points and must not be reported as zero.
        return 0.0, 0.0, max_drawdown
    returns = np.diff(equity) / equity[:-1]
    annualise = math.sqrt(YEAR_SECONDS / step_seconds)
    spread = float(returns.std())
    sharpe = float(returns.mean()) / spread * annualise if spread > 0.0 else 0.0
    downside = returns[returns < 0.0]
    downside_spread = math.sqrt(float((downside**2).mean())) if downside.size else 0.0
    sortino = float(returns.mean()) / downside_spread * annualise if downside_spread > 0.0 else 0.0
    return sharpe, sortino, max_drawdown


class Backtester:
    def __init__(
        self,
        settings: Settings,
        data: BacktestData,
        step_seconds: int = 300,
        refit_seconds: int = 3600,
    ) -> None:
        self._settings = settings
        self._data = data
        self._step = step_seconds
        self._refit = refit_seconds
        self._now = 0.0
        self._mode: Mode = "dry_run"

    def _clock(self) -> float:
        return self._now

    def _spot_at(self, asset: AssetType, ts: int) -> float | None:
        points = self._data.spot.get(asset)
        if points:
            index = bisect_right(points, (ts, math.inf)) - 1
            if index >= 0:
                return points[index][1]
        interval = self._settings.candle_interval_seconds
        closed = [bar for bar in self._data.candles.get(asset, []) if bar.ts + interval <= ts]
        return closed[-1].close if closed else None

    def _closed_candles(self, asset: AssetType, ts: int) -> list[Candle]:
        interval = self._settings.candle_interval_seconds
        bars = [bar for bar in self._data.candles.get(asset, []) if bar.ts + interval <= ts]
        return bars[-self._settings.candle_history :]

    def _winner(self, topic: BacktestTopic) -> int | None:
        if topic.resolved_option_id is not None:
            return topic.resolved_option_id
        final = self._spot_at(topic.meta.asset, topic.meta.end_time_utc)
        if final is None:
            return None
        for bucket in topic.meta.buckets:
            if bucket.lo < final <= bucket.hi:
                return bucket.option_id
        return None

    async def run(self) -> BacktestResult:
        settings = self._settings
        data = self._data
        ledger = Ledger(":memory:", clock=self._clock)
        risk = RiskManager(settings, ledger, self._mode, clock=self._clock)
        books: dict[int, LmsrBook] = {}
        local = LocalEstimator(
            books.get, settings.fee_rate, Millisats(settings.min_buy_commission_millisats)
        )
        optimizer = SlippageOptimizer(settings, local)
        starting = settings.bankroll_millisats

        opens = {tid: events[0][0] for tid, events in data.price_events.items() if events}
        if not opens:
            raise ValueError("the recording contains no prices")
        start = min(opens.values())
        end = max(topic.meta.end_time_utc for topic in data.topics.values())
        cursors = dict.fromkeys(opens, 0)
        latest: dict[int, dict[int, float]] = {tid: {} for tid in opens}
        slow_fits: dict[AssetType, SlowFit] = {}
        equity_curve: list[tuple[int, int]] = []
        calibrator = (
            OnlineCalibrator(
                prior=calibration_prior(settings),
                eta=settings.calibration_eta,
                share=settings.calibration_share,
                min_resolved=settings.calibration_min_resolved,
            )
            if settings.enable_calibration
            else None
        )

        for ts in range(start, end + self._step, self._step):
            self._now = float(ts)
            positions = ledger.positions(self._mode)

            # 1. Binary settlement of every position whose market has ended.
            for (topic_id, option_id), position in positions.items():
                topic = data.topics[topic_id]
                if ts < topic.meta.end_time_utc:
                    continue
                winner = self._winner(topic)
                ledger.record_settlement(
                    self._mode,
                    position,
                    won=winner is not None and option_id == winner,
                    fee_rate=settings.fee_rate,
                    note=f"resolved to option {winner}",
                )
            positions = ledger.positions(self._mode)
            if calibrator is not None:
                for topic_id in [
                    tid for tid, pending in calibrator.pending.items() if pending.end_time_utc <= ts
                ]:
                    winner = self._winner(data.topics[topic_id])
                    if winner is None:
                        calibrator.pending.pop(topic_id, None)
                    else:
                        calibrator.resolve(topic_id, winner)

            # 2. Rebuild each active book from the recording plus our own fills.
            books.clear()
            for topic_id, events in data.price_events.items():
                cursor = cursors[topic_id]
                while cursor < len(events) and events[cursor][0] <= ts:
                    latest[topic_id][events[cursor][1]] = events[cursor][2]
                    cursor += 1
                cursors[topic_id] = cursor
                meta = data.topics[topic_id].meta
                if not latest[topic_id] or ts >= meta.end_time_utc:
                    continue
                option_ids = [bucket.option_id for bucket in meta.buckets]
                prices = [latest[topic_id].get(option_id, 0.0) for option_id in option_ids]
                book = book_from_prices(
                    topic_id, option_ids, prices, settings.backtest_liquidity_b, ts=float(ts)
                )
                own = [
                    (option_id, position.contracts)
                    for (held_topic, option_id), position in positions.items()
                    if held_topic == topic_id
                ]
                books[topic_id] = book.applied(own) if own else book

            # 3. Walk-forward model state: only candles closed before `ts`.
            for asset in {data.topics[topic_id].meta.asset for topic_id in books}:
                current = slow_fits.get(asset)
                if current is None or ts - current.fitted_at >= self._refit:
                    candles = self._closed_candles(asset, ts)
                    if len(candles) >= 30:
                        slow_fits[asset] = fit_slow(asset, candles, settings, float(ts))

            # 4. Exits first (free up bankroll), then entries.
            for topic_id in sorted(books, key=lambda tid: data.topics[tid].meta.end_time_utc):
                meta = data.topics[topic_id].meta
                slow = slow_fits.get(meta.asset)
                spot = self._spot_at(meta.asset, ts)
                if slow is None or spot is None:
                    continue
                fast = FastInputs(spot, None, None, None, ("recording",), float(ts))
                key = OnlineCalibrator.key_for(meta.asset.value, meta.batch_id)
                view = calibrator.view(key) if calibrator is not None else None
                try:
                    forecast = build_forecast(meta, slow, fast, float(ts), settings, view)
                except ForecastError:
                    continue
                if (
                    calibrator is not None
                    and meta.seconds_to_expiry(float(ts)) >= settings.calibration_min_lead_seconds
                ):
                    calibrator.record(
                        topic_id,
                        PendingForecast(
                            key=key,
                            components=forecast.components,
                            pooled=forecast.probabilities,
                            option_ids=tuple(bucket.option_id for bucket in meta.buckets),
                            end_time_utc=meta.end_time_utc,
                            batch_id=meta.batch_id,
                        ),
                    )
                book = books[topic_id]

                for (held_topic, option_id), position in ledger.positions(self._mode).items():
                    if held_topic != topic_id:
                        continue
                    quote = local.sell(book, [(option_id, position.contracts)])
                    net = Millisats(int(quote.amount_millisats) - int(quote.commission_millisats))
                    probability = forecast.probability(option_id)
                    decision = evaluate_exit(
                        position, probability, book.price(option_id), net, settings
                    )
                    if decision.should_exit:
                        ledger.record_exit(
                            self._mode,
                            position,
                            position.contracts,
                            quote.amount_millisats,
                            quote.commission_millisats,
                            market_price=book.price(option_id),
                            model_probability=probability,
                            note=f"{decision.reason}: {decision.detail}",
                        )
                        book = book.applied([(option_id, position.contracts)], sign=-1.0)
                        books[topic_id] = book

                remaining = meta.seconds_to_expiry(float(ts))
                if not (
                    settings.min_seconds_to_expiry
                    <= remaining
                    <= settings.max_horizon_hours * 3600.0
                ):
                    continue
                if not risk.entries_allowed:
                    continue
                candidates = [
                    signal
                    for signal in build_signals(meta, book, forecast)
                    if float(signal.edge_points) >= settings.min_raw_edge_points
                ][: settings.max_candidates_per_topic]
                for signal in candidates:
                    held = ledger.positions(self._mode)
                    cash = ledger.cash_millisats(self._mode, starting)
                    equity = Millisats(int(cash) + self._liquidation_value(held, books, local))
                    budget = risk.entry_budget(topic_id, signal.option_id, equity, cash, held)
                    if int(budget) <= 0:
                        continue
                    result = await optimizer.optimize(
                        signal,
                        book=book,
                        bankroll=equity,
                        max_cost=budget,
                        kelly_multiplier=forecast.kelly_multiplier,
                        ambiguity=forecast.ambiguity,
                    )
                    plan = result.plan
                    if plan is None:
                        continue
                    if not risk.check_entry(plan, meta, equity, cash, held, float(ts)).approved:
                        continue
                    risk.note_order(plan.topic_id, plan.option_id, float(ts))
                    ledger.record_entry(
                        self._mode,
                        plan,
                        plan.cost_millisats,
                        plan.commission_millisats,
                        note="backtest fill on the local LS-LMSR replica",
                    )
                    book = book.applied([(plan.option_id, plan.contracts)])
                    books[topic_id] = book
                    break

            # 5. Mark to liquidation value and feed the drawdown breaker.
            held = ledger.positions(self._mode)
            cash = ledger.cash_millisats(self._mode, starting)
            equity_value = int(cash) + self._liquidation_value(held, books, local)
            risk.update_equity(Millisats(equity_value), float(ts))
            equity_curve.append((ts, equity_value))

        summary = ledger.summary(self._mode, starting)
        ending = equity_curve[-1][1]
        sharpe, sortino, max_drawdown = performance_metrics(equity_curve, self._step)
        records = tuple(
            ledger.records(
                self._mode, kinds=[RecordKind.ENTRY, RecordKind.EXIT, RecordKind.SETTLEMENT]
            )
        )
        ledger.close()
        return BacktestResult(
            starting_millisats=starting,
            ending_equity_millisats=Millisats(ending),
            pnl_millisats=Millisats(ending - int(starting)),
            roi=(ending - int(starting)) / int(starting),
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_drawdown,
            win_rate=summary.win_rate,
            entries=summary.entries,
            exits=summary.exits,
            settlements=summary.settlements,
            commission_millisats=summary.commission_millisats,
            equity_curve=tuple(equity_curve),
            records=records,
        )

    @staticmethod
    def _liquidation_value(
        positions: dict[PositionKey, Position], books: dict[int, LmsrBook], local: LocalEstimator
    ) -> int:
        total = 0
        for (topic_id, option_id), position in positions.items():
            book = books.get(topic_id)
            if book is None or option_id not in book:
                total += int(position.cost_basis_millisats)
                continue
            quote = local.sell(book, [(option_id, position.contracts)])
            total += int(quote.amount_millisats) - int(quote.commission_millisats)
        return total


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _lognormal_bucket_probabilities(
    buckets: Sequence[Bucket], centre: float, sigma: float
) -> np.ndarray:
    from scipy.special import ndtr

    lows = np.array([bucket.lo for bucket in buckets])
    highs = np.array([bucket.hi for bucket in buckets])
    scale = max(sigma, 1.0e-6)

    def cdf(prices: np.ndarray) -> np.ndarray:
        return np.asarray(ndtr((np.log(prices / centre) + 0.5 * scale**2) / scale))

    mass = np.maximum(cdf(highs) - cdf(lows), 0.0)
    return np.asarray(mass / mass.sum())


def synthetic_dataset(
    seed: int = 11,
    n_topics: int = 12,
    mispriced: bool = True,
    step_seconds: int = 300,
    hourly_vol: float = 0.005,
    history_hours: int = 400,
    topic_life_hours: int = 3,
    stale_lag_seconds: int = 2700,
    width_inflation: float = 1.5,
    start_ts: int = 1_800_000_000,
) -> BacktestData:
    """A self-contained BTC-like world with a known data-generating process.

    The underlying follows a driftless geometric Brownian motion. Each hourly
    topic quotes 100 buckets. With ``mispriced=True`` the market's distribution
    is *stale* (centred on the spot ``stale_lag_seconds`` ago) and *too wide*
    (volatility inflated by ``width_inflation``), and converges to the correct
    distribution as expiry approaches -- the pattern a calibrated model can
    harvest. With ``mispriced=False`` the market is always correctly priced
    and there is nothing to harvest; this is the control case.
    """
    rng = np.random.default_rng(seed)
    per_hour = 3600 // step_seconds
    test_hours = n_topics + topic_life_hours
    n_steps = (history_hours + test_hours) * per_hour
    step_vol = hourly_vol / math.sqrt(per_hour)
    log_path = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, step_vol, n_steps))])
    prices = 60_000.0 * np.exp(log_path - log_path[history_hours * per_hour])
    origin = start_ts - history_hours * 3600

    def price_at(ts: int) -> float:
        index = min(max((ts - origin) // step_seconds, 0), n_steps)
        return float(prices[index])

    data = BacktestData()
    candles: list[Candle] = []
    for hour in range(history_hours + test_hours):
        window = prices[hour * per_hour : (hour + 1) * per_hour + 1]
        candles.append(
            Candle(
                ts=origin + hour * 3600,
                open=float(window[0]),
                high=float(np.max(window)),
                low=float(np.min(window)),
                close=float(window[-1]),
                volume=1.0,
            )
        )
    data.candles[AssetType.BTC] = candles
    data.spot[AssetType.BTC] = [
        (origin + i * step_seconds, float(prices[i]))
        for i in range(history_hours * per_hour, n_steps + 1)
    ]

    buckets = tuple(
        Bucket(
            i + 1, f"{50_000 + 200 * i}-{50_200 + 200 * i}", 50_000.0 + 200 * i, 50_200.0 + 200 * i
        )
        for i in range(100)
    )
    for index in range(n_topics):
        topic_id = 1000 + index
        open_ts = start_ts + index * 3600
        end_ts = open_ts + topic_life_hours * 3600
        meta = TopicMeta(
            topic_id, "synthetic", f"Synthetic BTC {index}", AssetType.BTC, end_ts, buckets
        )
        final = price_at(end_ts)
        winner = next((b.option_id for b in buckets if b.lo < final <= b.hi), None)
        data.topics[topic_id] = BacktestTopic(meta, winner)
        events: list[PriceEvent] = []
        for ts in range(open_ts, end_ts, step_seconds):
            hours_left = (end_ts - ts) / 3600.0
            true_sigma = hourly_vol * math.sqrt(hours_left)
            true = _lognormal_bucket_probabilities(buckets, price_at(ts), true_sigma)
            if mispriced:
                progress = (ts - open_ts) / (end_ts - open_ts)
                blend = progress**2
                stale = _lognormal_bucket_probabilities(
                    buckets, price_at(ts - stale_lag_seconds), true_sigma * width_inflation
                )
                quoted = (1.0 - blend) * stale + blend * true
            else:
                quoted = true
            events.extend(
                (ts, bucket.option_id, round(100.0 * float(quoted[i]), 4))
                for i, bucket in enumerate(buckets)
                if 100.0 * quoted[i] >= MIN_RECORDED_PRICE
            )
        data.price_events[topic_id] = events
    return data
