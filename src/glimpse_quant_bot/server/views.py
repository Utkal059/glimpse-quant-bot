"""Pure translations from engine objects to API models.

Nothing in this module talks to a venue. Sizing previews run the engine's own
:class:`~glimpse_quant_bot.lmsr_optimizer.SlippageOptimizer` with a
:class:`~glimpse_quant_bot.lmsr_optimizer.LocalEstimator` as its pricing
source: the replica is exact, so the preview is the size the engine would pick
(before its server confirmation) at zero rate-limit cost.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from glimpse_markets.money import Millisats, PriceUnits

from glimpse_quant_bot.backtester import performance_metrics
from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.forecaster import EnsembleForecast, build_signals
from glimpse_quant_bot.ledger import LedgerSummary, PositionKey
from glimpse_quant_bot.lmsr import LmsrBook
from glimpse_quant_bot.lmsr_optimizer import (
    LocalEstimator,
    SlippageOptimizer,
    evaluate_exit,
    ladder_point,
    settlement_value,
)
from glimpse_quant_bot.server.schemas import (
    BucketColumns,
    DistributionResponse,
    EquityPoint,
    ExitRule,
    ForecastMeta,
    LadderPointView,
    LadderResponse,
    LadderRow,
    LedgerRecordView,
    LedgerSummaryView,
    MarketSummary,
    PerformanceMetrics,
    PositionView,
    PositiveEvBucket,
)
from glimpse_quant_bot.types import (
    BucketSignal,
    LadderPoint,
    LedgerRecord,
    Position,
    TopicMeta,
)

FloatArray = npt.NDArray[np.float64]
MAX_POSITIVE_EV_ROWS = 24
_KL_FLOOR = 1.0e-12


def local_estimator(book: LmsrBook, settings: Settings) -> LocalEstimator:
    return LocalEstimator(
        lambda topic_id: book if topic_id == book.topic_id else None,
        settings.fee_rate,
        Millisats(settings.min_buy_commission_millisats),
    )


def model_array(topic: TopicMeta, forecast: EnsembleForecast | None) -> FloatArray | None:
    if forecast is None:
        return None
    return np.array([forecast.probability(b.option_id) for b in topic.buckets], dtype=np.float64)


def aligned_market(topic: TopicMeta, book: LmsrBook) -> tuple[FloatArray, FloatArray]:
    """Market probabilities and marginal prices in ``topic.buckets`` order."""
    probabilities = book.probabilities()
    prices = book.prices()
    index = [book.index_of(bucket.option_id) for bucket in topic.buckets]
    return (
        np.asarray(probabilities[index], dtype=np.float64),
        np.asarray(prices[index], dtype=np.float64),
    )


def kl_divergence(model: FloatArray, market: FloatArray) -> float:
    """``KL(p_model || p_market)`` in nats."""
    p = np.clip(model, 0.0, None)
    q = np.clip(market, _KL_FLOOR, None)
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    p = p / total
    q = q / float(q.sum())
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def extractable_edge_millisats(liquidity_b: float, kl: float) -> Millisats:
    """``100 * b * KL`` sats (ARCHITECTURE.md §3.3), as integer millisats."""
    return Millisats(round(100.0 * liquidity_b * kl * 1000.0))


def net_edge(probability: FloatArray, price: FloatArray, fee_rate: float) -> FloatArray:
    """Marginal contract after the commission and the settlement fee."""
    return np.asarray(
        100.0 * probability * (1.0 - fee_rate) - price * (1.0 + fee_rate), dtype=np.float64
    )


def _finite(values: FloatArray, digits: int = 6) -> list[float]:
    return [round(float(v), digits) if math.isfinite(float(v)) else 0.0 for v in values]


def ladder_point_view(point: LadderPoint) -> LadderPointView:
    return LadderPointView(
        contracts=point.contracts,
        cost_millisats=int(point.cost_millisats),
        commission_millisats=int(point.commission_millisats),
        exec_price=float(point.exec_price),
        net_edge_points=float(point.net_edge_points),
        expected_profit_millisats=int(point.expected_profit_millisats),
        source=point.source,
    )


def forecast_meta(forecast: EnsembleForecast) -> ForecastMeta:
    return ForecastMeta(
        ts=forecast.ts,
        horizon_seconds=forecast.horizon_seconds,
        spot=forecast.spot,
        sigma=forecast.sigma,
        js_divergence=forecast.disagreement,
        kelly_multiplier_kappa=forecast.kelly_multiplier,
        ambiguity=forecast.ambiguity,
        weights=dict(forecast.weights),
        spot_sources=list(forecast.spot_sources),
        diagnostics={k: float(v) for k, v in forecast.diagnostics.items() if math.isfinite(v)},
    )


def market_summary(
    topic: TopicMeta,
    book: LmsrBook,
    forecast: EnsembleForecast | None,
    held_contracts: float,
    spot: float | None,
    settings: Settings,
    now: float,
) -> MarketSummary:
    market_p, prices = aligned_market(topic, book)
    model = model_array(topic, forecast)
    mode_index = int(np.argmax(market_p))
    kl: float | None = None
    best_bucket: str | None = None
    best_edge: float | None = None
    positive = 0
    if model is not None:
        kl = kl_divergence(model, market_p)
        edges = 100.0 * model - prices
        best = int(np.argmax(edges))
        best_bucket = topic.buckets[best].name
        best_edge = float(edges[best])
        nets = net_edge(model, prices, settings.fee_rate)
        positive = int(
            np.sum((edges >= settings.min_raw_edge_points) & (nets >= settings.min_net_edge_points))
        )
    return MarketSummary(
        topic_id=topic.topic_id,
        batch_id=topic.batch_id,
        title=topic.title,
        asset=topic.asset,
        end_time_utc=topic.end_time_utc,
        seconds_to_expiry=topic.seconds_to_expiry(now),
        n_buckets=len(topic.buckets),
        liquidity_b=book.liquidity,
        alpha=book.alpha,
        book_ts=book.ts,
        spot_price=forecast.spot if forecast is not None else spot,
        market_mode_bucket=topic.buckets[mode_index].name,
        forecast_available=forecast is not None,
        forecast_ts=forecast.ts if forecast is not None else None,
        js_divergence=forecast.disagreement if forecast is not None else None,
        kelly_multiplier_kappa=forecast.kelly_multiplier if forecast is not None else None,
        kl_divergence=kl,
        max_extractable_edge_millisats=(
            int(extractable_edge_millisats(book.liquidity, kl)) if kl is not None else None
        ),
        best_bucket=best_bucket,
        best_edge_points=best_edge,
        positive_ev_buckets=positive,
        held_contracts=held_contracts,
    )


async def engine_choice(
    signal: BucketSignal,
    book: LmsrBook,
    forecast: EnsembleForecast,
    settings: Settings,
    equity: Millisats,
    budget: Millisats,
) -> PositiveEvBucket:
    """What :meth:`SlippageOptimizer.optimize` would do for this bucket right now."""
    optimizer = SlippageOptimizer(settings, local_estimator(book, settings))
    result = await optimizer.optimize(
        signal,
        book=book,
        bankroll=equity,
        max_cost=budget,
        kelly_multiplier=forecast.kelly_multiplier,
        ambiguity=forecast.ambiguity,
    )
    market = float(signal.market_price)
    probability = signal.model_probability
    fair = 100.0 * probability * (1.0 - settings.fee_rate)
    marginal_net = fair - market * (1.0 + settings.fee_rate)
    plan = result.plan
    ladder = result.ladder or result.local_ladder
    slippage: int | None = None
    if plan is not None:
        slippage = int(plan.all_in_millisats) - round(plan.contracts * market * 1000.0)
    return PositiveEvBucket(
        option_id=signal.option_id,
        name=signal.name,
        lo=signal.lo,
        hi=signal.hi,
        market_price=market,
        model_probability=min(max(probability, 0.0), 1.0),
        edge_points=float(signal.edge_points),
        net_edge_points=marginal_net,
        recommended_contracts=plan.contracts if plan is not None else None,
        kelly_fraction=plan.kelly_fraction if plan is not None else None,
        budget_millisats=int(plan.budget_millisats) if plan is not None else int(budget),
        expected_profit_millisats=(
            int(plan.expected_profit_millisats) if plan is not None else None
        ),
        slippage_cost_millisats=slippage,
        rejection=result.rejection,
        ladder=[ladder_point_view(point) for point in ladder],
    )


async def distribution_view(
    topic: TopicMeta,
    book: LmsrBook,
    forecast: EnsembleForecast | None,
    settings: Settings,
    held: Mapping[int, float],
    spot: float | None,
    equity: Millisats,
    budget_for: Callable[[int], Millisats],
    now: float,
) -> DistributionResponse:
    market_p, prices = aligned_market(topic, book)
    model = model_array(topic, forecast)
    components: dict[str, list[float]] = {}
    positive: list[PositiveEvBucket] = []
    kl: float | None = None
    edges: FloatArray | None = None
    nets: FloatArray | None = None
    if forecast is not None and model is not None:
        kl = kl_divergence(model, market_p)
        edges = np.asarray(100.0 * model - prices, dtype=np.float64)
        nets = net_edge(model, prices, settings.fee_rate)
        for name, values in forecast.components.items():
            components[name] = _finite(
                np.array([values.get(b.option_id, 0.0) for b in topic.buckets]), 8
            )
        signals = {
            signal.option_id: signal
            for signal in build_signals(topic, book, forecast)
            if float(signal.edge_points) >= settings.min_raw_edge_points
        }
        ranked = sorted(
            (
                (float(nets[i]), bucket.option_id)
                for i, bucket in enumerate(topic.buckets)
                if bucket.option_id in signals and nets[i] >= settings.min_net_edge_points
            ),
            reverse=True,
        )[:MAX_POSITIVE_EV_ROWS]
        for _, option_id in ranked:
            positive.append(
                await engine_choice(
                    signals[option_id],
                    book,
                    forecast,
                    settings,
                    equity,
                    budget_for(option_id),
                )
            )
    columns = BucketColumns(
        option_id=[b.option_id for b in topic.buckets],
        name=[b.name for b in topic.buckets],
        lo=[b.lo for b in topic.buckets],
        hi=[b.hi for b in topic.buckets],
        market_price=_finite(prices, 4),
        market_probability=_finite(market_p, 8),
        model_probability=_finite(model, 8) if model is not None else None,
        edge_points=_finite(edges, 4) if edges is not None else None,
        net_edge_points=_finite(nets, 4) if nets is not None else None,
        held_contracts=[float(held.get(b.option_id, 0.0)) for b in topic.buckets],
    )
    return DistributionResponse(
        topic_id=topic.topic_id,
        title=topic.title,
        asset=topic.asset,
        end_time_utc=topic.end_time_utc,
        seconds_to_expiry=topic.seconds_to_expiry(now),
        liquidity_b=book.liquidity,
        alpha=book.alpha,
        book_ts=book.ts,
        fee_rate=settings.fee_rate,
        min_raw_edge_points=settings.min_raw_edge_points,
        min_net_edge_points=settings.min_net_edge_points,
        min_expected_gain_sats=settings.min_expected_gain_sats,
        spot_price=forecast.spot if forecast is not None else spot,
        forecast=forecast_meta(forecast) if forecast is not None else None,
        js_divergence=forecast.disagreement if forecast is not None else None,
        kelly_multiplier_kappa=forecast.kelly_multiplier if forecast is not None else None,
        kl_divergence=kl,
        max_extractable_edge_millisats=(
            int(extractable_edge_millisats(book.liquidity, kl)) if kl is not None else None
        ),
        buckets=columns,
        component_densities=components,
        positive_ev=positive,
    )


async def ladder_view(
    topic: TopicMeta,
    book: LmsrBook,
    option_id: int,
    forecast: EnsembleForecast | None,
    settings: Settings,
    equity: Millisats,
    budget: Millisats,
    max_contracts: int,
) -> LadderResponse:
    """``C(q + k e_i) - C(q)`` for ``k = 1..max_contracts`` on the local replica."""
    bucket = topic.bucket(option_id)
    if bucket is None or option_id not in book:
        raise KeyError(option_id)
    estimator = local_estimator(book, settings)
    market = book.price(option_id)
    probability = forecast.probability(option_id) if forecast is not None else None
    rows: list[LadderRow] = []
    previous_profit = 0
    breakeven: float | None = None
    best_profit: int | None = None
    best_size: float | None = None
    for k in range(1, max_contracts + 1):
        size = float(k)
        quote = estimator.buy(book, [(option_id, size)])
        point = ladder_point(
            size, quote, probability if probability is not None else 0.0, settings.fee_rate
        )
        profit = int(point.expected_profit_millisats) if probability is not None else 0
        marginal = profit - previous_profit
        previous_profit = profit
        if probability is not None:
            if breakeven is None and marginal < 0:
                breakeven = size - 1.0
            if best_profit is None or profit > best_profit:
                best_profit, best_size = profit, size
        rows.append(
            LadderRow(
                contracts=size,
                cost_millisats=int(quote.amount_millisats),
                commission_millisats=int(quote.commission_millisats),
                exec_price=float(point.exec_price),
                marginal_price=float(book.price_after_buy(option_id, size)),
                net_edge_points=float(point.net_edge_points) if probability is not None else 0.0,
                expected_profit_millisats=profit,
                marginal_profit_millisats=marginal,
                within_budget=int(point.all_in_millisats) <= int(budget),
            )
        )
    choice: PositiveEvBucket | None = None
    if forecast is not None and probability is not None:
        signal = BucketSignal(
            topic_id=topic.topic_id,
            option_id=option_id,
            name=bucket.name,
            lo=bucket.lo,
            hi=bucket.hi,
            market_price=market,
            model_probability=probability,
            edge_points=PriceUnits(100.0 * probability - float(market)),
            components={n: v.get(option_id, 0.0) for n, v in forecast.components.items()},
        )
        choice = await engine_choice(signal, book, forecast, settings, equity, budget)
    return LadderResponse(
        topic_id=topic.topic_id,
        option_id=option_id,
        bucket_name=bucket.name,
        market_price=float(market),
        model_probability=probability,
        fair_value_points=(
            float(settlement_value(probability, settings.fee_rate))
            if probability is not None
            else None
        ),
        budget_millisats=int(budget),
        max_contracts=max_contracts,
        optimal_contracts=best_size if best_profit is not None and best_profit > 0 else None,
        marginal_breakeven_contracts=breakeven,
        rows=rows,
        engine_choice=choice,
    )


def position_views(
    positions: Mapping[PositionKey, Position],
    topics: Mapping[int, TopicMeta],
    books: Mapping[int, LmsrBook],
    forecasts: Mapping[int, EnsembleForecast],
    settings: Settings,
    now: float,
) -> list[PositionView]:
    views: list[PositionView] = []
    for (topic_id, option_id), position in sorted(positions.items()):
        topic = topics.get(topic_id)
        book = books.get(topic_id)
        forecast = forecasts.get(topic_id)
        mark: float | None = None
        liquidation: int | None = None
        if book is not None and option_id in book:
            mark = float(book.price(option_id))
            try:
                estimator = local_estimator(book, settings)
                quote = estimator.sell(book, [(option_id, position.contracts)])
                liquidation = int(quote.amount_millisats) - int(quote.commission_millisats)
            except ValueError:
                liquidation = None
        probability = forecast.probability(option_id) if forecast is not None else None
        rule: ExitRule | None = None
        detail: str | None = None
        if probability is not None and mark is not None and liquidation is not None:
            decision = evaluate_exit(
                position, probability, PriceUnits(mark), Millisats(liquidation), settings
            )
            rule, detail = decision.reason, decision.detail
        entry = float(position.entry_price)
        views.append(
            PositionView(
                topic_id=topic_id,
                option_id=option_id,
                bucket_name=position.bucket_name,
                asset=topic.asset if topic is not None else None,
                title=topic.title if topic is not None else None,
                contracts=position.contracts,
                cost_basis_millisats=int(position.cost_basis_millisats),
                entry_price=entry,
                entry_model_probability=min(max(position.entry_model_probability, 0.0), 1.0),
                mark_price=mark,
                liquidation_value_millisats=liquidation,
                unrealized_pnl_millisats=(
                    liquidation - int(position.cost_basis_millisats)
                    if liquidation is not None
                    else None
                ),
                model_probability=probability,
                take_profit_trigger_price=(
                    max(0.0, 100.0 * probability - settings.take_profit_gap_points)
                    if probability is not None
                    else None
                ),
                invalidation_model_points=max(0.0, entry - settings.invalidation_drop_points),
                exit_rule=rule,
                exit_detail=detail,
                seconds_to_expiry=topic.seconds_to_expiry(now) if topic is not None else None,
            )
        )
    return views


def record_view(record: LedgerRecord) -> LedgerRecordView:
    return LedgerRecordView(
        record_id=record.record_id,
        ts=record.ts,
        mode=record.mode,
        kind=record.kind.value,
        topic_id=record.topic_id,
        option_id=record.option_id,
        bucket_name=record.bucket_name,
        contracts=record.contracts,
        cash_flow_millisats=int(record.cash_flow_millisats),
        commission_millisats=int(record.commission_millisats),
        exec_price=float(record.exec_price) if record.exec_price is not None else None,
        model_probability=record.model_probability,
        market_price=float(record.market_price) if record.market_price is not None else None,
        realized_pnl_millisats=int(record.realized_pnl_millisats),
        trade_id=record.trade_id,
        note=record.note,
    )


def summary_view(summary: LedgerSummary) -> LedgerSummaryView:
    return LedgerSummaryView(
        mode=summary.mode,
        starting_millisats=int(summary.starting_millisats),
        cash_millisats=int(summary.cash_millisats),
        open_cost_basis_millisats=int(summary.open_cost_basis_millisats),
        realized_pnl_millisats=int(summary.realized_pnl_millisats),
        commission_millisats=int(summary.commission_millisats),
        entries=summary.entries,
        exits=summary.exits,
        settlements=summary.settlements,
        wins=summary.wins,
        losses=summary.losses,
        open_positions=summary.open_positions,
        roi=summary.roi,
        win_rate=summary.win_rate,
    )


def equity_points(curve: Sequence[tuple[float, Millisats]]) -> list[EquityPoint]:
    return [EquityPoint(ts=ts, equity_millisats=int(value)) for ts, value in curve]


def performance(
    curve: Sequence[tuple[float, Millisats]], starting: Millisats | None = None
) -> PerformanceMetrics | None:
    """Sharpe, Sortino and maximum drawdown of an equity curve (backtester maths)."""
    if not curve:
        return None
    timestamps = np.array([ts for ts, _ in curve], dtype=np.float64)
    steps = np.diff(timestamps)
    step = int(max(1.0, float(np.median(steps)))) if steps.size else 60
    as_ints = [(int(ts), int(value)) for ts, value in curve]
    sharpe, sortino, max_drawdown = performance_metrics(as_ints, step)
    first = int(starting) if starting is not None else as_ints[0][1]
    last = as_ints[-1][1]
    return PerformanceMetrics(
        points=len(curve),
        starting_millisats=first,
        ending_millisats=last,
        pnl_millisats=last - first,
        roi=(last - first) / first if first else 0.0,
        sharpe=sharpe if math.isfinite(sharpe) else 0.0,
        sortino=sortino if math.isfinite(sortino) else 0.0,
        max_drawdown=max_drawdown if math.isfinite(max_drawdown) else 0.0,
    )
