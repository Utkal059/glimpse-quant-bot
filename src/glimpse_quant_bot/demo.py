"""Example-trades generator: one pass of the full pipeline against live markets.

``glimpse-quant-bot demo-trades`` scans the live Glimpse universe, forecasts
every nearby market from live Deribit / Binance / Coinbase / Kraken data,
ranks the mispricings, prices a full slippage ladder for the best ones on the
server's ``/trades/estimate`` endpoint, runs them through the risk gatekeeper
and "executes" the survivors through the SDK in dry-run mode. Everything is
written to a Markdown report.

It is always a dry run. No API key is needed and no order is ever placed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import glimpse_markets
from glimpse_markets import AsyncClient, QuoteMode
from glimpse_markets.models import BatchMarketItem
from glimpse_markets.money import Millisats

from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.execution import ExecutionRouter
from glimpse_quant_bot.external_feeds import MarketDataHub
from glimpse_quant_bot.forecaster import (
    EnsembleForecast,
    EnsembleForecaster,
    ForecastError,
    build_signals,
)
from glimpse_quant_bot.ledger import Ledger
from glimpse_quant_bot.lmsr import LmsrBook
from glimpse_quant_bot.lmsr_optimizer import (
    ExitDecision,
    LocalEstimator,
    OptimizationResult,
    ServerEstimator,
    SlippageOptimizer,
    evaluate_exit,
    ladder_point,
)
from glimpse_quant_bot.risk_manager import RiskDecision, RiskManager
from glimpse_quant_bot.strategy import build_topic, discover_batches, load_batch_topics
from glimpse_quant_bot.types import (
    AssetType,
    BucketSignal,
    LadderPoint,
    LedgerRecord,
    Mode,
    TopicMeta,
)

MODE: Mode = "dry_run"


@dataclass(slots=True)
class DemoCandidate:
    topic: TopicMeta
    book: LmsrBook
    forecast: EnsembleForecast
    signal: BucketSignal
    local_best_profit: int
    result: OptimizationResult | None = None
    decision: RiskDecision | None = None
    record: LedgerRecord | None = None
    exit_decision: ExitDecision | None = None
    exit_net_millisats: int = 0


@dataclass(slots=True)
class DemoReport:
    generated_at: float
    settings: Settings
    asset_rows: list[tuple[AssetType, EnsembleForecast]] = field(default_factory=list)
    signals: list[tuple[TopicMeta, EnsembleForecast, BucketSignal]] = field(default_factory=list)
    candidates: list[DemoCandidate] = field(default_factory=list)
    topics_scanned: int = 0
    cash_millisats: int = 0
    timeseries_model: str = "gjr_garch"


def _sats(millisats: int) -> str:
    return f"{millisats / 1000:,.1f}"


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


async def run_demo(
    settings: Settings,
    client: AsyncClient,
    ledger: Ledger,
    max_trades: int = 5,
    topics_per_batch: int = 3,
) -> DemoReport:
    """Scan, rank, price, gate and paper-execute. Returns everything for the report."""
    now = time.time()
    hub = MarketDataHub(settings.assets, enable_websocket=False)
    forecaster = EnsembleForecaster(hub, settings)
    risk = RiskManager(settings, ledger, MODE)
    router = ExecutionRouter(client, ledger, risk, MODE)
    optimizer = SlippageOptimizer(settings, ServerEstimator(client))
    local = LocalEstimator(
        lambda _topic_id: None, settings.fee_rate, Millisats(settings.min_buy_commission_millisats)
    )
    report = DemoReport(generated_at=now, settings=settings)
    report.timeseries_model = forecaster.timeseries_model
    try:
        batches = await discover_batches(client, settings)
        seen_assets: set[AssetType] = set()
        pool: list[DemoCandidate] = []
        for batch_id, asset in batches.items():
            loaded = await load_batch_topics(client, batch_id, asset, settings, now)
            for topic, book in loaded[:topics_per_batch]:
                try:
                    forecast = await forecaster.forecast(topic, now)
                except ForecastError:
                    continue
                report.topics_scanned += 1
                if asset not in seen_assets:
                    seen_assets.add(asset)
                    report.asset_rows.append((asset, forecast))
                signals = build_signals(topic, book, forecast)
                report.signals.extend((topic, forecast, signal) for signal in signals[:2])
                for signal in signals[: settings.max_candidates_per_topic]:
                    if float(signal.edge_points) < settings.min_raw_edge_points:
                        continue
                    profits = [
                        int(
                            ladder_point(
                                size,
                                local.buy(book, [(signal.option_id, size)]),
                                signal.model_probability,
                                settings.fee_rate,
                            ).expected_profit_millisats
                        )
                        for size in optimizer.candidate_sizes(signal, book)
                    ]
                    pool.append(DemoCandidate(topic, book, forecast, signal, max(profits)))

        report.signals.sort(key=lambda row: float(row[2].edge_points), reverse=True)
        pool.sort(key=lambda candidate: candidate.local_best_profit, reverse=True)

        for candidate in pool[:max_trades]:
            # The scan may be a minute old by now and active markets move.
            # Re-read this one book and re-derive the signal before pricing.
            refreshed = await _refresh(client, candidate, now)
            if not refreshed:
                continue
            positions = ledger.positions(MODE)
            cash = ledger.cash_millisats(MODE, settings.bankroll_millisats)
            equity = Millisats(
                int(cash) + sum(int(p.cost_basis_millisats) for p in positions.values())
            )
            signal = candidate.signal
            budget = risk.entry_budget(signal.topic_id, signal.option_id, equity, cash, positions)
            candidate.result = await optimizer.optimize(
                signal,
                book=candidate.book,
                bankroll=equity,
                max_cost=budget,
                kelly_multiplier=candidate.forecast.kelly_multiplier,
                full_server_ladder=True,
            )
            plan = candidate.result.plan
            if plan is None:
                continue
            ledger.record_signal(MODE, signal, note=candidate.topic.title)
            ledger.record_plan(MODE, plan)
            candidate.decision = risk.check_entry(
                plan, candidate.topic, equity, cash, positions, now
            )
            if not candidate.decision.approved:
                continue
            candidate.record = await router.enter(plan)
            if candidate.record is None:
                continue
            position = ledger.positions(MODE)[(plan.topic_id, plan.option_id)]
            quote = local.sell(candidate.book, [(plan.option_id, position.contracts)])
            candidate.exit_net_millisats = int(quote.amount_millisats) - int(
                quote.commission_millisats
            )
            candidate.exit_decision = evaluate_exit(
                position,
                signal.model_probability,
                signal.market_price,
                Millisats(candidate.exit_net_millisats),
                settings,
            )
        report.candidates = pool[:max_trades]
        report.cash_millisats = int(ledger.cash_millisats(MODE, settings.bankroll_millisats))
    finally:
        await hub.close()
    return report


async def _refresh(client: AsyncClient, candidate: DemoCandidate, now: float) -> bool:
    """Reload the candidate's book from ``market_quotes`` and rebuild its signal.

    Returns ``False`` (and records why) if the market can no longer be read
    or the edge has disappeared since the scan."""
    topic = candidate.topic
    quotes = await client.market_quotes(topic.topic_id)
    built = build_topic(
        BatchMarketItem(
            topic_id=topic.topic_id,
            batch_id=topic.batch_id,
            title=topic.title,
            outcomes=quotes.outcomes,
            num_outcomes=len(quotes.outcomes or []),
            end_time_utc=topic.end_time_utc,
        ),
        topic.asset,
        now,
    )
    if built is None or quotes.quote_mode != QuoteMode.LIVE.value:
        candidate.result = OptimizationResult(None, (), (), "market no longer live")
        return False
    candidate.book = built[1]
    fresh = next(
        (
            signal
            for signal in build_signals(topic, candidate.book, candidate.forecast)
            if signal.option_id == candidate.signal.option_id
        ),
        None,
    )
    if fresh is None:
        candidate.result = OptimizationResult(None, (), (), "bucket no longer quoted")
        return False
    candidate.signal = fresh
    return True


def _ladder_table(
    server: tuple[LadderPoint, ...], local: tuple[LadderPoint, ...], chosen: float | None
) -> list[str]:
    local_by_size = {point.contracts: point for point in local}
    lines = [
        "| contracts | server cost (sats) | local replica (sats) | commission (sats) | "
        "all-in price | net edge (pts) | expected profit (sats) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in server:
        twin = local_by_size.get(point.contracts)
        marker = " **k\\***" if chosen is not None and point.contracts == chosen else ""
        lines.append(
            f"| {point.contracts:g}{marker} | {_sats(int(point.cost_millisats))} | "
            f"{_sats(int(twin.cost_millisats)) if twin else '-'} | "
            f"{_sats(int(point.commission_millisats))} | {float(point.exec_price):.2f} | "
            f"{float(point.net_edge_points):+.2f} | {_sats(int(point.expected_profit_millisats))} |"
        )
    return lines


def render_report(report: DemoReport) -> str:
    settings = report.settings
    executed = [c for c in report.candidates if c.record is not None]
    rejected = [c for c in report.candidates if c.record is None]
    spent = sum(-int(c.record.cash_flow_millisats) for c in executed if c.record is not None)
    expected = sum(
        int(c.result.plan.expected_profit_millisats)
        for c in executed
        if c.result is not None and c.result.plan is not None
    )
    lines: list[str] = [
        "# Sample trades",
        "",
        "> Auto-generated by `glimpse-quant-bot demo-trades`. Re-run the command to refresh it.",
        "",
        f"- **Generated:** {_utc(report.generated_at)}",
        "- **Mode:** DRY RUN. No order was placed and no API key was used. Signals come from "
        "live market data; fills are priced by Glimpse's own `/trades/estimate` endpoint "
        "through `AsyncClient.enter_multi_topic_multi_leg(dry_run=True)`.",
        f"- **SDK:** `glimpse-markets {glimpse_markets.__version__}`; time-series component: "
        f"`{report.timeseries_model}`",
        f"- **Bankroll:** {settings.bankroll_sats:,} sats (paper)",
        f"- **Thresholds:** raw edge >= {settings.min_raw_edge_points:g} pts, net edge >= "
        f"{settings.min_net_edge_points:g} pts, expected gain >= "
        f"{settings.min_expected_gain_sats:g} sats, Kelly fraction {settings.kelly_fraction:g}",
        f"- **Scanned:** {report.topics_scanned} live markets; "
        f"{len(report.candidates)} candidates priced on the server; "
        f"{len(executed)} paper trades executed",
        "",
        "Prices are on Glimpse's 0-100 scale (1 point on 1 contract = 1 sat). A winning "
        f"contract pays 100 points less the {100 * settings.fee_rate:g}% settlement fee.",
        "",
        "## 1. Market data snapshot",
        "",
        "| asset | spot (USD) | spot sources | ensemble weights | model disagreement (JSD) | "
        "Kelly multiplier |",
        "|---|---:|---|---|---:|---:|",
    ]
    for asset, forecast in report.asset_rows:
        weights = ", ".join(f"{name} {weight:.0%}" for name, weight in forecast.weights.items())
        lines.append(
            f"| {asset.value} | {forecast.spot:,.2f} | {', '.join(forecast.spot_sources)} | "
            f"{weights} | {forecast.disagreement:.3f} | {forecast.kelly_multiplier:.2f} |"
        )

    lines += [
        "",
        "## 2. Signals detected",
        "",
        "Largest model-versus-market gaps across the scanned markets (top 15).",
        "",
        "| asset | market | expires | bucket | market price | model | raw edge (pts) | "
        "component probabilities |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for topic, _forecast, signal in report.signals[:15]:
        components = ", ".join(
            f"{name} {100 * value:.1f}" for name, value in signal.components.items()
        )
        lines.append(
            f"| {topic.asset.value} | {topic.title} (#{topic.topic_id}) | "
            f"{_utc(topic.end_time_utc)} | {signal.name} | {float(signal.market_price):.2f} | "
            f"{100 * signal.model_probability:.2f} | {float(signal.edge_points):+.2f} | "
            f"{components} |"
        )

    lines += ["", "## 3. Paper trades executed", ""]
    if not executed:
        lines += [
            "No candidate cleared every threshold in this run. See section 4 for the reasons.",
            "",
        ]
    for index, candidate in enumerate(executed, start=1):
        result, record = candidate.result, candidate.record
        if result is None or result.plan is None or record is None:
            continue
        plan, topic, forecast = result.plan, candidate.topic, candidate.forecast
        lines += [
            f"### Trade {index}: BUY {plan.contracts:g} x `{plan.bucket_name}` on "
            f"{topic.asset.value} #{topic.topic_id}",
            "",
            f"- **Market:** {topic.title}, expires {_utc(topic.end_time_utc)} "
            f"({forecast.horizon_seconds / 3600:.1f} h); spot {forecast.spot:,.2f}; "
            f"LS-LMSR depth b = {candidate.book.liquidity:.1f} contracts",
            f"- **Signal:** model {100 * plan.model_probability:.2f}% vs market "
            f"{float(plan.market_price):.2f} -> raw edge "
            f"{float(candidate.signal.edge_points):+.2f} pts",
            f"- **Sizing:** fractional Kelly allows {100 * plan.kelly_fraction:.2f}% of bankroll "
            f"(budget {_sats(int(plan.budget_millisats))} sats after the bucket/topic caps); "
            "slippage ladder below",
            "",
            *_ladder_table(result.ladder, result.local_ladder, plan.contracts),
            "",
            f"- **Chosen:** k\\* = {plan.contracts:g} contracts, the rung with the highest "
            f"expected profit that clears every threshold: all-in {float(plan.exec_price):.2f} "
            f"pts/contract, net edge {float(plan.net_edge_points):+.2f} pts, expected profit "
            f"{_sats(int(plan.expected_profit_millisats))} sats",
            f"- **Risk gate:** approved; **fill (paper):** paid "
            f"{_sats(-int(record.cash_flow_millisats))} sats including "
            f"{_sats(int(record.commission_millisats))} sats commission "
            f"(ledger record #{record.record_id})",
        ]
        if candidate.exit_decision is not None:
            lines.append(
                f"- **Exit monitor:** `{candidate.exit_decision.reason}` -- "
                f"{candidate.exit_decision.detail}; selling now would net "
                f"{_sats(candidate.exit_net_millisats)} sats. The live strategy re-evaluates "
                f"this every {settings.tick_interval_seconds:g} s and exits on convergence "
                f"(market >= model - {settings.take_profit_gap_points:g}) or invalidation "
                f"(model < entry - {settings.invalidation_drop_points:g})."
            )
        lines.append("")

    lines += ["## 4. Candidates priced but not traded", ""]
    if not rejected:
        lines += ["None.", ""]
    else:
        lines += [
            "| asset | market | bucket | market | model | best server rung | reason |",
            "|---|---|---|---:|---:|---|---|",
        ]
        for candidate in rejected:
            result = candidate.result
            ladder = result.ladder if result is not None else ()
            best = max(ladder, key=lambda p: int(p.expected_profit_millisats), default=None)
            best_text = (
                "-"
                if best is None
                else f"{best.contracts:g} @ {float(best.exec_price):.2f}, "
                f"EV {_sats(int(best.expected_profit_millisats))} sats"
            )
            if candidate.decision is not None and not candidate.decision.approved:
                reason = "risk gate: " + "; ".join(candidate.decision.reasons)
            elif result is not None and result.rejection:
                reason = result.rejection
            else:
                reason = "not filled"
            lines.append(
                f"| {candidate.topic.asset.value} | #{candidate.topic.topic_id} | "
                f"{candidate.signal.name} | {float(candidate.signal.market_price):.2f} | "
                f"{100 * candidate.signal.model_probability:.2f} | {best_text} | {reason} |"
            )
        lines.append("")

    bankroll = int(settings.bankroll_millisats)
    lines += [
        "## 5. Paper portfolio after the run",
        "",
        f"- Trades: {len(executed)}; capital deployed: {_sats(spent)} sats "
        f"({100 * spent / bankroll:.2f}% of bankroll); "
        f"cash left: {_sats(report.cash_millisats)} sats",
        f"- Sum of expected profit at entry: {_sats(expected)} sats "
        f"({100 * expected / bankroll:.3f}% of bankroll)",
        f"- Caps in force: {100 * settings.max_bucket_fraction:g}% per bucket, "
        f"{100 * settings.max_topic_fraction:g}% per topic, "
        f"{100 * settings.daily_drawdown_limit:g}% daily drawdown kill-switch",
        "",
        "Expected profit is a model estimate, not a result: each position settles at 0 or "
        "100. Run `glimpse-quant-bot run` to hold these to convergence or expiry in paper "
        "mode and `glimpse-quant-bot report` to see realised P&L.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -e .",
        f"glimpse-quant-bot demo-trades --max-trades {max(len(report.candidates), 1)} "
        f"--min-expected-gain-sats {settings.min_expected_gain_sats:g}",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(report: DemoReport, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(report), encoding="utf-8")
    return target
