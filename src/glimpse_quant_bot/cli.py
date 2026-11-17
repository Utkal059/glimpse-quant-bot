"""Command-line interface.

    glimpse-quant-bot run            stream-driven trading loop (paper unless armed)
    glimpse-quant-bot demo-trades    one-shot scan + paper trades -> Markdown report
    glimpse-quant-bot record         record the live feed for later backtests
    glimpse-quant-bot backtest       walk-forward replay of a recording
    glimpse-quant-bot report         ledger summary and recent fills
    glimpse-quant-bot reconcile      align the live ledger with the exchange

Real orders require BOTH ``--live`` on ``run`` and ``GLIMPSE_LIVE_TRADING=true``
in the environment. Every other command is read-only or paper-only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog
import websockets
from dotenv import load_dotenv
from glimpse_markets import AsyncClient, StrategyRunner
from glimpse_markets.forecasting.recorder import MarketRecorder
from glimpse_markets.ratelimit import AsyncRateLimiter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from glimpse_quant_bot.backtester import (
    BacktestData,
    Backtester,
    BacktestResult,
    BacktestTopic,
    load_dataset,
    save_dataset,
    synthetic_dataset,
)
from glimpse_quant_bot.config import LiveTradingNotArmedError, Settings, load_settings
from glimpse_quant_bot.demo import run_demo, write_report
from glimpse_quant_bot.external_feeds import FeedError, MarketDataHub
from glimpse_quant_bot.ledger import Ledger
from glimpse_quant_bot.risk_manager import RiskManager
from glimpse_quant_bot.strategy import GlimpseQuantStrategy, discover_batches, load_batch_topics
from glimpse_quant_bot.types import Mode, RecordKind, TopicMeta

console = Console()
log = structlog.get_logger(__name__)


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def make_client(settings: Settings, mode: Mode) -> AsyncClient:
    """Build the SDK client. ``dry_run`` is on unless the mode is explicitly live."""
    load_dotenv()
    return AsyncClient.from_env(
        dry_run=mode != "live",
        rate_limiter=AsyncRateLimiter(max_requests=settings.glimpse_requests_per_minute),
    )


def _apply_overrides(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "min_expected_gain_sats", None) is not None:
        overrides["min_expected_gain_sats"] = args.min_expected_gain_sats
    if getattr(args, "data_dir", None) is not None:
        overrides["data_dir"] = Path(args.data_dir)
    if getattr(args, "max_horizon_hours", None) is not None:
        overrides["max_horizon_hours"] = args.max_horizon_hours
    return load_settings(**overrides)


def _sats(millisats: int) -> str:
    return f"{millisats / 1000:,.1f}"


def print_ledger_report(ledger: Ledger, settings: Settings, mode: Mode, limit: int = 15) -> None:
    summary = ledger.summary(mode, settings.bankroll_millisats)
    table = Table(title=f"Ledger summary ({mode})", show_header=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("Starting bankroll (sats)", _sats(int(summary.starting_millisats)))
    table.add_row("Cash (sats)", _sats(int(summary.cash_millisats)))
    table.add_row("Open cost basis (sats)", _sats(int(summary.open_cost_basis_millisats)))
    table.add_row("Realised P&L (sats)", _sats(int(summary.realized_pnl_millisats)))
    table.add_row("Realised ROI", f"{100 * summary.roi:.3f}%")
    table.add_row("Fees paid (sats)", _sats(int(summary.commission_millisats)))
    table.add_row(
        "Entries / exits / settlements",
        f"{summary.entries} / {summary.exits} / {summary.settlements}",
    )
    table.add_row("Win rate (closed)", f"{100 * summary.win_rate:.1f}%")
    table.add_row("Open positions", str(summary.open_positions))
    console.print(table)

    fills = ledger.records(
        mode, kinds=[RecordKind.ENTRY, RecordKind.EXIT, RecordKind.SETTLEMENT], limit=limit
    )
    if not fills:
        console.print("No fills recorded yet.")
        return
    recent = Table(title=f"Last {len(fills)} fills")
    for column in (
        "id",
        "kind",
        "topic",
        "bucket",
        "contracts",
        "cash flow (sats)",
        "price",
        "P&L (sats)",
    ):
        recent.add_column(column, justify="right")
    for record in fills:
        recent.add_row(
            str(record.record_id),
            record.kind.value,
            str(record.topic_id),
            record.bucket_name or "",
            f"{record.contracts:g}",
            _sats(int(record.cash_flow_millisats)),
            "-" if record.exec_price is None else f"{float(record.exec_price):.2f}",
            _sats(int(record.realized_pnl_millisats)),
        )
    console.print(recent)


# --- run --------------------------------------------------------------------


async def cmd_run(args: argparse.Namespace) -> int:
    settings = _apply_overrides(args)
    configure_logging(settings)
    try:
        mode = settings.resolve_mode(args.live)
    except LiveTradingNotArmedError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2
    banner = (
        "[bold red]LIVE TRADING -- real Bitcoin at risk[/bold red]"
        if mode == "live"
        else "[bold green]DRY RUN -- paper trading, no orders are placed[/bold green]"
    )
    console.print(Panel(banner, title="glimpse-quant-bot"))

    client = make_client(settings, mode)
    strategy = GlimpseQuantStrategy(settings, mode)
    deadline = time.monotonic() + args.duration if args.duration else None
    backoff = 2.0
    try:
        while True:
            runner = StrategyRunner(
                strategy, client=client, tick_interval=settings.tick_interval_seconds
            )
            try:
                if deadline is None:
                    await runner.run()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    await asyncio.wait_for(runner.run(), timeout=remaining)
            except TimeoutError:
                break
            except (websockets.WebSocketException, OSError) as exc:
                log.warning("run.stream_dropped", error=repr(exc), retry_in=backoff)
            else:
                log.warning("run.stream_ended", retry_in=backoff)
            await asyncio.sleep(backoff)
            backoff = min(60.0, backoff * 2.0)
    finally:
        print_ledger_report(strategy.ledger, settings, mode)
        await strategy.close()
        await client.close()
    return 0


# --- demo-trades ------------------------------------------------------------


async def cmd_demo_trades(args: argparse.Namespace) -> int:
    settings = _apply_overrides(args)
    configure_logging(settings)
    console.print(
        Panel("[bold green]DRY RUN -- no orders are placed[/bold green]", title="demo-trades")
    )
    ledger_path = settings.data_dir / f"demo-{int(time.time())}.sqlite"
    ledger = Ledger(ledger_path, settings.data_dir / "demo-audit.ndjson")
    client = make_client(settings, "dry_run")
    try:
        report = await run_demo(
            settings,
            client,
            ledger,
            max_trades=args.max_trades,
            topics_per_batch=args.topics_per_batch,
        )
    finally:
        await client.close()
    target = write_report(report, args.out)
    executed = sum(1 for candidate in report.candidates if candidate.record is not None)
    console.print(
        f"Scanned {report.topics_scanned} markets, priced {len(report.candidates)} candidates, "
        f"executed {executed} paper trades."
    )
    print_ledger_report(ledger, settings, "dry_run")
    ledger.close()
    console.print(f"Report written to [bold]{target}[/bold]; ledger at {ledger_path}")
    return 0


# --- record -----------------------------------------------------------------


async def _resolved_options(client: AsyncClient, topics: Sequence[TopicMeta]) -> dict[int, int]:
    winners: dict[int, int] = {}
    for batch_id in sorted({topic.batch_id for topic in topics if topic.batch_id}):
        page = await client.batch_resolved_markets_page(batch_id)
        for market in page.markets or []:
            if market.topic_id is not None and market.resolved_option_id is not None:
                winners[market.topic_id] = market.resolved_option_id
    return winners


async def cmd_record(args: argparse.Namespace) -> int:
    settings = _apply_overrides(args)
    configure_logging(settings)
    out_dir = (
        Path(args.out) if args.out else settings.recordings_dir / time.strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = MarketRecorder(
        max_points_per_series=1_000_000, persist_path=out_dir / "prices.ndjson"
    )
    client = make_client(settings, "dry_run")
    hub = MarketDataHub(settings.assets, enable_websocket=False)
    data = BacktestData()
    topics: dict[int, TopicMeta] = {}
    updates = 0
    deadline = time.monotonic() + args.duration
    next_spot = 0.0
    console.print(f"Recording for {args.duration:.0f} s into {out_dir} ...")
    try:
        batches = await discover_batches(client, settings)
        for batch_id, asset in batches.items():
            for topic, _book in await load_batch_topics(
                client, batch_id, asset, settings, time.time()
            ):
                topics[topic.topic_id] = topic
        async with client.stream_market_updates() as stream:
            iterator = stream.__aiter__()
            while time.monotonic() < deadline:
                if time.monotonic() >= next_spot:
                    next_spot = time.monotonic() + args.spot_interval
                    for asset in settings.assets:
                        with contextlib.suppress(FeedError):
                            quote = await hub.spot(asset)
                            data.spot.setdefault(asset, []).append((int(quote.ts), quote.price))
                try:
                    update = await asyncio.wait_for(iterator.__anext__(), timeout=5.0)
                except TimeoutError:
                    continue
                except StopAsyncIteration:
                    break
                if update.topic_id in topics:
                    recorder.record(update)
                    updates += 1
        for asset in settings.assets:
            with contextlib.suppress(FeedError):
                data.candles[asset] = list(
                    await hub.candles(
                        asset, settings.candle_interval_seconds, settings.candle_history
                    )
                )
        winners = await _resolved_options(client, list(topics.values()))
        for topic_id, topic in topics.items():
            data.topics[topic_id] = BacktestTopic(topic, winners.get(topic_id))
        save_dataset(data, out_dir, include_prices=False)
    finally:
        await hub.close()
        await client.close()
    console.print(
        f"Recorded {updates} market updates across {len(topics)} topics. "
        f"Backtest with: glimpse-quant-bot backtest --data-dir {out_dir}"
    )
    return 0


# --- backtest ---------------------------------------------------------------


def print_backtest(result: BacktestResult) -> None:
    table = Table(title="Backtest result", show_header=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("Starting bankroll (sats)", _sats(int(result.starting_millisats)))
    table.add_row("Ending equity (sats)", _sats(int(result.ending_equity_millisats)))
    table.add_row("Net P&L (sats)", _sats(int(result.pnl_millisats)))
    table.add_row("ROI", f"{100 * result.roi:.3f}%")
    table.add_row("Sharpe (annualised)", f"{result.sharpe:.2f}")
    table.add_row("Sortino (annualised)", f"{result.sortino:.2f}")
    table.add_row("Max drawdown", f"{100 * result.max_drawdown:.3f}%")
    table.add_row("Win rate (closed trades)", f"{100 * result.win_rate:.1f}%")
    table.add_row(
        "Entries / exits / settlements", f"{result.entries} / {result.exits} / {result.settlements}"
    )
    table.add_row("Fees paid (sats)", _sats(int(result.commission_millisats)))
    console.print(table)


async def cmd_backtest(args: argparse.Namespace) -> int:
    settings = _apply_overrides(args)
    configure_logging(settings)
    if args.synthetic:
        data = synthetic_dataset(seed=args.seed, mispriced=not args.control)
        console.print(
            "Synthetic recording: "
            + ("correctly priced market (control)" if args.control else "stale, over-wide market")
        )
    elif args.data_dir:
        root = Path(args.data_dir)
        data = load_dataset(
            root / "prices.ndjson",
            root / "meta.json",
            root / "candles.ndjson",
            root / "spot.ndjson",
        )
    else:
        console.print("[red]Pass --synthetic or --data-dir DIR[/red]")
        return 2
    result = await Backtester(settings, data, step_seconds=args.step_seconds).run()
    print_backtest(result)
    if args.json:
        payload = {
            "pnl_sats": int(result.pnl_millisats) / 1000,
            "roi": result.roi,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "entries": result.entries,
            "exits": result.exits,
            "settlements": result.settlements,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


# --- report / reconcile -----------------------------------------------------


async def cmd_report(args: argparse.Namespace) -> int:
    settings = _apply_overrides(args)
    ledger_path = Path(args.ledger) if args.ledger else settings.ledger_path
    if not ledger_path.exists():
        console.print(f"No ledger at {ledger_path}. Run `glimpse-quant-bot run` first.")
        return 1
    ledger = Ledger(ledger_path)
    print_ledger_report(ledger, settings, "live" if args.live else "dry_run", limit=args.limit)
    ledger.close()
    return 0


async def cmd_reconcile(args: argparse.Namespace) -> int:
    settings = _apply_overrides(args)
    configure_logging(settings)
    if settings.glimpse_api_key is None:
        console.print(
            "[red]GLIMPSE_API_KEY is required: reconcile reads your live portfolio.[/red]"
        )
        return 2
    ledger = Ledger(settings.ledger_path, settings.audit_path)
    risk = RiskManager(settings, ledger, "live")
    client = make_client(settings, "dry_run")
    try:
        report = await risk.reconcile(client)
    finally:
        await client.close()
    if not report.ok:
        console.print(f"[red]{report.note}[/red]")
        ledger.close()
        return 1
    table = Table(title="Reconciliation: ledger vs portfolio_active()")
    for column in ("topic", "option", "ledger contracts", "exchange contracts"):
        table.add_column(column, justify="right")
    for item in report.discrepancies:
        table.add_row(
            str(item.topic_id),
            str(item.option_id),
            f"{item.ledger_contracts:g}",
            f"{item.actual_contracts:g}",
        )
    console.print(table if report.discrepancies else "Ledger and exchange agree.")
    console.print(report.note)
    ledger.close()
    return 0


# --- entry point --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glimpse-quant-bot",
        description="Probability-calibration and LS-LMSR execution engine for Glimpse markets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-dir", help="directory for the ledger and recordings")
        command.add_argument(
            "--min-expected-gain-sats",
            type=float,
            help="override the minimum expected profit per trade (default 25)",
        )
        command.add_argument("--max-horizon-hours", type=float, help="override the horizon cap")

    run = sub.add_parser("run", help="run the trading loop (paper unless armed for live)")
    common(run)
    run.add_argument(
        "--live",
        action="store_true",
        help="place REAL orders; also requires GLIMPSE_LIVE_TRADING=true",
    )
    run.add_argument("--duration", type=float, help="stop after this many seconds")
    run.set_defaults(handler=cmd_run)

    demo = sub.add_parser("demo-trades", help="scan live markets and write example paper trades")
    common(demo)
    demo.add_argument("--max-trades", type=int, default=5)
    demo.add_argument("--topics-per-batch", type=int, default=3)
    demo.add_argument("--out", default="examples/sample_trades.md")
    demo.set_defaults(handler=cmd_demo_trades)

    record = sub.add_parser("record", help="record the live feed into a backtest dataset")
    common(record)
    record.add_argument("--duration", type=float, default=3600.0)
    record.add_argument("--spot-interval", type=float, default=30.0)
    record.add_argument("--out", help="output directory (default: data/recordings/<timestamp>)")
    record.set_defaults(handler=cmd_record)

    backtest = sub.add_parser("backtest", help="walk-forward replay of a recording")
    common(backtest)
    backtest.add_argument(
        "--synthetic", action="store_true", help="use the built-in synthetic world"
    )
    backtest.add_argument(
        "--control", action="store_true", help="synthetic: correctly priced market"
    )
    backtest.add_argument("--seed", type=int, default=11)
    backtest.add_argument("--step-seconds", type=int, default=300)
    backtest.add_argument("--json", help="also write the metrics to this JSON file")
    backtest.set_defaults(handler=cmd_backtest)

    report = sub.add_parser("report", help="ledger summary and recent fills")
    common(report)
    report.add_argument("--ledger", help="path to a ledger file (default: data/ledger.sqlite)")
    report.add_argument(
        "--live", action="store_true", help="show live fills instead of paper fills"
    )
    report.add_argument("--limit", type=int, default=15)
    report.set_defaults(handler=cmd_report)

    reconcile = sub.add_parser("reconcile", help="align the live ledger with the exchange")
    common(reconcile)
    reconcile.set_defaults(handler=cmd_reconcile)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(asyncio.run(args.handler(args)))
    except KeyboardInterrupt:
        console.print("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
