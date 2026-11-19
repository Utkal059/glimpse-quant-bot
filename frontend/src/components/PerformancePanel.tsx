"use client";

import { useCallback, useEffect, useState } from "react";

import { api, errorText } from "@/lib/api";
import type { BacktestResultsResponse, EquityPoint, ModeName, PerformanceMetrics } from "@/lib/contract";
import { formatClock, formatDateTime, formatNumber, formatPercent, formatSats } from "@/lib/format";
import { millisats, toSats } from "@/lib/units";

import { LineChart } from "./LineChart";
import { useToast } from "./Toaster";
import { Button, Delta, EmptyState, Stat, Toggle } from "./ui";

const REFRESH_MS = 15_000;

function Metrics({ metrics }: { metrics: PerformanceMetrics }) {
  return (
    <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
      <Stat label="P&L" value={<Delta value={metrics.pnl_millisats}>{formatSats(millisats(Math.abs(metrics.pnl_millisats)))}</Delta>} />
      <Stat label="ROI" value={<Delta value={metrics.roi}>{formatPercent(Math.abs(metrics.roi * 100))}</Delta>} />
      <Stat label="Sharpe" value={<span className="num">{formatNumber(metrics.sharpe, 2)}</span>} hint="annualised" />
      <Stat label="Sortino" value={<span className="num">{formatNumber(metrics.sortino, 2)}</span>} />
      <Stat label="Max drawdown" value={<span className="num">{formatPercent(-metrics.max_drawdown * 100)}</span>} />
      <Stat label="Equity points" value={<span className="num">{metrics.points}</span>} />
    </div>
  );
}

function EquityCurve({ curve, label }: { curve: readonly EquityPoint[]; label: string }) {
  if (curve.length < 2) {
    return <EmptyState title="Not enough equity snapshots yet">The engine records equity every tick; the curve fills in as it runs.</EmptyState>;
  }
  const spanHours = ((curve[curve.length - 1]?.ts ?? 0) - (curve[0]?.ts ?? 0)) / 3600;
  return (
    <LineChart
      ariaLabel={label}
      height={200}
      xs={curve.map((p) => p.ts)}
      xFormat={(ts) => (spanHours > 36 ? formatDateTime(ts).slice(5, 16) : formatClock(ts))}
      yFormat={(y) => `${Math.round(y).toLocaleString("en-US")}`}
      series={[{ key: "equity", label: "equity (sats)", color: "var(--color-s1)", values: curve.map((p) => toSats(p.equity_millisats)) }]}
    />
  );
}

export function PerformancePanel({ mode }: { mode: ModeName }) {
  const toast = useToast();
  const [data, setData] = useState<BacktestResultsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [control, setControl] = useState(false);
  const [starting, setStarting] = useState(false);

  const load = useCallback(async () => {
    try {
      setData(await api.backtestResults(mode));
      setError(null);
    } catch (err) {
      setError(errorText(err));
    }
  }, [mode]);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), data?.backtest_running === true ? 3_000 : REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [load, data?.backtest_running]);

  async function runBacktest() {
    setStarting(true);
    try {
      const result = await api.runBacktest(control);
      toast.push("info", "Backtest started", result.message);
      void load();
    } catch (err) {
      toast.push("critical", "Backtest refused", errorText(err));
    } finally {
      setStarting(false);
    }
  }

  const latest = data?.latest_backtest ?? null;

  return (
    <div className="grid gap-4 p-3 xl:grid-cols-2">
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-[11px] font-semibold tracking-[0.08em] text-ink-2 uppercase">{mode === "live" ? "Live" : "Paper"} equity</h3>
          {error !== null && <span className="text-[11px] text-critical-ink">✕ {error}</span>}
        </div>
        {data?.metrics !== null && data?.metrics !== undefined && <Metrics metrics={data.metrics} />}
        <EquityCurve curve={data?.equity_curve ?? []} label={`${mode} equity curve in sats`} />
      </section>

      <section className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-[11px] font-semibold tracking-[0.08em] text-ink-2 uppercase">Walk-forward backtest</h3>
          <span className="ml-auto" />
          <Toggle checked={control} onChange={setControl}>
            Control (fair market)
          </Toggle>
          <Button size="sm" tone="primary" disabled={starting || data?.backtest_running === true} onClick={() => void runBacktest()}>
            {data?.backtest_running === true ? "Running…" : "Run synthetic backtest"}
          </Button>
        </div>
        {latest === null ? (
          <EmptyState title="No backtest yet">
            Runs the unmodified strategy over the built-in synthetic world in a worker thread. The control run prices
            the market fairly and should not make money — a sanity check on the edge.
          </EmptyState>
        ) : (
          <>
            <p className="text-xs text-ink-3">
              {latest.control ? "Control run (fair market)" : "Mispriced synthetic world"} · seed {latest.seed} · step {latest.step_seconds}s
              · {latest.entries} entries · {latest.settlements} settlements · win rate {formatPercent(latest.win_rate * 100, { digits: 0 })} ·
              fees {formatSats(latest.commission_millisats)} · finished {formatClock(latest.finished_at)}
            </p>
            <Metrics metrics={latest.metrics} />
            <EquityCurve curve={latest.equity_curve} label="Backtest equity curve in sats" />
          </>
        )}
      </section>
    </div>
  );
}
