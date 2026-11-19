"use client";

import type { ExposureRow, RiskMetrics } from "@/lib/contract";
import { formatPercent, formatSats } from "@/lib/format";
import { millisats } from "@/lib/units";

import { Delta, EmptyState, Meter, Panel, Stat } from "./ui";

function ExposureBar({ row }: { row: ExposureRow }) {
  const tone = row.utilization >= 1 ? "critical" : row.utilization >= 0.7 ? "warn" : "s1";
  return (
    <div className="space-y-1">
      <div className="flex items-baseline gap-2 text-xs">
        <span className="min-w-0 flex-1 truncate text-ink-2" title={row.label}>
          {row.asset !== null && <span className="mr-1 text-[10px] font-bold text-ink-3">{row.asset}</span>}
          {row.label}
        </span>
        <span className="num text-ink">{formatPercent(row.fraction_of_equity * 100, { digits: 1 })}</span>
        <span className="num text-[11px] text-ink-3">/ {formatPercent(row.cap_fraction * 100, { digits: 0 })}</span>
        {row.utilization >= 1 && <span className="text-[10px] font-bold text-critical-ink">✕ CAP</span>}
      </div>
      <Meter fraction={row.utilization} tone={tone} label={`${row.label} exposure versus its cap`} height={5} />
    </div>
  );
}

export function ExposurePanel({ risk }: { risk: RiskMetrics | null }) {
  if (risk === null) {
    return (
      <Panel title="Risk & exposure">
        <EmptyState title="Waiting for risk metrics" />
      </Panel>
    );
  }
  const pnl = risk.realized_pnl_millisats + risk.unrealized_pnl_millisats;
  return (
    <Panel title="Risk & exposure" subtitle={risk.mode === "live" ? "live book" : "paper book"}>
      <div className="grid grid-cols-2 gap-3">
        <Stat label="Equity" value={<span className="num">{formatSats(risk.equity_millisats)}</span>} />
        <Stat label="Cash" value={<span className="num">{formatSats(risk.cash_millisats)}</span>} />
        <Stat
          label="Total P&L"
          value={
            <Delta value={pnl}>
              {formatSats(millisats(Math.abs(pnl)))}
            </Delta>
          }
          hint={`realized ${formatSats(risk.realized_pnl_millisats, { signed: true, unit: false })} · open ${formatSats(risk.unrealized_pnl_millisats, { signed: true, unit: false })}`}
        />
        <Stat
          label="Fractional Kelly"
          value={<span className="num">{risk.kelly_fraction.toFixed(2)}×</span>}
          hint="before the κ agreement multiplier"
        />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 border-t border-line pt-3">
        <div className="space-y-1">
          <div className="flex justify-between text-[11px]">
            <span className="text-ink-3">Open positions</span>
            <span className="num text-ink-2">
              {risk.open_positions} / {risk.max_open_positions}
            </span>
          </div>
          <Meter fraction={risk.open_positions / Math.max(1, risk.max_open_positions)} tone="s1" label="Open positions versus the maximum" height={5} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-[11px]">
            <span className="text-ink-3">Orders / min</span>
            <span className="num text-ink-2">
              {risk.fills_last_minute} / {risk.max_orders_per_minute}
            </span>
          </div>
          <Meter
            fraction={risk.fills_last_minute / Math.max(1, risk.max_orders_per_minute)}
            tone={risk.fills_last_minute >= risk.max_orders_per_minute ? "warn" : "s1"}
            label="Orders in the last minute versus the limit"
            height={5}
          />
        </div>
      </div>

      <div className="mt-4 space-y-3">
        <div>
          <div className="mb-1.5 flex justify-between text-[11px] text-ink-3">
            <span>Per bucket</span>
            <span>cap {formatPercent(risk.max_bucket_fraction * 100, { digits: 0 })} of equity</span>
          </div>
          <div className="space-y-2">
            {risk.bucket_exposures.length === 0 && <p className="text-xs text-ink-3">No open bucket exposure.</p>}
            {risk.bucket_exposures.slice(0, 6).map((row) => (
              <ExposureBar key={`${row.topic_id}-${row.option_id ?? "t"}`} row={row} />
            ))}
          </div>
        </div>
        <div>
          <div className="mb-1.5 flex justify-between text-[11px] text-ink-3">
            <span>Per topic</span>
            <span>cap {formatPercent(risk.max_topic_fraction * 100, { digits: 0 })} of equity</span>
          </div>
          <div className="space-y-2">
            {risk.topic_exposures.length === 0 && <p className="text-xs text-ink-3">No open topic exposure.</p>}
            {risk.topic_exposures.slice(0, 6).map((row) => (
              <ExposureBar key={`t-${row.topic_id}`} row={row} />
            ))}
          </div>
        </div>
      </div>
    </Panel>
  );
}
