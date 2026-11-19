"use client";

import type { PositionView } from "@/lib/contract";
import { formatContracts, formatCountdown, formatPoints, formatProbability, formatSats } from "@/lib/format";
import { millisats } from "@/lib/units";

import { Delta, EmptyState, Pill } from "./ui";

const RULE_LABEL: Record<string, { text: string; tone: "good" | "critical" | "neutral" }> = {
  take_profit: { text: "▲ take profit", tone: "good" },
  invalidation: { text: "✕ invalidated", tone: "critical" },
  hold: { text: "hold", tone: "neutral" },
};

export function PositionsTable({ positions, onInspect }: { positions: readonly PositionView[]; onInspect: (topicId: number, optionId: number) => void }) {
  if (positions.length === 0) {
    return <EmptyState title="No open positions">Entries appear here the moment the ExecutionRouter fills them.</EmptyState>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[920px] text-xs">
        <thead className="text-[11px] text-ink-3">
          <tr className="text-right">
            <th className="px-3 py-2 text-left font-medium">Bucket</th>
            <th className="px-2 font-medium">Contracts</th>
            <th className="px-2 font-medium" title="All-in entry price">Entry</th>
            <th className="px-2 font-medium" title="Current marginal LS-LMSR quote">Mark</th>
            <th className="px-2 font-medium" title="Take profit when the quote reaches 100·p_model − 1.0">TP trigger</th>
            <th className="px-2 font-medium" title="Invalidate when 100·p_model falls below this">Invalidation</th>
            <th className="px-2 font-medium">Model p</th>
            <th className="px-2 font-medium">Cost basis</th>
            <th className="px-2 font-medium" title="Liquidation value after commission minus cost basis">Unrealized</th>
            <th className="px-2 text-left font-medium">Exit rule</th>
            <th className="px-3 font-medium">Expiry</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => {
            const rule = p.exit_rule === null ? null : RULE_LABEL[p.exit_rule];
            const pnl = p.unrealized_pnl_millisats;
            return (
              <tr
                key={`${p.topic_id}-${p.option_id}`}
                onClick={() => onInspect(p.topic_id, p.option_id)}
                className="num cursor-pointer border-t border-line text-right hover:bg-hover"
              >
                <td className="px-3 py-1.5 text-left">
                  <div className="font-medium text-ink">
                    {p.asset !== null && <span className="mr-1.5 text-[10px] font-bold text-ink-3">{p.asset}</span>}
                    {p.bucket_name}
                  </div>
                  {p.title !== null && <div className="truncate text-[11px] text-ink-3">{p.title}</div>}
                </td>
                <td className="px-2 text-ink">{formatContracts(p.contracts)}</td>
                <td className="px-2 text-ink-2">{formatPoints(p.entry_price)}</td>
                <td className="px-2 text-ink">{formatPoints(p.mark_price)}</td>
                <td className="px-2 text-ink-2">{formatPoints(p.take_profit_trigger_price)}</td>
                <td className="px-2 text-ink-2">{formatPoints(p.invalidation_model_points)}</td>
                <td className="px-2 text-ink-2">{formatProbability(p.model_probability, 1)}</td>
                <td className="px-2 text-ink-2">{formatSats(p.cost_basis_millisats, { unit: false })}</td>
                <td className="px-2">
                  {pnl === null ? (
                    <span className="text-ink-3">—</span>
                  ) : (
                    <Delta value={pnl}>{formatSats(millisats(Math.abs(pnl)), { unit: false })}</Delta>
                  )}
                </td>
                <td className="px-2 text-left">
                  {rule === null || rule === undefined ? (
                    <span className="text-ink-3">—</span>
                  ) : (
                    <span title={p.exit_detail ?? undefined}>
                      <Pill tone={rule.tone}>{rule.text}</Pill>
                    </span>
                  )}
                </td>
                <td className="px-3 text-ink-2">{formatCountdown(p.seconds_to_expiry)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
