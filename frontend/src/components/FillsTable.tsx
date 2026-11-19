"use client";

import { useEffect, useMemo, useState } from "react";

import { api, errorText } from "@/lib/api";
import type { LedgerRecordView, LedgerSummaryView, ModeName } from "@/lib/contract";
import { formatClock, formatContracts, formatPercent, formatPoints, formatSats } from "@/lib/format";
import { millisats } from "@/lib/units";

import { Button, Delta, EmptyState, Stat, cx } from "./ui";

const KIND_STYLE: Record<string, string> = {
  entry: "text-s1",
  exit: "text-s2",
  settlement: "text-ink",
  adjustment: "text-warn",
};

export function FillsTable({ live, mode }: { live: readonly LedgerRecordView[]; mode: ModeName }) {
  const [older, setOlder] = useState<readonly LedgerRecordView[]>([]);
  const [summary, setSummary] = useState<LedgerSummaryView | null>(null);
  const [total, setTotal] = useState<number | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The summary is cheap (server memory) — refresh it whenever a fill streams in.
  const newest = live[0]?.record_id ?? 0;
  useEffect(() => {
    let cancelled = false;
    api
      .ledger({ mode, limit: 1 })
      .then((page) => {
        if (cancelled) return;
        setSummary(page.summary);
        setTotal(page.total);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(errorText(err));
      });
    return () => {
      cancelled = true;
    };
  }, [mode, newest]);

  useEffect(() => {
    setOlder([]);
    setCursor(null);
  }, [mode]);

  const rows = useMemo(() => {
    const byId = new Map<number, LedgerRecordView>();
    for (const record of [...live, ...older]) if (record.mode === mode) byId.set(record.record_id, record);
    return [...byId.values()].sort((a, b) => b.record_id - a.record_id);
  }, [live, older, mode]);

  async function loadOlder() {
    const oldest = cursor ?? rows[rows.length - 1]?.record_id;
    if (oldest === undefined) return;
    setLoading(true);
    try {
      const page = await api.ledger({ mode, beforeId: oldest, limit: 100 });
      setOlder((current) => [...current, ...page.records]);
      setCursor(page.next_before_id);
      setTotal(page.total);
      setError(null);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col">
      {summary !== null && (
        <div className="grid grid-cols-3 gap-3 border-b border-line px-3 py-2 sm:grid-cols-6">
          <Stat
            label="Realized P&L"
            value={
              <Delta value={summary.realized_pnl_millisats}>
                {formatSats(millisats(Math.abs(summary.realized_pnl_millisats)))}
              </Delta>
            }
          />
          <Stat label="ROI" value={<Delta value={summary.roi}>{formatPercent(Math.abs(summary.roi * 100))}</Delta>} />
          <Stat label="Win rate" value={<span className="num">{formatPercent(summary.win_rate * 100, { digits: 0 })}</span>} hint={`${summary.wins}W · ${summary.losses}L`} />
          <Stat label="Commission paid" value={<span className="num">{formatSats(summary.commission_millisats)}</span>} />
          <Stat label="Entries / exits" value={<span className="num">{summary.entries} / {summary.exits}</span>} hint={`${summary.settlements} settled`} />
          <Stat label="Records" value={<span className="num">{total ?? "—"}</span>} hint="append-only ledger" />
        </div>
      )}
      {error !== null && <p className="px-3 py-2 text-xs text-critical-ink">✕ {error}</p>}
      {rows.length === 0 ? (
        <EmptyState title="No fills yet">Every fill is appended to the SQLite + NDJSON ledger and streamed here.</EmptyState>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-xs">
            <thead className="text-[11px] text-ink-3">
              <tr className="text-right">
                <th className="px-3 py-2 text-left font-medium">Time</th>
                <th className="px-2 text-left font-medium">Kind</th>
                <th className="px-2 text-left font-medium">Bucket</th>
                <th className="px-2 font-medium">Contracts</th>
                <th className="px-2 font-medium">Exec</th>
                <th className="px-2 font-medium">Quote</th>
                <th className="px-2 font-medium">Cash flow</th>
                <th className="px-2 font-medium">Fee</th>
                <th className="px-2 font-medium">Realized</th>
                <th className="px-3 text-left font-medium">Note</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.record_id} className="num border-t border-line text-right">
                  <td className="px-3 py-1.5 text-left text-ink-2">{formatClock(r.ts)}</td>
                  <td className={cx("px-2 text-left font-semibold uppercase", KIND_STYLE[r.kind] ?? "text-ink-2")}>{r.kind}</td>
                  <td className="px-2 text-left text-ink">{r.bucket_name ?? "—"}</td>
                  <td className="px-2 text-ink">{formatContracts(r.contracts)}</td>
                  <td className="px-2 text-ink-2">{formatPoints(r.exec_price)}</td>
                  <td className="px-2 text-ink-2">{formatPoints(r.market_price)}</td>
                  <td className="px-2">
                    <Delta value={r.cash_flow_millisats}>{formatSats(millisats(Math.abs(r.cash_flow_millisats)), { unit: false })}</Delta>
                  </td>
                  <td className="px-2 text-ink-3">{formatSats(r.commission_millisats, { unit: false })}</td>
                  <td className="px-2">
                    {r.realized_pnl_millisats === 0 ? (
                      <span className="text-ink-3">—</span>
                    ) : (
                      <Delta value={r.realized_pnl_millisats}>{formatSats(millisats(Math.abs(r.realized_pnl_millisats)), { unit: false })}</Delta>
                    )}
                  </td>
                  <td className="max-w-[320px] truncate px-3 text-left text-ink-3" title={r.note}>
                    {r.note}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {rows.length > 0 && (total === null || rows.length < total) && (
        <div className="flex justify-center border-t border-line py-2">
          <Button size="sm" tone="ghost" disabled={loading} onClick={() => void loadOlder()}>
            {loading ? "Loading…" : "Load older fills"}
          </Button>
        </div>
      )}
    </div>
  );
}
