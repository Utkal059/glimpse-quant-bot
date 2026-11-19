"use client";

import { useCallback, useEffect, useState } from "react";

import { api, errorText } from "@/lib/api";
import type { HealthResponse, LadderResponse, TestOrderResult } from "@/lib/contract";
import { LIVE_ORDER_PHRASE } from "@/lib/contract";
import { formatContracts, formatPoints, formatProbability, formatSats } from "@/lib/format";
import { toSats } from "@/lib/units";

import { LineChart } from "./LineChart";
import { useToast } from "./Toaster";
import { Button, Field, Pill, Stat, cx, inputClass } from "./ui";

const REFRESH_MS = 5_000;
const MAX_CONTRACTS = 50;
const QUEUED_AFTER_MS = 2_500;

/**
 * Bucket inspector: the LS-LMSR cost of buying k = 1..50 contracts, where the
 * marginal edge turns negative, and a guarded test-order ticket. Tickets go
 * through the server's RiskManager and ExecutionRouter like any engine order;
 * in live mode they additionally need a typed phrase.
 */
export function LadderDrawer({
  topicId,
  optionId,
  feeRate,
  health,
  onClose,
}: {
  topicId: number;
  optionId: number;
  feeRate: number;
  health: HealthResponse | null;
  onClose: () => void;
}) {
  const toast = useToast();
  const [ladder, setLadder] = useState<LadderResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [contracts, setContracts] = useState("1");
  const [phrase, setPhrase] = useState("");
  const [busy, setBusy] = useState(false);
  const [queued, setQueued] = useState(false);
  const [result, setResult] = useState<TestOrderResult | null>(null);

  // An approved ticket shares the engine's Glimpse REST budget, so it can wait for a slot.
  useEffect(() => {
    if (!busy) {
      setQueued(false);
      return;
    }
    const timer = window.setTimeout(() => setQueued(true), QUEUED_AFTER_MS);
    return () => window.clearTimeout(timer);
  }, [busy]);

  const load = useCallback(async () => {
    try {
      const next = await api.ladder(topicId, optionId, MAX_CONTRACTS);
      setLadder(next);
      setError(null);
    } catch (err) {
      setError(errorText(err));
    }
  }, [topicId, optionId]);

  useEffect(() => {
    setLadder(null);
    setResult(null);
    setPhrase("");
    void load();
    const timer = window.setInterval(() => void load(), REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [load]);

  useEffect(() => {
    if (ladder?.optimal_contracts !== null && ladder?.optimal_contracts !== undefined) {
      setContracts((current) => (current === "1" ? String(Math.max(1, Math.round(ladder.optimal_contracts ?? 1))) : current));
    }
  }, [ladder?.optimal_contracts]);

  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const live = health?.mode === "LIVE_ARMED";
  const running = health?.engine_state === "running";
  const size = Number(contracts);
  const sizeValid = Number.isFinite(size) && size > 0 && size <= 400;
  const canSend = running && sizeValid && !busy && (!live || phrase === LIVE_ORDER_PHRASE) && !(health?.kill_switch_engaged ?? false);

  async function send() {
    setBusy(true);
    setResult(null);
    try {
      const response = await api.testOrder({
        topic_id: topicId,
        option_id: optionId,
        contracts: size,
        confirmation_phrase: live ? phrase : null,
      });
      setResult(response);
      if (response.ok) toast.push("good", live ? "Live order filled" : "Paper order filled", response.message);
      else toast.push("warn", "Order not filled", response.reasons.join("; ") || response.message);
      void load();
    } catch (err) {
      toast.push("critical", "Order refused", errorText(err));
    } finally {
      setBusy(false);
      setPhrase("");
    }
  }

  const rows = ladder?.rows ?? [];
  const xs = rows.map((row) => row.contracts);
  const fair = ladder?.fair_value_points ?? null;

  return (
    <aside
      aria-label="Bucket inspector"
      className="fixed inset-y-0 right-0 z-40 flex w-full max-w-[560px] flex-col border-l border-line-strong bg-panel shadow-[-24px_0_48px_rgb(0_0_0/0.45)]"
    >
      <header className="flex items-start gap-3 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <p className="text-[11px] tracking-[0.08em] text-ink-3 uppercase">Bucket inspector · topic {topicId}</p>
          <h2 className="truncate text-base font-semibold">{ladder?.bucket_name ?? "Loading…"}</h2>
        </div>
        <Button tone="ghost" size="sm" onClick={onClose} aria-label="Close inspector">
          ✕
        </Button>
      </header>

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {error !== null && <p className="rounded-md border border-critical/50 bg-critical/10 px-3 py-2 text-xs text-critical-ink">✕ {error}</p>}

        {ladder !== null && (
          <>
            <div className="grid grid-cols-3 gap-3">
              <Stat label="Market quote" value={<span className="num">{formatPoints(ladder.market_price)} pts</span>} />
              <Stat label="Model probability" value={<span className="num">{formatProbability(ladder.model_probability, 2)}</span>} />
              <Stat label="Fair value (net of fees)" value={<span className="num">{formatPoints(ladder.fair_value_points)} pts</span>} />
              <Stat label="Risk budget" value={<span className="num">{formatSats(ladder.budget_millisats)}</span>} hint="15% bucket · 35% topic · cash" />
              <Stat
                label="Optimal size k*"
                value={<span className="num">{ladder.optimal_contracts === null ? "—" : formatContracts(ladder.optimal_contracts)}</span>}
              />
              <Stat
                label="Marginal edge < 0 at"
                value={
                  <span className="num">
                    {ladder.marginal_breakeven_contracts === null ? "never ≤ 50" : `k = ${formatContracts(ladder.marginal_breakeven_contracts)}`}
                  </span>
                }
              />
            </div>

            <section>
              <h3 className="mb-1 text-[11px] font-semibold tracking-[0.08em] text-ink-2 uppercase">LS-LMSR cost curve</h3>
              <p className="mb-2 text-xs text-ink-3">
                Average all-in price (cost + 2% commission) and the bucket&apos;s marginal price after buying k contracts,
                against the model&apos;s fair value. Past the red line the next contract loses money in expectation.
              </p>
              <LineChart
                ariaLabel="Average all-in price and marginal price versus contracts"
                xs={xs}
                height={190}
                xFormat={(x) => `k=${formatContracts(x)}`}
                yFormat={(y) => y.toFixed(2)}
                series={[
                  { key: "avg", label: "avg all-in", color: "var(--color-s1)", values: rows.map((r) => r.exec_price) },
                  {
                    key: "marginal",
                    label: "marginal × (1+fee)",
                    color: "var(--color-s2)",
                    values: rows.map((r) => r.marginal_price * (1 + feeRate)),
                  },
                ]}
                refs={[
                  ...(fair === null ? [] : [{ axis: "y" as const, value: fair, label: `fair ${fair.toFixed(2)}`, tone: "good" as const }]),
                  ...(ladder.marginal_breakeven_contracts === null
                    ? []
                    : [{ axis: "x" as const, value: ladder.marginal_breakeven_contracts, label: "edge < 0", tone: "critical" as const }]),
                ]}
              />
            </section>

            <section>
              <h3 className="mb-2 text-[11px] font-semibold tracking-[0.08em] text-ink-2 uppercase">Expected profit by size</h3>
              <LineChart
                ariaLabel="Expected profit in sats versus contracts"
                xs={xs}
                height={160}
                includeZero
                xFormat={(x) => `k=${formatContracts(x)}`}
                yFormat={(y) => `${Math.round(y)}`}
                series={[
                  {
                    key: "ep",
                    label: "expected profit (sats)",
                    color: "var(--color-s1)",
                    values: rows.map((r) => (r.within_budget ? toSats(r.expected_profit_millisats) : null)),
                  },
                ]}
                refs={
                  ladder.optimal_contracts === null
                    ? []
                    : [{ axis: "x", value: ladder.optimal_contracts, label: "k*", tone: "good" }]
                }
              />
              <p className="mt-1 text-[11px] text-ink-3">Sizes beyond the risk budget are omitted.</p>
            </section>

            <details className="rounded-md border border-line">
              <summary className="cursor-pointer px-3 py-2 text-xs text-ink-2">Full ladder (k = 1…{ladder.max_contracts})</summary>
              <div className="max-h-64 overflow-auto">
                <table className="w-full text-[11px]">
                  <thead className="sticky top-0 bg-raised text-ink-3">
                    <tr className="text-right">
                      <th className="px-2 py-1 font-medium">k</th>
                      <th className="px-2 font-medium">Cost</th>
                      <th className="px-2 font-medium">Fee</th>
                      <th className="px-2 font-medium">Avg</th>
                      <th className="px-2 font-medium">Net edge</th>
                      <th className="px-2 font-medium">EP</th>
                      <th className="px-2 font-medium">Marg. EP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr
                        key={row.contracts}
                        className={cx(
                          "num border-t border-line text-right",
                          !row.within_budget && "text-ink-3 opacity-60",
                          row.contracts === ladder.optimal_contracts && "bg-good/10",
                        )}
                      >
                        <td className="px-2 py-0.5">{formatContracts(row.contracts)}</td>
                        <td className="px-2">{formatSats(row.cost_millisats, { unit: false })}</td>
                        <td className="px-2">{formatSats(row.commission_millisats, { unit: false })}</td>
                        <td className="px-2">{formatPoints(row.exec_price)}</td>
                        <td className="px-2">{formatPoints(row.net_edge_points, { signed: true })}</td>
                        <td className="px-2">{formatSats(row.expected_profit_millisats, { unit: false, signed: true })}</td>
                        <td className={cx("px-2", row.marginal_profit_millisats < 0 && "text-critical-ink")}>
                          {formatSats(row.marginal_profit_millisats, { unit: false, signed: true })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </>
        )}
      </div>

      <footer className={cx("space-y-3 border-t px-4 py-3", live ? "border-critical/60 bg-critical/5" : "border-line")}>
        <div className="flex items-center gap-2">
          <h3 className="text-[11px] font-semibold tracking-[0.08em] text-ink-2 uppercase">Test order</h3>
          {live ? <Pill tone="critical">● LIVE — real sats</Pill> : <Pill tone="good">● PAPER</Pill>}
          <span className="ml-auto text-[11px] text-ink-3">RiskManager → ExecutionRouter → Ledger</span>
        </div>
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (canSend) void send();
          }}
        >
          <div className="w-28">
            <Field label="Contracts">
              <input
                className={`${inputClass} num`}
                inputMode="decimal"
                value={contracts}
                onChange={(event) => setContracts(event.target.value)}
                aria-invalid={!sizeValid}
              />
            </Field>
          </div>
          {live && (
            <div className="min-w-48 flex-1">
              <Field label={`Type ${LIVE_ORDER_PHRASE}`}>
                <input
                  className={`${inputClass} mono`}
                  value={phrase}
                  onChange={(event) => setPhrase(event.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                  placeholder={LIVE_ORDER_PHRASE}
                />
              </Field>
            </div>
          )}
          <Button type="submit" tone={live ? "critical" : "primary"} disabled={!canSend} className="ml-auto">
            {busy ? (queued ? "Queued…" : "Sending…") : live ? "Send LIVE order" : "Send paper order"}
          </Button>
        </form>
        {!running && <p className="text-[11px] text-ink-3">Start the engine to send test orders.</p>}
        {busy && queued && (
          <p role="status" className="text-[11px] text-ink-3">
            Passed the RiskManager pre-check; waiting for a slot in the Glimpse REST budget (
            {health?.rate_limit.limit_per_minute ?? 50}/min, shared with the engine) to get the venue&apos;s quote.
          </p>
        )}
        {result !== null && (
          <div
            className={cx(
              "rounded-md border px-3 py-2 text-xs",
              result.ok ? "border-good/50 bg-good/10 text-ink" : "border-warn/50 bg-warn/10 text-ink",
            )}
          >
            <p className="font-medium">
              {result.ok ? "✓ " : "! "}
              {result.message}
              {result.fill !== null && result.fill.exec_price !== null && ` at ${formatPoints(result.fill.exec_price)} pts all-in`}
            </p>
            {result.reasons.length > 0 && (
              <ul className="mt-1 list-disc pl-4 text-ink-2">
                {result.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </footer>
    </aside>
  );
}
