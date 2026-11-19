"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { API_URL, WS_URL } from "@/lib/config";
import type { ModeName } from "@/lib/contract";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { useTelemetrySocket } from "@/hooks/useTelemetrySocket";

import type { RunParams } from "./BotControls";
import { BotControls } from "./BotControls";
import { DistributionPanel } from "./DistributionPanel";
import { EnsemblePanel } from "./EnsemblePanel";
import { ExposurePanel } from "./ExposurePanel";
import { FillsTable } from "./FillsTable";
import { LadderDrawer } from "./LadderDrawer";
import { LiveArming } from "./LiveArming";
import { LogStream } from "./LogStream";
import { MarketScreener } from "./MarketScreener";
import { PerformancePanel } from "./PerformancePanel";
import { PositionsTable } from "./PositionsTable";
import { ToastProvider } from "./Toaster";
import { TopRibbon } from "./TopRibbon";
import { Button, cx } from "./ui";

type Tab = "positions" | "fills" | "performance" | "logs";

const TOPIC_KEY = "gqt.focusTopic";

function readStoredTopic(): number | null {
  try {
    const raw = window.localStorage.getItem(TOPIC_KEY);
    const value = raw === null ? Number.NaN : Number(raw);
    return Number.isInteger(value) && value > 0 ? value : null;
  } catch {
    return null;
  }
}

function storeTopic(topicId: number | null) {
  try {
    if (topicId === null) window.localStorage.removeItem(TOPIC_KEY);
    else window.localStorage.setItem(TOPIC_KEY, String(topicId));
  } catch {
    // storage unavailable (private mode): focus just isn't remembered
  }
}

function TerminalInner() {
  const [focusTopic, setFocusTopic] = useState<number | null>(null);
  const [inspect, setInspect] = useState<{ topicId: number; optionId: number } | null>(null);
  const [armingOpen, setArmingOpen] = useState(false);
  const [tab, setTab] = useState<Tab>("positions");
  const [params, setParams] = useState<RunParams>({ minExpectedGainSats: 25, maxHorizonHours: 30, assets: ["BTC", "ETH", "SOL", "GOLD"] });
  const paramsSeeded = useRef(false);

  const { state, reconnect } = useTelemetrySocket(WS_URL, focusTopic);
  const { health, markets, risk, positions, fills, logs, distribution, connection, latencyMs } = state;

  useEffect(() => {
    setFocusTopic(readStoredTopic());
  }, []);

  // Seed the editable run parameters from the server once.
  useEffect(() => {
    if (health === null || paramsSeeded.current) return;
    paramsSeeded.current = true;
    setParams({
      minExpectedGainSats: health.min_expected_gain_sats,
      maxHorizonHours: health.max_horizon_hours,
      assets: health.assets.length > 0 ? health.assets : ["BTC", "ETH", "SOL", "GOLD"],
    });
  }, [health]);

  // Keep the focus on a live topic: default to the richest edge.
  const topics = markets?.topics;
  useEffect(() => {
    if (topics === undefined || topics.length === 0) return;
    if (focusTopic !== null && topics.some((t) => t.topic_id === focusTopic)) return;
    const best = [...topics].sort(
      (a, b) => (b.max_extractable_edge_millisats ?? -1) - (a.max_extractable_edge_millisats ?? -1),
    )[0];
    if (best !== undefined) setFocusTopic(best.topic_id);
  }, [topics, focusTopic]);

  const selectTopic = useCallback((topicId: number) => {
    setFocusTopic(topicId);
    storeTopic(topicId);
    setInspect(null);
  }, []);

  const summary = useMemo(
    () => topics?.find((topic) => topic.topic_id === focusTopic) ?? null,
    [topics, focusTopic],
  );
  const dist = distribution !== null && distribution.topic_id === focusTopic ? distribution : null;
  const mode: ModeName = risk?.mode ?? (health?.dry_run === false ? "live" : "dry_run");
  const riskLocks = logs.reduce((n, entry) => (entry.category === "RISK_LOCK" ? n + 1 : n), 0);

  const wide = useMediaQuery("(min-width: 1536px)");
  const sidePanels = (
    <>
      <BotControls health={health} params={params} onParamsChange={setParams} onArmLive={() => setArmingOpen(true)} />
      <EnsemblePanel dist={dist} timeseriesModel={health?.timeseries_model ?? null} />
      <ExposurePanel risk={risk} />
    </>
  );

  const tabs: ReadonlyArray<{ key: Tab; label: string; badge?: number }> = [
    { key: "positions", label: "Positions", badge: positions.length },
    { key: "fills", label: "Fills & ledger" },
    { key: "performance", label: "Performance" },
    { key: "logs", label: "Logs", badge: riskLocks },
  ];

  return (
    <div className="flex min-h-dvh flex-col">
      <TopRibbon health={health} risk={risk} tickers={markets?.tickers ?? []} connection={connection} latencyMs={latencyMs} />

      {connection !== "open" && (
        <div role="alert" className="flex flex-wrap items-center gap-3 border-b border-warn/40 bg-warn/10 px-4 py-2 text-sm">
          <span className="font-semibold text-warn">! {connection === "connecting" ? "Connecting to the engine…" : "Engine unreachable — retrying"}</span>
          <span className="text-ink-2">
            Expecting <code className="mono text-xs">{WS_URL}</code>. Start it with{" "}
            <code className="mono rounded bg-raised px-1.5 py-0.5 text-xs">glimpse-quant-bot serve --demo</code> (simulated) or{" "}
            <code className="mono rounded bg-raised px-1.5 py-0.5 text-xs">glimpse-quant-bot serve</code> (live feed, paper trading).
          </span>
          <Button size="sm" tone="warn" className="ml-auto" onClick={reconnect}>
            Retry now
          </Button>
        </div>
      )}

      {health?.mode === "LIVE_ARMED" && (
        <div role="alert" className="border-b border-critical/60 bg-critical/15 px-4 py-1.5 text-center text-xs font-semibold tracking-wide text-critical-ink">
          ● LIVE TRADING — orders spend real sats on Glimpse. The kill switch is in the top-right corner.
        </div>
      )}

      <main className="grid flex-1 items-start gap-3 p-3 lg:grid-cols-[300px_minmax(0,1fr)] 2xl:grid-cols-[300px_minmax(0,1fr)_340px]">
        <div className="flex h-[440px] min-w-0 flex-col lg:sticky lg:top-[calc(var(--ribbon-h,56px)+12px)] lg:h-[calc(100dvh-var(--ribbon-h,56px)-24px)]">
          <MarketScreener topics={topics ?? []} selectedTopicId={focusTopic} onSelect={selectTopic} />
        </div>

        <div className="flex min-w-0 flex-col gap-3">
          <DistributionPanel
            dist={dist}
            summary={summary}
            selectedOptionId={inspect?.topicId === focusTopic ? inspect.optionId : null}
            onSelectBucket={(optionId) => focusTopic !== null && setInspect({ topicId: focusTopic, optionId })}
          />

          {!wide && <div className="grid min-w-0 items-start gap-3 md:grid-cols-2 xl:grid-cols-3">{sidePanels}</div>}

        <section className="min-w-0 rounded-lg border border-line bg-panel">
          <div role="tablist" aria-label="Portfolio" className="flex items-center gap-1 border-b border-line px-2">
            {tabs.map((item) => (
              <button
                key={item.key}
                type="button"
                role="tab"
                aria-selected={tab === item.key}
                onClick={() => setTab(item.key)}
                className={cx(
                  "relative inline-flex h-10 items-center gap-1.5 px-3 text-xs font-medium transition-colors",
                  tab === item.key ? "text-ink after:absolute after:inset-x-2 after:bottom-0 after:h-0.5 after:rounded-full after:bg-s1" : "text-ink-3 hover:text-ink-2",
                )}
              >
                {item.label}
                {item.badge !== undefined && item.badge > 0 && (
                  <span className={cx("num rounded-full px-1.5 text-[10px]", item.key === "logs" ? "bg-critical/25 text-critical-ink" : "bg-raised text-ink-2")}>
                    {item.badge}
                  </span>
                )}
              </button>
            ))}
            <span className="ml-auto pr-2 text-[11px] text-ink-3">{mode === "live" ? "live ledger" : "paper ledger"} · API {API_URL}</span>
          </div>
          <div role="tabpanel">
            {tab === "positions" && <PositionsTable positions={positions} onInspect={(topicId, optionId) => { selectTopic(topicId); setInspect({ topicId, optionId }); }} />}
            {tab === "fills" && <FillsTable live={fills} mode={mode} />}
            {tab === "performance" && <PerformancePanel mode={mode} />}
            {tab === "logs" && <LogStream logs={logs} />}
          </div>
        </section>
        </div>

        {wide && <div className="flex min-w-0 flex-col gap-3">{sidePanels}</div>}
      </main>

      {inspect !== null && (
        <LadderDrawer
          key={`${inspect.topicId}-${inspect.optionId}`}
          topicId={inspect.topicId}
          optionId={inspect.optionId}
          feeRate={dist?.fee_rate ?? 0.02}
          health={health}
          onClose={() => setInspect(null)}
        />
      )}

      <LiveArming open={armingOpen} onClose={() => setArmingOpen(false)} health={health} params={params} />
    </div>
  );
}

export function Terminal() {
  return (
    <ToastProvider>
      <TerminalInner />
    </ToastProvider>
  );
}
