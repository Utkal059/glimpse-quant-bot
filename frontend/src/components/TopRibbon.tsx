"use client";

import { useEffect, useRef } from "react";

import type { HealthResponse, RiskMetrics, SpotTicker } from "@/lib/contract";
import { formatPercent, formatPrice } from "@/lib/format";
import type { ConnectionStatus } from "@/hooks/useTelemetrySocket";

import { KillSwitch } from "./KillSwitch";
import { Delta, Meter, Pill, cx } from "./ui";

const ASSET_LABEL: Record<string, string> = { BTC: "BTC", ETH: "ETH", SOL: "SOL", GOLD: "GOLD" };

function Ticker({ ticker }: { ticker: SpotTicker }) {
  return (
    <div className="flex min-w-0 items-baseline gap-2 px-3" title={ticker.sources.join(", ") || "no feed yet"}>
      <span className="text-[11px] font-semibold text-ink-3">{ASSET_LABEL[ticker.asset] ?? ticker.asset}</span>
      <span className="num text-sm font-semibold text-ink">{formatPrice(ticker.price)}</span>
      <Delta value={ticker.change_pct} className="text-[11px]">
        {formatPercent(ticker.change_pct === null ? null : Math.abs(ticker.change_pct))}
      </Delta>
    </div>
  );
}

function LinkStatus({ status, latencyMs }: { status: ConnectionStatus; latencyMs: number | null }) {
  const live = status === "open";
  const tone = !live ? "critical" : latencyMs === null ? "neutral" : latencyMs < 150 ? "good" : latencyMs < 500 ? "warn" : "serious";
  const dot: Record<typeof tone, string> = {
    good: "bg-good",
    warn: "bg-warn",
    serious: "bg-serious",
    critical: "bg-critical",
    neutral: "bg-ink-3",
  };
  return (
    <div className="flex items-center gap-2 px-3" title="WebSocket round-trip to the engine">
      <span className={cx("size-2 rounded-full", dot[tone], live && "animate-pulse")} aria-hidden />
      <span className="text-[11px] text-ink-3">WS</span>
      <span className="num text-xs font-medium text-ink-2">
        {live ? (latencyMs === null ? "…" : `${Math.round(latencyMs)} ms`) : status === "connecting" ? "connecting" : "offline"}
      </span>
    </div>
  );
}

export function SafetyPill({ health }: { health: HealthResponse | null }) {
  if (health === null) return <Pill tone="neutral">NO ENGINE</Pill>;
  if (health.mode === "LIVE_ARMED") {
    return (
      <Pill tone="critical" title="Real orders are being placed on Glimpse" className="animate-pulse">
        <span aria-hidden>●</span> LIVE ARMED
      </Pill>
    );
  }
  if (health.arming.pending) {
    return (
      <Pill tone="warn" title="An arming token is pending (expires in 60 s)">
        <span aria-hidden>◐</span> ARMING…
      </Pill>
    );
  }
  return (
    <Pill tone="good" title="Dry run: every order is simulated against the live book">
      <span aria-hidden>●</span> PAPER MODE
    </Pill>
  );
}

function DrawdownGauge({ risk }: { risk: RiskMetrics | null }) {
  if (risk === null) return null;
  const used = risk.drawdown_limit > 0 ? risk.intraday_drawdown / risk.drawdown_limit : 0;
  const tone = risk.breaker_tripped ? "critical" : used >= 0.66 ? "warn" : "s1";
  return (
    <div className="flex w-44 flex-col gap-1 px-3" title="Intraday drawdown from the UTC-day equity peak vs the breaker">
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="text-ink-3">Drawdown</span>
        <span className="num font-medium text-ink-2">
          {formatPercent(-risk.intraday_drawdown * 100, { digits: 2 })}
          <span className="text-ink-3"> / {formatPercent(-risk.drawdown_limit * 100, { digits: 0 })}</span>
        </span>
      </div>
      <Meter fraction={used} tone={tone} label="Intraday drawdown versus the breaker limit" />
    </div>
  );
}

function LockStatus({ health }: { health: HealthResponse | null }) {
  if (health === null) return null;
  if (health.kill_switch_engaged) return <Pill tone="critical">⏻ KILL SWITCH ENGAGED</Pill>;
  if (health.breaker_tripped)
    return (
      <Pill tone="critical" title={health.breaker_reason ?? undefined}>
        ✕ BREAKER TRIPPED
      </Pill>
    );
  if (health.trading_locked)
    return (
      <Pill tone="serious" title={health.lock_reason ?? undefined}>
        ! LOCKED
      </Pill>
    );
  return (
    <Pill tone="neutral" title="RiskManager accepts new entries">
      ✓ Entries open
    </Pill>
  );
}

function EngineBadge({ health }: { health: HealthResponse | null }) {
  const state = health?.engine_state ?? "stopped";
  const tone =
    state === "running" ? "info" : state === "error" || state === "halted" ? "critical" : state === "starting" ? "warn" : "neutral";
  return (
    <Pill tone={tone} title={health?.last_error ?? undefined}>
      {state.toUpperCase()}
      {health?.venue === "simulated" && <span className="font-normal text-ink-3">· SIM</span>}
    </Pill>
  );
}

export function TopRibbon({
  health,
  risk,
  tickers,
  connection,
  latencyMs,
}: {
  health: HealthResponse | null;
  risk: RiskMetrics | null;
  tickers: readonly SpotTicker[];
  connection: ConnectionStatus;
  latencyMs: number | null;
}) {
  const ref = useRef<HTMLElement>(null);
  // Publish the ribbon height so sticky columns sit just below it, however it wraps.
  useEffect(() => {
    const element = ref.current;
    if (element === null) return;
    const observer = new ResizeObserver(() => {
      document.documentElement.style.setProperty("--ribbon-h", `${element.offsetHeight}px`);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <header ref={ref} className="sticky top-0 z-30 border-b border-line bg-page/95 backdrop-blur">
      <div className="flex min-h-12 flex-wrap items-center gap-y-2 px-3 py-1.5">
        <div className="flex items-center gap-2 pr-3">
          <span aria-hidden className="grid size-7 place-items-center rounded-md bg-s1/15 text-sm font-black text-s1">
            G
          </span>
          <div className="leading-tight">
            <div className="text-sm font-semibold">Glimpse Quant</div>
            <div className="text-[10px] tracking-[0.14em] text-ink-3 uppercase">Terminal</div>
          </div>
        </div>
        <div className="flex min-w-0 flex-1 flex-wrap items-center divide-x divide-line border-x border-line">
          {tickers.map((ticker) => (
            <Ticker key={ticker.asset} ticker={ticker} />
          ))}
          {tickers.length === 0 && <span className="px-3 text-xs text-ink-3">waiting for spot feeds…</span>}
        </div>
        <div className="flex flex-wrap items-center gap-y-2 divide-x divide-line">
          <LinkStatus status={connection} latencyMs={latencyMs} />
          <div className="flex items-center gap-2 px-3">
            <SafetyPill health={health} />
            <EngineBadge health={health} />
            <LockStatus health={health} />
          </div>
          <DrawdownGauge risk={risk} />
          <div className="pl-3">
            <KillSwitch engaged={health?.kill_switch_engaged ?? false} compact />
          </div>
        </div>
      </div>
    </header>
  );
}
