"use client";

import type { DistributionResponse } from "@/lib/contract";
import { formatDuration, formatNumber, formatPrice } from "@/lib/format";

import { COMPONENTS } from "./DistributionChart";
import { EmptyState, Meter, Panel, Stat } from "./ui";

/** JSD at which κ hits its 0.25 floor: 1 - 2·JSD = 0.25. */
const JSD_FLOOR = 0.375;

function componentMeta(key: string) {
  return COMPONENTS.find((c) => c.key === key) ?? { key, label: key, colorVar: "--color-ink-3" };
}

export function EnsemblePanel({ dist, timeseriesModel }: { dist: DistributionResponse | null; timeseriesModel: string | null }) {
  const forecast = dist?.forecast ?? null;
  if (dist === null || forecast === null) {
    return (
      <Panel title="Ensemble">
        <EmptyState title="No forecast yet">Model agreement and Kelly scaling appear once the focused topic is forecast.</EmptyState>
      </Panel>
    );
  }
  const jsd = forecast.js_divergence;
  const kappa = forecast.kelly_multiplier_kappa;
  const tone = kappa >= 0.75 ? "s1" : kappa > 0.25 ? "warn" : "critical";
  const weights = Object.entries(forecast.weights).sort((a, b) => b[1] - a[1]);
  const age = Math.max(0, Date.now() / 1000 - forecast.ts);

  return (
    <Panel title="Ensemble" subtitle={`time-series leg: ${timeseriesModel ?? "—"}`}>
      <div className="flex items-end gap-4">
        <div>
          <div className="text-[11px] text-ink-3">Kelly multiplier κ</div>
          <div className="text-3xl leading-none font-semibold">{kappa.toFixed(2)}</div>
        </div>
        <p className="pb-0.5 text-xs text-ink-3">
          {kappa >= 0.999
            ? "Models agree: full fractional-Kelly size."
            : kappa <= 0.2501
              ? "Models disagree strongly: size floored at 25%."
              : `Model disagreement trims size to ${Math.round(kappa * 100)}%.`}
        </p>
      </div>

      <div className="mt-3 space-y-1">
        <div className="flex justify-between text-[11px]">
          <span className="text-ink-3">Jensen–Shannon divergence</span>
          <span className="num text-ink-2">{formatNumber(jsd, 3)}</span>
        </div>
        <Meter fraction={jsd / JSD_FLOOR} marks={[0.25 / JSD_FLOOR]} tone={tone} label="Jensen-Shannon divergence between ensemble components" />
        <div className="flex justify-between text-[10px] text-ink-3">
          <span>0 · κ=1</span>
          <span>0.25 · κ=0.5</span>
          <span>0.375 · κ floor</span>
        </div>
      </div>

      <div className="mt-4 space-y-2">
        <div className="text-[11px] text-ink-3">Component weights</div>
        {weights.map(([key, weight]) => {
          const meta = componentMeta(key);
          return (
            <div key={key} className="grid grid-cols-[96px_1fr_44px] items-center gap-2 text-xs">
              <span className="flex items-center gap-1.5 truncate text-ink-2">
                <span aria-hidden className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: `var(${meta.colorVar})` }} />
                {meta.label}
              </span>
              <div className="h-1.5 overflow-hidden rounded-full bg-hover">
                <div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(1, weight)) * 100}%`, background: `var(${meta.colorVar})` }} />
              </div>
              <span className="num text-right text-ink">{(weight * 100).toFixed(0)}%</span>
            </div>
          );
        })}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3 border-t border-line pt-3">
        <Stat label="Spot" value={<span className="num">{formatPrice(forecast.spot)}</span>} />
        <Stat label="σ (horizon)" value={<span className="num">{formatNumber(forecast.sigma * 100, 2)}%</span>} />
        <Stat label="Horizon" value={<span className="num">{formatDuration(forecast.horizon_seconds)}</span>} />
        <Stat label="Ambiguity ε" value={<span className="num">{formatNumber(forecast.ambiguity, 4)}</span>} />
        <Stat label="Forecast age" value={<span className="num">{formatDuration(age)}</span>} />
        <Stat label="Spot feeds" value={<span className="text-xs">{forecast.spot_sources.join(", ") || "—"}</span>} />
      </div>
    </Panel>
  );
}
