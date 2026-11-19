"use client";

import { useState } from "react";

import { api, errorText } from "@/lib/api";
import type { AssetType, HealthResponse } from "@/lib/contract";
import { ASSET_TYPE_VALUES } from "@/lib/contract";
import { formatDuration } from "@/lib/format";

import { useToast } from "./Toaster";
import { Button, Field, Meter, Panel, Pill, Stat, cx, inputClass } from "./ui";

export interface RunParams {
  readonly minExpectedGainSats: number;
  readonly maxHorizonHours: number;
  readonly assets: readonly AssetType[];
}

export function BotControls({
  health,
  params,
  onParamsChange,
  onArmLive,
}: {
  health: HealthResponse | null;
  params: RunParams;
  onParamsChange: (params: RunParams) => void;
  onArmLive: () => void;
}) {
  const toast = useToast();
  const [busy, setBusy] = useState<string | null>(null);

  const state = health?.engine_state ?? "stopped";
  const active = state === "running" || state === "starting";
  const halted = health?.kill_switch_engaged ?? false;
  const eligibility = health?.live_eligibility ?? null;

  async function run(label: string, action: () => Promise<{ message: string }>, tone: "good" | "info" = "info") {
    setBusy(label);
    try {
      const result = await action();
      toast.push(tone, label, result.message);
    } catch (error) {
      toast.push("critical", `${label} failed`, errorText(error));
    } finally {
      setBusy(null);
    }
  }

  const gainValid = Number.isFinite(params.minExpectedGainSats) && params.minExpectedGainSats >= 0 && params.minExpectedGainSats <= 1_000_000;
  const horizonValid = Number.isFinite(params.maxHorizonHours) && params.maxHorizonHours > 0 && params.maxHorizonHours <= 720;

  function toggleAsset(asset: AssetType) {
    const has = params.assets.includes(asset);
    if (has && params.assets.length === 1) return;
    onParamsChange({
      ...params,
      assets: has ? params.assets.filter((a) => a !== asset) : [...params.assets, asset],
    });
  }

  const rate = health?.rate_limit ?? null;

  return (
    <Panel
      title="Engine"
      subtitle={health === null ? "not connected" : `${health.venue} venue · v${health.version}`}
      actions={
        <Pill tone={active ? "info" : state === "error" || state === "halted" ? "critical" : "neutral"}>{state.toUpperCase()}</Pill>
      }
    >
      <div className="grid grid-cols-3 gap-3">
        <Stat label="Uptime" value={<span className="num">{health?.started_at === null || health === null ? "—" : formatDuration(health.server_time - health.started_at)}</span>} />
        <Stat label="Fills" value={<span className="num">{health?.fills ?? "—"}</span>} />
        <Stat label="Topics" value={<span className="num">{health?.topics ?? "—"}</span>} />
      </div>

      {rate !== null && (
        <div className="mt-3 space-y-1">
          <div className="flex justify-between text-[11px]">
            <span className="text-ink-3">Glimpse REST budget</span>
            <span className="num text-ink-2">
              {rate.used_last_minute} / {rate.limit_per_minute} per min
            </span>
          </div>
          <Meter
            fraction={rate.used_last_minute / Math.max(1, rate.limit_per_minute)}
            tone={rate.remaining <= 5 ? "warn" : "s1"}
            label="Glimpse requests used in the last minute"
            height={5}
          />
          <p className="text-[10px] text-ink-3">The terminal reads the engine&apos;s memory; refreshing this page never spends this budget.</p>
        </div>
      )}

      {health?.last_error !== null && health?.last_error !== undefined && (
        <p className="mt-3 rounded-md border border-critical/50 bg-critical/10 px-2.5 py-1.5 text-xs break-words text-critical-ink">
          ✕ {health.last_error}
        </p>
      )}

      <fieldset className="mt-4 space-y-3 border-t border-line pt-3" disabled={active}>
        <legend className="sr-only">Run parameters</legend>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Min expected gain (sats)">
            <input
              className={cx(inputClass, "num", !gainValid && "border-critical")}
              inputMode="decimal"
              value={Number.isFinite(params.minExpectedGainSats) ? String(params.minExpectedGainSats) : ""}
              onChange={(event) => onParamsChange({ ...params, minExpectedGainSats: Number(event.target.value) })}
            />
          </Field>
          <Field label="Max horizon (hours)">
            <input
              className={cx(inputClass, "num", !horizonValid && "border-critical")}
              inputMode="decimal"
              value={Number.isFinite(params.maxHorizonHours) ? String(params.maxHorizonHours) : ""}
              onChange={(event) => onParamsChange({ ...params, maxHorizonHours: Number(event.target.value) })}
            />
          </Field>
        </div>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Assets to trade">
          {ASSET_TYPE_VALUES.map((asset) => {
            const on = params.assets.includes(asset);
            return (
              <button
                key={asset}
                type="button"
                aria-pressed={on}
                onClick={() => toggleAsset(asset)}
                className={cx(
                  "h-6 rounded-md border px-2 text-[11px] font-semibold disabled:opacity-50",
                  on ? "border-s1/60 bg-s1/15 text-ink" : "border-line text-ink-3",
                )}
              >
                {asset}
              </button>
            );
          })}
        </div>
        <p className="text-[10px] text-ink-3">
          Only these three knobs are adjustable. Risk caps (15% bucket, 35% topic, −6% drawdown) are server-side and
          cannot be changed from the UI.
        </p>
      </fieldset>

      <div className="mt-3 flex flex-wrap gap-2">
        {active ? (
          <Button tone="neutral" disabled={busy !== null} onClick={() => void run("Engine stopped", () => api.stop())}>
            ■ Stop
          </Button>
        ) : (
          <Button
            tone="good"
            disabled={busy !== null || halted || !gainValid || !horizonValid || health === null}
            onClick={() =>
              void run(
                "Paper engine started",
                () =>
                  api.start({
                    dry_run: true,
                    min_expected_gain_sats: params.minExpectedGainSats,
                    max_horizon_hours: params.maxHorizonHours,
                    assets: [...params.assets],
                  }),
                "good",
              )
            }
          >
            ▶ Start paper
          </Button>
        )}
        <Button
          tone="neutral"
          disabled={busy !== null || health === null}
          onClick={() =>
            void run("Reconciled", async () => {
              const result = await api.reconcile();
              const drift = result.discrepancies.length;
              return { message: `${result.note}${drift > 0 ? ` (${drift} discrepancies)` : ""}` };
            })
          }
        >
          ⟳ Reconcile
        </Button>
        <Button
          tone="warn"
          className="ml-auto"
          disabled={active || halted || health === null}
          onClick={onArmLive}
          title={
            eligibility === null || eligibility.eligible
              ? "Two-key live arming"
              : `Live trading unavailable: ${eligibility.blockers.join("; ")}`
          }
        >
          {eligibility?.eligible === true ? "Arm live…" : "Live: locked"}
        </Button>
      </div>
    </Panel>
  );
}
