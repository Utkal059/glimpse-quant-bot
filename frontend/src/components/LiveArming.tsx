"use client";

import { useEffect, useState } from "react";

import { api, errorText } from "@/lib/api";
import type { HealthResponse, LiveEligibility } from "@/lib/contract";
import { ARM_PHRASE } from "@/lib/contract";

import { useToast } from "./Toaster";
import { Button, Dialog, Field, Pill, cx, inputClass } from "./ui";

type Step = "checklist" | "phrase" | "confirm";

const CHECKS: ReadonlyArray<{ key: keyof Omit<LiveEligibility, "eligible" | "blockers">; label: string; how: string }> = [
  { key: "server_flag", label: "Server flag", how: "GLIMPSE_LIVE_TRADING=true in the server's .env" },
  { key: "api_key_present", label: "API key", how: "GLIMPSE_API_KEY set on the server" },
  { key: "serve_allow_live", label: "Live allowed at launch", how: "server started with `serve --allow-live`" },
  { key: "real_venue", label: "Real venue", how: "not the simulated --demo venue" },
  { key: "kill_switch_clear", label: "Kill switch clear", how: "release the kill switch first" },
];

/**
 * Two-key live arming, as three explicit steps:
 *  1. every server-side key is turned (env flag, API key, --allow-live, real venue);
 *  2. the operator types the arming phrase -> the server issues a 60 s single-use token;
 *  3. the operator reviews the run parameters and starts LIVE with that token.
 * Closing the dialog at any point revokes the token.
 */
export function LiveArming({
  open,
  onClose,
  health,
  params,
}: {
  open: boolean;
  onClose: () => void;
  health: HealthResponse | null;
  params: { minExpectedGainSats: number; maxHorizonHours: number; assets: readonly string[] };
}) {
  const toast = useToast();
  const [step, setStep] = useState<Step>("checklist");
  const [phrase, setPhrase] = useState("");
  const [token, setToken] = useState<{ value: string; expiresAt: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [acknowledged, setAcknowledged] = useState(false);

  useEffect(() => {
    if (!open) return;
    setStep("checklist");
    setPhrase("");
    setToken(null);
    setAcknowledged(false);
  }, [open]);

  useEffect(() => {
    if (token === null) return;
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 250);
    return () => window.clearInterval(timer);
  }, [token]);

  const eligibility = health?.live_eligibility ?? null;
  const remaining = token === null ? 0 : Math.max(0, token.expiresAt - now);
  const expired = token !== null && remaining <= 0;

  async function close() {
    if (token !== null) {
      try {
        await api.disarm();
      } catch {
        // the token expires on its own within 60 s
      }
    }
    onClose();
  }

  async function arm() {
    setBusy(true);
    try {
      const response = await api.arm(phrase);
      setToken({ value: response.arming_token, expiresAt: response.expires_at });
      setNow(Date.now() / 1000);
      setStep("confirm");
    } catch (error) {
      toast.push("critical", "Arming refused", errorText(error));
    } finally {
      setBusy(false);
    }
  }

  async function startLive() {
    if (token === null) return;
    setBusy(true);
    try {
      const result = await api.start({
        dry_run: false,
        arming_token: token.value,
        min_expected_gain_sats: params.minExpectedGainSats,
        max_horizon_hours: params.maxHorizonHours,
      });
      toast.push("warn", "LIVE trading started", result.message);
      setToken(null);
      onClose();
    } catch (error) {
      toast.push("critical", "Live start refused", errorText(error));
      setToken(null);
      setStep("phrase");
      setPhrase("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onClose={() => void close()} title="Arm live trading" tone="warn" width="max-w-xl">
      <ol className="mb-4 flex items-center gap-2 text-[11px] text-ink-3" aria-label="Arming steps">
        {(["checklist", "phrase", "confirm"] as const).map((name, index) => (
          <li key={name} className="flex items-center gap-2">
            <span
              className={cx(
                "grid size-5 place-items-center rounded-full border text-[10px] font-bold",
                step === name ? "border-warn bg-warn/20 text-warn" : "border-line-strong",
              )}
            >
              {index + 1}
            </span>
            <span className={step === name ? "text-ink" : undefined}>
              {name === "checklist" ? "Server keys" : name === "phrase" ? "Typed arming" : "Review & start"}
            </span>
            {index < 2 && <span aria-hidden>→</span>}
          </li>
        ))}
      </ol>

      {step === "checklist" && (
        <div className="flex flex-col gap-4">
          <p className="text-sm text-ink-2">
            Live orders spend real sats. Every key below is enforced on the server; the UI cannot bypass any of
            them.
          </p>
          <ul className="divide-y divide-line rounded-lg border border-line">
            {CHECKS.map((check) => {
              const ok = eligibility?.[check.key] ?? false;
              return (
                <li key={check.key} className="flex items-center gap-3 px-3 py-2">
                  <span
                    aria-hidden
                    className={cx(
                      "grid size-5 place-items-center rounded-full text-[11px] font-bold",
                      ok ? "bg-good/20 text-good-ink" : "bg-critical/20 text-critical-ink",
                    )}
                  >
                    {ok ? "✓" : "✕"}
                  </span>
                  <span className="flex-1 text-sm">{check.label}</span>
                  <span className={cx("text-xs", ok ? "text-ink-3" : "text-critical-ink")}>
                    {ok ? "ok" : check.how}
                  </span>
                </li>
              );
            })}
          </ul>
          <label className="flex items-start gap-2 text-sm text-ink-2">
            <input
              type="checkbox"
              className="mt-0.5 accent-[var(--color-warn)]"
              checked={acknowledged}
              onChange={(event) => setAcknowledged(event.target.checked)}
            />
            I understand live mode places real orders on Glimpse, still capped at 15% of equity per bucket, 35%
            per topic and a −6% intraday drawdown breaker.
          </label>
          <div className="flex justify-end gap-2">
            <Button tone="ghost" onClick={() => void close()}>
              Cancel
            </Button>
            <Button
              tone="warn"
              disabled={eligibility?.eligible !== true || !acknowledged}
              onClick={() => setStep("phrase")}
            >
              Continue
            </Button>
          </div>
        </div>
      )}

      {step === "phrase" && (
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (phrase === ARM_PHRASE) void arm();
          }}
        >
          <p className="text-sm text-ink-2">
            The server answers with a single-use arming token that expires after 60 seconds.
          </p>
          <Field label={`Type ${ARM_PHRASE}`}>
            <input
              className={`${inputClass} mono`}
              value={phrase}
              onChange={(event) => setPhrase(event.target.value)}
              autoComplete="off"
              spellCheck={false}
              placeholder={ARM_PHRASE}
              autoFocus
            />
          </Field>
          <div className="flex justify-end gap-2">
            <Button tone="ghost" onClick={() => setStep("checklist")}>
              Back
            </Button>
            <Button tone="warn" type="submit" disabled={busy || phrase !== ARM_PHRASE}>
              {busy ? "Arming…" : "Request arming token"}
            </Button>
          </div>
        </form>
      )}

      {step === "confirm" && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <Pill tone={expired ? "critical" : "warn"}>
              {expired ? "token expired" : `token valid ${Math.ceil(remaining)}s`}
            </Pill>
            <span className="text-xs text-ink-3">single use · revoked if you close this dialog</span>
          </div>
          <dl className="grid grid-cols-3 gap-3 rounded-lg border border-line bg-page p-3 text-sm">
            <div>
              <dt className="text-[11px] text-ink-3">Min expected gain</dt>
              <dd className="num font-semibold">{params.minExpectedGainSats} sats</dd>
            </div>
            <div>
              <dt className="text-[11px] text-ink-3">Max horizon</dt>
              <dd className="num font-semibold">{params.maxHorizonHours} h</dd>
            </div>
            <div>
              <dt className="text-[11px] text-ink-3">Assets</dt>
              <dd className="font-semibold">{params.assets.join(" · ")}</dd>
            </div>
          </dl>
          <div className="flex justify-end gap-2">
            <Button tone="ghost" onClick={() => void close()}>
              Abort
            </Button>
            <Button tone="critical" disabled={busy || expired} onClick={() => void startLive()} className="font-bold">
              {busy ? "Starting…" : "Start LIVE engine"}
            </Button>
          </div>
        </div>
      )}
    </Dialog>
  );
}
