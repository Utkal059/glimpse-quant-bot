"use client";

import { useState } from "react";

import { api, errorText } from "@/lib/api";
import { RELEASE_PHRASE } from "@/lib/contract";

import { useToast } from "./Toaster";
import { Button, Dialog, Field, inputClass } from "./ui";

/**
 * The kill switch is deliberately *easy* to engage (two clicks, Enter confirms)
 * and deliberately *hard* to release (typed phrase). Engaging cancels the
 * runner, locks the RiskManager and revokes any pending live arming.
 */
export function KillSwitch({ engaged, compact = false }: { engaged: boolean; compact?: boolean }) {
  const toast = useToast();
  const [confirming, setConfirming] = useState(false);
  const [releasing, setReleasing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [reason, setReason] = useState("");
  const [phrase, setPhrase] = useState("");

  async function engage() {
    setBusy(true);
    try {
      const result = await api.killSwitch(reason.trim() || "operator kill switch");
      toast.push("critical", "Kill switch engaged", result.message);
      setConfirming(false);
      setReason("");
    } catch (error) {
      toast.push("critical", "Kill switch failed", errorText(error));
    } finally {
      setBusy(false);
    }
  }

  async function release() {
    setBusy(true);
    try {
      const result = await api.releaseKillSwitch(phrase);
      toast.push("warn", "Kill switch released", result.message);
      setReleasing(false);
      setPhrase("");
    } catch (error) {
      toast.push("critical", "Release refused", errorText(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      {engaged ? (
        <Button tone="warn" size={compact ? "sm" : "md"} onClick={() => setReleasing(true)}>
          <span aria-hidden>⏻</span> Release kill switch…
        </Button>
      ) : (
        <Button
          tone="critical"
          size={compact ? "sm" : "md"}
          onClick={() => setConfirming(true)}
          className="font-bold tracking-wide"
          title="Stop the engine, lock trading and revoke live arming"
        >
          <span aria-hidden>⏻</span> KILL SWITCH
        </Button>
      )}

      <Dialog open={confirming} onClose={() => setConfirming(false)} title="Engage the kill switch?" tone="critical">
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            void engage();
          }}
        >
          <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
            <li>Cancels the trading runner immediately (no new orders).</li>
            <li>Locks the RiskManager and revokes any pending live arming.</li>
            <li>Open positions stay as they are; Glimpse fills instantly, so there are no resting orders.</li>
          </ul>
          <Field label="Reason (optional, written to the ledger)">
            <input
              className={inputClass}
              value={reason}
              maxLength={200}
              onChange={(event) => setReason(event.target.value)}
              placeholder="operator kill switch"
            />
          </Field>
          <div className="flex justify-end gap-2">
            <Button tone="ghost" onClick={() => setConfirming(false)}>
              Cancel
            </Button>
            <Button tone="critical" type="submit" disabled={busy} autoFocus className="font-bold">
              {busy ? "Engaging…" : "Engage kill switch"}
            </Button>
          </div>
        </form>
      </Dialog>

      <Dialog open={releasing} onClose={() => setReleasing(false)} title="Release the kill switch" tone="warn">
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (phrase === RELEASE_PHRASE) void release();
          }}
        >
          <p className="text-sm text-ink-2">
            Releasing does not restart anything: the engine stays stopped until you start it, in paper mode
            unless you arm live again.
          </p>
          <Field label={`Type ${RELEASE_PHRASE} to confirm`}>
            <input
              className={`${inputClass} mono`}
              value={phrase}
              onChange={(event) => setPhrase(event.target.value)}
              autoComplete="off"
              spellCheck={false}
              placeholder={RELEASE_PHRASE}
            />
          </Field>
          <div className="flex justify-end gap-2">
            <Button tone="ghost" onClick={() => setReleasing(false)}>
              Cancel
            </Button>
            <Button tone="warn" type="submit" disabled={busy || phrase !== RELEASE_PHRASE}>
              Release
            </Button>
          </div>
        </form>
      </Dialog>
    </>
  );
}
