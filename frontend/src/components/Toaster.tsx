"use client";

import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";

import { cx } from "./ui";

export type ToastTone = "info" | "good" | "warn" | "critical";

interface Toast {
  readonly id: number;
  readonly tone: ToastTone;
  readonly title: string;
  readonly detail?: string;
}

interface ToastApi {
  readonly push: (tone: ToastTone, title: string, detail?: string) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

const ICONS: Record<ToastTone, string> = { info: "ℹ", good: "✓", warn: "!", critical: "✕" };

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<readonly Toast[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (tone: ToastTone, title: string, detail?: string) => {
      const id = nextId.current++;
      const toast: Toast = detail === undefined ? { id, tone, title } : { id, tone, title, detail };
      setToasts((current) => [...current.slice(-3), toast]);
      window.setTimeout(() => dismiss(id), tone === "critical" ? 9000 : 5000);
    },
    [dismiss],
  );

  const api = useMemo<ToastApi>(() => ({ push }), [push]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-[min(380px,calc(100%-2rem))] flex-col gap-2"
      >
        {toasts.map((toast) => (
          <div
            key={toast.id}
            role="status"
            className={cx(
              "pointer-events-auto flex gap-3 rounded-lg border bg-raised px-3.5 py-3 shadow-xl",
              toast.tone === "good" && "border-good/50",
              toast.tone === "warn" && "border-warn/50",
              toast.tone === "critical" && "border-critical/60",
              toast.tone === "info" && "border-line-strong",
            )}
          >
            <span
              aria-hidden
              className={cx(
                "mt-0.5 grid size-5 shrink-0 place-items-center rounded-full text-[11px] font-bold",
                toast.tone === "good" && "bg-good/20 text-good-ink",
                toast.tone === "warn" && "bg-warn/20 text-warn",
                toast.tone === "critical" && "bg-critical/25 text-critical-ink",
                toast.tone === "info" && "bg-s1/20 text-ink",
              )}
            >
              {ICONS[toast.tone]}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-ink">{toast.title}</p>
              {toast.detail !== undefined && (
                <p className="mt-0.5 text-xs break-words text-ink-2">{toast.detail}</p>
              )}
            </div>
            <button
              type="button"
              aria-label="Dismiss"
              onClick={() => dismiss(toast.id)}
              className="h-5 text-ink-3 hover:text-ink"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const api = useContext(ToastContext);
  if (api === null) throw new Error("useToast must be used inside <ToastProvider>");
  return api;
}
