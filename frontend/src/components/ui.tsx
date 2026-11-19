"use client";

import type { ButtonHTMLAttributes, ReactNode } from "react";
import { useEffect, useId, useRef } from "react";

import { MINUS } from "@/lib/format";

export function cx(...parts: ReadonlyArray<string | false | null | undefined>): string {
  return parts.filter((part): part is string => typeof part === "string" && part.length > 0).join(" ");
}

// --- layout ---------------------------------------------------------------------

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={cx(
        "flex min-w-0 flex-col rounded-lg border border-line bg-panel shadow-[0_1px_0_rgb(255_255_255/0.03)_inset]",
        className,
      )}
    >
      {title !== undefined && (
        <header className="flex min-h-10 items-center gap-3 border-b border-line px-3 py-2">
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[11px] font-semibold tracking-[0.08em] text-ink-2 uppercase">
              {title}
            </h2>
            {subtitle !== undefined && (
              <p className="truncate text-xs text-ink-3">{subtitle}</p>
            )}
          </div>
          {actions !== undefined && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cx("min-h-0 flex-1", bodyClassName ?? "p-3")}>{children}</div>
    </section>
  );
}

// --- controls -------------------------------------------------------------------

type Tone = "neutral" | "primary" | "good" | "warn" | "critical" | "ghost";

const BUTTON_TONES: Record<Tone, string> = {
  neutral: "border-line-strong bg-raised text-ink hover:bg-hover",
  primary: "border-s1/60 bg-s1/15 text-ink hover:bg-s1/25",
  good: "border-good/60 bg-good/15 text-good-ink hover:bg-good/25",
  warn: "border-warn/60 bg-warn/15 text-warn hover:bg-warn/25",
  critical: "border-critical/70 bg-critical/20 text-critical-ink hover:bg-critical/30",
  ghost: "border-transparent bg-transparent text-ink-2 hover:bg-hover hover:text-ink",
};

export function Button({
  tone = "neutral",
  size = "md",
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { tone?: Tone; size?: "sm" | "md" }) {
  return (
    <button
      type="button"
      {...props}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md border font-medium whitespace-nowrap transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-45",
        size === "sm" ? "h-7 px-2.5 text-xs" : "h-8 px-3 text-sm",
        BUTTON_TONES[tone],
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Pill({
  tone,
  children,
  title,
  className,
}: {
  tone: "neutral" | "good" | "warn" | "serious" | "critical" | "info";
  children: ReactNode;
  title?: string;
  className?: string;
}) {
  const tones: Record<typeof tone, string> = {
    neutral: "border-line-strong text-ink-2",
    good: "border-good/50 bg-good/12 text-good-ink",
    warn: "border-warn/50 bg-warn/12 text-warn",
    serious: "border-serious/50 bg-serious/12 text-serious",
    critical: "border-critical/60 bg-critical/18 text-critical-ink",
    info: "border-s1/50 bg-s1/12 text-ink",
  };
  return (
    <span
      title={title}
      className={cx(
        "inline-flex h-6 items-center gap-1.5 rounded-full border px-2.5 text-[11px] font-semibold tracking-wide whitespace-nowrap",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/** A signed value that never relies on colour alone: sign + arrow + colour. */
export function Delta({
  value,
  children,
  className,
}: {
  value: number | null | undefined;
  children: ReactNode;
  className?: string;
}) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return <span className={cx("text-ink-3", className)}>—</span>;
  }
  const up = value > 0;
  const down = value < 0;
  return (
    <span
      className={cx(
        "num inline-flex items-center gap-1",
        up && "text-good-ink",
        down && "text-critical-ink",
        !up && !down && "text-ink-2",
        className,
      )}
    >
      <span aria-hidden className="text-[0.7em]">
        {up ? "▲" : down ? "▼" : "■"}
      </span>
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  hint,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx("min-w-0", className)}>
      <div className="truncate text-[11px] text-ink-3">{label}</div>
      <div className="truncate text-sm font-semibold text-ink">{value}</div>
      {hint !== undefined && <div className="truncate text-[11px] text-ink-3">{hint}</div>}
    </div>
  );
}

/** Horizontal meter. `fraction` 0..1 of the track; `marks` are reference ticks. */
export function Meter({
  fraction,
  tone,
  marks = [],
  label,
  height = 6,
}: {
  fraction: number;
  tone: "s1" | "good" | "warn" | "critical" | "s2";
  marks?: readonly number[];
  label: string;
  height?: number;
}) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(fraction) ? fraction : 0));
  const fills: Record<typeof tone, string> = {
    s1: "bg-s1",
    s2: "bg-s2",
    good: "bg-good",
    warn: "bg-warn",
    critical: "bg-critical",
  };
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped * 100)}
      className="relative w-full overflow-hidden rounded-full bg-hover"
      style={{ height }}
    >
      <div
        className={cx("h-full rounded-full transition-[width] duration-500", fills[tone])}
        style={{ width: `${clamped * 100}%` }}
      />
      {marks.map((mark) => (
        <span
          key={mark}
          aria-hidden
          className="absolute top-0 h-full w-px bg-ink-2/70"
          style={{ left: `${Math.max(0, Math.min(1, mark)) * 100}%` }}
        />
      ))}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: ReadonlyArray<{ value: T; label: ReactNode; title?: string }>;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-md border border-line-strong bg-page p-0.5">
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={option.title}
            onClick={() => onChange(option.value)}
            className={cx(
              "h-6 rounded px-2 text-xs whitespace-nowrap transition-colors",
              active ? "bg-raised font-semibold text-ink" : "text-ink-3 hover:text-ink",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  children,
  swatch,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  children: ReactNode;
  swatch?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={cx(
        "inline-flex h-6 items-center gap-1.5 rounded-md border px-2 text-xs transition-colors",
        checked ? "border-line-strong bg-raised text-ink" : "border-line text-ink-3 hover:text-ink-2",
      )}
    >
      {swatch !== undefined && (
        <span
          aria-hidden
          className="h-0.5 w-3 rounded-full"
          style={{ background: swatch, opacity: checked ? 1 : 0.4 }}
        />
      )}
      {children}
    </button>
  );
}

// --- dialogs ----------------------------------------------------------------------

export function Dialog({
  open,
  onClose,
  title,
  tone = "neutral",
  children,
  width = "max-w-lg",
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  tone?: "neutral" | "warn" | "critical";
  children: ReactNode;
  width?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (dialog === null) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  const accent =
    tone === "critical" ? "border-critical/70" : tone === "warn" ? "border-warn/60" : "border-line-strong";

  return (
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      onClose={onClose}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      className={cx(
        "m-auto w-[calc(100%-2rem)] rounded-xl border bg-panel p-0 text-ink shadow-2xl",
        width,
        accent,
      )}
    >
      {open && (
        <div className="flex flex-col">
          <header className="flex items-center gap-3 border-b border-line px-5 py-3.5">
            <h2 id={titleId} className="flex-1 text-sm font-semibold">
              {title}
            </h2>
            <button
              type="button"
              aria-label="Close"
              onClick={onClose}
              className="grid size-7 place-items-center rounded-md text-ink-3 hover:bg-hover hover:text-ink"
            >
              ✕
            </button>
          </header>
          <div className="px-5 py-4">{children}</div>
        </div>
      )}
    </dialog>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] font-medium text-ink-3">{label}</span>
      {children}
      {hint !== undefined && <span className="text-[11px] text-ink-3">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "h-8 w-full rounded-md border border-line-strong bg-page px-2.5 text-sm text-ink placeholder:text-ink-3 focus:border-s1 focus:outline-none";

export function EmptyState({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex h-full min-h-28 flex-col items-center justify-center gap-1.5 px-4 py-6 text-center">
      <p className="text-sm font-medium text-ink-2">{title}</p>
      {children !== undefined && <div className="max-w-md text-xs text-ink-3">{children}</div>}
    </div>
  );
}

export function signedText(value: number, text: string): string {
  if (value > 0) return `+${text}`;
  if (value < 0) return `${MINUS}${text}`;
  return text;
}
