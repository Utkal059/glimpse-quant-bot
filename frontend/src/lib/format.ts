import type { Millisats, PriceUnits, Probability } from "./units";
import { toSats } from "./units";

export const MINUS = "−";

const intFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const compactFormat = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function withSign(text: string, value: number, always: boolean): string {
  if (value < 0) return `${MINUS}${text}`;
  if (always && value > 0) return `+${text}`;
  return text;
}

/**
 * "12,345 sats". Amounts under 100 sats keep one decimal ("13.1 sats"): three
 * decimals ("13.123") read as thirteen thousand under a European locale.
 */
export function formatSats(
  value: Millisats | null | undefined,
  opts: { signed?: boolean; unit?: boolean; compact?: boolean } = {},
): string {
  if (value === null || value === undefined) return "—";
  const sats = toSats(value);
  const magnitude = Math.abs(sats);
  let text: string;
  if (opts.compact === true && magnitude >= 10_000) text = compactFormat.format(magnitude);
  else if (magnitude >= 100 || Number.isInteger(magnitude)) text = intFormat.format(magnitude);
  else text = magnitude.toLocaleString("en-US", { maximumFractionDigits: 1 });
  const out = withSign(text, sats, opts.signed ?? false);
  return opts.unit === false ? out : `${out} sats`;
}

/** Quote-scale value, e.g. "15.06" (points). */
export function formatPoints(
  value: PriceUnits | null | undefined,
  opts: { digits?: number; signed?: boolean } = {},
): string {
  if (value === null || value === undefined) return "—";
  const digits = opts.digits ?? 2;
  return withSign(Math.abs(value).toFixed(digits), value, opts.signed ?? false);
}

export function formatProbability(value: Probability | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "—";
  const pct = value * 100;
  if (pct > 0 && pct < 0.1) return "<0.1%";
  return `${pct.toFixed(digits)}%`;
}

export function formatPercent(
  value: number | null | undefined,
  opts: { digits?: number; signed?: boolean } = {},
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const digits = opts.digits ?? 2;
  return `${withSign(Math.abs(value).toFixed(digits), value, opts.signed ?? false)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatContracts(value: number): string {
  return Number.isInteger(value) ? intFormat.format(value) : value.toFixed(2);
}

/** Spot / bucket price with sensible precision per magnitude. */
export function formatPrice(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const digits = value >= 1000 ? 0 : value >= 10 ? 2 : 4;
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatCountdown(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds <= 0) return "expired";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  return `${m}:${String(r).padStart(2, "0")}`;
}

export function formatClock(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString("en-GB", { hour12: false });
}

export function formatDateTime(ts: number): string {
  const d = new Date(ts * 1000);
  return `${d.toLocaleDateString("en-CA")} ${d.toLocaleTimeString("en-GB", { hour12: false })}`;
}

export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const d = Math.floor(s / 86_400);
  const h = Math.floor((s % 86_400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s % 60}s`;
}
