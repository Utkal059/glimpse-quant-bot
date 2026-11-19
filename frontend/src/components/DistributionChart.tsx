"use client";

import type { KeyboardEvent, PointerEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { DistributionResponse, PositiveEvBucket } from "@/lib/contract";
import { formatContracts, formatNumber, formatPoints, formatPrice, formatProbability, formatSats } from "@/lib/format";
import type { PriceUnits, Probability } from "@/lib/units";
import { pointsToProbability } from "@/lib/units";

export type ChartView = "focus" | "full";

/** Component densities keyed by engine name; colours follow the fixed series order. */
export const COMPONENTS: ReadonlyArray<{ key: string; label: string; colorVar: string }> = [
  { key: "options_rnd", label: "Options RND", colorVar: "--color-s3" },
  { key: "har_microstructure", label: "HAR-RV", colorVar: "--color-s4" },
  { key: "gjr_garch", label: "GJR-GARCH", colorVar: "--color-s5" },
  { key: "timesfm", label: "TimesFM", colorVar: "--color-s5" },
];

const FOCUS_HALF_WIDTH = 30;
const HEIGHT = 360;
const PAD = { left: 48, right: 12, top: 18, bottom: 30 } as const;

interface Palette {
  s1: string;
  s2: string;
  good: string;
  ink: string;
  ink2: string;
  ink3: string;
  line: string;
  lineStrong: string;
  panel: string;
  components: Record<string, string>;
}

function readPalette(element: Element): Palette {
  const style = getComputedStyle(element);
  const v = (name: string) => style.getPropertyValue(name).trim();
  const components: Record<string, string> = {};
  for (const component of COMPONENTS) components[component.key] = v(component.colorVar);
  return {
    s1: v("--color-s1"),
    s2: v("--color-s2"),
    good: v("--color-good"),
    ink: v("--color-ink"),
    ink2: v("--color-ink-2"),
    ink3: v("--color-ink-3"),
    line: v("--color-line"),
    lineStrong: v("--color-line-strong"),
    panel: v("--color-panel"),
    components,
  };
}

function niceStep(max: number, ticks: number): number {
  const raw = max / ticks;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  const nice = unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 2.5 ? 2.5 : unit <= 5 ? 5 : 10;
  return nice * power;
}

/** Index of the bucket that contains the spot price (or the market mode). */
export function centerIndex(dist: DistributionResponse): number {
  const { lo, hi, market_price } = dist.buckets;
  const spot = dist.spot_price ?? dist.forecast?.spot ?? null;
  if (spot !== null) {
    for (let i = 0; i < lo.length; i += 1) {
      if ((lo[i] ?? Infinity) <= spot && spot < (hi[i] ?? -Infinity)) return i;
    }
  }
  let best = 0;
  for (let i = 1; i < market_price.length; i += 1) {
    if ((market_price[i] ?? 0) > (market_price[best] ?? 0)) best = i;
  }
  return best;
}

export function visibleRange(dist: DistributionResponse, view: ChartView): readonly [number, number] {
  const n = dist.buckets.option_id.length;
  if (view === "full" || n <= FOCUS_HALF_WIDTH * 2 + 1) return [0, Math.max(0, n - 1)];
  const center = centerIndex(dist);
  const start = Math.max(0, Math.min(n - 1 - FOCUS_HALF_WIDTH * 2, center - FOCUS_HALF_WIDTH));
  return [start, start + FOCUS_HALF_WIDTH * 2];
}

interface Hover {
  index: number;
  x: number;
  y: number;
}

export function DistributionChart({
  dist,
  view,
  showModel,
  components,
  selectedOptionId,
  onSelect,
}: {
  dist: DistributionResponse;
  view: ChartView;
  showModel: boolean;
  components: ReadonlySet<string>;
  selectedOptionId: number | null;
  onSelect: (optionId: number) => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<Hover | null>(null);

  const [i0, i1] = visibleRange(dist, view);
  const count = i1 - i0 + 1;

  const evByOption = useMemo(() => {
    const map = new Map<number, PositiveEvBucket>();
    for (const bucket of dist.positive_ev) map.set(bucket.option_id, bucket);
    return map;
  }, [dist.positive_ev]);

  const activeComponents = useMemo(
    () => COMPONENTS.filter((c) => components.has(c.key) && dist.component_densities[c.key] !== undefined),
    [components, dist.component_densities],
  );

  const yMax = useMemo(() => {
    let max = 0;
    const { market_price, model_probability } = dist.buckets;
    for (let i = i0; i <= i1; i += 1) {
      max = Math.max(max, (market_price[i] ?? 0) / 100);
      if (showModel && model_probability !== null) max = Math.max(max, model_probability[i] ?? 0);
      for (const component of activeComponents) {
        max = Math.max(max, dist.component_densities[component.key]?.[i] ?? 0);
      }
    }
    return Math.max(max * 1.12, 0.002);
  }, [dist, i0, i1, showModel, activeComponents]);

  useEffect(() => {
    const element = wrapRef.current;
    if (element === null) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry !== undefined) setWidth(Math.floor(entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const plotW = Math.max(1, width - PAD.left - PAD.right);
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const barW = plotW / count;
  const xOf = useCallback((index: number) => PAD.left + (index - i0) * barW, [i0, barW]);
  const yOf = useCallback((value: number) => PAD.top + plotH * (1 - value / yMax), [plotH, yMax]);

  // --- draw -------------------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null || width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(HEIGHT * dpr);
    const ctx = canvas.getContext("2d");
    if (ctx === null) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, HEIGHT);
    const p = readPalette(canvas);
    const { lo, market_price, model_probability, option_id, held_contracts } = dist.buckets;
    const baseline = PAD.top + plotH;

    // grid + y labels (hairlines, recessive)
    const step = niceStep(yMax, 4);
    ctx.font = "11px system-ui, -apple-system, Segoe UI, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (let value = 0; value <= yMax + 1e-12; value += step) {
      const y = Math.round(yOf(value)) + 0.5;
      ctx.strokeStyle = value === 0 ? p.lineStrong : p.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + plotW, y);
      ctx.stroke();
      ctx.fillStyle = p.ink3;
      const pct = value * 100;
      ctx.fillText(`${pct < 1 && pct > 0 ? pct.toFixed(1) : pct.toFixed(pct < 10 ? 1 : 0)}%`, PAD.left - 6, y);
    }

    // +EV bands behind everything
    for (let i = i0; i <= i1; i += 1) {
      const id = option_id[i];
      if (id === undefined || !evByOption.has(id)) continue;
      ctx.fillStyle = p.good;
      ctx.globalAlpha = 0.12;
      ctx.fillRect(xOf(i), PAD.top, Math.max(1, barW), plotH);
      ctx.globalAlpha = 1;
    }

    // market quote bars (quote / 100 = the price you pay as a probability)
    const gap = barW >= 7 ? 2 : barW >= 3.5 ? 1 : 0;
    const radius = barW >= 8 ? 2 : 0;
    for (let i = i0; i <= i1; i += 1) {
      const value = (market_price[i] ?? 0) / 100;
      const x = xOf(i) + gap / 2;
      const w = Math.max(0.6, barW - gap);
      const y = yOf(value);
      const h = Math.max(0.5, baseline - y);
      const hovered = hover?.index === i;
      ctx.fillStyle = p.s1;
      ctx.globalAlpha = hover === null || hovered ? 0.9 : 0.55;
      ctx.beginPath();
      if (radius > 0) ctx.roundRect(x, y, w, h, [radius, radius, 0, 0]);
      else ctx.rect(x, y, w, h);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (option_id[i] === selectedOptionId) {
        ctx.strokeStyle = p.ink;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x - 1, y - 1, w + 2, h + 1);
      }
      if ((held_contracts[i] ?? 0) > 0) {
        ctx.fillStyle = p.ink;
        ctx.beginPath();
        ctx.arc(xOf(i) + barW / 2, baseline - 5, Math.min(3, Math.max(1.5, barW / 3)), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // +EV markers on top
    ctx.fillStyle = p.good;
    for (let i = i0; i <= i1; i += 1) {
      const id = option_id[i];
      if (id === undefined || !evByOption.has(id)) continue;
      const cx = xOf(i) + barW / 2;
      const size = Math.min(5, Math.max(3, barW / 2));
      ctx.beginPath();
      ctx.moveTo(cx, PAD.top - 2);
      ctx.lineTo(cx - size, PAD.top - 2 - size * 1.4);
      ctx.lineTo(cx + size, PAD.top - 2 - size * 1.4);
      ctx.closePath();
      ctx.fill();
    }

    const drawLine = (values: readonly number[], color: string, lineWidth: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let i = i0; i <= i1; i += 1) {
        const x = xOf(i) + barW / 2;
        const y = yOf(values[i] ?? 0);
        if (i === i0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    for (const component of activeComponents) {
      const values = dist.component_densities[component.key];
      if (values !== undefined) drawLine(values, p.components[component.key] ?? p.ink3, 1.5);
    }
    if (showModel && model_probability !== null) drawLine(model_probability, p.s2, 2);

    // spot marker
    const spot = dist.spot_price ?? dist.forecast?.spot ?? null;
    if (spot !== null) {
      const { hi } = dist.buckets;
      for (let i = i0; i <= i1; i += 1) {
        const l = lo[i] ?? 0;
        const h = hi[i] ?? 0;
        if (l <= spot && spot < h && h > l) {
          const x = Math.round(xOf(i) + ((spot - l) / (h - l)) * barW) + 0.5;
          ctx.strokeStyle = p.ink2;
          ctx.globalAlpha = 0.7;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, PAD.top);
          ctx.lineTo(x, baseline);
          ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.fillStyle = p.ink2;
          ctx.textAlign = x > PAD.left + plotW - 80 ? "right" : "left";
          ctx.textBaseline = "top";
          ctx.fillText(`spot ${formatPrice(spot)}`, x + (ctx.textAlign === "right" ? -4 : 4), PAD.top + 2);
          break;
        }
      }
    }

    // hover crosshair
    if (hover !== null) {
      const x = Math.round(xOf(hover.index) + barW / 2) + 0.5;
      ctx.strokeStyle = p.ink3;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, PAD.top);
      ctx.lineTo(x, baseline);
      ctx.stroke();
    }

    // x labels: bucket lower edges
    ctx.fillStyle = p.ink3;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const labelEvery = Math.max(1, Math.ceil(count / Math.max(2, Math.floor(plotW / 90))));
    for (let i = i0; i <= i1; i += labelEvery) {
      ctx.fillText(formatPrice(lo[i] ?? 0), xOf(i) + barW / 2, baseline + 8);
    }
  }, [dist, width, view, hover, selectedOptionId, showModel, activeComponents, evByOption, i0, i1, count, barW, plotW, plotH, yMax, xOf, yOf]);

  // --- interaction ---------------------------------------------------------------------
  const indexAt = (clientX: number): number | null => {
    const canvas = canvasRef.current;
    if (canvas === null) return null;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left - PAD.left;
    if (x < 0 || x > plotW) return null;
    return Math.max(i0, Math.min(i1, i0 + Math.floor(x / barW)));
  };

  const onPointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    const index = indexAt(event.clientX);
    const rect = event.currentTarget.getBoundingClientRect();
    if (index === null) setHover(null);
    else setHover({ index, x: event.clientX - rect.left, y: event.clientY - rect.top });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    const current = hover?.index ?? centerIndex(dist);
    let next: number | null = null;
    if (event.key === "ArrowLeft") next = current - (event.shiftKey ? 5 : 1);
    else if (event.key === "ArrowRight") next = current + (event.shiftKey ? 5 : 1);
    else if (event.key === "Home") next = i0;
    else if (event.key === "End") next = i1;
    else if (event.key === "Escape") {
      setHover(null);
      return;
    } else if (event.key === "Enter" || event.key === " ") {
      const id = dist.buckets.option_id[current];
      if (id !== undefined) onSelect(id);
      event.preventDefault();
      return;
    }
    if (next === null) return;
    event.preventDefault();
    const clamped = Math.max(i0, Math.min(i1, next));
    setHover({ index: clamped, x: xOf(clamped) + barW / 2, y: PAD.top + 24 });
  };

  const tooltip = hover === null ? null : buildTooltip(dist, hover.index, evByOption, activeComponents);
  const tooltipLeft = hover === null ? 0 : hover.x + 16 + 260 > width ? Math.max(4, hover.x - 16 - 260) : hover.x + 16;

  return (
    <div ref={wrapRef} className="relative w-full select-none" style={{ height: HEIGHT }}>
      <canvas
        ref={canvasRef}
        tabIndex={0}
        role="img"
        aria-label={`Bucket distribution for ${dist.title}: market quotes as bars, ensemble model as a line. Arrow keys move, Enter inspects a bucket.`}
        className="block h-full w-full cursor-crosshair rounded-md"
        style={{ height: HEIGHT }}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHover(null)}
        onClick={(event) => {
          const index = indexAt(event.clientX);
          const id = index === null ? undefined : dist.buckets.option_id[index];
          if (id !== undefined) onSelect(id);
        }}
        onKeyDown={onKeyDown}
        onBlur={() => setHover(null)}
      />
      {tooltip !== null && hover !== null && (
        <div
          role="tooltip"
          className="pointer-events-none absolute z-10 w-[260px] rounded-lg border border-line-strong bg-raised/95 p-3 text-xs shadow-2xl backdrop-blur"
          style={{ left: tooltipLeft, top: Math.max(4, Math.min(HEIGHT - 250, hover.y - 40)) }}
        >
          {tooltip}
        </div>
      )}
    </div>
  );
}

function Row({ value, label, swatch }: { value: string; label: string; swatch?: string }) {
  return (
    <div className="flex items-center gap-2 py-0.5">
      {swatch !== undefined && <span aria-hidden className="h-0.5 w-3 rounded-full" style={{ background: swatch }} />}
      <span className="num font-semibold text-ink">{value}</span>
      <span className="ml-auto text-ink-3">{label}</span>
    </div>
  );
}

function buildTooltip(
  dist: DistributionResponse,
  index: number,
  evByOption: ReadonlyMap<number, PositiveEvBucket>,
  activeComponents: ReadonlyArray<{ key: string; label: string; colorVar: string }>,
) {
  const b = dist.buckets;
  const id = b.option_id[index];
  const price = b.market_price[index];
  if (id === undefined || price === undefined) return null;
  const model: Probability | undefined = b.model_probability?.[index];
  const edge: PriceUnits | undefined = b.edge_points?.[index];
  const net: PriceUnits | undefined = b.net_edge_points?.[index];
  const ev = evByOption.get(id);
  const held = b.held_contracts[index] ?? 0;
  return (
    <>
      <div className="mb-2 flex items-baseline gap-2">
        <span className="text-sm font-semibold text-ink">{b.name[index]}</span>
        {ev !== undefined && <span className="rounded bg-good/20 px-1.5 text-[10px] font-bold text-good-ink">▲ +EV</span>}
        {held > 0 && <span className="ml-auto text-[11px] text-ink-2">held {formatContracts(held)}</span>}
      </div>
      <Row value={`${formatPoints(price)} pts`} label="market quote" swatch="var(--color-s1)" />
      <Row value={formatProbability(pointsToProbability(price), 2)} label="implied by quote" />
      {model !== undefined && <Row value={formatProbability(model, 2)} label="ensemble model" swatch="var(--color-s2)" />}
      {activeComponents.map((component) => {
        const value = dist.component_densities[component.key]?.[index];
        return value === undefined ? null : (
          <Row key={component.key} value={formatProbability(value, 2)} label={component.label} swatch={`var(${component.colorVar})`} />
        );
      })}
      <div className="my-1.5 border-t border-line" />
      {edge !== undefined && <Row value={`${formatPoints(edge, { signed: true })} pts`} label="raw edge" />}
      {net !== undefined && <Row value={`${formatPoints(net, { signed: true })} pts`} label="net of 2% + 2% fees" />}
      <Row
        value={`${formatNumber(dist.js_divergence, 3)} / ${formatNumber(dist.kelly_multiplier_kappa, 2)}`}
        label="JSD / Kelly κ"
      />
      {ev !== undefined && (
        <>
          <div className="my-1.5 border-t border-line" />
          <Row value={ev.recommended_contracts === null ? "—" : formatContracts(ev.recommended_contracts)} label="Kelly size k*" />
          <Row value={formatSats(ev.slippage_cost_millisats)} label="slippage + fees" />
          <Row value={formatSats(ev.expected_profit_millisats, { signed: true })} label="expected profit" />
          {ev.rejection !== null && <p className="mt-1 text-[11px] text-warn">! {ev.rejection}</p>}
        </>
      )}
      <p className="mt-2 text-[10px] text-ink-3">Click or press Enter to open the cost ladder</p>
    </>
  );
}
