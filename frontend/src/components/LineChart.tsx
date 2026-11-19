"use client";

import type { PointerEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

export interface LineSeries {
  readonly key: string;
  readonly label: string;
  /** CSS colour (a series token, e.g. `var(--color-s1)`). */
  readonly color: string;
  readonly values: readonly (number | null)[];
}

export interface ReferenceLine {
  readonly axis: "x" | "y";
  readonly value: number;
  readonly label: string;
  readonly tone?: "neutral" | "good" | "critical";
}

const PAD = { left: 52, right: 14, top: 12, bottom: 26 } as const;

function niceTicks(min: number, max: number, count: number): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const raw = (max - min) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  const step = (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * power;
  const ticks: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}

/**
 * One-axis line chart (never dual-axis) with a snapping crosshair and a readout
 * listing every series at the hovered x. Reference lines are labelled hairlines.
 */
export function LineChart({
  xs,
  series,
  height = 180,
  xFormat,
  yFormat,
  refs = [],
  ariaLabel,
  includeZero = false,
}: {
  xs: readonly number[];
  series: readonly LineSeries[];
  height?: number;
  xFormat: (x: number) => string;
  yFormat: (y: number) => string;
  refs?: readonly ReferenceLine[];
  ariaLabel: string;
  includeZero?: boolean;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);

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

  const domain = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series) {
      for (const v of s.values) {
        if (v === null || !Number.isFinite(v)) continue;
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
    for (const ref of refs) {
      if (ref.axis === "y") {
        lo = Math.min(lo, ref.value);
        hi = Math.max(hi, ref.value);
      }
    }
    if (includeZero) {
      lo = Math.min(lo, 0);
      hi = Math.max(hi, 0);
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1 };
    if (lo === hi) return { lo: lo - 1, hi: hi + 1 };
    const pad = (hi - lo) * 0.08;
    return { lo: lo - pad, hi: hi + pad };
  }, [series, refs, includeZero]);

  const xMin = xs[0] ?? 0;
  const xMax = xs[xs.length - 1] ?? 1;
  const plotW = Math.max(1, width - PAD.left - PAD.right);
  const plotH = height - PAD.top - PAD.bottom;
  const xOf = (x: number) => PAD.left + (xMax === xMin ? plotW / 2 : ((x - xMin) / (xMax - xMin)) * plotW);
  const yOf = (y: number) => PAD.top + plotH * (1 - (y - domain.lo) / (domain.hi - domain.lo));

  const yTicks = niceTicks(domain.lo, domain.hi, 4);
  const xTickCount = Math.max(2, Math.floor(plotW / 90));
  const xTickIdx = xs.length <= 1 ? [0] : Array.from({ length: xTickCount + 1 }, (_, k) => Math.round((k * (xs.length - 1)) / xTickCount));

  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    if (xs.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < xs.length; i += 1) {
      const d = Math.abs(xOf(xs[i] ?? 0) - x);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    setHover(best);
  };

  const path = (values: readonly (number | null)[]) => {
    let d = "";
    let pen = false;
    for (let i = 0; i < xs.length; i += 1) {
      const v = values[i];
      if (v === null || v === undefined || !Number.isFinite(v)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${xOf(xs[i] ?? 0).toFixed(1)},${yOf(v).toFixed(1)}`;
      pen = true;
    }
    return d;
  };

  const hoverX = hover === null ? null : xs[hover];
  const readoutLeft = hoverX === undefined || hoverX === null ? 0 : xOf(hoverX) > width - 200 ? xOf(hoverX) - 196 : xOf(hoverX) + 12;

  return (
    <div ref={wrapRef} className="relative w-full" style={{ height }}>
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={ariaLabel}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          className="block"
        >
          {yTicks.map((tick) => (
            <g key={tick}>
              <line
                x1={PAD.left}
                x2={PAD.left + plotW}
                y1={yOf(tick)}
                y2={yOf(tick)}
                stroke={tick === 0 ? "var(--color-line-strong)" : "var(--color-line)"}
                strokeWidth={1}
              />
              <text x={PAD.left - 6} y={yOf(tick)} textAnchor="end" dominantBaseline="middle" fontSize={10} fill="var(--color-ink-3)" className="num">
                {yFormat(tick)}
              </text>
            </g>
          ))}
          {xTickIdx.map((i) => {
            const x = xs[i];
            return x === undefined ? null : (
              <text key={i} x={xOf(x)} y={height - 8} textAnchor="middle" fontSize={10} fill="var(--color-ink-3)" className="num">
                {xFormat(x)}
              </text>
            );
          })}
          {refs.map((ref) => {
            const color =
              ref.tone === "good" ? "var(--color-good)" : ref.tone === "critical" ? "var(--color-critical)" : "var(--color-ink-3)";
            if (ref.axis === "y") {
              const y = yOf(ref.value);
              return (
                <g key={`y${ref.label}`}>
                  <line x1={PAD.left} x2={PAD.left + plotW} y1={y} y2={y} stroke={color} strokeWidth={1} opacity={0.8} />
                  <text x={PAD.left + plotW - 4} y={y - 4} textAnchor="end" fontSize={10} fill={color}>
                    {ref.label}
                  </text>
                </g>
              );
            }
            const x = xOf(ref.value);
            return (
              <g key={`x${ref.label}`}>
                <line x1={x} x2={x} y1={PAD.top} y2={PAD.top + plotH} stroke={color} strokeWidth={1} opacity={0.8} />
                <text x={x + 4} y={PAD.top + 10} fontSize={10} fill={color}>
                  {ref.label}
                </text>
              </g>
            );
          })}
          {series.map((s) => (
            <path key={s.key} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {hoverX !== undefined && hoverX !== null && (
            <g>
              <line x1={xOf(hoverX)} x2={xOf(hoverX)} y1={PAD.top} y2={PAD.top + plotH} stroke="var(--color-ink-3)" strokeWidth={1} />
              {series.map((s) => {
                const v = hover === null ? null : s.values[hover];
                return v === null || v === undefined ? null : (
                  <circle key={s.key} cx={xOf(hoverX)} cy={yOf(v)} r={4} fill={s.color} stroke="var(--color-panel)" strokeWidth={2} />
                );
              })}
            </g>
          )}
        </svg>
      )}
      {hover !== null && hoverX !== undefined && hoverX !== null && (
        <div
          className="pointer-events-none absolute top-2 z-10 w-[184px] rounded-md border border-line-strong bg-raised/95 px-2.5 py-2 text-[11px] shadow-xl"
          style={{ left: readoutLeft }}
        >
          <div className="mb-1 font-semibold text-ink">{xFormat(hoverX)}</div>
          {series.map((s) => {
            const v = s.values[hover];
            return (
              <div key={s.key} className="flex items-center gap-1.5">
                <span aria-hidden className="h-0.5 w-3 rounded-full" style={{ background: s.color }} />
                <span className="num font-semibold text-ink">{v === null || v === undefined ? "—" : yFormat(v)}</span>
                <span className="ml-auto truncate text-ink-3">{s.label}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
