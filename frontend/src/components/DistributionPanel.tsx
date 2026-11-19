"use client";

import { useMemo, useState } from "react";

import type { DistributionResponse, MarketSummary } from "@/lib/contract";
import {
  formatContracts,
  formatCountdown,
  formatNumber,
  formatPoints,
  formatPrice,
  formatProbability,
  formatSats,
} from "@/lib/format";

import type { ChartView } from "./DistributionChart";
import { COMPONENTS, DistributionChart, visibleRange } from "./DistributionChart";
import { EmptyState, Panel, Segmented, Stat, Toggle, cx } from "./ui";

export function DistributionPanel({
  dist,
  summary,
  selectedOptionId,
  onSelectBucket,
}: {
  dist: DistributionResponse | null;
  summary: MarketSummary | null;
  selectedOptionId: number | null;
  onSelectBucket: (optionId: number) => void;
}) {
  const [view, setView] = useState<ChartView>("focus");
  const [showModel, setShowModel] = useState(true);
  const [components, setComponents] = useState<ReadonlySet<string>>(() => new Set());
  const [tableView, setTableView] = useState(false);

  const available = useMemo(
    () => COMPONENTS.filter((c) => dist?.component_densities[c.key] !== undefined),
    [dist],
  );

  function toggleComponent(key: string, on: boolean) {
    setComponents((current) => {
      const next = new Set(current);
      if (on) next.add(key);
      else next.delete(key);
      return next;
    });
  }

  if (dist === null) {
    return (
      <Panel title="Distribution" className="min-h-[460px]">
        <EmptyState title={summary === null ? "Select a market" : "Loading the 500-bucket distribution…"}>
          {summary === null
            ? "Pick a topic on the left to stream its LS-LMSR book against the ensemble forecast."
            : "Streaming from the engine's in-memory book — no extra Glimpse requests."}
        </EmptyState>
      </Panel>
    );
  }

  const forecastReady = dist.forecast !== null;
  const ev = dist.positive_ev;

  return (
    <Panel
      title={dist.title}
      subtitle={`${dist.asset} · topic ${dist.topic_id} · expires in ${formatCountdown(dist.seconds_to_expiry)}`}
      actions={
        <Segmented<ChartView>
          label="Zoom"
          value={view}
          onChange={setView}
          options={[
            { value: "focus", label: "Spot ±30", title: "61 buckets around spot" },
            { value: "full", label: "All 500", title: "Every bucket" },
          ]}
        />
      }
      bodyClassName="flex flex-col gap-3 p-3"
    >
      <div className="grid grid-cols-3 gap-x-4 gap-y-2 sm:grid-cols-6">
        <Stat label="Spot" value={<span className="num">{formatPrice(dist.spot_price)}</span>} />
        <Stat label="Liquidity b" value={<span className="num">{formatNumber(dist.liquidity_b, 1)}</span>} />
        <Stat
          label="KL(model‖market)"
          value={<span className="num">{formatNumber(dist.kl_divergence, 3)}</span>}
          hint="nats"
        />
        <Stat
          label="Max extractable"
          value={<span className="num">{formatSats(dist.max_extractable_edge_millisats, { compact: true })}</span>}
          hint="100·b·KL, gross"
        />
        <Stat
          label="JSD → κ"
          value={
            <span className="num">
              {formatNumber(dist.js_divergence, 3)} → {formatNumber(dist.kelly_multiplier_kappa, 2)}
            </span>
          }
        />
        <Stat
          label="+EV buckets"
          value={<span className={cx("num", ev.length > 0 && "text-good-ink")}>{ev.length}</span>}
          hint={`net ≥ ${formatNumber(dist.min_net_edge_points, 1)} pts`}
        />
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 inline-flex items-center gap-1.5 text-xs text-ink-2">
          <span aria-hidden className="h-2.5 w-2.5 rounded-sm bg-s1" />
          Market quote
        </span>
        <Toggle checked={showModel} onChange={setShowModel} swatch="var(--color-s2)">
          Ensemble
        </Toggle>
        {available.map((component) => (
          <Toggle
            key={component.key}
            checked={components.has(component.key)}
            onChange={(on) => toggleComponent(component.key, on)}
            swatch={`var(${component.colorVar})`}
          >
            {component.label}
          </Toggle>
        ))}
        <span className="ml-1 inline-flex items-center gap-1 text-xs text-ink-2" title="Positive expected value after the 2% commission and 2% settlement fee">
          <span aria-hidden className="text-good-ink">▲</span> +EV after fees
        </span>
        <span className="ml-auto" />
        <Toggle checked={tableView} onChange={setTableView}>
          Table view
        </Toggle>
      </div>

      {!forecastReady && (
        <p className="rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
          ! The ensemble forecast for this topic is still being computed; market quotes are live.
        </p>
      )}

      {tableView ? (
        <BucketTable dist={dist} view={view} onSelect={onSelectBucket} selectedOptionId={selectedOptionId} />
      ) : (
        <DistributionChart
          dist={dist}
          view={view}
          showModel={showModel}
          components={components}
          selectedOptionId={selectedOptionId}
          onSelect={onSelectBucket}
        />
      )}

      <div className="rounded-md border border-line">
        <div className="flex items-center justify-between border-b border-line px-3 py-1.5">
          <h3 className="text-[11px] font-semibold tracking-[0.08em] text-ink-2 uppercase">Positive-EV buckets</h3>
          <span className="text-[11px] text-ink-3">sized by the engine&apos;s optimiser under current caps</span>
        </div>
        {ev.length === 0 ? (
          <p className="px-3 py-3 text-xs text-ink-3">
            No bucket clears the net-edge threshold after fees right now. That is the normal state of a fairly
            priced book.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-[11px] text-ink-3">
                <tr className="text-right">
                  <th className="px-3 py-1.5 text-left font-medium">Bucket</th>
                  <th className="px-2 font-medium">Quote</th>
                  <th className="px-2 font-medium">Model</th>
                  <th className="px-2 font-medium">Net edge</th>
                  <th className="px-2 font-medium">k*</th>
                  <th className="px-2 font-medium">Slippage</th>
                  <th className="px-3 font-medium">Exp. profit</th>
                </tr>
              </thead>
              <tbody>
                {ev.map((bucket) => (
                  <tr
                    key={bucket.option_id}
                    onClick={() => onSelectBucket(bucket.option_id)}
                    className={cx(
                      "num cursor-pointer border-t border-line text-right hover:bg-hover",
                      bucket.option_id === selectedOptionId && "bg-s1/10",
                    )}
                  >
                    <td className="px-3 py-1.5 text-left font-medium text-ink">
                      {bucket.name}
                      {bucket.rejection !== null && (
                        <span className="ml-2 text-[10px] text-warn" title={bucket.rejection}>
                          ! blocked
                        </span>
                      )}
                    </td>
                    <td className="px-2 text-ink-2">{formatPoints(bucket.market_price)}</td>
                    <td className="px-2 text-ink-2">{formatProbability(bucket.model_probability, 1)}</td>
                    <td className="px-2 text-good-ink">+{formatPoints(bucket.net_edge_points)}</td>
                    <td className="px-2 text-ink">
                      {bucket.recommended_contracts === null ? "—" : formatContracts(bucket.recommended_contracts)}
                    </td>
                    <td className="px-2 text-ink-2">{formatSats(bucket.slippage_cost_millisats, { unit: false })}</td>
                    <td className="px-3 text-ink">{formatSats(bucket.expected_profit_millisats, { signed: true })}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Panel>
  );
}

function BucketTable({
  dist,
  view,
  onSelect,
  selectedOptionId,
}: {
  dist: DistributionResponse;
  view: ChartView;
  onSelect: (optionId: number) => void;
  selectedOptionId: number | null;
}) {
  const [i0, i1] = visibleRange(dist, view);
  const b = dist.buckets;
  const rows: number[] = [];
  for (let i = i0; i <= i1; i += 1) rows.push(i);
  return (
    <div className="max-h-[360px] overflow-auto rounded-md border border-line">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-raised text-[11px] text-ink-3">
          <tr className="text-right">
            <th className="px-3 py-1.5 text-left font-medium">Bucket</th>
            <th className="px-2 font-medium">Quote (pts)</th>
            <th className="px-2 font-medium">Implied</th>
            <th className="px-2 font-medium">Model</th>
            <th className="px-2 font-medium">Raw edge</th>
            <th className="px-3 font-medium">Net edge</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((i) => {
            const id = b.option_id[i];
            const net = b.net_edge_points?.[i];
            return (
              <tr
                key={id ?? i}
                onClick={() => id !== undefined && onSelect(id)}
                className={cx(
                  "num cursor-pointer border-t border-line text-right hover:bg-hover",
                  id === selectedOptionId && "bg-s1/10",
                )}
              >
                <td className="px-3 py-1 text-left text-ink">{b.name[i]}</td>
                <td className="px-2 text-ink-2">{formatPoints(b.market_price[i])}</td>
                <td className="px-2 text-ink-2">{formatProbability(b.market_probability[i], 2)}</td>
                <td className="px-2 text-ink-2">{formatProbability(b.model_probability?.[i], 2)}</td>
                <td className="px-2 text-ink-2">{formatPoints(b.edge_points?.[i], { signed: true })}</td>
                <td className={cx("px-3", net !== undefined && net > 0 ? "text-good-ink" : "text-ink-3")}>
                  {formatPoints(net, { signed: true })}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
