"use client";

import { useMemo, useState } from "react";

import type { AssetType, MarketSummary } from "@/lib/contract";
import { ASSET_TYPE_VALUES } from "@/lib/contract";
import { formatCountdown, formatNumber, formatPoints, formatSats } from "@/lib/format";

import { EmptyState, Panel, Segmented, cx } from "./ui";

type SortKey = "edge" | "expiry" | "kappa";

function edgeOf(topic: MarketSummary): number {
  return topic.max_extractable_edge_millisats ?? -1;
}

export function MarketScreener({
  topics,
  selectedTopicId,
  onSelect,
}: {
  topics: readonly MarketSummary[];
  selectedTopicId: number | null;
  onSelect: (topicId: number) => void;
}) {
  const [sort, setSort] = useState<SortKey>("edge");
  const [assets, setAssets] = useState<ReadonlySet<AssetType>>(() => new Set(ASSET_TYPE_VALUES));

  const rows = useMemo(() => {
    const filtered = topics.filter((topic) => assets.has(topic.asset));
    const sorted = [...filtered];
    if (sort === "edge") sorted.sort((a, b) => edgeOf(b) - edgeOf(a));
    if (sort === "expiry") sorted.sort((a, b) => a.seconds_to_expiry - b.seconds_to_expiry);
    if (sort === "kappa")
      sorted.sort((a, b) => (b.kelly_multiplier_kappa ?? -1) - (a.kelly_multiplier_kappa ?? -1));
    return sorted;
  }, [topics, assets, sort]);

  function toggleAsset(asset: AssetType) {
    setAssets((current) => {
      const next = new Set(current);
      if (next.has(asset) && next.size > 1) next.delete(asset);
      else next.add(asset);
      return next;
    });
  }

  return (
    <Panel
      title="Markets"
      subtitle={`${rows.length} live topics · edge = 100·b·KL`}
      bodyClassName="flex min-h-0 flex-col"
      className="h-full min-h-0"
    >
      <div className="flex flex-col gap-2 border-b border-line px-3 py-2">
        <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by asset">
          {ASSET_TYPE_VALUES.map((asset) => {
            const on = assets.has(asset);
            return (
              <button
                key={asset}
                type="button"
                aria-pressed={on}
                onClick={() => toggleAsset(asset)}
                className={cx(
                  "h-6 rounded-md border px-2 text-[11px] font-semibold",
                  on ? "border-line-strong bg-raised text-ink" : "border-line text-ink-3",
                )}
              >
                {asset}
              </button>
            );
          })}
        </div>
        <Segmented<SortKey>
          label="Sort markets"
          value={sort}
          onChange={setSort}
          options={[
            { value: "edge", label: "Max edge" },
            { value: "expiry", label: "Expiry" },
            { value: "kappa", label: "κ (agreement)" },
          ]}
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto" role="listbox" aria-label="Markets">
        {rows.length === 0 && (
          <EmptyState title="No markets yet">The engine publishes topics once it has loaded the batches.</EmptyState>
        )}
        {rows.map((topic) => {
          const selected = topic.topic_id === selectedTopicId;
          const edge = topic.max_extractable_edge_millisats;
          return (
            <button
              key={topic.topic_id}
              type="button"
              role="option"
              aria-selected={selected}
              onClick={() => onSelect(topic.topic_id)}
              className={cx(
                "flex w-full flex-col gap-1 border-b border-line px-3 py-2 text-left transition-colors",
                selected ? "bg-s1/10 shadow-[inset_2px_0_0_var(--color-s1)]" : "hover:bg-hover",
              )}
            >
              <div className="flex items-center gap-2">
                <span className="rounded bg-raised px-1.5 py-0.5 text-[10px] font-bold text-ink-2">{topic.asset}</span>
                <span className="min-w-0 flex-1 truncate text-xs font-medium text-ink">{topic.title}</span>
                <span className="num text-[11px] text-ink-3" title="time to expiry">
                  {formatCountdown(topic.seconds_to_expiry)}
                </span>
              </div>
              <div className="grid grid-cols-4 gap-2 text-[11px]">
                <span className="text-ink-3" title="Max extractable edge, 100·b·KL(p_model‖p_market), gross of fees">
                  edge{" "}
                  <span className={cx("num font-semibold", edge !== null && edge > 0 ? "text-ink" : "text-ink-3")}>
                    {edge === null ? "—" : formatSats(edge, { unit: false, compact: true })}
                  </span>
                </span>
                <span className="text-ink-3" title="LS-LMSR liquidity b">
                  b <span className="num text-ink-2">{formatNumber(topic.liquidity_b, 1)}</span>
                </span>
                <span className="text-ink-3" title="Kelly multiplier κ = clip(1 − 2·JSD, 0.25, 1)">
                  κ <span className="num text-ink-2">{formatNumber(topic.kelly_multiplier_kappa, 2)}</span>
                </span>
                <span className="text-ink-3" title="Buckets with positive net edge after fees">
                  +EV{" "}
                  <span className={cx("num font-semibold", topic.positive_ev_buckets > 0 ? "text-good-ink" : "text-ink-3")}>
                    {topic.positive_ev_buckets}
                  </span>
                </span>
              </div>
              {topic.best_bucket !== null && topic.best_edge_points !== null && (
                <div className="truncate text-[11px] text-ink-3">
                  best {topic.best_bucket} ·{" "}
                  <span className="num text-ink-2">{formatPoints(topic.best_edge_points, { signed: true })} pts</span>
                  {topic.held_contracts > 0 && <span className="text-s1"> · holding {topic.held_contracts}</span>}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </Panel>
  );
}
