"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { LogCategory, LogEntry } from "@/lib/contract";
import { formatClock } from "@/lib/format";

import { EmptyState, cx, inputClass } from "./ui";

const CATEGORIES: ReadonlyArray<{ key: LogCategory; label: string; className: string }> = [
  { key: "INFO", label: "INFO", className: "text-ink-2 border-line-strong" },
  { key: "SIGNAL", label: "SIGNAL", className: "text-s1 border-s1/50" },
  { key: "EXECUTION", label: "EXECUTION", className: "text-s2 border-s2/50" },
  { key: "RISK_LOCK", label: "RISK_LOCK", className: "text-critical-ink border-critical/60" },
];

const LEVEL_STYLE: Record<string, string> = {
  debug: "text-ink-3",
  info: "text-ink-2",
  warning: "text-warn",
  error: "text-critical-ink",
  critical: "text-critical-ink font-semibold",
};

export function LogStream({ logs }: { logs: readonly LogEntry[] }) {
  const [enabled, setEnabled] = useState<ReadonlySet<LogCategory>>(() => new Set<LogCategory>(["INFO", "SIGNAL", "EXECUTION", "RISK_LOCK"]));
  const [query, setQuery] = useState("");
  const [follow, setFollow] = useState(true);
  const scroller = useRef<HTMLDivElement>(null);

  const counts = useMemo(() => {
    const out: Record<LogCategory, number> = { INFO: 0, SIGNAL: 0, EXECUTION: 0, RISK_LOCK: 0 };
    for (const entry of logs) out[entry.category] += 1;
    return out;
  }, [logs]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return logs.filter(
      (entry) => enabled.has(entry.category) && (needle.length === 0 || entry.message.toLowerCase().includes(needle)),
    );
  }, [logs, enabled, query]);

  useEffect(() => {
    const element = scroller.current;
    if (follow && element !== null) element.scrollTop = element.scrollHeight;
  }, [visible, follow]);

  function toggle(category: LogCategory) {
    setEnabled((current) => {
      const next = new Set(current);
      if (next.has(category)) next.delete(category);
      else next.add(category);
      return next;
    });
  }

  return (
    <div className="flex flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-2">
        {CATEGORIES.map((category) => {
          const on = enabled.has(category.key);
          return (
            <button
              key={category.key}
              type="button"
              aria-pressed={on}
              onClick={() => toggle(category.key)}
              className={cx(
                "inline-flex h-6 items-center gap-1.5 rounded-md border px-2 font-mono text-[10px] font-semibold",
                on ? category.className : "border-line text-ink-3 opacity-60",
              )}
            >
              {category.label}
              <span className="num font-normal text-ink-3">{counts[category.key]}</span>
            </button>
          );
        })}
        <input
          className={cx(inputClass, "h-7 max-w-64 text-xs")}
          placeholder="Filter logs…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          aria-label="Filter logs"
        />
        <label className="ml-auto flex items-center gap-1.5 text-[11px] text-ink-3">
          <input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} />
          Follow
        </label>
      </div>
      <div
        ref={scroller}
        role="log"
        aria-live="off"
        className="h-[320px] overflow-y-auto px-3 py-2 font-mono text-[11px] leading-5"
        onWheel={(event) => {
          if (event.deltaY < 0) setFollow(false);
        }}
      >
        {visible.length === 0 && <EmptyState title="No log lines match" />}
        {visible.map((entry) => {
          const category = CATEGORIES.find((c) => c.key === entry.category);
          return (
            <div key={entry.seq} className="flex gap-2 hover:bg-hover">
              <span className="shrink-0 text-ink-3">{formatClock(entry.ts)}</span>
              <span className={cx("w-[76px] shrink-0 font-semibold", category?.className.split(" ")[0])}>{entry.category}</span>
              <span className={cx("min-w-0 break-all", LEVEL_STYLE[entry.level] ?? "text-ink-2")}>{entry.message}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
