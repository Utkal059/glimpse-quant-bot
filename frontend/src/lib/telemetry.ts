/**
 * `/ws/telemetry` protocol. Every server frame is `{type, ts, data}`; the data
 * shapes are the generated API contract, so the socket and REST paths share
 * one set of types.
 */

import type {
  DistributionResponse,
  HealthResponse,
  LedgerRecordView,
  LogEntry,
  MarketsResponse,
  PositionView,
  RiskMetrics,
} from "./contract";

export interface HelloData {
  readonly engine_status: HealthResponse;
  readonly market_quotes: MarketsResponse;
  readonly risk_metrics: RiskMetrics;
  readonly active_positions: readonly PositionView[];
  readonly ledger_fills: readonly LedgerRecordView[];
  readonly system_logs: readonly LogEntry[];
}

export interface PongData {
  readonly t: number;
}

interface FrameOf<T extends string, D> {
  readonly type: T;
  readonly ts: number;
  readonly data: D;
}

export type TelemetryFrame =
  | FrameOf<"hello", HelloData>
  | FrameOf<"engine_status", HealthResponse>
  | FrameOf<"market_quotes", MarketsResponse>
  | FrameOf<"ensemble_distribution", DistributionResponse>
  | FrameOf<"risk_metrics", RiskMetrics>
  | FrameOf<"active_positions", readonly PositionView[]>
  | FrameOf<"ledger_fills", readonly LedgerRecordView[]>
  | FrameOf<"system_logs", LogEntry>
  | FrameOf<"pong", PongData>;

export type TelemetryChannel = TelemetryFrame["type"];

const CHANNELS: ReadonlySet<string> = new Set<TelemetryChannel>([
  "hello",
  "engine_status",
  "market_quotes",
  "ensemble_distribution",
  "risk_metrics",
  "active_positions",
  "ledger_fills",
  "system_logs",
  "pong",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Parse one frame. The envelope is checked structurally; the payload is trusted
 * to match the contract (the server builds it from the same Pydantic models).
 * Unknown channels are dropped so a newer server never breaks an older UI.
 */
export function parseFrame(text: string): TelemetryFrame | null {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(raw)) return null;
  const { type, ts, data } = raw;
  if (typeof type !== "string" || !CHANNELS.has(type)) return null;
  if (typeof ts !== "number" || data === undefined) return null;
  return raw as unknown as TelemetryFrame;
}

/** Client -> server messages. */
export type ClientMessage =
  | { readonly type: "focus"; readonly topic_id: number | null }
  | { readonly type: "ping"; readonly t: number };
