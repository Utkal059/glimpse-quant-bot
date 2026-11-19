"use client";

/**
 * `useTelemetrySocket` — one auto-reconnecting WebSocket to `/ws/telemetry`,
 * folded into a single immutable terminal state.
 *
 * - Exponential backoff with full jitter (0.5 s → 15 s), reset on a good open.
 * - Liveness: a ping every 5 s; the pong round-trip is the displayed latency.
 *   No frame for 12 s means a half-open socket, which is torn down and retried.
 * - The focused topic is re-sent after every reconnect, so the server keeps
 *   streaming that topic's 500-bucket distribution.
 * - Nothing here calls Glimpse: the server streams in-memory snapshots fed by
 *   its own WebSocket feed and 45 s REST poller.
 */

import { useCallback, useEffect, useReducer, useRef } from "react";

import type {
  DistributionResponse,
  HealthResponse,
  LedgerRecordView,
  LogEntry,
  MarketsResponse,
  PositionView,
  RiskMetrics,
} from "@/lib/contract";
import type { ClientMessage, TelemetryFrame } from "@/lib/telemetry";
import { parseFrame } from "@/lib/telemetry";

export type ConnectionStatus = "connecting" | "open" | "reconnecting" | "closed";

export interface TerminalState {
  readonly connection: ConnectionStatus;
  readonly latencyMs: number | null;
  readonly lastFrameAt: number | null;
  readonly reconnects: number;
  readonly health: HealthResponse | null;
  readonly markets: MarketsResponse | null;
  readonly risk: RiskMetrics | null;
  readonly positions: readonly PositionView[];
  readonly fills: readonly LedgerRecordView[];
  readonly logs: readonly LogEntry[];
  readonly distribution: DistributionResponse | null;
}

const MAX_FILLS = 500;
const MAX_LOGS = 1500;
const PING_INTERVAL_MS = 5_000;
const STALE_AFTER_MS = 12_000;
const BACKOFF_MIN_MS = 500;
const BACKOFF_MAX_MS = 15_000;

const initialState: TerminalState = {
  connection: "connecting",
  latencyMs: null,
  lastFrameAt: null,
  reconnects: 0,
  health: null,
  markets: null,
  risk: null,
  positions: [],
  fills: [],
  logs: [],
  distribution: null,
};

type Action =
  | { readonly kind: "status"; readonly status: ConnectionStatus }
  | { readonly kind: "frame"; readonly frame: TelemetryFrame; readonly receivedAt: number }
  | { readonly kind: "focus"; readonly topicId: number | null };

function mergeFills(
  current: readonly LedgerRecordView[],
  incoming: readonly LedgerRecordView[],
): readonly LedgerRecordView[] {
  const byId = new Map<number, LedgerRecordView>();
  for (const record of current) byId.set(record.record_id, record);
  for (const record of incoming) byId.set(record.record_id, record);
  return [...byId.values()].sort((a, b) => b.record_id - a.record_id).slice(0, MAX_FILLS);
}

function applyFrame(state: TerminalState, frame: TelemetryFrame): TerminalState {
  switch (frame.type) {
    case "hello":
      return {
        ...state,
        health: frame.data.engine_status,
        markets: frame.data.market_quotes,
        risk: frame.data.risk_metrics,
        positions: frame.data.active_positions,
        fills: mergeFills([], frame.data.ledger_fills),
        logs: frame.data.system_logs.slice(-MAX_LOGS),
      };
    case "engine_status":
      return { ...state, health: frame.data };
    case "market_quotes":
      return { ...state, markets: frame.data };
    case "ensemble_distribution":
      return { ...state, distribution: frame.data };
    case "risk_metrics":
      return { ...state, risk: frame.data };
    case "active_positions":
      return { ...state, positions: frame.data };
    case "ledger_fills":
      return { ...state, fills: mergeFills(state.fills, frame.data) };
    case "system_logs": {
      const logs =
        state.logs.length >= MAX_LOGS
          ? [...state.logs.slice(state.logs.length - MAX_LOGS + 1), frame.data]
          : [...state.logs, frame.data];
      return { ...state, logs };
    }
    case "pong":
      return state;
  }
}

function reducer(state: TerminalState, action: Action): TerminalState {
  switch (action.kind) {
    case "status":
      return {
        ...state,
        connection: action.status,
        reconnects: action.status === "reconnecting" ? state.reconnects + 1 : state.reconnects,
        latencyMs: action.status === "open" ? state.latencyMs : null,
      };
    case "focus":
      // Drop a distribution that belongs to another topic straight away.
      return state.distribution !== null && state.distribution.topic_id !== action.topicId
        ? { ...state, distribution: null }
        : state;
    case "frame": {
      const next = applyFrame(state, action.frame);
      if (action.frame.type === "pong") {
        const latency = Math.max(0, performance.now() - action.frame.data.t);
        return { ...next, latencyMs: latency, lastFrameAt: action.receivedAt };
      }
      return { ...next, lastFrameAt: action.receivedAt };
    }
  }
}

export interface TelemetrySocket {
  readonly state: TerminalState;
  /** Force a reconnect now (e.g. from a "retry" button). */
  readonly reconnect: () => void;
}

export function useTelemetrySocket(url: string, focusTopicId: number | null): TelemetrySocket {
  const [state, dispatch] = useReducer(reducer, initialState);
  const socketRef = useRef<WebSocket | null>(null);
  const focusRef = useRef<number | null>(focusTopicId);
  const attemptRef = useRef(0);
  const lastFrameRef = useRef(0);
  const retryTimer = useRef<number | null>(null);
  const disposedRef = useRef(false);
  const connectRef = useRef<() => void>(() => undefined);

  const send = useCallback((message: ClientMessage) => {
    const socket = socketRef.current;
    if (socket !== null && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (disposedRef.current || retryTimer.current !== null) return;
    const attempt = attemptRef.current;
    attemptRef.current = attempt + 1;
    const ceiling = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS * 2 ** attempt);
    const delay = BACKOFF_MIN_MS / 2 + Math.random() * ceiling; // full jitter
    dispatch({ kind: "status", status: "reconnecting" });
    retryTimer.current = window.setTimeout(() => {
      retryTimer.current = null;
      connectRef.current();
    }, delay);
  }, []);

  const connect = useCallback(() => {
    if (disposedRef.current) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(url);
    } catch {
      scheduleReconnect();
      return;
    }
    socketRef.current = socket;
    socket.onopen = () => {
      attemptRef.current = 0;
      lastFrameRef.current = Date.now();
      dispatch({ kind: "status", status: "open" });
      send({ type: "focus", topic_id: focusRef.current });
      send({ type: "ping", t: performance.now() });
    };
    socket.onmessage = (event: MessageEvent<unknown>) => {
      if (typeof event.data !== "string") return;
      const frame = parseFrame(event.data);
      lastFrameRef.current = Date.now();
      if (frame !== null) dispatch({ kind: "frame", frame, receivedAt: Date.now() });
    };
    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      scheduleReconnect();
    };
    socket.onerror = () => {
      socket.close();
    };
  }, [url, send, scheduleReconnect]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    disposedRef.current = false;
    connect();
    const heartbeat = window.setInterval(() => {
      const socket = socketRef.current;
      if (socket === null || socket.readyState !== WebSocket.OPEN) return;
      if (Date.now() - lastFrameRef.current > STALE_AFTER_MS) {
        socket.close(); // half-open: onclose schedules the retry
        return;
      }
      send({ type: "ping", t: performance.now() });
    }, PING_INTERVAL_MS);
    return () => {
      disposedRef.current = true;
      window.clearInterval(heartbeat);
      if (retryTimer.current !== null) window.clearTimeout(retryTimer.current);
      retryTimer.current = null;
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket !== null) {
        socket.onclose = null;
        socket.close();
      }
      dispatch({ kind: "status", status: "closed" });
    };
  }, [connect, send]);

  useEffect(() => {
    focusRef.current = focusTopicId;
    dispatch({ kind: "focus", topicId: focusTopicId });
    send({ type: "focus", topic_id: focusTopicId });
  }, [focusTopicId, send]);

  const reconnect = useCallback(() => {
    attemptRef.current = 0;
    if (retryTimer.current !== null) {
      window.clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }
    const socket = socketRef.current;
    if (socket !== null) socket.close();
    else connectRef.current();
  }, []);

  return { state, reconnect };
}
