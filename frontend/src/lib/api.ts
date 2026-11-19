/**
 * Typed REST client for `glimpse-quant-bot serve`.
 *
 * Every GET reads the server's in-memory state (never Glimpse), so polling or
 * refreshing the page costs no venue rate limit. Every POST carries the
 * operator-intent header the server requires for state changes; requests go
 * through the controller, i.e. the RiskManager and ExecutionRouter.
 */

import { API_URL } from "./config";
import type {
  ActionResult,
  ArmResponse,
  BacktestResultsResponse,
  ErrorDetail,
  LadderResponse,
  LedgerPage,
  ModeName,
  ReconcileResult,
  StartRequest,
  TestOrderRequest,
  TestOrderResult,
} from "./contract";

export class ApiError extends Error {
  readonly status: number;
  readonly reasons: readonly string[];

  constructor(status: number, detail: string, reasons: readonly string[] = []) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.reasons = reasons;
  }
}

function isErrorDetail(value: unknown): value is ErrorDetail {
  return (
    typeof value === "object" &&
    value !== null &&
    "detail" in value &&
    typeof (value as { detail: unknown }).detail === "string"
  );
}

function validationMessage(value: unknown): string | null {
  // FastAPI 422s carry `detail: [{loc, msg}]`.
  if (typeof value !== "object" || value === null || !("detail" in value)) return null;
  const detail = (value as { detail: unknown }).detail;
  if (!Array.isArray(detail)) return null;
  const messages = detail
    .map((item: unknown) =>
      typeof item === "object" && item !== null && "msg" in item
        ? String((item as { msg: unknown }).msg)
        : null,
    )
    .filter((msg): msg is string => msg !== null);
  return messages.length > 0 ? messages.join("; ") : null;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, cache: "no-store" });
  } catch {
    throw new ApiError(0, `Cannot reach the engine at ${API_URL}`);
  }
  let body: unknown = null;
  const text = await response.text();
  if (text.length > 0) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }
  if (!response.ok) {
    if (isErrorDetail(body)) {
      const reasons = Array.isArray(body.reasons) ? body.reasons : [];
      throw new ApiError(response.status, body.detail, reasons);
    }
    throw new ApiError(
      response.status,
      validationMessage(body) ?? `Request failed (${response.status})`,
    );
  }
  return body as T;
}

function get<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}

function post<T>(path: string, body?: object): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-GQB-Intent": "operator" },
    body: JSON.stringify(body ?? {}),
  });
}

export const api = {
  ladder(topicId: number, optionId: number, maxContracts = 50): Promise<LadderResponse> {
    return get(`/api/ladder/${topicId}/${optionId}?max_contracts=${maxContracts}`);
  },
  ledger(params: { mode?: ModeName; beforeId?: number; limit?: number } = {}): Promise<LedgerPage> {
    const query = new URLSearchParams();
    if (params.mode !== undefined) query.set("mode", params.mode);
    if (params.beforeId !== undefined) query.set("before_id", String(params.beforeId));
    query.set("limit", String(params.limit ?? 50));
    return get(`/api/ledger?${query.toString()}`);
  },
  backtestResults(mode?: ModeName): Promise<BacktestResultsResponse> {
    return get(`/api/backtest/results${mode === undefined ? "" : `?mode=${mode}`}`);
  },

  start(body: StartRequest): Promise<ActionResult> {
    return post("/api/bot/start", body);
  },
  stop(): Promise<ActionResult> {
    return post("/api/bot/stop");
  },
  arm(confirmationPhrase: string): Promise<ArmResponse> {
    return post("/api/bot/arm", { confirmation_phrase: confirmationPhrase });
  },
  disarm(): Promise<ActionResult> {
    return post("/api/bot/disarm");
  },
  reconcile(): Promise<ReconcileResult> {
    return post("/api/bot/reconcile");
  },
  killSwitch(reason: string): Promise<ActionResult> {
    return post("/api/bot/kill-switch", { reason });
  },
  releaseKillSwitch(confirmationPhrase: string): Promise<ActionResult> {
    return post("/api/bot/kill-switch/release", { confirmation_phrase: confirmationPhrase });
  },
  testOrder(body: TestOrderRequest): Promise<TestOrderResult> {
    return post("/api/orders/test", body);
  },
  runBacktest(control: boolean): Promise<ActionResult> {
    return post("/api/backtest/run", { control });
  },
};

export function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    return error.reasons.length > 0 ? `${error.message}: ${error.reasons.join("; ")}` : error.message;
  }
  if (error instanceof Error) return error.message;
  return "Unexpected error";
}
