/**
 * Endpoints of `glimpse-quant-bot serve`. `NEXT_PUBLIC_*` values are inlined at
 * build time, so they must be referenced literally here.
 */
export const API_URL: string = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

export const WS_URL: string =
  process.env.NEXT_PUBLIC_WS_URL ?? `${API_URL.replace(/^http/, "ws")}/ws/telemetry`;
