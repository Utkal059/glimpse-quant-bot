"""Non-blocking telemetry bus: a log ring buffer and a fan-out to WebSocket clients.

Design constraints:

* **Never block the trading loop.** ``publish`` is synchronous and O(clients);
  each client owns a bounded queue and loses its *oldest* frames when it falls
  behind. A stalled browser costs memory bounded by ``QUEUE_SIZE`` frames.
* **No extra venue traffic.** Frames are built from in-memory engine state by
  the controller's publisher; nothing here calls Glimpse.
* **Thread-safe log capture.** structlog may emit from worker threads (model
  fits run in ``asyncio.to_thread``); those records are marshalled onto the
  event loop with ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from glimpse_quant_bot.server.schemas import LogCategory, LogEntry, LogLevel

QUEUE_SIZE = 256
LOG_BUFFER = 2000
HELLO_LOG_LINES = 200

_LEVELS: dict[str, LogLevel] = {
    "debug": "debug",
    "info": "info",
    "warning": "warning",
    "warn": "warning",
    "error": "error",
    "critical": "critical",
    "exception": "error",
}
_SKIP_FIELDS = frozenset({"event", "level", "timestamp", "log_level"})


def classify_event(event: str, level: LogLevel) -> LogCategory:
    """Map a structlog event name onto the terminal's four log streams."""
    if event.startswith(("risk.", "execution.ambiguous")) or "locked" in event:
        return "RISK_LOCK"
    if event.startswith("execution.") or event in ("strategy.settled", "operator.order"):
        return "EXECUTION"
    if event.startswith(("strategy.candidate", "coherence.", "calibration.", "signal.")):
        return "SIGNAL"
    if event.startswith("operator.") and level in ("warning", "error", "critical"):
        return "RISK_LOCK"
    return "INFO"


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:237] + "..."
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= 240 else text[:237] + "..."


@dataclass(slots=True, eq=False)
class Subscriber:
    """One WebSocket client."""

    queue: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(QUEUE_SIZE))
    focus_topic: int | None = None
    dropped: int = 0
    last_distribution_key: tuple[float, float] | None = None

    def offer(self, frame: str) -> None:
        """Enqueue without ever waiting; evict the oldest frame when full."""
        while True:
            try:
                self.queue.put_nowait(frame)
                return
            except asyncio.QueueFull:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:  # pragma: no cover - raced with a reader
                    return


class TelemetryHub:
    """Fan-out of JSON frames plus a ring buffer of structured log lines."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._subscribers: set[Subscriber] = set()
        self._logs: deque[LogEntry] = deque(maxlen=LOG_BUFFER)
        self._seq = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self.muted_threads: set[int] = set()
        """Threads whose logs stay out of the terminal (e.g. a background backtest)."""

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._loop_thread = threading.get_ident()

    # --- subscribers ----------------------------------------------------------

    @property
    def client_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> Subscriber:
        subscriber = Subscriber()
        self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.discard(subscriber)

    def subscribers(self) -> list[Subscriber]:
        return list(self._subscribers)

    @staticmethod
    def frame(channel: str, data: Any, ts: float) -> str:
        return json.dumps({"type": channel, "ts": ts, "data": data}, separators=(",", ":"))

    def publish(self, channel: str, data: Any) -> None:
        """Send one frame to every client (non-blocking)."""
        if not self._subscribers:
            return
        encoded = self.frame(channel, data, self._clock())
        for subscriber in list(self._subscribers):
            subscriber.offer(encoded)

    def send(self, subscriber: Subscriber, channel: str, data: Any) -> None:
        subscriber.offer(self.frame(channel, data, self._clock()))

    # --- logs -------------------------------------------------------------------

    def recent_logs(self, limit: int = HELLO_LOG_LINES) -> list[LogEntry]:
        if limit <= 0:
            return []
        return list(self._logs)[-limit:]

    def record_log(self, event_dict: Mapping[str, Any]) -> None:
        """Capture one structlog event (callable from any thread)."""
        if threading.get_ident() in self.muted_threads:
            return
        if self._loop is not None and threading.get_ident() != self._loop_thread:
            snapshot = dict(event_dict)
            try:
                self._loop.call_soon_threadsafe(self._append_log, snapshot)
            except RuntimeError:  # loop closed during shutdown
                return
            return
        self._append_log(event_dict)

    def _append_log(self, event_dict: Mapping[str, Any]) -> None:
        event = str(event_dict.get("event", ""))
        level = _LEVELS.get(str(event_dict.get("level", "info")).lower(), "info")
        fields = {
            str(key): _short(value)
            for key, value in event_dict.items()
            if key not in _SKIP_FIELDS and not str(key).startswith("_")
        }
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        self._seq += 1
        entry = LogEntry(
            seq=self._seq,
            ts=self._clock(),
            level=level,
            category=classify_event(event, level),
            event=event,
            message=f"{event} {detail}".strip(),
            fields=fields,
        )
        self._logs.append(entry)
        self.publish("system_logs", entry.model_dump(mode="json"))

    def processor(
        self, _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """structlog processor: tap every event, pass it through unchanged."""
        with contextlib.suppress(Exception):  # telemetry must never break logging
            self.record_log(event_dict)
        return event_dict
