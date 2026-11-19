"""Append-only trade ledger: SQLite for queries, NDJSON for audit.

Every signal, estimate, fill, settlement, reconciliation and risk event is one
immutable row. Rows are never updated or deleted -- SQLite triggers reject both
-- so positions, cash and P&L are always *derived* by replaying the rows. That
makes the ledger a trustworthy reference to reconcile against the exchange
after a ``GlimpseAmbiguousTradeStateError``: whatever the bot believed before
the failed call is exactly what the rows say.

Money is stored as integer millisats and quotes as 0-100 price points, in
separate columns, mirroring the SDK's ``Millisats`` / ``PriceUnits`` split.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glimpse_markets.money import Millisats, PriceUnits

from glimpse_quant_bot.types import (
    PAYOUT_PRICE,
    BucketSignal,
    LedgerRecord,
    Mode,
    Position,
    RecordKind,
    TradePlan,
    price_points_to_millisats,
)

PositionKey = tuple[int, int]
"""``(topic_id, option_id)``."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    mode TEXT NOT NULL,
    kind TEXT NOT NULL,
    topic_id INTEGER,
    option_id INTEGER,
    bucket_name TEXT,
    contracts REAL NOT NULL DEFAULT 0,
    cash_flow_millisats INTEGER NOT NULL DEFAULT 0,
    commission_millisats INTEGER NOT NULL DEFAULT 0,
    exec_price REAL,
    model_probability REAL,
    market_price REAL,
    realized_pnl_millisats INTEGER NOT NULL DEFAULT 0,
    trade_id TEXT,
    note TEXT NOT NULL DEFAULT '',
    payload TEXT
);
CREATE INDEX IF NOT EXISTS records_mode_kind ON records (mode, kind);
CREATE TRIGGER IF NOT EXISTS records_no_update BEFORE UPDATE ON records
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS records_no_delete BEFORE DELETE ON records
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
"""

_POSITION_KINDS = (
    RecordKind.ENTRY.value,
    RecordKind.EXIT.value,
    RecordKind.SETTLEMENT.value,
    RecordKind.ADJUSTMENT.value,
)
_EPSILON = 1.0e-9


@dataclass(frozen=True, slots=True)
class LedgerSummary:
    mode: Mode
    starting_millisats: Millisats
    cash_millisats: Millisats
    open_cost_basis_millisats: Millisats
    realized_pnl_millisats: Millisats
    commission_millisats: Millisats
    entries: int
    exits: int
    settlements: int
    wins: int
    losses: int
    open_positions: int

    @property
    def roi(self) -> float:
        """Realised return on the starting bankroll."""
        return int(self.realized_pnl_millisats) / int(self.starting_millisats)

    @property
    def win_rate(self) -> float:
        closed = self.wins + self.losses
        return self.wins / closed if closed else 0.0


class Ledger:
    """Append-only store of everything the bot decided and did."""

    def __init__(
        self,
        db_path: Path | str,
        audit_path: Path | str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._clock = clock
        self._audit_path = Path(audit_path) if audit_path is not None else None
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        if self._audit_path is not None:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    # --- writing ------------------------------------------------------------

    def _append(
        self,
        mode: Mode,
        kind: RecordKind,
        *,
        topic_id: int | None = None,
        option_id: int | None = None,
        bucket_name: str | None = None,
        contracts: float = 0.0,
        cash_flow: Millisats = Millisats(0),  # noqa: B008
        commission: Millisats = Millisats(0),  # noqa: B008
        exec_price: PriceUnits | None = None,
        model_probability: float | None = None,
        market_price: PriceUnits | None = None,
        realized_pnl: Millisats = Millisats(0),  # noqa: B008
        trade_id: str | None = None,
        note: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> LedgerRecord:
        ts = self._clock()
        payload_json = json.dumps(payload, sort_keys=True) if payload is not None else None
        cursor = self._db.execute(
            "INSERT INTO records (ts, mode, kind, topic_id, option_id, bucket_name, contracts,"
            " cash_flow_millisats, commission_millisats, exec_price, model_probability,"
            " market_price, realized_pnl_millisats, trade_id, note, payload)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts,
                mode,
                kind.value,
                topic_id,
                option_id,
                bucket_name,
                contracts,
                int(cash_flow),
                int(commission),
                None if exec_price is None else float(exec_price),
                model_probability,
                None if market_price is None else float(market_price),
                int(realized_pnl),
                trade_id,
                note,
                payload_json,
            ),
        )
        record = LedgerRecord(
            record_id=int(cursor.lastrowid or 0),
            ts=ts,
            mode=mode,
            kind=kind,
            topic_id=topic_id,
            option_id=option_id,
            bucket_name=bucket_name,
            contracts=contracts,
            cash_flow_millisats=cash_flow,
            commission_millisats=commission,
            exec_price=exec_price,
            model_probability=model_probability,
            market_price=market_price,
            realized_pnl_millisats=realized_pnl,
            trade_id=trade_id,
            note=note,
        )
        if self._audit_path is not None:
            line = {
                "record_id": record.record_id,
                "ts": ts,
                "mode": mode,
                "kind": kind.value,
                "topic_id": topic_id,
                "option_id": option_id,
                "bucket_name": bucket_name,
                "contracts": contracts,
                "cash_flow_millisats": int(cash_flow),
                "commission_millisats": int(commission),
                "exec_price": None if exec_price is None else float(exec_price),
                "model_probability": model_probability,
                "market_price": None if market_price is None else float(market_price),
                "realized_pnl_millisats": int(realized_pnl),
                "trade_id": trade_id,
                "note": note,
                "payload": payload,
            }
            with self._audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, sort_keys=True) + "\n")
        return record

    def record_signal(self, mode: Mode, signal: BucketSignal, note: str = "") -> LedgerRecord:
        return self._append(
            mode,
            RecordKind.SIGNAL,
            topic_id=signal.topic_id,
            option_id=signal.option_id,
            bucket_name=signal.name,
            model_probability=signal.model_probability,
            market_price=signal.market_price,
            note=note,
            payload={
                "edge_points": float(signal.edge_points),
                "components": dict(signal.components),
            },
        )

    def record_plan(self, mode: Mode, plan: TradePlan) -> LedgerRecord:
        return self._append(
            mode,
            RecordKind.ESTIMATE,
            topic_id=plan.topic_id,
            option_id=plan.option_id,
            bucket_name=plan.bucket_name,
            contracts=plan.contracts,
            commission=plan.commission_millisats,
            exec_price=plan.exec_price,
            model_probability=plan.model_probability,
            market_price=plan.market_price,
            note=plan.rationale,
            payload={
                "cost_millisats": int(plan.cost_millisats),
                "expected_profit_millisats": int(plan.expected_profit_millisats),
                "net_edge_points": float(plan.net_edge_points),
                "kelly_fraction": plan.kelly_fraction,
                "budget_millisats": int(plan.budget_millisats),
                "ladder": [
                    {
                        "contracts": point.contracts,
                        "cost_millisats": int(point.cost_millisats),
                        "commission_millisats": int(point.commission_millisats),
                        "exec_price": float(point.exec_price),
                        "net_edge_points": float(point.net_edge_points),
                        "expected_profit_millisats": int(point.expected_profit_millisats),
                        "source": point.source,
                    }
                    for point in plan.ladder
                ],
            },
        )

    def record_entry(
        self,
        mode: Mode,
        plan: TradePlan,
        cost: Millisats,
        commission: Millisats,
        trade_id: str | None = None,
        note: str = "",
    ) -> LedgerRecord:
        paid = Millisats(int(cost) + int(commission))
        return self._append(
            mode,
            RecordKind.ENTRY,
            topic_id=plan.topic_id,
            option_id=plan.option_id,
            bucket_name=plan.bucket_name,
            contracts=plan.contracts,
            cash_flow=Millisats(-int(paid)),
            commission=commission,
            exec_price=PriceUnits(int(paid) / (plan.contracts * 1000.0)),
            model_probability=plan.model_probability,
            market_price=plan.market_price,
            trade_id=trade_id,
            note=note,
        )

    def record_exit(
        self,
        mode: Mode,
        position: Position,
        contracts: float,
        proceeds: Millisats,
        commission: Millisats,
        market_price: PriceUnits | None = None,
        model_probability: float | None = None,
        trade_id: str | None = None,
        note: str = "",
    ) -> LedgerRecord:
        sold = min(contracts, position.contracts)
        net = Millisats(int(proceeds) - int(commission))
        basis_sold = round(int(position.cost_basis_millisats) * sold / position.contracts)
        return self._append(
            mode,
            RecordKind.EXIT,
            topic_id=position.topic_id,
            option_id=position.option_id,
            bucket_name=position.bucket_name,
            contracts=sold,
            cash_flow=net,
            commission=commission,
            exec_price=PriceUnits(int(net) / (sold * 1000.0)),
            model_probability=model_probability,
            market_price=market_price,
            realized_pnl=Millisats(int(net) - basis_sold),
            trade_id=trade_id,
            note=note,
        )

    def record_settlement(
        self, mode: Mode, position: Position, won: bool, fee_rate: float, note: str = ""
    ) -> LedgerRecord:
        """Binary settlement: a winning contract pays 100 points less the fee."""
        payout = (
            price_points_to_millisats(
                PriceUnits(float(PAYOUT_PRICE) * (1.0 - fee_rate)), position.contracts
            )
            if won
            else Millisats(0)
        )
        commission = (
            Millisats(
                int(price_points_to_millisats(PAYOUT_PRICE, position.contracts)) - int(payout)
            )
            if won
            else Millisats(0)
        )
        return self._append(
            mode,
            RecordKind.SETTLEMENT,
            topic_id=position.topic_id,
            option_id=position.option_id,
            bucket_name=position.bucket_name,
            contracts=position.contracts,
            cash_flow=payout,
            commission=commission,
            realized_pnl=Millisats(int(payout) - int(position.cost_basis_millisats)),
            note=note or ("won" if won else "lost"),
        )

    def record_adjustment(
        self,
        mode: Mode,
        topic_id: int,
        option_id: int,
        bucket_name: str,
        contracts_delta: float,
        cash_flow: Millisats,
        note: str,
    ) -> LedgerRecord:
        """Align the ledger with the exchange after a reconciliation."""
        return self._append(
            mode,
            RecordKind.ADJUSTMENT,
            topic_id=topic_id,
            option_id=option_id,
            bucket_name=bucket_name,
            contracts=contracts_delta,
            cash_flow=cash_flow,
            note=note,
        )

    def record_event(
        self,
        mode: Mode,
        kind: RecordKind,
        note: str,
        payload: Mapping[str, Any] | None = None,
        topic_id: int | None = None,
        option_id: int | None = None,
    ) -> LedgerRecord:
        return self._append(
            mode, kind, topic_id=topic_id, option_id=option_id, note=note, payload=payload
        )

    def record_equity(
        self, mode: Mode, cash: Millisats, positions_value: Millisats
    ) -> LedgerRecord:
        return self._append(
            mode,
            RecordKind.EQUITY,
            payload={
                "cash_millisats": int(cash),
                "positions_value_millisats": int(positions_value),
                "equity_millisats": int(cash) + int(positions_value),
            },
        )

    # --- reading ------------------------------------------------------------

    def _rows(self, mode: Mode | None, kinds: Sequence[str] | None) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if mode is not None:
            clauses.append("mode = ?")
            params.append(mode)
        if kinds:
            clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
            params.extend(kinds)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return list(
            self._db.execute(f"SELECT * FROM records{where} ORDER BY record_id", params).fetchall()
        )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> LedgerRecord:
        return LedgerRecord(
            record_id=int(row["record_id"]),
            ts=float(row["ts"]),
            mode=row["mode"],
            kind=RecordKind(row["kind"]),
            topic_id=row["topic_id"],
            option_id=row["option_id"],
            bucket_name=row["bucket_name"],
            contracts=float(row["contracts"]),
            cash_flow_millisats=Millisats(int(row["cash_flow_millisats"])),
            commission_millisats=Millisats(int(row["commission_millisats"])),
            exec_price=None if row["exec_price"] is None else PriceUnits(float(row["exec_price"])),
            model_probability=row["model_probability"],
            market_price=(
                None if row["market_price"] is None else PriceUnits(float(row["market_price"]))
            ),
            realized_pnl_millisats=Millisats(int(row["realized_pnl_millisats"])),
            trade_id=row["trade_id"],
            note=str(row["note"]),
        )

    def records(
        self,
        mode: Mode | None = None,
        kinds: Sequence[RecordKind] | None = None,
        limit: int | None = None,
    ) -> list[LedgerRecord]:
        rows = self._rows(mode, [kind.value for kind in kinds] if kinds else None)
        selected = rows[-limit:] if limit is not None else rows
        return [self._to_record(row) for row in selected]

    @staticmethod
    def _filter(
        mode: Mode | None, kinds: Sequence[RecordKind] | None
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if mode is not None:
            clauses.append("mode = ?")
            params.append(mode)
        if kinds:
            clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
            params.extend(kind.value for kind in kinds)
        return clauses, params

    def page(
        self,
        mode: Mode | None = None,
        kinds: Sequence[RecordKind] | None = None,
        limit: int = 50,
        before_id: int | None = None,
    ) -> list[LedgerRecord]:
        """Newest-first keyset page: at most ``limit`` rows with ``record_id < before_id``.

        Read-only; used by the terminal so that paging a long ledger never loads
        every row into memory.
        """
        clauses, params = self._filter(mode, kinds)
        if before_id is not None:
            clauses.append("record_id < ?")
            params.append(before_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.execute(
            f"SELECT * FROM records{where} ORDER BY record_id DESC LIMIT ?",
            [*params, max(0, limit)],
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def records_after(
        self,
        after_id: int,
        mode: Mode | None = None,
        kinds: Sequence[RecordKind] | None = None,
        limit: int = 200,
    ) -> list[LedgerRecord]:
        """Rows appended after ``after_id``, oldest first (incremental tailing)."""
        clauses, params = self._filter(mode, kinds)
        clauses.append("record_id > ?")
        params.append(after_id)
        rows = self._db.execute(
            f"SELECT * FROM records WHERE {' AND '.join(clauses)} ORDER BY record_id LIMIT ?",
            [*params, max(0, limit)],
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def count(self, mode: Mode | None = None, kinds: Sequence[RecordKind] | None = None) -> int:
        clauses, params = self._filter(mode, kinds)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._db.execute(f"SELECT COUNT(*) AS n FROM records{where}", params).fetchone()
        return int(row["n"])

    def last_record_id(self) -> int:
        row = self._db.execute("SELECT COALESCE(MAX(record_id), 0) AS n FROM records").fetchone()
        return int(row["n"])

    def equity_curve(self, mode: Mode, limit: int = 5000) -> list[tuple[float, Millisats]]:
        """``(ts, equity)`` from the periodic equity snapshots, oldest first."""
        rows = self._db.execute(
            "SELECT ts, payload FROM ("
            " SELECT record_id, ts, payload FROM records WHERE mode = ? AND kind = ?"
            " ORDER BY record_id DESC LIMIT ?"
            ") ORDER BY record_id",
            (mode, RecordKind.EQUITY.value, max(0, limit)),
        ).fetchall()
        curve: list[tuple[float, Millisats]] = []
        for row in rows:
            if row["payload"] is None:
                continue
            payload = json.loads(row["payload"])
            equity = payload.get("equity_millisats")
            if isinstance(equity, int):
                curve.append((float(row["ts"]), Millisats(equity)))
        return curve

    def payload(self, record_id: int) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT payload FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        if row is None or row["payload"] is None:
            return {}
        loaded: dict[str, Any] = json.loads(row["payload"])
        return loaded

    def positions(self, mode: Mode) -> dict[PositionKey, Position]:
        """Open positions, derived by replaying fills in order."""
        contracts: dict[PositionKey, float] = {}
        basis: dict[PositionKey, float] = {}
        weighted_probability: dict[PositionKey, float] = {}
        names: dict[PositionKey, str] = {}
        for row in self._rows(mode, _POSITION_KINDS):
            if row["topic_id"] is None or row["option_id"] is None:
                continue
            key = (int(row["topic_id"]), int(row["option_id"]))
            held = contracts.get(key, 0.0)
            kind = row["kind"]
            amount = float(row["contracts"])
            names[key] = row["bucket_name"] or names.get(key, "")
            is_adjustment = kind == RecordKind.ADJUSTMENT.value
            if kind == RecordKind.ENTRY.value or (is_adjustment and amount > 0.0):
                contracts[key] = held + amount
                basis[key] = basis.get(key, 0.0) + max(0.0, -float(row["cash_flow_millisats"]))
                weighted_probability[key] = weighted_probability.get(key, 0.0) + amount * float(
                    row["model_probability"] or 0.0
                )
            elif kind == RecordKind.SETTLEMENT.value:
                contracts[key] = 0.0
            elif held > _EPSILON:
                removed = min(abs(amount), held)
                keep = (held - removed) / held
                contracts[key] = held - removed
                basis[key] = basis.get(key, 0.0) * keep
                weighted_probability[key] = weighted_probability.get(key, 0.0) * keep
        return {
            key: Position(
                topic_id=key[0],
                option_id=key[1],
                bucket_name=names.get(key, ""),
                contracts=held,
                cost_basis_millisats=Millisats(round(basis.get(key, 0.0))),
                entry_model_probability=weighted_probability.get(key, 0.0) / held,
            )
            for key, held in contracts.items()
            if held > _EPSILON
        }

    def cash_millisats(self, mode: Mode, starting: Millisats) -> Millisats:
        row = self._db.execute(
            f"SELECT COALESCE(SUM(cash_flow_millisats), 0) AS total FROM records"
            f" WHERE mode = ? AND kind IN ({','.join('?' for _ in _POSITION_KINDS)})",
            (mode, *_POSITION_KINDS),
        ).fetchone()
        return Millisats(int(starting) + int(row["total"]))

    def realized_pnl_millisats(self, mode: Mode) -> Millisats:
        row = self._db.execute(
            "SELECT COALESCE(SUM(realized_pnl_millisats), 0) AS total FROM records WHERE mode = ?",
            (mode,),
        ).fetchone()
        return Millisats(int(row["total"]))

    def summary(self, mode: Mode, starting: Millisats) -> LedgerSummary:
        entries = exits = settlements = wins = losses = 0
        commission = 0
        for row in self._rows(mode, _POSITION_KINDS):
            kind = row["kind"]
            commission += int(row["commission_millisats"])
            if kind == RecordKind.ENTRY.value:
                entries += 1
                continue
            if kind == RecordKind.EXIT.value:
                exits += 1
            elif kind == RecordKind.SETTLEMENT.value:
                settlements += 1
            else:
                continue
            if int(row["realized_pnl_millisats"]) > 0:
                wins += 1
            else:
                losses += 1
        open_positions = self.positions(mode)
        return LedgerSummary(
            mode=mode,
            starting_millisats=starting,
            cash_millisats=self.cash_millisats(mode, starting),
            open_cost_basis_millisats=Millisats(
                sum(int(position.cost_basis_millisats) for position in open_positions.values())
            ),
            realized_pnl_millisats=self.realized_pnl_millisats(mode),
            commission_millisats=Millisats(commission),
            entries=entries,
            exits=exits,
            settlements=settlements,
            wins=wins,
            losses=losses,
            open_positions=len(open_positions),
        )
