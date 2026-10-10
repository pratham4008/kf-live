"""SQLite storage. The API reads only from here, so page views never hit the broker."""
from __future__ import annotations

import json
import os
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (
    ts TEXT NOT NULL, trading_day TEXT NOT NULL, equity REAL NOT NULL, bench_price REAL);
CREATE TABLE IF NOT EXISTS signals (
    ts TEXT NOT NULL, trading_day TEXT NOT NULL, price_y REAL, price_x REAL,
    z REAL, beta REAL, alpha REAL, prev_position INTEGER, new_position INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS trades (
    ts TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, qty INTEGER NOT NULL,
    order_id TEXT, status TEXT, fill_price REAL);
"""


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # meta -------------------------------------------------------------
    def get(self, key: str, default=None):
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                               "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                               (key, json.dumps(value)))
            self._conn.commit()

    # rows -------------------------------------------------------------
    def _insert(self, table: str, row: dict) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock:
            self._conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
            self._conn.commit()

    def add_snapshot(self, row: dict) -> None:
        self._insert("snapshots", row)

    def add_signal(self, row: dict) -> None:
        self._insert("signals", row)

    def add_trade(self, row: dict) -> None:
        self._insert("trades", row)

    def _select(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def snapshots(self) -> list[dict]:
        return self._select("SELECT ts, trading_day, equity, bench_price FROM snapshots ORDER BY rowid")

    def recent_trades(self, limit: int = 15) -> list[dict]:
        return self._select("SELECT * FROM trades ORDER BY rowid DESC LIMIT ?", (limit,))

    def trade_count(self) -> int:
        return self._select("SELECT COUNT(*) AS n FROM trades")[0]["n"]

    def last_signal(self) -> dict | None:
        rows = self._select("SELECT * FROM signals ORDER BY rowid DESC LIMIT 1")
        return rows[0] if rows else None
