"""Orchestration: one research-identical decision per trading day, plus equity snapshots.

Timeline of a trading day (alpaca mode):
  - every SNAPSHOT_SECONDS while the market is open: record account equity and benchmark price
  - once, DECISION_MINUTES_BEFORE_CLOSE before the close: rerun the research pipeline over
    every daily close since ANCHOR_START plus today's latest price, then trade toward the
    research backtest's target weights. Trading at the close of day t is the live equivalent
    of the backtest's execution_lag=1 (weights set at t earn the t -> t+1 return).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from metrics import downsample, summarize
from strategy import FLAT, LABELS, compute_signal, target_shares

log = logging.getLogger("engine")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Engine:
    def __init__(self, settings, broker, store):
        self.s = settings
        self.broker = broker
        self.store = store

    # setup ------------------------------------------------------------
    def bootstrap(self) -> None:
        if self.store.get("inception") is None:
            equity = self.broker.equity()
            bench = self.broker.latest_prices([self.s.benchmark])[self.s.benchmark]
            self.store.set("inception", {"ts": now_iso(), "equity": equity, "bench_price": bench})
            self.store.set("position", FLAT)
            log.info("Inception recorded: equity %.2f", equity)

    # loop body ----------------------------------------------------------
    def tick(self) -> None:
        clock = self.broker.clock()
        self.store.set("market_open", clock.is_open)
        if not clock.is_open:
            return
        self.record_snapshot(clock.trading_day)
        if (clock.minutes_to_close <= self.s.decision_minutes_before_close
                and self.store.get("last_decision_day") != clock.trading_day):
            self.decide(clock.trading_day)
        self.store.set("last_tick", now_iso())

    def record_snapshot(self, trading_day: str) -> None:
        equity = self.broker.equity()
        bench = self.broker.latest_prices([self.s.benchmark])[self.s.benchmark]
        self.store.add_snapshot({"ts": now_iso(), "trading_day": trading_day,
                                 "equity": equity, "bench_price": bench})
        self.store.set("positions", self.broker.positions())

    def history_with_today(self, trading_day: str) -> tuple[pd.DataFrame, dict]:
        y, x = self.s.symbol_y, self.s.symbol_x
        rows = [r for r in self.broker.daily_closes([y, x], self.s.anchor_start) if r[0] < trading_day]
        live = self.broker.latest_prices([y, x])
        rows.append((trading_day, live))
        frame = pd.DataFrame([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))
        return frame, live

    def decide(self, trading_day: str) -> None:
        y_sym, x_sym = self.s.symbol_y, self.s.symbol_x
        closes, px = self.history_with_today(trading_day)
        sig = compute_signal(closes, y_sym, x_sym, delta=self.s.delta, r=self.s.r,
                             zscore_window=self.s.zscore_window, entry=self.s.entry_z,
                             exit_=self.s.exit_z, stop=self.s.stop_z,
                             rebalance_hedge=self.s.rebalance_hedge)
        current = self.store.get("position", FLAT)
        target = sig["position"]

        if target != current:
            note = f"{LABELS[current]} -> {LABELS[target]}"
        elif target != FLAT and self.s.rebalance_hedge:
            note = "re-hedge"
        else:
            note = "hold"

        if note != "hold":
            ty, tx = target_shares(target, sig["hedge_beta"], px[y_sym], px[x_sym],
                                   self.broker.equity(), self.s.gross_exposure)
            held = self.broker.positions()
            for sym, tgt in ((y_sym, ty), (x_sym, tx)):
                diff = tgt - held.get(sym, {}).get("qty", 0)
                closing = tgt == 0
                if diff and (closing or abs(diff) * px[sym] >= self.s.min_trade_dollars):
                    res = self.broker.submit_market(sym, diff)
                    self.store.add_trade({"ts": now_iso(), "symbol": sym,
                                          "side": "buy" if diff > 0 else "sell",
                                          "qty": abs(diff), **res})
            self.store.set("position", target)
            self.store.set("positions", self.broker.positions())   # reflect today's fills

        self.store.add_signal({"ts": now_iso(), "trading_day": trading_day,
                               "price_y": px[y_sym], "price_x": px[x_sym], "z": sig["z"],
                               "beta": sig["hedge_beta"], "alpha": None,
                               "prev_position": current, "new_position": target, "note": note})
        self.store.set("observations", sig["observations"])
        self.store.set("last_decision_day", trading_day)
        log.info("%s z=%s beta=%.4f %s", trading_day,
                 "n/a" if sig["z"] is None else f"{sig['z']:.2f}", sig["hedge_beta"], note)

    # public view ----------------------------------------------------------
    def public_snapshot(self) -> dict:
        inception = self.store.get("inception") or {}
        snaps = self.store.snapshots()
        inc_eq = inception.get("equity") or 0.0
        bench0 = inception.get("bench_price")
        curve = [{"t": s["ts"], "day": s["trading_day"], "equity": s["equity"],
                  "benchmark": (inc_eq * s["bench_price"] / bench0) if bench0 and s["bench_price"] else None}
                 for s in snaps]
        sig = self.store.last_signal()
        position = self.store.get("position", FLAT)
        return {
            "mode": "paper" if self.s.mode == "alpaca" else "simulation",
            "strategy": {"name": "Kalman-filtered pairs", "y": self.s.symbol_y, "x": self.s.symbol_x,
                         "benchmark": self.s.benchmark, "entry_z": self.s.entry_z,
                         "exit_z": self.s.exit_z, "stop_z": self.s.stop_z,
                         "zscore_window": self.s.zscore_window,
                         "rebalance_hedge": self.s.rebalance_hedge,
                         "gross_exposure": self.s.gross_exposure,
                         "anchor_start": self.s.anchor_start},
            "inception": inception or None,
            "updated_at": self.store.get("last_tick"),
            "market_open": self.store.get("market_open", False),
            "metrics": {**summarize(snaps, inc_eq, bench0), "n_trades": self.store.trade_count()},
            "equity_curve": downsample(curve),
            "signal": {"z": sig["z"] if sig else None, "beta": sig["beta"] if sig else None,
                       "as_of": sig["trading_day"] if sig else None, "note": sig["note"] if sig else None,
                       "position": LABELS[position],
                       "warming_up": bool(sig) and sig["z"] is None,
                       "observations": self.store.get("observations", 0)},
            "positions": self.store.get("positions", {}),
            "trades": self.store.recent_trades(),
            "last_error": self.store.get("last_error"),
        }
