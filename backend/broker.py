"""Broker adapters.

AlpacaBroker talks to Alpaca's paper-trading API. SimBroker is a synthetic
market for developing and testing offline; its results are labelled
"simulation" everywhere and must never be shown as live performance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

ET = ZoneInfo("America/New_York")


@dataclass
class Clock:
    is_open: bool
    minutes_to_close: float
    trading_day: str          # YYYY-MM-DD in New York time


class AlpacaBroker:
    def __init__(self, key: str, secret: str, live_feed: str = "iex", history_feed: str = "sip"):
        if not key or not secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET are required in alpaca mode")
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self.trading = TradingClient(key, secret, paper=True)   # paper=True is deliberate and fixed
        self.data = StockHistoricalDataClient(key, secret)
        pick = lambda f: DataFeed.IEX if f == "iex" else DataFeed.SIP
        self.live_feed, self.history_feed = pick(live_feed), pick(history_feed)

    def clock(self) -> Clock:
        c = self.trading.get_clock()
        now = c.timestamp
        return Clock(is_open=bool(c.is_open),
                     minutes_to_close=(c.next_close - now).total_seconds() / 60.0,
                     trading_day=now.astimezone(ET).date().isoformat())

    def equity(self) -> float:
        return float(self.trading.get_account().equity)

    def positions(self) -> dict[str, dict]:
        out = {}
        for p in self.trading.get_all_positions():
            out[p.symbol] = {"qty": int(float(p.qty)), "market_value": float(p.market_value),
                             "avg_entry_price": float(p.avg_entry_price),
                             "unrealized_pl": float(p.unrealized_pl)}
        return out

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        from alpaca.data.requests import StockLatestTradeRequest
        res = self.data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbols, feed=self.live_feed))
        return {s: float(res[s].price) for s in symbols}

    def daily_closes(self, symbols: list[str], since: str) -> list[tuple[str, dict[str, float]]]:
        """Completed, split- and dividend-adjusted daily closes from `since` (YYYY-MM-DD),
        oldest first, only for dates where every symbol has a bar. Today's in-progress
        bar is excluded; the engine appends today's live price itself."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        start = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
        end = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
        df = self.data.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=end,
            adjustment="all", feed=self.history_feed)).df
        today = datetime.now(ET).date().isoformat()
        by_day: dict[str, dict[str, float]] = {}
        for (sym, ts), row in df.iterrows():
            day = ts.astimezone(ET).date().isoformat()
            if day < today:
                by_day.setdefault(day, {})[sym] = float(row["close"])
        rows = [(d, v) for d, v in sorted(by_day.items()) if all(s in v for s in symbols)]
        return rows

    def submit_market(self, symbol: str, qty: int) -> dict:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        order = self.trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY))
        fill, status = None, str(order.status)
        for _ in range(10):                      # wait up to ~10s for the paper fill
            time.sleep(1)
            o = self.trading.get_order_by_id(order.id)
            status = str(o.status.value if hasattr(o.status, "value") else o.status)
            if o.filled_avg_price is not None:
                fill = float(o.filled_avg_price)
                break
        return {"order_id": str(order.id), "status": status, "fill_price": fill}


class SimBroker:
    """Synthetic cointegrated pair for offline development. One advance() = one trading day."""

    def __init__(self, symbols: tuple[str, str, str], capital: float,
                 history: int = 300, horizon: int = 5000, seed: int = 7):
        self.y_sym, self.x_sym, self.b_sym = symbols
        rng = np.random.default_rng(seed)
        n = history + horizon
        x = 30 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
        beta = 1.2 + np.cumsum(rng.normal(0, 0.002, n))
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.4)
        y = beta * x + 2.0 + spread
        bench = 450 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        self.series = {self.y_sym: y, self.x_sym: x, self.b_sym: bench}
        self.true_beta = beta
        self.history = history
        self.t = history
        self.cash = float(capital)
        self.pos: dict[str, int] = {}
        self.start = datetime(2026, 1, 2)

    def _day(self, t: int) -> str:
        return (self.start + timedelta(days=t)).date().isoformat()

    def advance(self) -> None:
        self.t += 1

    def clock(self) -> Clock:
        return Clock(is_open=True, minutes_to_close=5.0, trading_day=self._day(self.t))

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: float(self.series[s][self.t]) for s in symbols}

    def equity(self) -> float:
        return self.cash + sum(q * float(self.series[s][self.t]) for s, q in self.pos.items())

    def positions(self) -> dict[str, dict]:
        return {s: {"qty": q, "market_value": q * float(self.series[s][self.t]),
                    "avg_entry_price": None, "unrealized_pl": None}
                for s, q in self.pos.items() if q != 0}

    def daily_closes(self, symbols: list[str], since: str) -> list[tuple[str, dict[str, float]]]:
        return [(self._day(t), {s: float(self.series[s][t]) for s in symbols}) for t in range(self.t)]

    def submit_market(self, symbol: str, qty: int) -> dict:
        price = float(self.series[symbol][self.t])
        self.cash -= qty * price
        self.pos[symbol] = self.pos.get(symbol, 0) + qty
        return {"order_id": f"sim-{self.t}-{symbol}", "status": "filled", "fill_price": price}


def make_broker(settings):
    if settings.mode == "alpaca":
        return AlpacaBroker(settings.api_key, settings.api_secret,
                            settings.live_feed, settings.history_feed)
    if settings.mode == "sim":
        return SimBroker((settings.symbol_y, settings.symbol_x, settings.benchmark),
                         settings.starting_capital)
    raise ValueError(f"MODE must be 'alpaca' or 'sim', got {settings.mode!r}")
