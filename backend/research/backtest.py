"""Causal, cost-aware, dollar-neutral backtest for a single Kalman pair.

Guards against the two mistakes that flatter pairs backtests: look-ahead (the
weight is lagged before it meets the return) and free rebalancing (costs are
charged on the full change in dollar weights, so daily re-hedging is paid for).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    cost_bps: float = 1.0
    rebalance_hedge: bool = True   # True: re-hedge daily; False: freeze beta at entry
    execution_lag: int = 1


@dataclass
class BacktestResult:
    returns: pd.Series
    gross_returns: pd.Series
    equity: pd.Series
    weight_y: pd.Series
    weight_x: pd.Series
    turnover: pd.Series
    positions: pd.Series


def _target_weights(position: pd.Series, beta: pd.Series,
                    rebalance_hedge: bool) -> tuple[pd.Series, pd.Series]:
    """Dollar weights on (y, x) scaled to unit gross. Long spread => long y,
    short beta*x. If not re-hedging, beta is frozen at the value held on entry."""
    pos = position.to_numpy(dtype=float)
    b = beta.to_numpy(dtype=float)

    if not rebalance_hedge:
        b_eff = b.copy()
        held = np.nan
        for t in range(len(pos)):
            if pos[t] != 0 and (t == 0 or pos[t - 1] != pos[t]):
                held = b[t]
            b_eff[t] = held if pos[t] != 0 else b[t]
        b = b_eff

    gross = 1.0 + np.abs(b)
    idx = position.index
    return (pd.Series(pos / gross, index=idx, name="w_y"),
            pd.Series(pos * (-b) / gross, index=idx, name="w_x"))


def run_backtest(log_prices: pd.DataFrame, y: str, x: str,
                 positions: pd.Series, beta: pd.Series,
                 cfg: BacktestConfig | None = None) -> BacktestResult:
    """Backtest one pair. Returns = lagged weights . next-period log returns,
    net of turnover costs; reported per dollar of gross exposure."""
    cfg = cfg or BacktestConfig()

    df = log_prices[[y, x]].dropna()
    ry = df[y].diff()                    # return realised on day t
    rx = df[x].diff()

    wy, wx = _target_weights(positions.reindex(df.index).fillna(0.0),
                             beta.reindex(df.index).ffill(),
                             cfg.rebalance_hedge)

    # Lag the weights (not lead the returns) to keep the fill causal.
    wy_eff = wy.shift(cfg.execution_lag).fillna(0.0)
    wx_eff = wx.shift(cfg.execution_lag).fillna(0.0)

    gross_ret = (wy_eff * ry + wx_eff * rx).rename("gross_returns")

    turnover = ((wy_eff.diff().abs().fillna(wy_eff.abs()))
                + (wx_eff.diff().abs().fillna(wx_eff.abs()))).rename("turnover")
    net_ret = (gross_ret - turnover * (cfg.cost_bps / 1e4)).rename("returns").fillna(0.0)

    return BacktestResult(
        returns=net_ret,
        gross_returns=gross_ret.fillna(0.0),
        equity=(1.0 + net_ret).cumprod().rename("equity"),
        weight_y=wy_eff, weight_x=wx_eff, turnover=turnover,
        positions=positions.reindex(df.index).fillna(0.0),
    )
