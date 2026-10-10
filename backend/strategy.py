"""Glue between the research code and live orders.

compute_signal runs the research pipeline unchanged (log prices -> KalmanHedge ->
rolling_zscore -> generate_positions) over the full history since a fixed anchor
date. Because every step is causal, extending the history by one day never
changes earlier positions, so the live position on any day equals the position
the research backtest would hold on that day.

target_shares converts the research backtest's dollar weights into share counts.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.kalman import KalmanHedge
from research.signals import SignalConfig, generate_positions, rolling_zscore

FLAT, LONG, SHORT = 0, 1, -1
LABELS = {FLAT: "flat", LONG: "long spread", SHORT: "short spread"}


def compute_signal(closes: pd.DataFrame, y: str, x: str, *, delta: float, r: float,
                   zscore_window: int, entry: float, exit_: float, stop: float,
                   rebalance_hedge: bool = True) -> dict:
    """closes: price levels indexed by date, oldest first, with columns y and x."""
    logp = np.log(closes[[y, x]].astype(float))
    res = KalmanHedge(delta=delta, r=r).filter(logp[y], logp[x])
    z = rolling_zscore(res.spread, window=zscore_window)
    pos = generate_positions(z, SignalConfig(entry, exit_, stop))

    position = int(pos.iloc[-1])
    beta_now = float(res.beta.iloc[-1])
    hedge_beta = beta_now
    if not rebalance_hedge and position != 0:
        # Same rule as research backtest._target_weights: freeze beta on entry day.
        p = pos.to_numpy()
        t = len(p) - 1
        while t > 0 and p[t - 1] == p[t]:
            t -= 1
        hedge_beta = float(res.beta.iloc[t])

    z_last = float(z.iloc[-1])
    return {
        "position": position,
        "beta": beta_now,
        "hedge_beta": hedge_beta,
        "z": None if math.isnan(z_last) else z_last,
        "spread": float(res.spread.iloc[-1]),
        "observations": len(logp),
        "positions": pos,          # full path, used by tests
    }


def target_shares(position: int, beta: float, price_y: float, price_x: float,
                  equity: float, gross_exposure: float) -> tuple[int, int]:
    """Research weights: w_y = pos / (1 + |b|), w_x = -pos * b / (1 + |b|), unit gross.
    Scaled here to equity * gross_exposure of gross dollars, truncated to whole shares."""
    if position == FLAT or price_y <= 0 or price_x <= 0:
        return 0, 0
    gross = equity * gross_exposure
    denom = 1.0 + abs(beta)
    w_y = position / denom
    w_x = -position * beta / denom
    return int(w_y * gross / price_y), int(w_x * gross / price_x)
