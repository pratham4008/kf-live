"""Turn the Kalman spread into discrete trading states via a rolling z-score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rolling_zscore(spread: pd.Series, window: int = 60,
                   min_periods: int | None = None) -> pd.Series:
    """Trailing z-score of the spread level (causal). I standardise the level
    rather than the filter's predicted variance because the trade is a bet on
    the level reverting; the hedge ratio stays fully dynamic."""
    mp = min_periods or window
    mu = spread.rolling(window, min_periods=mp).mean()
    sd = spread.rolling(window, min_periods=mp).std(ddof=1)
    return ((spread - mu) / sd).rename("zscore")


@dataclass
class SignalConfig:
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    allow_short: bool = True

    def __post_init__(self) -> None:
        if not (self.exit < self.entry < self.stop):
            raise ValueError("require exit < entry < stop")


def generate_positions(zscore: pd.Series, cfg: SignalConfig | None = None) -> pd.Series:
    """Map z-score to spread positions in {-1, 0, +1} with hysteresis. +1 longs
    the spread (long y, short x); high z shorts it, low z longs it."""
    cfg = cfg or SignalConfig()
    z = zscore.to_numpy(dtype=float)
    pos = np.zeros(len(z))
    state = 0

    for t in range(len(z)):
        zt = z[t]
        if np.isnan(zt):
            pos[t] = state
            continue
        if state == 0:
            if zt >= cfg.entry:
                state = -1
            elif zt <= -cfg.entry and cfg.allow_short:
                state = 1
        else:
            if abs(zt) >= cfg.stop or abs(zt) <= cfg.exit:
                state = 0
        pos[t] = state

    return pd.Series(pos, index=zscore.index, name="position")
