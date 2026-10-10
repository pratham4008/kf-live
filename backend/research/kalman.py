"""Dynamic hedge-ratio estimation via a Kalman filter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class KalmanHedgeResult:
    beta: pd.Series          # filtered hedge ratio
    alpha: pd.Series         # filtered intercept
    spread: pd.Series        # forecast error e_t (the tradable spread)
    spread_var: pd.Series    # forecast variance S_t
    zscore: pd.Series        # e_t / sqrt(S_t)
    state_cov_trace: pd.Series = field(default=None)


class KalmanHedge:
    """State-space local regression y_t = beta_t x_t + alpha_t, coefficients a
    random walk. I use the Chan delta trick so one knob (`delta`) sets how fast
    the hedge ratio may move. The filter is strictly causal."""

    def __init__(self, delta: float = 1e-4, r: float = 1e-3,
                 p0: float = 1.0, beta0: float | None = None,
                 alpha0: float = 0.0) -> None:
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must lie in (0, 1)")
        if r <= 0:
            raise ValueError("observation noise r must be positive")
        self.delta = float(delta)
        self.r = float(r)
        self.p0 = float(p0)
        self.beta0 = beta0
        self.alpha0 = float(alpha0)

    def filter(self, y: pd.Series, x: pd.Series) -> KalmanHedgeResult:
        y, x = self._align(y, x)
        n = len(y)
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)

        Q = (self.delta / (1.0 - self.delta)) * np.eye(2)
        R = self.r

        beta_seed = self.beta0
        if beta_seed is None:
            beta_seed = yv[0] / xv[0] if xv[0] != 0 else 1.0
        theta = np.array([beta_seed, self.alpha0], dtype=float)
        P = self.p0 * np.eye(2)

        betas = np.empty(n); alphas = np.empty(n)
        spreads = np.empty(n); spread_vars = np.empty(n)
        zs = np.empty(n); traces = np.empty(n)

        for t in range(n):
            H = np.array([xv[t], 1.0])
            P_pred = P + Q                     # random-walk transition
            e = yv[t] - H @ theta              # innovation = spread
            S = H @ P_pred @ H + R
            K = (P_pred @ H) / S
            theta = theta + K * e
            P = P_pred - np.outer(K, H) @ P_pred

            betas[t] = theta[0]; alphas[t] = theta[1]
            spreads[t] = e; spread_vars[t] = S
            zs[t] = e / np.sqrt(S); traces[t] = np.trace(P)

        idx = y.index
        return KalmanHedgeResult(
            beta=pd.Series(betas, index=idx, name="beta"),
            alpha=pd.Series(alphas, index=idx, name="alpha"),
            spread=pd.Series(spreads, index=idx, name="spread"),
            spread_var=pd.Series(spread_vars, index=idx, name="spread_var"),
            zscore=pd.Series(zs, index=idx, name="zscore"),
            state_cov_trace=pd.Series(traces, index=idx, name="P_trace"),
        )

    @staticmethod
    def _align(y: pd.Series, x: pd.Series) -> tuple[pd.Series, pd.Series]:
        df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if len(df) < 2:
            raise ValueError("need at least two overlapping observations")
        return df["y"], df["x"]


def rolling_ols_hedge(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling-OLS hedge ratio -- the naive time-varying beta the filter beats."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    cov = df["y"].rolling(window).cov(df["x"])
    var = df["x"].rolling(window).var()
    return (cov / var).rename("beta_ols")
