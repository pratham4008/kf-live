import hashlib
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app, run_loop  # noqa: E402
from broker import SimBroker  # noqa: E402
from config import load_settings  # noqa: E402
from metrics import max_drawdown, summarize  # noqa: E402
from research.backtest import _target_weights  # noqa: E402
from research.kalman import KalmanHedge  # noqa: E402
from research.signals import SignalConfig, generate_positions, rolling_zscore  # noqa: E402
from store import Store  # noqa: E402
from strategy import FLAT, LONG, SHORT, compute_signal, target_shares  # noqa: E402

PARAMS = dict(delta=1e-4, r=1e-3, zscore_window=60, entry=2.0, exit_=0.5, stop=4.0)

# SHA-256 of the files copied from github.com/pratham4008/kalman-pairs-trading (src/, commit cd24be7).
# If one of these tests fails, the vendored research code was edited; re-copy it from the research repo.
VENDORED = {
    "kalman.py": "60756e4ae32da538a7f178c2b046c85d6ec281ea22ca06b27c7b62abf8ba53c5",
    "signals.py": "4e80ec0f861436fe1b5ec51404f58e9a923d06049f722cf9b49b3de293cacd90",
    "backtest.py": "41c4504e50334a3b9dc0e143cd64a35190ce8b8351621d8bef4b7b8136802a59",
}


def _prices(n=900, seed=3):
    """Cointegrated price levels with a drifting log hedge ratio."""
    rng = np.random.default_rng(seed)
    lx = np.cumsum(rng.normal(0, 0.01, n)) + np.log(30)
    beta = np.linspace(0.8, 1.2, n)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.93 * noise[t - 1] + rng.normal(0, 0.008)
    ly = beta * lx + 0.3 + noise
    idx = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame({"Y": np.exp(ly), "X": np.exp(lx)}, index=idx)


# research code is used unmodified ------------------------------------------------
@pytest.mark.parametrize("name", sorted(VENDORED))
def test_vendored_research_code_unmodified(name):
    digest = hashlib.sha256((ROOT / "research" / name).read_bytes()).hexdigest()
    assert digest == VENDORED[name]


def test_signal_matches_research_pipeline_on_full_history():
    closes = _prices()
    sig = compute_signal(closes, "Y", "X", **PARAMS)
    logp = np.log(closes)
    res = KalmanHedge(delta=1e-4, r=1e-3).filter(logp["Y"], logp["X"])
    pos = generate_positions(rolling_zscore(res.spread, 60), SignalConfig(2.0, 0.5, 4.0))
    assert sig["position"] == int(pos.iloc[-1])
    assert sig["beta"] == pytest.approx(float(res.beta.iloc[-1]), abs=0)


def test_live_position_each_day_equals_backtest_position():
    """The key guarantee: deciding with history up to day t reproduces the full
    backtest's position on day t, so the live record is the researched strategy."""
    closes = _prices()
    full = compute_signal(closes, "Y", "X", **PARAMS)["positions"]
    assert full.abs().sum() > 0, "fixture should produce trades"
    for t in range(200, len(closes), 7):
        live = compute_signal(closes.iloc[: t + 1], "Y", "X", **PARAMS)
        assert live["position"] == int(full.iloc[t]), t


# sizing matches research weights -----------------------------------------------------
@pytest.mark.parametrize("position", [LONG, SHORT])
def test_share_targets_match_research_dollar_weights(position):
    beta, py, px, equity = 0.93, 24.0, 31.0, 100_000.0
    ny, nx = target_shares(position, beta, py, px, equity, gross_exposure=1.0)
    wy, wx = _target_weights(pd.Series([float(position)]), pd.Series([beta]), True)
    assert ny * py == pytest.approx(wy.iloc[0] * equity, abs=py)
    assert nx * px == pytest.approx(wx.iloc[0] * equity, abs=px)
    assert np.sign(ny) == position and np.sign(nx) == -position


def test_flat_means_no_shares():
    assert target_shares(FLAT, 1.1, 20, 30, 100_000, 1.0) == (0, 0)


def test_frozen_hedge_uses_entry_day_beta():
    closes = _prices()
    full = compute_signal(closes, "Y", "X", **PARAMS)["positions"].to_numpy()
    held = [t for t in range(1, len(full)) if full[t] != 0 and full[t - 1] == full[t]]
    t = held[len(held) // 2]
    entry = t
    while full[entry - 1] == full[t]:
        entry -= 1
    frozen = compute_signal(closes.iloc[: t + 1], "Y", "X", **PARAMS, rebalance_hedge=False)
    logp = np.log(closes)
    betas = KalmanHedge(delta=1e-4, r=1e-3).filter(logp["Y"], logp["X"]).beta
    assert frozen["hedge_beta"] == pytest.approx(float(betas.iloc[entry]))


# metrics --------------------------------------------------------------------------
def test_max_drawdown():
    assert max_drawdown([100, 120, 90, 130, 117]) == pytest.approx(-0.25)


def test_sharpe_withheld_on_short_history():
    snaps = [{"ts": f"2026-01-{d:02d}T20:00:00", "trading_day": f"2026-01-{d:02d}",
              "equity": 100_000 + d * 10, "bench_price": 500 + d} for d in range(1, 6)]
    m = summarize(snaps, 100_000, 500)
    assert m["sharpe"] is None and "20 trading days" in m["sharpe_note"]
    assert m["total_return"] == pytest.approx(0.0005)
    assert m["benchmark_return"] == pytest.approx(0.01)


def test_empty_summary():
    assert summarize([], 0, None)["equity"] is None


# end to end ------------------------------------------------------------------------
def test_simulated_run_and_api(tmp_path):
    from fastapi.testclient import TestClient
    s = replace(load_settings(), mode="sim", db_path=str(tmp_path / "t.db"), snapshot_seconds=0)
    broker = SimBroker((s.symbol_y, s.symbol_x, s.benchmark), 100_000)
    store = Store(s.db_path)
    app = create_app(s, broker, store, start_loop=False)
    run_loop(app.state.engine, s, threading.Event(), max_ticks=250)

    body = TestClient(app).get("/api/snapshot").json()
    assert body["mode"] == "simulation"
    assert body["last_error"] is None
    assert body["inception"]["equity"] == pytest.approx(100_000)
    assert len(body["equity_curve"]) == 250
    assert body["metrics"]["n_trades"] > 0
    assert body["strategy"]["entry_z"] == 2.0 and body["strategy"]["exit_z"] == 0.5

    signals = store._select("SELECT trading_day, note FROM signals ORDER BY rowid")
    assert len({r["trading_day"] for r in signals}) == 250
    assert body["signal"]["as_of"] == signals[-1]["trading_day"]
    held = {k: v["qty"] for k, v in body["positions"].items() if v["qty"]}
    assert (body["signal"]["position"] == "flat") == (not held)

    # the live path must equal the research path on the same simulated closes
    closes = pd.DataFrame(
        {sym: broker.series[sym][: broker.t] for sym in (s.symbol_y, s.symbol_x)},
        index=pd.bdate_range("2000-01-03", periods=broker.t))
    research = compute_signal(closes, s.symbol_y, s.symbol_x, delta=s.delta, r=s.r,
                              zscore_window=s.zscore_window, entry=s.entry_z,
                              exit_=s.exit_z, stop=s.stop_z)["positions"].iloc[-250:]
    live = [r["new_position"] for r in store._select("SELECT new_position FROM signals ORDER BY rowid")]
    assert live == [int(v) for v in research]
