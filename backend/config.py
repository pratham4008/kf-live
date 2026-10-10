"""Settings for the live Kalman pairs service, read from environment variables.

Every tunable lives here so the deployed strategy is fully described by one .env
file. Defaults mirror DEFAULTS in the research repo's config.py; keep the two in
sync so the live run is the strategy that was researched.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return cast(raw)


@dataclass(frozen=True)
class Settings:
    mode: str                      # "alpaca" (paper trading) or "sim" (offline dev only)
    api_key: str
    api_secret: str
    live_feed: str                 # feed for today's prices; "iex" on Alpaca's free plan
    history_feed: str              # feed for completed daily bars; "sip" is free for past data
    symbol_y: str                  # dependent leg (y in the research code)
    symbol_x: str                  # hedge leg (x in the research code)
    benchmark: str
    anchor_start: str              # the filter always runs from this date, like the research backtest
    starting_capital: float        # only used by the simulator; live results use real account equity
    gross_exposure: float          # gross dollars in the pair as a fraction of equity
    delta: float                   # research config: delta
    r: float                       # research config: r (observation noise)
    zscore_window: int
    entry_z: float
    exit_z: float
    stop_z: float
    rebalance_hedge: bool          # research config: rebalance_hedge
    min_trade_dollars: float       # skip re-hedge orders smaller than this
    decision_minutes_before_close: int
    snapshot_seconds: int
    db_path: str
    allowed_origins: tuple[str, ...]


def load_settings() -> Settings:
    origins = _env("ALLOWED_ORIGINS", "*")
    return Settings(
        mode=_env("MODE", "sim").lower(),
        api_key=_env("ALPACA_API_KEY", ""),
        api_secret=_env("ALPACA_API_SECRET", ""),
        live_feed=_env("ALPACA_LIVE_FEED", "iex").lower(),
        history_feed=_env("ALPACA_HISTORY_FEED", "sip").lower(),
        symbol_y=_env("SYMBOL_Y", "EWA").upper(),
        symbol_x=_env("SYMBOL_X", "EWC").upper(),
        benchmark=_env("BENCHMARK", "SPY").upper(),
        anchor_start=_env("ANCHOR_START", "2016-01-04"),
        starting_capital=_env("STARTING_CAPITAL", 100_000.0, float),
        gross_exposure=_env("GROSS_EXPOSURE", 1.0, float),
        delta=_env("KF_DELTA", 1e-4, float),
        r=_env("KF_R", 1e-3, float),
        zscore_window=_env("ZSCORE_WINDOW", 60, int),
        entry_z=_env("ENTRY_Z", 2.0, float),
        exit_z=_env("EXIT_Z", 0.5, float),
        stop_z=_env("STOP_Z", 4.0, float),
        rebalance_hedge=_env("REBALANCE_HEDGE", True, lambda v: v.strip().lower() in ("1", "true", "yes")),
        min_trade_dollars=_env("MIN_TRADE_DOLLARS", 200.0, float),
        decision_minutes_before_close=_env("DECISION_MINUTES_BEFORE_CLOSE", 10, int),
        snapshot_seconds=_env("SNAPSHOT_SECONDS", 300, int),
        db_path=_env("DB_PATH", "data/live.db"),
        allowed_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )
