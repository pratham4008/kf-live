"""Performance metrics from live snapshots only. Nothing here touches backtest data."""
from __future__ import annotations

import math
import statistics

MIN_DAYS_FOR_SHARPE = 20


def max_drawdown(values: list[float]) -> float:
    peak, worst = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, v / peak - 1.0)
    return worst


def daily_closes(snapshots: list[dict]) -> list[float]:
    """Last equity reading of each trading day (New York date)."""
    by_day: dict[str, float] = {}
    for s in snapshots:
        by_day[s["trading_day"]] = s["equity"]
    return [by_day[d] for d in sorted(by_day)]


def summarize(snapshots: list[dict], inception_equity: float, bench_start: float | None) -> dict:
    if not snapshots or inception_equity <= 0:
        return {"equity": None, "total_return": None, "benchmark_return": None,
                "excess_return": None, "max_drawdown": None, "sharpe": None,
                "sharpe_note": "No live data yet.", "days_live": 0}

    last = snapshots[-1]
    total = last["equity"] / inception_equity - 1.0
    bench = None
    if bench_start and last.get("bench_price"):
        bench = last["bench_price"] / bench_start - 1.0

    closes = daily_closes(snapshots)
    rets = [b / a - 1.0 for a, b in zip(closes, closes[1:]) if a > 0]
    sharpe, note = None, None
    if len(rets) < MIN_DAYS_FOR_SHARPE:
        note = f"Shown after {MIN_DAYS_FOR_SHARPE} trading days of returns ({len(rets)} so far)."
    else:
        sd = statistics.stdev(rets)
        sharpe = statistics.fmean(rets) / sd * math.sqrt(252) if sd > 0 else None

    return {
        "equity": last["equity"],
        "total_return": total,
        "benchmark_return": bench,
        "excess_return": (total - bench) if bench is not None else None,
        "max_drawdown": max_drawdown([inception_equity] + [s["equity"] for s in snapshots]),
        "sharpe": sharpe,
        "sharpe_note": note,
        "days_live": len(closes),
    }


def downsample(points: list, max_points: int = 400) -> list:
    if len(points) <= max_points:
        return points
    step = math.ceil(len(points) / max_points)
    out = points[::step]
    if out[-1] is not points[-1]:
        out.append(points[-1])
    return out
