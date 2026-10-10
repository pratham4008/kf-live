# Kalman pairs, live on paper

The strategy from [kalman-pairs-trading](https://github.com/pratham4008/kalman-pairs-trading) running forward on Alpaca paper trading, with a widget that shows its live results on my portfolio site.

```
backend/            FastAPI service, trading loop, SQLite storage, tests
backend/research/   unmodified copies of the research repo's kalman.py, signals.py, backtest.py
backend/fly.toml    Fly.io deployment config
frontend/           live-strategy-widget.html  (open with ?demo to preview on simulated data)
```

## The live run is the researched strategy, not a re-implementation

The service imports `KalmanHedge`, `rolling_zscore` and `generate_positions` from the research repo unchanged. Each trading day, shortly before the close, it:

1. loads every adjusted daily close since `ANCHOR_START`, plus today's latest price;
2. runs the research pipeline on log prices: Kalman hedge ratio, 60-day rolling z-score of the spread, then entry, exit and stop bands with hysteresis;
3. converts the research backtest's dollar weights into share counts and trades toward them, re-hedging daily as `rebalance_hedge=True` does in the backtest.

Because every step is causal and the filter always starts from the same anchor date, adding one more day of data never changes an earlier position. So the live position on any day is exactly what `run_backtest.py` would hold that day. Trading at the close of day t is the live counterpart of the backtest's `execution_lag=1`.

The tests enforce this:

- the vendored research files must match their SHA-256 hashes, so they can't be edited by accident;
- running the pipeline on history up to day t gives the full backtest's position on day t;
- share targets reproduce the backtest's `_target_weights` in dollars;
- in a 250-day simulated run, every live decision equals the research path.

## Before going live

1. **Choose the pair with the research repo.** The repo leaves its results tables blank, so no pair is recorded yet. Run `scripts/screen_pairs.py`, then `scripts/run_backtest.py` on the passing pairs, and pick one on its out-of-sample scorecard. Record that choice and its scorecard in the research README before going live, so the live record can be compared with it.
2. **Set the pair and parameters** in `.env`. The parameter defaults already match the research `config.py`.
3. **Reset the Alpaca paper account to $100,000.** The service records the account equity at its first run as the starting point.

If the backtest is holding a position on the first live day, the service enters that position immediately, because it follows the backtest's state.

## Run it locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                  # 13 tests, no network needed

MODE=sim SNAPSHOT_SECONDS=1 uvicorn app:create_app --factory --port 8000
# then open http://localhost:8000/api/snapshot
```

In `sim` mode the service runs on a synthetic market, one day per tick. The API marks this data as `"mode": "simulation"` and the widget shows a banner saying so. Never present simulation numbers as results.

## Dry run against Alpaca (market hours only)

```bash
cp .env.example .env        # fill in keys, pair, parameters
set -a; source .env; set +a
DB_PATH=data/test.db DECISION_MINUTES_BEFORE_CLOSE=400 \
  uvicorn app:create_app --factory --port 8000
```

`DECISION_MINUTES_BEFORE_CLOSE=400` forces a decision right away so you can see orders go through. Check `/api/snapshot` and the Alpaca dashboard. Afterwards, close any positions, reset the paper account to $100,000, and delete `data/test.db` so the real run starts clean.

## Deploy on Fly.io

Any host that keeps a container running and offers a persistent volume works (Fly.io, Railway, Render, a small VPS). Serverless platforms that sleep between requests won't, because the trading loop has to keep running.

```bash
cd backend
fly auth login
fly launch --copy-config --no-deploy        # keep fly.toml; change `app` if the name is taken
fly volumes create kf_data --region ewr --size 1
fly secrets set ALPACA_API_KEY=... ALPACA_API_SECRET=... \
  SYMBOL_Y=... SYMBOL_X=... KF_DELTA=... KF_R=... ANCHOR_START=...
fly deploy
fly logs                                     # look for "Inception recorded"
curl https://<app>.fly.dev/health
```

Any `.env` setting not in `fly.toml` or set as a secret falls back to the default in `config.py`.

The SQLite file on `/data` is the entire track record, so back it up weekly:

```bash
fly ssh sftp get /data/live.db ./backups/live-$(date +%F).db
```

## Embed it

Copy the font `<link>` tags, the `<style>` block, the `<section id="kf-live">` element and the `<script>` block from `frontend/live-strategy-widget.html` into the portfolio page, then set `API_BASE` in the script to the deployed URL. All CSS is scoped under `#kf-live`. `ALLOWED_ORIGINS` must include the portfolio's origin so the browser can call the API.

## Choices made for credibility

- **Live results only.** The equity curve starts at the first live run, with no backtest attached to the front of it.
- **The account is the source of truth.** Equity comes from Alpaca's account balance, not my own arithmetic.
- **An honest benchmark.** SPY buy-and-hold with the same starting capital over the same period.
- **Sharpe ratio withheld early.** It appears only after 20 trading days of returns.
- **Every decision is logged.** The `signals` table records prices, z-score, hedge ratio and action for every trading day, including days with no trade.

## Where live differs from the backtest

- **Price source.** History comes from Alpaca's adjusted daily bars, while the research used Yahoo Finance. Small differences in adjusted prices can shift the hedge ratio and z-score slightly.
- **Execution price.** Orders go out about ten minutes before the close at the latest trade price, not the official close.
- **Whole shares and a minimum order size.** Positions are rounded down to whole shares, and re-hedge orders under `MIN_TRADE_DOLLARS` are skipped to avoid dozens of tiny orders.
- **Costs.** Alpaca paper fills charge no commission, and the backtest charged 1bp per leg. Borrow fees for the short leg aren't modelled in either.
- **Missed days.** If the service is down at decision time, that day is skipped. The next decision still recomputes the correct position from the full history.

## Known limitations

- The Alpaca adapter follows the `alpaca-py` SDK but hasn't been run against a live account yet. Dry-run it during market hours first.
- Alpaca's free plan serves recent prices from IEX only, a single exchange, so live prices can differ slightly from the consolidated tape.
- In the research `signals.py`, a position can be entered from flat when |z| is already beyond the stop band; the stop then closes it the next day. The live service inherits this. If you fix it in the research repo, re-copy `signals.py` and update its hash in `tests/test_core.py`.
