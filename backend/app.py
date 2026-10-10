"""HTTP API plus the background loop, in one process.

Run:  uvicorn app:create_app --factory --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import threading
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from broker import make_broker
from config import load_settings
from engine import Engine, now_iso
from store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

CLOSED_POLL_SECONDS = 900


def run_loop(engine: Engine, settings, stop: threading.Event, max_ticks: int | None = None) -> None:
    try:
        engine.bootstrap()
    except Exception:
        engine.store.set("last_error", {"ts": now_iso(), "error": traceback.format_exc(limit=3)})
        log.exception("Bootstrap failed")
        return
    ticks = 0
    while not stop.is_set():
        wait = settings.snapshot_seconds
        try:
            engine.tick()
            if not engine.store.get("market_open", False):
                wait = CLOSED_POLL_SECONDS
            if settings.mode == "sim":
                engine.broker.advance()
        except Exception:
            engine.store.set("last_error", {"ts": now_iso(), "error": traceback.format_exc(limit=3)})
            log.exception("Tick failed")
        ticks += 1
        if max_ticks is not None and ticks >= max_ticks:
            break
        stop.wait(wait)


def create_app(settings=None, broker=None, store=None, start_loop: bool = True) -> FastAPI:
    settings = settings or load_settings()
    store = store or Store(settings.db_path)
    broker = broker or make_broker(settings)
    engine = Engine(settings, broker, store)
    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(_app):
        thread = None
        if start_loop:
            thread = threading.Thread(target=run_loop, args=(engine, settings, stop), daemon=True)
            thread.start()
        yield
        stop.set()
        if thread:
            thread.join(timeout=5)

    app = FastAPI(title="Kalman pairs live", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                       allow_methods=["GET"], allow_headers=["*"])
    app.state.engine = engine

    @app.get("/health")
    def health():
        return {"ok": True, "mode": settings.mode, "last_tick": store.get("last_tick")}

    @app.get("/api/snapshot")
    def snapshot():
        return engine.public_snapshot()

    return app

