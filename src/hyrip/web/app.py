"""Web app assembly: FastAPI + the SPA from web/static, run through `hyrip serve`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..service.session import SessionService
from .routes import router


class GenerationLimiter:
    """One in-flight paper generation app-wide."""

    def __init__(self, limit: int = 1):
        self.semaphore = asyncio.Semaphore(limit)


def build_app(service: SessionService, config: Config) -> FastAPI:
    app = FastAPI(title="hyrip", docs_url=None, redoc_url=None)
    app.state.service = service
    app.state.config = config
    app.state.limiter = GenerationLimiter()
    app.include_router(router)
    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


def run_server(service: SessionService, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    config = service.config
    uvicorn.run(
        build_app(service, config),
        host=host,
        port=port,
        log_level="info",
    )
