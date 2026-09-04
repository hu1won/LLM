"""FastAPI application for the LLMBench web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llmbench import __version__
from llmbench.web.api import router

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    application = FastAPI(
        title="LLMBench",
        version=__version__,
        description="Local LLM fine-tuning workbench",
    )
    application.include_router(router, prefix="/api")

    if STATIC_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @application.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return application


app = create_app()
