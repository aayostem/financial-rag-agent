# =============================================================================
# Financial RAG Agent — FastAPI Application Server
# src/financial_rag/api/server.py
#
# Entry point for the API. Wires together:
#   - FastAPI app with lifespan context manager
#   - CORS middleware
#   - Request logging middleware
#   - Global exception handlers
#   - API router (routes.py)
#
# Run:
#   uvicorn financial_rag.api.server:app --reload --port 8000
# =============================================================================

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from financial_rag.api.dependencies import (
    initialise_dependencies,
    shutdown_dependencies,
)
from financial_rag.api.middleware import (
    RequestLoggingMiddleware,
    register_exception_handlers,
)
from financial_rag.api.routes import router
from financial_rag.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan — startup and shutdown
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    Replaces the deprecated @app.on_event("startup") / ("shutdown") pattern.

    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.
    """
    settings = get_settings()

    logger.info(
        "Starting %s v%s [%s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.APP_ENV,
    )

    await initialise_dependencies()

    logger.info("Application ready — listening on %s:%d", settings.API_HOST, settings.API_PORT)

    yield  # ← application runs here

    logger.info("Shutting down %s...", settings.APP_NAME)
    await shutdown_dependencies()
    logger.info("Shutdown complete")


# =============================================================================
# Application factory
# =============================================================================


def create_app() -> FastAPI:
    """
    Build and configure the FastAPI application.

    Separated from module-level instantiation so the factory can be
    called in tests with overridden settings.
    """
    settings = get_settings()

    app = FastAPI(
        title="Financial RAG Analyst API",
        description=(
            "Production-grade financial analysis using RAG over SEC filings. "
            "Ingests 10-K, 10-Q, and 8-K documents and answers natural language "
            "financial questions grounded in primary source documents."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS_LIST,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Request logging ───────────────────────────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Root ──────────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs" if settings.DEBUG else "disabled",
        }

    return app


# =============================================================================
# Module-level app instance — used by uvicorn
# =============================================================================

app = create_app()


# =============================================================================
# Dev entrypoint
# =============================================================================


def main() -> None:
    """
    Development entrypoint.
    In production, run via: uvicorn financial_rag.api.server:app
    """
    settings = get_settings()
    uvicorn.run(
        "financial_rag.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
