# =============================================================================
# Financial RAG Agent — API Middleware
# src/financial_rag/api/middleware.py
#
# Request-level middleware applied to every HTTP request:
#   - Structured JSON request/response logging
#   - Request timing (X-Process-Time header)
#   - Global exception handler (returns RFC 7807 problem detail)
#   - Correlation ID injection (X-Request-ID header)
# =============================================================================

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)


# =============================================================================
# Request logging + timing middleware
# =============================================================================


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request and response with structured fields.
    Adds X-Request-ID and X-Process-Time headers to every response.

    Log format:
        request:  method, path, query, request_id
        response: status_code, process_time_ms, request_id
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        t0 = time.monotonic()

        # Attach request_id to request state for use in route handlers
        request.state.request_id = request_id

        logger.info(
            "request_start",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query),
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            process_ms = int((time.monotonic() - t0) * 1000)
            logger.error(
                "request_unhandled_error",
                extra={
                    "request_id": request_id,
                    "process_ms": process_ms,
                    "error": str(exc),
                },
            )
            raise

        process_ms = int((time.monotonic() - t0) * 1000)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_ms}ms"

        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_ms": process_ms,
            },
        )

        return response


# =============================================================================
# Global exception handlers
# =============================================================================


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register application-wide exception handlers on the FastAPI app.
    Returns RFC 7807 Problem Detail JSON for all unhandled errors.
    Call this in server.py after creating the FastAPI instance.
    """

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "unhandled_exception",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "type": "about:blank",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "An unexpected error occurred. Check server logs.",
                "instance": request.url.path,
                "request_id": request_id,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            "validation_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "error": str(exc),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "type": "about:blank",
                "title": "Validation Error",
                "status": 422,
                "detail": str(exc),
                "instance": request.url.path,
                "request_id": request_id,
            },
        )
