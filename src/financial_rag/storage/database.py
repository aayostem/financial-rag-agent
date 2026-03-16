# =============================================================================
# Financial RAG Agent — Database Client
# src/financial_rag/storage/database.py
#
# Async PostgreSQL client built on SQLAlchemy 2.0 + asyncpg.
# Provides:
#   - Connection pool lifecycle management
#   - Typed session context manager
#   - Health check
#   - pgvector extension verification
# =============================================================================

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from financial_rag.config import get_settings
from financial_rag.utils.exceptions import DatabaseConnectionError, DatabaseQueryError

logger = logging.getLogger(__name__)


# =============================================================================
# ORM Base — shared by all models
# =============================================================================


class Base(DeclarativeBase):
    """
    Declarative base for all SQLAlchemy ORM models.
    Import this in repository files to define table mappings.
    """

    pass


# =============================================================================
# Engine factory
# =============================================================================


def _build_engine() -> AsyncEngine:
    """
    Build the async SQLAlchemy engine from application settings.

    Pool configuration:
      pool_size      — persistent connections kept alive
      max_overflow   — extra connections allowed above pool_size under load
      pool_recycle   — recycle connections older than N seconds (avoids stale)
      pool_pre_ping  — validate connection before handing it out from pool
      connect_args   — asyncpg-specific: per-query and connect timeouts
    """
    settings = get_settings()

    engine = create_async_engine(
        settings.DATABASE_URL,
        # ── Pool ──────────────────────────────────────────────────────────────
        pool_size=settings.DB_POOL_MIN_SIZE,
        max_overflow=settings.DB_POOL_MAX_SIZE - settings.DB_POOL_MIN_SIZE,
        pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
        pool_pre_ping=True,  # evict dead connections silently
        pool_timeout=settings.DB_CONNECT_TIMEOUT_SECONDS,
        # ── asyncpg-specific ──────────────────────────────────────────────────
        connect_args={
            "command_timeout": settings.DB_QUERY_TIMEOUT_SECONDS,
            "server_settings": {
                # Surface query-level notices in Python's logging
                "application_name": settings.APP_NAME,
            },
        },
        # ── Logging ───────────────────────────────────────────────────────────
        echo=settings.DEBUG,  # log all SQL in debug mode only
        echo_pool=settings.DEBUG,
    )

    logger.info(
        "Database engine created — host=%s db=%s pool_size=%d max_overflow=%d",
        settings.POSTGRES_HOST,
        settings.POSTGRES_DB,
        settings.DB_POOL_MIN_SIZE,
        settings.DB_POOL_MAX_SIZE - settings.DB_POOL_MIN_SIZE,
    )
    return engine


# =============================================================================
# DatabaseClient — lifecycle owner
# =============================================================================


class DatabaseClient:
    """
    Owns the engine and session factory for the application lifetime.

    Instantiate once at startup via get_db_client().
    Never instantiate directly in request handlers.

    Usage:
        client = await get_db_client()

        async with client.session() as session:
            result = await session.execute(text("SELECT 1"))

        async with client.connection() as conn:
            await conn.run_sync(Base.metadata.create_all)
    """

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Initialise the engine and session factory.
        Call once at application startup.
        """
        if self._engine is not None:
            logger.warning(
                "DatabaseClient.connect() called on already-connected client"
            )
            return

        try:
            self._engine = _build_engine()
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,  # avoid lazy-load after commit
                autobegin=True,
                autoflush=False,
            )
            # Verify the connection is actually reachable
            await self._verify_connection()
            logger.info("Database connection pool established")

        except Exception as exc:
            logger.error("Failed to establish database connection: %s", exc)
            raise DatabaseConnectionError(
                f"Cannot connect to PostgreSQL at "
                f"{get_settings().POSTGRES_HOST}:{get_settings().POSTGRES_PORT} — {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """
        Dispose the engine and drain the connection pool.
        Call once at application shutdown.
        """
        if self._engine is None:
            return
        await self._engine.dispose()
        self._engine = None
        self._session_factory = None
        logger.info("Database connection pool closed")

    # ── Session context manager ───────────────────────────────────────────────

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Yield a transactional AsyncSession.

        Commits on clean exit, rolls back on any exception, always closes.

        Example:
            async with db.session() as session:
                session.add(obj)
                # commit happens automatically on exit
        """
        if self._session_factory is None:
            raise DatabaseConnectionError(
                "DatabaseClient is not connected. Call connect() first."
            )

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as exc:
                await session.rollback()
                logger.error("Session rolled back due to: %s", exc)
                raise DatabaseQueryError(str(exc)) from exc
            finally:
                await session.close()

    # ── Raw connection context manager ────────────────────────────────────────

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[AsyncConnection, None]:
        """
        Yield a raw AsyncConnection for DDL operations (schema creation, migrations).
        Not for use in request handlers — use session() instead.
        """
        if self._engine is None:
            raise DatabaseConnectionError(
                "DatabaseClient is not connected. Call connect() first."
            )
        async with self._engine.begin() as conn:
            yield conn

    # ── Health & diagnostics ──────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """
        Run a lightweight liveness probe against the database.

        Returns a dict suitable for inclusion in /health endpoint responses.
        Raises DatabaseConnectionError if the database is unreachable.
        """
        if self._engine is None:
            return {"status": "disconnected", "error": "Client not initialised"}

        try:
            async with self._engine.connect() as conn:
                row = await conn.execute(
                    text("SELECT version(), pg_postmaster_start_time()")
                )
                version, start_time = row.one()

                pool = self._engine.pool
                return {
                    "status": "healthy",
                    "postgres_version": version.split(" ")[1],
                    "server_start_time": str(start_time),
                    "pool_size": pool.size(),
                    "pool_checked_out": pool.checkedout(),
                    "pool_overflow": pool.overflow(),
                }
        except Exception as exc:
            logger.error("Database health check failed: %s", exc)
            raise DatabaseConnectionError(f"Health check failed: {exc}") from exc

    async def verify_pgvector(self) -> bool:
        """
        Verify that the pgvector extension is installed and accessible.
        Called at startup before any embedding operations.
        """
        try:
            async with self._engine.connect() as conn:  # type: ignore[union-attr]
                result = await conn.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                )
                installed = result.scalar() is not None
                if installed:
                    logger.info("pgvector extension verified")
                else:
                    logger.error(
                        "pgvector extension NOT found. "
                        "Run: CREATE EXTENSION IF NOT EXISTS vector;"
                    )
                return installed
        except Exception as exc:
            logger.error("Failed to verify pgvector: %s", exc)
            return False

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _verify_connection(self) -> None:
        """Ping the database once to confirm connectivity at startup."""
        assert self._engine is not None
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            raise DatabaseConnectionError(
                f"Database connectivity check failed: {exc}"
            ) from exc

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise DatabaseConnectionError("DatabaseClient is not connected.")
        return self._engine


# =============================================================================
# Singleton accessor
# =============================================================================

# Module-level singleton — initialised once at startup via connect()
_db_client: DatabaseClient | None = None


async def get_db_client() -> DatabaseClient:
    """
    Return the application-level DatabaseClient singleton.

    Called by FastAPI dependencies and application startup.
    The client must have connect() called before first use.
    """
    global _db_client
    if _db_client is None:
        _db_client = DatabaseClient()
    return _db_client


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: yield a database session per request.

    Usage in a route:
        @router.get("/items")
        async def list_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    client = await get_db_client()
    async with client.session() as session:
        yield session
