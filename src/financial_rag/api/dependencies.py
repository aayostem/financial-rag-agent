# =============================================================================
# Financial RAG Agent — FastAPI Dependencies
# src/financial_rag/api/dependencies.py
#
# Dependency injection functions used across route handlers.
# All expensive objects (QueryEngine, VectorStore) are instantiated once
# at startup and reused via module-level singletons.
# =============================================================================

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends

from financial_rag.retrieval.query_engine import QueryEngine
from financial_rag.storage.cache import CacheClient, get_cache_client
from financial_rag.storage.database import DatabaseClient, get_db_client
from financial_rag.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

# =============================================================================
# Module-level singletons — initialised at startup, reused per request
# =============================================================================

_query_engine: QueryEngine | None = None
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Return the application-level VectorStore singleton.
    Instantiated once at startup via initialise_dependencies().
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_query_engine() -> QueryEngine:
    """
    Return the application-level QueryEngine singleton.
    Instantiated once at startup via initialise_dependencies().
    """
    global _query_engine
    if _query_engine is None:
        vs = get_vector_store()
        _query_engine = QueryEngine(vector_store=vs)
    return _query_engine


# =============================================================================
# FastAPI dependency functions
# Used with Depends() in route handlers.
# =============================================================================


async def db_client() -> DatabaseClient:
    """Yield the database client. Raises if not connected."""
    return await get_db_client()


async def cache_client() -> CacheClient:
    """Yield the cache client. Raises if not connected."""
    return await get_cache_client()


async def query_engine() -> QueryEngine:
    """Yield the query engine singleton."""
    return get_query_engine()


async def vector_store() -> VectorStore:
    """Yield the vector store singleton."""
    return get_vector_store()


# =============================================================================
# Annotated type aliases for cleaner route signatures
# =============================================================================

DBClient = Annotated[DatabaseClient, Depends(db_client)]
CacheClient_ = Annotated[CacheClient, Depends(cache_client)]
Engine = Annotated[QueryEngine, Depends(query_engine)]
Store = Annotated[VectorStore, Depends(vector_store)]


# =============================================================================
# Startup / shutdown helpers — called from server lifespan
# =============================================================================


async def initialise_dependencies() -> None:
    """
    Initialise all application dependencies at startup.
    Called once from the FastAPI lifespan context manager.
    """
    # Database
    db = await get_db_client()
    await db.connect()
    logger.info("Database client connected")

    # Verify pgvector extension
    await db.verify_pgvector()

    # Cache
    cache = await get_cache_client()
    await cache.connect()
    logger.info("Cache client connected")

    # Pre-warm singletons so first request isn't slow
    get_vector_store()
    get_query_engine()
    logger.info("QueryEngine and VectorStore initialised")


async def shutdown_dependencies() -> None:
    """
    Gracefully close all connections at shutdown.
    Called once from the FastAPI lifespan context manager.
    """
    db = await get_db_client()
    await db.disconnect()
    logger.info("Database client disconnected")

    cache = await get_cache_client()
    await cache.disconnect()
    logger.info("Cache client disconnected")
