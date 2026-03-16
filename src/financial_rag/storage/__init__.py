# =============================================================================
# Financial RAG Agent — Storage Package
# src/financial_rag/storage/__init__.py
# =============================================================================

from .cache import (
    NS_ANALYSIS,
    NS_CHUNKS,
    NS_EMBEDDINGS,
    NS_MARKET,
    NS_QUERY,
    CacheClient,
    build_key,
    get_cache_client,
)
from .database import Base, DatabaseClient, get_db_client, get_session

__all__ = [
    # Database
    "Base",
    "DatabaseClient",
    "get_db_client",
    "get_session",
    # Cache
    "CacheClient",
    "get_cache_client",
    "build_key",
    # Cache namespaces
    "NS_CHUNKS",
    "NS_EMBEDDINGS",
    "NS_QUERY",
    "NS_ANALYSIS",
    "NS_MARKET",
]
