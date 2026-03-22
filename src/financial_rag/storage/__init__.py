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

# AFTER — isort sorts alphabetically, ignoring comments
__all__ = [
    "NS_ANALYSIS",
    "NS_ANALYSIS",
    "NS_CHUNKS",
    "NS_EMBEDDINGS",
    "NS_MARKET",
    "NS_QUERY",
    "Base",
    "BaseRepository",  # if present, else skip
    "CacheClient",
    "DatabaseClient",
    "build_key",
    "get_cache_client",
    "get_db_client",
    "get_session",
]
