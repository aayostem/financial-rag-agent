# =============================================================================
# Financial RAG Agent — Cache Client
# src/financial_rag/storage/cache.py
#
# Async Redis client providing:
#   - Typed get/set/delete with automatic JSON serialisation
#   - Namespaced key builder (prevents key collisions across features)
#   - TTL management with per-call overrides
#   - Health check
#   - Connection pool lifecycle management
# =============================================================================

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from financial_rag.config import get_settings
from financial_rag.utils.exceptions import CacheConnectionError, CacheOperationError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# =============================================================================
# Key namespace constants
# All cache keys are namespaced to prevent collisions.
# Format: finrag:{namespace}:{identifier}
# =============================================================================

NS_CHUNKS = "chunks"  # finrag:chunks:{ticker}:{hash}
NS_EMBEDDINGS = "embeddings"  # finrag:embeddings:{hash}
NS_QUERY = "query"  # finrag:query:{hash}
NS_ANALYSIS = "analysis"  # finrag:analysis:{session_id}
NS_MARKET = "market"  # finrag:market:{ticker}
NS_HEALTH = "health"  # finrag:health:last_check


def build_key(*parts: str) -> str:
    """
    Build a namespaced cache key.

    Example:
        build_key(NS_CHUNKS, "AAPL", "abc123")
        → "finrag:chunks:AAPL:abc123"
    """
    return "finrag:" + ":".join(parts)


# =============================================================================
# CacheClient — lifecycle owner
# =============================================================================


class CacheClient:
    """
    Async Redis client with typed operations and structured error handling.

    Instantiate once at startup via get_cache_client().
    Never instantiate directly in request handlers.

    Usage:
        client = await get_cache_client()

        await client.set(build_key(NS_QUERY, query_hash), result, ttl=3600)
        result = await client.get(build_key(NS_QUERY, query_hash))
    """

    def __init__(self) -> None:
        self._pool: ConnectionPool | None = None
        self._redis: Redis | None = None  # type: ignore[type-arg]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Initialise the Redis connection pool.
        Call once at application startup.
        """
        if self._redis is not None:
            logger.warning("CacheClient.connect() called on already-connected client")
            return

        settings = get_settings()

        try:
            self._pool = ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
                socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT_SECONDS,
                decode_responses=True,  # always return str, never bytes
                health_check_interval=30,  # background ping every 30s
            )
            self._redis = Redis(connection_pool=self._pool)

            # Verify connectivity immediately
            await self._redis.ping()
            logger.info(
                "Redis connection pool established — host=%s port=%d db=%d",
                settings.REDIS_HOST,
                settings.REDIS_PORT,
                settings.REDIS_DB,
            )
        except RedisError as exc:
            logger.error("Failed to connect to Redis: %s", exc)
            raise CacheConnectionError(
                f"Cannot connect to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT} — {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """
        Close all connections in the pool.
        Call once at application shutdown.
        """
        if self._redis is None:
            return
        await self._redis.aclose()
        if self._pool is not None:
            await self._pool.aclose()
        self._redis = None
        self._pool = None
        logger.info("Redis connection pool closed")

    # ── Core typed operations ─────────────────────────────────────────────────

    async def get(self, key: str) -> Any | None:
        """
        Retrieve and deserialise a value from cache.

        Returns None if the key does not exist or has expired.
        Raises CacheOperationError on Redis errors.
        """
        self._assert_connected()
        try:
            raw = await self._redis.get(key)  # type: ignore[union-attr]
            if raw is None:
                return None
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Cache value for key '%s' is not valid JSON: %s", key, exc)
            return None
        except RedisError as exc:
            logger.error("Cache GET failed for key '%s': %s", key, exc)
            raise CacheOperationError(f"GET {key} failed: {exc}") from exc

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """
        Serialise and store a value in cache.

        Args:
            key:   Cache key (use build_key() to namespace)
            value: Any JSON-serialisable value
            ttl:   Time-to-live in seconds. Falls back to settings default.

        Returns True on success, raises CacheOperationError on failure.
        """
        self._assert_connected()
        settings = get_settings()
        effective_ttl = ttl if ttl is not None else settings.REDIS_DEFAULT_TTL_SECONDS

        try:
            serialised = json.dumps(value, default=str)
            await self._redis.setex(  # type: ignore[union-attr]
                name=key,
                time=effective_ttl,
                value=serialised,
            )
            logger.debug("Cache SET key='%s' ttl=%ds", key, effective_ttl)
            return True
        except (TypeError, ValueError) as exc:
            logger.error("Cannot serialise value for key '%s': %s", key, exc)
            raise CacheOperationError(
                f"Value for key '{key}' is not JSON-serialisable: {exc}"
            ) from exc
        except RedisError as exc:
            logger.error("Cache SET failed for key '%s': %s", key, exc)
            raise CacheOperationError(f"SET {key} failed: {exc}") from exc

    async def delete(self, key: str) -> bool:
        """
        Delete a key from cache.
        Returns True if the key existed, False if it did not.
        """
        self._assert_connected()
        try:
            deleted = await self._redis.delete(key)  # type: ignore[union-attr]
            return bool(deleted)
        except RedisError as exc:
            logger.error("Cache DELETE failed for key '%s': %s", key, exc)
            raise CacheOperationError(f"DELETE {key} failed: {exc}") from exc

    async def exists(self, key: str) -> bool:
        """Return True if the key exists in cache."""
        self._assert_connected()
        try:
            return bool(await self._redis.exists(key))  # type: ignore[union-attr]
        except RedisError as exc:
            raise CacheOperationError(f"EXISTS {key} failed: {exc}") from exc

    async def expire(self, key: str, ttl: int) -> bool:
        """Reset the TTL on an existing key. Returns False if key not found."""
        self._assert_connected()
        try:
            return bool(
                await self._redis.expire(key, ttl)  # type: ignore[union-attr]
            )
        except RedisError as exc:
            raise CacheOperationError(f"EXPIRE {key} failed: {exc}") from exc

    async def clear_namespace(self, namespace: str) -> int:
        """
        Delete all keys under a namespace prefix.
        Uses SCAN to avoid blocking the Redis server.

        Returns the number of keys deleted.

        Example:
            await cache.clear_namespace("finrag:query")
        """
        self._assert_connected()
        pattern = f"finrag:{namespace}:*"
        deleted = 0
        try:
            async for key in self._redis.scan_iter(pattern):  # type: ignore[union-attr]
                await self._redis.delete(key)  # type: ignore[union-attr]
                deleted += 1
            logger.info("Cleared %d keys matching pattern '%s'", deleted, pattern)
            return deleted
        except RedisError as exc:
            raise CacheOperationError(f"clear_namespace '{namespace}' failed: {exc}") from exc

    # ── Health ────────────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """
        Run a lightweight liveness probe against Redis.
        Returns a dict suitable for /health endpoint responses.
        """
        if self._redis is None:
            return {"status": "disconnected", "error": "Client not initialised"}

        try:
            await self._redis.ping()
            info = await self._redis.info("server")
            pool_stats = self._pool_stats()

            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                **pool_stats,
            }
        except RedisError as exc:
            logger.error("Redis health check failed: %s", exc)
            raise CacheConnectionError(f"Health check failed: {exc}") from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assert_connected(self) -> None:
        if self._redis is None:
            raise CacheConnectionError("CacheClient is not connected. Call connect() first.")

    def _pool_stats(self) -> dict[str, Any]:
        """Extract connection pool statistics."""
        if self._pool is None:
            return {}
        stats: dict[str, Any] = {
            "pool_max_connections": self._pool.max_connections,
        }
        # _created_connections is internal — not guaranteed across redis-py versions
        created = getattr(self._pool, "_created_connections", None)
        if created is not None:
            stats["pool_created_connections"] = created
        return stats


# =============================================================================
# Singleton accessor
# =============================================================================

_cache_client: CacheClient | None = None


async def get_cache_client() -> CacheClient:
    """
    Return the application-level CacheClient singleton.

    The client must have connect() called before first use.
    Called by FastAPI dependencies and application startup.
    """
    global _cache_client
    if _cache_client is None:
        _cache_client = CacheClient()
    return _cache_client
