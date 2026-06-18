

## Step 6 — Write `settings.py`

Create `src/financial_rag/config/settings.py`:

```python
# src/financial_rag/config/settings.py
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Single source of truth for all configuration.
    Reads from environment variables and .env file.
    Secrets use SecretStr — never exposed in logs or repr().
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # ── Environment ───────────────────────────────────────────────────────────
    APP_ENV: Literal["development", "staging", "production", "testing"] = Field(
        default="development"
    )
    APP_NAME: str = Field(default="financial-rag-agent")
    APP_VERSION: str = Field(default="0.1.0")

    # ── Derived flags — auto-set from APP_ENV ─────────────────────────────────
    DEBUG: bool = Field(default=False)
    TESTING: bool = Field(default=False)
    MOCK_EXTERNAL_APIS: bool = Field(default=False)

    # ── Paths ─────────────────────────────────────────────────────────────────
    @computed_field
    @property
    def PROJECT_ROOT(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @computed_field
    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    # ── API Server ────────────────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1, le=65535)

    # ── LLM / Embedding ───────────────────────────────────────────────────────
    OPENAI_API_KEY: SecretStr | None = Field(default=None)
    GROQ_API_KEY: SecretStr | None = Field(default=None)
    EMBEDDING_PROVIDER: Literal["openai", "local"] = Field(default="local")
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    EMBEDDING_DIMENSIONS: int = Field(default=384)
    LLM_PROVIDER: Literal["openai", "anthropic", "local"] = Field(default="openai")
    LLM_MODEL: str = Field(default="gpt-3.5-turbo")
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=1)

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432, ge=1, le=65535)
    POSTGRES_USER: str = Field(default="finrag")
    POSTGRES_PASSWORD: SecretStr = Field(...)   # REQUIRED — no default
    POSTGRES_DB: str = Field(default="financial_rag")
    DB_POOL_MIN_SIZE: int = Field(default=2, ge=1)
    DB_POOL_MAX_SIZE: int = Field(default=10, ge=2)

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """Async DSN for SQLAlchemy + asyncpg."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Sync DSN for Alembic migrations (psycopg2)."""
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_PASSWORD: SecretStr = Field(...)      # REQUIRED — no default
    REDIS_MAX_CONNECTIONS: int = Field(default=20, ge=1)
    REDIS_DEFAULT_TTL_SECONDS: int = Field(default=3600, ge=60)

    @computed_field
    @property
    def REDIS_URL(self) -> str:
        return (
            f"redis://:{self.REDIS_PASSWORD.get_secret_value()}"
            f"@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    CHUNK_SIZE_TOKENS: int = Field(default=512, ge=64, le=2048)
    CHUNK_OVERLAP_TOKENS: int = Field(default=50, ge=0, le=200)
    TOP_K_RESULTS: int = Field(default=5, ge=1, le=50)
    HYBRID_SEARCH_ALPHA: float = Field(default=0.7, ge=0.0, le=1.0)
    MAX_CONTEXT_TOKENS: int = Field(default=6000, ge=1000)

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: Literal["json", "console"] = Field(default="console")

    # ── SEC EDGAR ─────────────────────────────────────────────────────────────
    EDGAR_USER_AGENT: str = Field(default="financial-rag-agent contact@example.com")
    EDGAR_RATE_LIMIT_RPS: int = Field(default=8, ge=1, le=10)

    # =========================================================================
    # Validators — consolidated into ONE method to avoid Pydantic v2 shadowing
    # =========================================================================
    @model_validator(mode="after")
    def _apply_all_defaults_and_validate(self) -> Settings:
        """
        Single validator that handles:
          1. ENV-driven flag defaults (DEBUG, TESTING, MOCK_EXTERNAL_APIS)
          2. LOG_LEVEL / LOG_FORMAT defaults
          3. Chunk size sanity check
          4. Provider API key presence check (skipped in testing)
          5. Production hardening guards

        WHY ONE METHOD?
        In Pydantic v2, defining multiple @model_validator(mode="after") methods
        with the same name on the same class means only the last definition runs
        — the others are silently overwritten. Consolidating into one method
        guarantees all logic executes on every instantiation.
        """
        is_dev  = self.APP_ENV == "development"
        is_test = self.APP_ENV == "testing"
        is_prod = self.APP_ENV == "production"

        # 1. ENV-driven flags
        # object.__setattr__ is required because Pydantic v2 models are
        # immutable (frozen) during validation — direct assignment raises an error.
        if is_dev or is_test:
            object.__setattr__(self, "DEBUG", True)
        if is_test:
            object.__setattr__(self, "TESTING", True)
            object.__setattr__(self, "MOCK_EXTERNAL_APIS", True)

        # 2. Log defaults
        if is_prod and self.LOG_LEVEL == "INFO":
            object.__setattr__(self, "LOG_LEVEL", "WARNING")
            object.__setattr__(self, "LOG_FORMAT", "json")
        elif (is_dev or is_test) and self.LOG_LEVEL == "INFO":
            object.__setattr__(self, "LOG_LEVEL", "DEBUG")

        # 3. Chunk sanity
        issues: list[str] = []
        if self.CHUNK_OVERLAP_TOKENS >= self.CHUNK_SIZE_TOKENS:
            issues.append(
                f"CHUNK_OVERLAP_TOKENS ({self.CHUNK_OVERLAP_TOKENS}) must be "
                f"< CHUNK_SIZE_TOKENS ({self.CHUNK_SIZE_TOKENS})"
            )

        # 4. Provider key checks (skip in testing / mock mode)
        if not (is_test or self.MOCK_EXTERNAL_APIS):
            if self.EMBEDDING_PROVIDER == "openai" and not self.OPENAI_API_KEY:
                issues.append("OPENAI_API_KEY required when EMBEDDING_PROVIDER='openai'")
            if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
                issues.append("OPENAI_API_KEY required when LLM_PROVIDER='openai'")
            if self.LLM_PROVIDER == "anthropic" and not self.GROQ_API_KEY:
                issues.append("GROQ_API_KEY required when LLM_PROVIDER='anthropic'")

        # 5. Production hardening
        if is_prod:
            if self.DEBUG:
                issues.append("DEBUG=True in production")
            if self.POSTGRES_HOST == "localhost":
                issues.append("POSTGRES_HOST is 'localhost' — use a real host in production")

        if issues:
            raise ValueError(
                "Configuration errors:\n" + "\n".join(f"  • {i}" for i in issues)
            )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton — instantiated exactly once per process.

    In tests, call get_settings.cache_clear() before injecting env overrides:
        monkeypatch.setenv("APP_ENV", "testing")
        get_settings.cache_clear()
        settings = get_settings()
    """
    instance = Settings()
    logger.info(
        "Settings loaded — env=%s debug=%s embedding=%s llm=%s",
        instance.APP_ENV,
        instance.DEBUG,
        instance.EMBEDDING_MODEL,
        instance.LLM_MODEL,
    )
    return instance
```

Create `src/financial_rag/config/__init__.py`:

```python
from financial_rag.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
```

**Quick smoke test:**

```bash
python -c "
from financial_rag.config import get_settings
s = get_settings()
print(f'env={s.APP_ENV} debug={s.DEBUG}')
print(f'db_url starts with postgresql+asyncpg: {s.DATABASE_URL.startswith(\"postgresql+asyncpg\")}')
"
```

Expected output:
```
env=development debug=True
db_url starts with postgresql+asyncpg: True
```

---

## Step 7 — Key Design Patterns in `settings.py` (Tutorial Callouts)

Before writing any more code, understand three patterns you will see throughout this entire project:

### Pattern 1: `SecretStr` — secrets that never leak

```python
POSTGRES_PASSWORD: SecretStr = Field(...)
```

```python
# This is safe — SecretStr masks the value
print(settings.POSTGRES_PASSWORD)         # **********
print(repr(settings.POSTGRES_PASSWORD))   # SecretStr('**********')

# This is how you actually use the value in code
url = settings.POSTGRES_PASSWORD.get_secret_value()
```

### Pattern 2: `computed_field` — derived values, never stored

```python
@computed_field
@property
def DATABASE_URL(self) -> str:
    return f"postgresql+asyncpg://{self.POSTGRES_USER}:..."
```

`DATABASE_URL` is built on-demand from individual fields. It is never set directly in `.env`. This means the password is assembled at runtime and never cached in a plain string that might appear in a stack trace.

### Pattern 3: `lru_cache(maxsize=1)` — one instance per process

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    ...
```

`Settings()` reads and validates every field from the environment. `lru_cache` ensures this expensive work happens exactly once. In tests, you clear the cache before changing environment variables:

```python
get_settings.cache_clear()
```

---

## Step 8 — Write `storage/database.py`

Create `src/financial_rag/storage/database.py`:

```python
# src/financial_rag/storage/database.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from financial_rag.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class DatabaseClient:
    """
    Async PostgreSQL client wrapping SQLAlchemy 2.0.

    Lifecycle:
        await client.connect()   # call once at app startup
        await client.disconnect() # call once at app shutdown

    Usage:
        async with client.session() as session:   # ORM operations
            session.add(model)

        async with client.connection() as conn:   # raw SQL / DDL
            await conn.execute(text("SELECT 1"))
    """

    def __init__(self):
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        settings = get_settings()
        self._engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.DB_POOL_MIN_SIZE,
            max_overflow=settings.DB_POOL_MAX_SIZE - settings.DB_POOL_MIN_SIZE,
            pool_pre_ping=True,   # silently recycles stale connections
            echo=settings.DEBUG,  # log SQL in development, silent in production
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,  # prevents lazy-load errors after commit
        )
        await self._verify_connection()
        logger.info("Database connected — pool_size=%d", settings.DB_POOL_MIN_SIZE)

    async def disconnect(self) -> None:
        if self._engine is None:
            return
        await self._engine.dispose()
        self._engine = None
        self._session_factory = None
        logger.info("Database disconnected")

    def _assert_connected(self) -> None:
        if self._session_factory is None:
            raise RuntimeError("DatabaseClient.connect() was not called at startup.")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Yield a transactional AsyncSession.
        Commits on clean exit, rolls back on any exception.
        """
        self._assert_connected()
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[AsyncConnection, None]:
        """
        Yield a raw AsyncConnection for DDL, migrations, and bulk operations.
        """
        self._assert_connected()
        async with self._engine.begin() as conn:
            yield conn

    async def _verify_connection(self) -> None:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            logger.error("Database connectivity check failed: %s", exc)
            raise


# =============================================================================
# Singleton
# =============================================================================

_db_client: DatabaseClient | None = None


async def get_db_client() -> DatabaseClient:
    """Return the application-level DatabaseClient singleton."""
    global _db_client
    if _db_client is None:
        _db_client = DatabaseClient()
        await _db_client.connect()
    return _db_client


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields one session per HTTP request.

    Usage:
        @router.get("/")
        async def handler(session: AsyncSession = Depends(get_session)):
            ...
    """
    client = await get_db_client()
    async with client.session() as session:
        yield session
```

---

## Step 9 — Write `storage/cache.py`

Create `src/financial_rag/storage/cache.py`:

```python
# src/financial_rag/storage/cache.py
from __future__ import annotations

import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from financial_rag.config import get_settings

logger = logging.getLogger(__name__)


def build_key(*parts: str) -> str:
    """Build a namespaced cache key: finrag:{part1}:{part2}..."""
    return "finrag:" + ":".join(parts)


class CacheClient:
    """
    Async Redis client with JSON serialisation and graceful degradation.

    All read/write failures log an error and return a safe default
    (None for reads, False for writes) — cache failures never crash the app.
    The database is always the source of truth.

    Lifecycle:
        await client.connect()    # call once at app startup
        await client.disconnect() # call once at app shutdown
    """

    def __init__(self):
        self._redis: Redis | None = None

    async def connect(self) -> None:
        settings = get_settings()
        self._redis = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
        )
        await self._redis.ping()
        logger.info("Redis connected")

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis disconnected")

    def _assert_connected(self) -> None:
        if self._redis is None:
            raise RuntimeError("CacheClient.connect() was not called at startup.")

    async def get(self, key: str) -> Any | None:
        self._assert_connected()
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Evicting malformed cache key '%s'", key)
            await self._redis.delete(key)
            return None
        except RedisError as exc:
            logger.error("Cache GET failed for '%s': %s", key, exc)
            return None  # degrade gracefully — caller will re-compute

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        self._assert_connected()
        settings = get_settings()
        effective_ttl = ttl if ttl is not None else settings.REDIS_DEFAULT_TTL_SECONDS
        try:
            serialized = json.dumps(value, default=str)
            await self._redis.setex(key, effective_ttl, serialized)
            return True
        except (TypeError, ValueError) as exc:
            logger.error("Cannot serialise value for key '%s': %s", key, exc)
            return False
        except RedisError as exc:
            logger.error("Cache SET failed for '%s': %s", key, exc)
            return False  # write failure is non-fatal

    async def delete(self, key: str) -> bool:
        self._assert_connected()
        try:
            return bool(await self._redis.delete(key))
        except RedisError as exc:
            logger.error("Cache DELETE failed for '%s': %s", key, exc)
            return False

    async def exists(self, key: str) -> bool:
        self._assert_connected()
        try:
            return bool(await self._redis.exists(key))
        except RedisError as exc:
            logger.error("Cache EXISTS failed for '%s': %s", key, exc)
            return False

    async def clear_namespace(self, namespace: str) -> int:
        """
        Delete all keys under a namespace prefix.
        Uses SCAN (non-blocking) instead of KEYS (blocks the entire Redis server).
        """
        self._assert_connected()
        pattern = f"finrag:{namespace}:*"
        deleted = 0
        try:
            async for key in self._redis.scan_iter(pattern):
                await self._redis.delete(key)
                deleted += 1
            logger.info("Cleared %d keys matching '%s'", deleted, pattern)
            return deleted
        except RedisError as exc:
            logger.error("Cache clear_namespace failed for '%s': %s", namespace, exc)
            return deleted


# =============================================================================
# Singleton
# =============================================================================

_cache_client: CacheClient | None = None


async def get_cache_client() -> CacheClient:
    """Return the application-level CacheClient singleton."""
    global _cache_client
    if _cache_client is None:
        _cache_client = CacheClient()
        await _cache_client.connect()
    return _cache_client
```

Update `src/financial_rag/storage/__init__.py`:

```python
from financial_rag.storage.cache import CacheClient, get_cache_client
from financial_rag.storage.database import DatabaseClient, get_db_client, get_session

__all__ = [
    "CacheClient", "get_cache_client",
    "DatabaseClient", "get_db_client", "get_session",
]
```

---

## Step 10 — Create the Docker Infrastructure

### 10a — `docker-compose.yml`

Create `docker-compose.yml` in the project root:

```yaml
# docker-compose.yml
# Phase 1 — Core services only: PostgreSQL + Redis
services:

  postgres:
    image: pgvector/pgvector:pg16
    container_name: finrag-postgres
    environment:
      POSTGRES_DB:       ${POSTGRES_DB:-financial_rag}
      POSTGRES_USER:     ${POSTGRES_USER:-finrag}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set in .env}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/docker/init:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U finrag -d financial_rag"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: finrag-redis
    command: >
      redis-server
        --appendonly yes
        --requirepass ${REDIS_PASSWORD:?REDIS_PASSWORD must be set in .env}
        --maxmemory 512mb
        --maxmemory-policy allkeys-lru
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  postgres_data:
    name: finrag_postgres_data
  redis_data:
    name: finrag_redis_data
```

> **Key things to understand:**
> - `${POSTGRES_PASSWORD:?Required}` — Docker Compose will refuse to start if this variable is not set. Fail-fast at startup, not at connection time.
> - `./infrastructure/docker/init:/docker-entrypoint-initdb.d:ro` — every `.sql` file in this directory runs automatically on first container start. `:ro` means read-only — the container cannot modify your init scripts.
> - `allkeys-lru` — when Redis hits its memory limit, it evicts the least recently used key. Correct policy for a cache (vs a primary store).

### 10b — The Database Init Script

Create `infrastructure/docker/init/01_create_schema.sql`:

```sql
-- infrastructure/docker/init/01_create_schema.sql
-- Runs automatically on first postgres container start.
-- Prefix with 01_ — Docker runs init scripts in alphabetical order.

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector: stores and searches embeddings
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- gen_random_uuid() for primary keys
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- trigram index for BM25 text search
CREATE EXTENSION IF NOT EXISTS btree_gin;    -- GIN index support for JSONB

-- ── filings ───────────────────────────────────────────────────────────────────
-- One row per SEC filing document (10-K, 10-Q, etc.)
CREATE TABLE IF NOT EXISTS filings (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker           VARCHAR(10) NOT NULL,
    filing_type      VARCHAR(20) NOT NULL,
    fiscal_year      SMALLINT,
    fiscal_quarter   SMALLINT    CHECK (fiscal_quarter BETWEEN 1 AND 4),
    filed_at         DATE,
    source_url       TEXT,
    file_hash        VARCHAR(64) UNIQUE,   -- SHA-256 for deduplication
    pages            INTEGER,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ingested_by      VARCHAR(100),
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE,
    CONSTRAINT filings_filing_type_check
        CHECK (filing_type IN ('10-K', '10-Q', '8-K', '20-F', 'DEF 14A', 'S-1'))
);

CREATE INDEX IF NOT EXISTS idx_filings_ticker    ON filings (ticker, fiscal_year DESC);
CREATE INDEX IF NOT EXISTS idx_filings_type_year ON filings (filing_type, fiscal_year DESC);
CREATE INDEX IF NOT EXISTS idx_filings_active    ON filings (ticker) WHERE is_active = TRUE;

-- ── financial_chunks ──────────────────────────────────────────────────────────
-- One row per text chunk extracted from a filing.
-- The embedding column stores the vector representation of chunk_text.
-- vector(3072) matches text-embedding-3-large.
-- In development with all-MiniLM-L6-v2 you get 384 dims —
-- this schema uses 3072 for production readiness.
CREATE TABLE IF NOT EXISTS financial_chunks (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id       UUID         NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    ticker          VARCHAR(10)  NOT NULL,
    filing_type     VARCHAR(20)  NOT NULL,
    fiscal_year     SMALLINT,
    section         VARCHAR(100),
    chunk_index     INTEGER      NOT NULL CHECK (chunk_index >= 0),
    chunk_text      TEXT         NOT NULL,
    token_count     INTEGER      CHECK (token_count IS NULL OR token_count > 0),
    embedding       vector(3072) NOT NULL,
    metrics         JSONB        NOT NULL DEFAULT '{}',
    entities        JSONB        NOT NULL DEFAULT '{}',
    sentiment_score REAL         CHECK (sentiment_score BETWEEN -1.0 AND 1.0),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(50)  NOT NULL DEFAULT 'text-embedding-3-large'
);

CREATE INDEX IF NOT EXISTS idx_chunks_ticker_year    ON financial_chunks (ticker, fiscal_year DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_filing_section ON financial_chunks (filing_id, section);
CREATE INDEX IF NOT EXISTS idx_chunks_metrics        ON financial_chunks USING GIN (metrics);
CREATE INDEX IF NOT EXISTS idx_chunks_entities       ON financial_chunks USING GIN (entities);
-- GIN trigram index — enables LIKE '%revenue%' to use an index instead of seq scan
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm      ON financial_chunks USING GIN (chunk_text gin_trgm_ops);

-- ── analysis_history ──────────────────────────────────────────────────────────
-- Append-only audit trail of every query answered by the agent.
CREATE TABLE IF NOT EXISTS analysis_history (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker           VARCHAR(10),
    question         TEXT        NOT NULL,
    answer           TEXT        NOT NULL,
    analysis_style   VARCHAR(20) NOT NULL DEFAULT 'analyst',
    agent_type       VARCHAR(50) NOT NULL,
    search_type      VARCHAR(20) NOT NULL DEFAULT 'similarity',
    latency_ms       INTEGER     NOT NULL,
    source_chunk_ids UUID[]      DEFAULT '{}',
    real_time_used   BOOLEAN     NOT NULL DEFAULT FALSE,
    error            TEXT,
    session_id       UUID,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT analysis_style_check
        CHECK (analysis_style IN ('analyst', 'executive', 'risk')),
    CONSTRAINT search_type_check
        CHECK (search_type IN ('similarity', 'mmr'))
);

CREATE INDEX IF NOT EXISTS idx_analysis_ticker_time ON analysis_history (ticker, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_session     ON analysis_history (session_id)
    WHERE session_id IS NOT NULL;

-- ── schema_migrations ─────────────────────────────────────────────────────────
-- Manual migration tracking (Alembic added in Phase 3).
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(20) PRIMARY KEY,
    description TEXT,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO schema_migrations (version, description)
VALUES ('001', 'Initial schema: filings, financial_chunks, analysis_history')
ON CONFLICT (version) DO NOTHING;
```

---

## Step 11 — Start Docker Services and Verify

```bash
# Start services (reads .env automatically)
docker compose up -d

# Watch until both are healthy (takes ~20 seconds)
docker compose ps
```

Expected output:
```
NAME               STATUS
finrag-postgres    Up (healthy)
finrag-redis       Up (healthy)
```

Verify the schema was created:

```bash
docker exec finrag-postgres psql -U finrag -d financial_rag -c "\dt"
```

Expected:
```
              List of relations
 Schema |       Name         | Type  | Owner
--------+--------------------+-------+-------
 public | analysis_history   | table | finrag
 public | financial_chunks   | table | finrag
 public | filings            | table | finrag
 public | schema_migrations  | table | finrag
```

Verify the pgvector extension:

```bash
docker exec finrag-postgres psql -U finrag -d financial_rag \
  -c "SELECT extname FROM pg_extension WHERE extname IN ('vector','pg_trgm','uuid-ossp');"
```

Verify Redis:

```bash
# Load environment variables from .env
export $(grep -v '^#' .env | xargs)
echo $REDIS_PASSWORD  # Should show the password
docker exec finrag-redis redis-cli -a "$REDIS_PASSWORD" ping #or simply use the password
# → PONG
```

---

## Step 12 — Write the Phase 1 Verification Test

Create `tests/integration/test_phase1_foundation.py`:

```python
# tests/integration/test_phase1_foundation.py
"""
Phase 1 end-to-end verification.
Requires running Docker services (docker compose up -d).
Run with: pytest tests/integration/test_phase1_foundation.py -v
"""
import pytest
from sqlalchemy import text

from financial_rag.config import get_settings
from financial_rag.storage.cache import CacheClient, build_key
from financial_rag.storage.database import DatabaseClient


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
async def db(settings):
    """Connects a fresh DatabaseClient and tears it down after the test."""
    client = DatabaseClient()
    await client.connect()
    yield client
    await client.disconnect()


@pytest.fixture
async def cache(settings):
    """Connects a fresh CacheClient and tears it down after the test."""
    client = CacheClient()
    await client.connect()
    yield client
    await client.disconnect()


class TestSettings:
    def test_env_is_development(self, settings):
        assert settings.APP_ENV == "development"

    def test_debug_auto_enabled_in_development(self, settings):
        assert settings.DEBUG is True

    def test_database_url_uses_asyncpg(self, settings):
        assert settings.DATABASE_URL.startswith("postgresql+asyncpg://")

    def test_database_url_sync_uses_psycopg2(self, settings):
        assert settings.DATABASE_URL_SYNC.startswith("postgresql+psycopg2://")

    def test_redis_url_format(self, settings):
        assert settings.REDIS_URL.startswith("redis://:")

    def test_password_not_exposed_in_repr(self, settings):
        assert settings.POSTGRES_PASSWORD.get_secret_value() not in repr(settings)

    def test_chunk_overlap_less_than_chunk_size(self, settings):
        assert settings.CHUNK_OVERLAP_TOKENS < settings.CHUNK_SIZE_TOKENS


class TestDatabase:
    @pytest.mark.integration
    async def test_connection_is_live(self, db):
        async with db.connection() as conn:
            result = await conn.execute(text("SELECT 1 AS val"))
            row = result.fetchone()
        assert row.val == 1

    @pytest.mark.integration
    async def test_pgvector_extension_installed(self, db):
        async with db.connection() as conn:
            result = await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            row = result.fetchone()
        assert row is not None, "pgvector extension not found"

    @pytest.mark.integration
    async def test_all_tables_exist(self, db):
        expected = {"filings", "financial_chunks", "analysis_history", "schema_migrations"}
        async with db.connection() as conn:
            result = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            tables = {row.tablename for row in result.fetchall()}
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    @pytest.mark.integration
    async def test_session_commits_and_rolls_back(self, db):
        # Insert a migration record then verify it's there
        async with db.session() as session:
            await session.execute(
                text(
                    "INSERT INTO schema_migrations (version, description) "
                    "VALUES ('test-001', 'phase1 test') "
                    "ON CONFLICT (version) DO NOTHING"
                )
            )
        async with db.connection() as conn:
            result = await conn.execute(
                text("SELECT version FROM schema_migrations WHERE version = 'test-001'")
            )
            row = result.fetchone()
        assert row is not None

        # Clean up
        async with db.session() as session:
            await session.execute(
                text("DELETE FROM schema_migrations WHERE version = 'test-001'")
            )


class TestCache:
    @pytest.mark.integration
    async def test_ping(self, cache):
        # connect() already called ping() — if we got here, it works
        assert cache._redis is not None

    @pytest.mark.integration
    async def test_set_and_get(self, cache):
        key = build_key("test", "phase1")
        await cache.set(key, {"value": 42}, ttl=60)
        result = await cache.get(key)
        assert result == {"value": 42}
        await cache.delete(key)

    @pytest.mark.integration
    async def test_get_missing_key_returns_none(self, cache):
        result = await cache.get(build_key("test", "nonexistent-xyz"))
        assert result is None

    @pytest.mark.integration
    async def test_delete_key(self, cache):
        key = build_key("test", "delete-me")
        await cache.set(key, "temporary", ttl=60)
        deleted = await cache.delete(key)
        assert deleted is True
        assert await cache.get(key) is None

    @pytest.mark.integration
    async def test_key_namespacing(self):
        key = build_key("query", "abc123")
        assert key == "finrag:query:abc123"

    @pytest.mark.integration
    async def test_clear_namespace(self, cache):
        for i in range(3):
            await cache.set(build_key("cleartest", str(i)), i, ttl=60)
        count = await cache.clear_namespace("cleartest")
        assert count == 3
```

Run the tests:

```bash
pytest tests/integration/test_phase1_foundation.py -v -m integration
```

Expected output:
```
tests/integration/test_phase1_foundation.py::TestSettings::test_env_is_development PASSED
tests/integration/test_phase1_foundation.py::TestSettings::test_debug_auto_enabled_in_development PASSED
tests/integration/test_phase1_foundation.py::TestSettings::test_database_url_uses_asyncpg PASSED
...
tests/integration/test_phase1_foundation.py::TestDatabase::test_connection_is_live PASSED
tests/integration/test_phase1_foundation.py::TestDatabase::test_pgvector_extension_installed PASSED
tests/integration/test_phase1_foundation.py::TestDatabase::test_all_tables_exist PASSED
tests/integration/test_phase1_foundation.py::TestCache::test_set_and_get PASSED
...
====== 14 passed in 1.23s ======
```

---

## Final File Tree for Phase 1

```
financial-rag-agent/
├── .env                              # ← never committed
├── .env.example                      # ← committed
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
│
├── infrastructure/
│   └── docker/
│       └── init/
│           └── 01_create_schema.sql
│
├── src/
│   └── financial_rag/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py           ← Component #1
│       └── storage/
│           ├── __init__.py
│           ├── database.py           ← Component #4
│           └── cache.py              ← Component #5
│
└── tests/
    └── integration/
        └── test_phase1_foundation.py
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `POSTGRES_PASSWORD must be set` | `.env` not found or empty | Check working directory; ensure `.env` exists |
| `connection refused` on port 5432 | Docker not started | `docker compose up -d` |
| `ModuleNotFoundError: financial_rag` | Not installed editable | `pip install -e ".[dev]"` |
| `asyncio_mode` warning | Missing pytest-asyncio config | Already in `pyproject.toml` — reinstall dev deps |
| `vector type not found` | pgvector extension not created | Check `01_create_schema.sql` is in `init/` directory |
| `Configuration errors: CHUNK_OVERLAP...` | Overlap ≥ chunk size in `.env` | Reduce `CHUNK_OVERLAP_TOKENS` below `CHUNK_SIZE_TOKENS` |

---

## What's Next — Phase 2

Phase 2 builds the SEC EDGAR ingestion pipeline on top of this foundation:

- Token bucket rate limiter (respecting EDGAR's 10 req/s hard limit)
- CIK resolution: ticker symbol → EDGAR company ID
- Filing discovery and download with SHA-256 deduplication
- HTML parser to strip SGML/XBRL wrappers
- Section detection (MD&A, Risk Factors, Financial Statements)
- Text normalization and metric extraction (revenue, EPS, margins)

Every component will write to the `filings` table you created in this phase.
