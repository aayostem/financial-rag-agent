# =============================================================================
# Financial RAG Agent — Unified Settings
# src/financial_rag/config/settings.py
#
# Single source of truth for all configuration.
# Environment-specific behaviour is driven entirely by APP_ENV + .env values.
# No subclasses. No conditional imports. No split files.
#
# Usage:
#   from financial_rag.config import get_settings
#   settings = get_settings()
# =============================================================================

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Unified application settings backed by environment variables and .env file.

    All secrets use SecretStr — values are never exposed in logs or repr().
    Required production secrets have no defaults — the app will refuse to
    start if they are missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # =========================================================================
    # Environment
    # =========================================================================
    APP_ENV: Literal["development", "staging", "production", "testing"] = Field(
        default="development",
        description="Runtime environment. Drives defaults for debug, logging, models.",
    )
    APP_NAME: str = Field(default="financial-rag-agent")
    APP_VERSION: str = Field(default="0.1.0")

    # =========================================================================
    # Derived flags — set automatically from APP_ENV, overridable via .env
    # =========================================================================
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode. Auto-true in development/testing.",
    )
    TESTING: bool = Field(
        default=False,
        description="Enable testing mode. Allows mock injection.",
    )
    MOCK_EXTERNAL_APIS: bool = Field(
        default=False,
        description="Replace all external API calls with mocks. Auto-true in testing.",
    )

    @model_validator(mode="after")
    def apply_env_defaults(self) -> Settings:
        """
        Derive sensible flag defaults from APP_ENV without subclassing.
        Only overrides values that were not explicitly set in the environment.
        """
        is_dev = self.APP_ENV == "development"
        is_test = self.APP_ENV == "testing"

        # DEBUG: true in dev and test unless explicitly overridden
        if is_dev or is_test:
            object.__setattr__(self, "DEBUG", True)

        # TESTING + MOCK_EXTERNAL_APIS: true in test unless explicitly overridden
        if is_test:
            object.__setattr__(self, "TESTING", True)
            object.__setattr__(self, "MOCK_EXTERNAL_APIS", True)

        return self

    # =========================================================================
    # Paths — computed, never stored, never overridden in subclasses
    # =========================================================================
    @computed_field
    @property
    def PROJECT_ROOT(self) -> Path:
        """Absolute path to the project root (where pyproject.toml lives)."""
        return Path(__file__).resolve().parents[3]

    @computed_field
    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @computed_field
    @property
    def RAW_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @computed_field
    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @computed_field
    @property
    def VECTOR_STORE_DIR(self) -> Path:
        """
        Test environment uses /tmp to avoid polluting real data.
        All other environments use the project data directory.
        """
        if self.APP_ENV == "testing":
            return Path("/tmp/finrag_test/vector_store")
        return self.DATA_DIR / "vector_store"

    # =========================================================================
    # API Server
    # =========================================================================
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    API_WORKERS: int = Field(default=1, ge=1, description="Increase to 4+ in production")
    # Stored as raw str — prevents pydantic-settings from calling json.loads()
    # on comma-separated values like "https://a.com,https://b.com".
    # Use CORS_ORIGINS_LIST computed property in application code.
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Comma-separated CORS origins. Example: https://a.com,https://b.com",
    )

    @computed_field
    @property
    def CORS_ORIGINS_LIST(self) -> list[str]:
        """Parsed list of CORS origins for use in FastAPI CORSMiddleware."""
        if not self.CORS_ORIGINS or not self.CORS_ORIGINS.strip():
            return []
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # =========================================================================
    # OpenAI
    # =========================================================================
    OPENAI_API_KEY: SecretStr | None = Field(
        default=None,
        description="Required when EMBEDDING_PROVIDER or LLM_PROVIDER is 'openai'.",
    )
    ANTHROPIC_API_KEY: SecretStr | None = Field(
        default=None,
        description="Required when LLM_PROVIDER is 'anthropic'.",
    )

    # =========================================================================
    # Embedding
    # =========================================================================
    EMBEDDING_PROVIDER: Literal["openai", "local"] = Field(
        default="local",
        description="Use 'openai' (text-embedding-3-large) in production.",
    )
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "Model name. "
            "Production: 'text-embedding-3-large' (3072 dims). "
            "Development/testing: 'all-MiniLM-L6-v2' (384 dims)."
        ),
    )
    EMBEDDING_DIMENSIONS: int = Field(
        default=384,
        description="Must match the model. 3072 for text-embedding-3-large.",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Chunks per OpenAI embedding API call.",
    )

    # =========================================================================
    # LLM
    # =========================================================================
    LLM_PROVIDER: Literal["openai", "anthropic", "local"] = Field(default="openai")
    LLM_MODEL: str = Field(
        default="gpt-3.5-turbo",
        description="Production: 'gpt-4o'. Development: 'gpt-3.5-turbo'.",
    )
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=1)
    LLM_REQUEST_TIMEOUT: int = Field(default=60, ge=5, description="Seconds")
    LLM_BASE_URL: str = Field(default="https://api.groq.com/openai/v1")

    # =========================================================================
    # PostgreSQL + pgvector
    # =========================================================================
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432, ge=1, le=65535)
    POSTGRES_USER: str = Field(default="finrag")
    POSTGRES_PASSWORD: SecretStr = Field(
        ...,  # REQUIRED — no default, app will not start without it
        description="PostgreSQL password. Must be set in .env.",
    )
    POSTGRES_DB: str = Field(default="financial_rag")

    DB_POOL_MIN_SIZE: int = Field(default=2, ge=1)
    DB_POOL_MAX_SIZE: int = Field(default=10, ge=2)
    DB_POOL_RECYCLE_SECONDS: int = Field(default=1800, ge=60)
    DB_QUERY_TIMEOUT_SECONDS: int = Field(default=30, ge=1)
    DB_CONNECT_TIMEOUT_SECONDS: int = Field(default=10, ge=1)

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """
        Async-compatible PostgreSQL DSN for SQLAlchemy + asyncpg.
        Returns a plain string — SQLAlchemy 2.0 accepts both str and URL.
        """
        return (
            f"postgresql+asyncpg://"
            f"{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}"
            f"/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Sync DSN for Alembic migrations (psycopg2)."""
        return (
            f"postgresql+psycopg2://"
            f"{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}"
            f"/{self.POSTGRES_DB}"
        )

    # =========================================================================
    # Redis
    # =========================================================================
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_PASSWORD: SecretStr = Field(
        ...,  # REQUIRED — no default
        description="Redis password. Must be set in .env.",
    )
    REDIS_MAX_CONNECTIONS: int = Field(default=20, ge=1)
    REDIS_SOCKET_TIMEOUT_SECONDS: int = Field(default=5, ge=1)
    REDIS_CONNECT_TIMEOUT_SECONDS: int = Field(default=5, ge=1)
    REDIS_DEFAULT_TTL_SECONDS: int = Field(
        default=3600,
        ge=60,
        description="Default cache TTL. 1 hour.",
    )

    @computed_field
    @property
    def REDIS_URL(self) -> str:
        """
        Redis DSN with auth. Returns plain string.
        Format: redis://:password@host:port/db
        """
        return (
            f"redis://:{self.REDIS_PASSWORD.get_secret_value()}"
            f"@{self.REDIS_HOST}:{self.REDIS_PORT}"
            f"/{self.REDIS_DB}"
        )

    # =========================================================================
    # Processing & Retrieval
    # =========================================================================
    CHUNK_SIZE_TOKENS: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Target chunk size in tokens. 512 is optimal for financial text.",
    )
    CHUNK_OVERLAP_TOKENS: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Overlap between consecutive chunks in tokens.",
    )
    TOP_K_RESULTS: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks retrieved per query.",
    )
    HYBRID_SEARCH_ALPHA: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector vs keyword in hybrid search. 1.0 = pure vector.",
    )
    VECTOR_SEARCH_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score. Below this, fall back to hybrid.",
    )
    MAX_CONTEXT_TOKENS: int = Field(
        default=6000,
        ge=1000,
        description="Maximum tokens passed to LLM as context.",
    )

    # =========================================================================
    # Logging
    # =========================================================================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level. DEBUG in development, WARNING in production.",
    )
    LOG_FORMAT: Literal["json", "console"] = Field(
        default="console",
        description="'json' for production log aggregators, 'console' for development.",
    )

    @model_validator(mode="after")
    def apply_log_defaults(self) -> Settings:
        """Set log level and format from APP_ENV if not explicitly configured."""
        if self.APP_ENV == "production":
            if self.LOG_LEVEL == "INFO":  # only override if still at default
                object.__setattr__(self, "LOG_LEVEL", "WARNING")
            if self.LOG_FORMAT == "console":
                object.__setattr__(self, "LOG_FORMAT", "json")
        elif self.APP_ENV in ("development", "testing"):
            if self.LOG_LEVEL == "INFO":
                object.__setattr__(self, "LOG_LEVEL", "DEBUG")
        return self

    # =========================================================================
    # Rate Limiting (all environments — controlled by values)
    # =========================================================================
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        ge=1,
        description="Max requests per RATE_LIMIT_PERIOD. Lower in production via .env.",
    )
    RATE_LIMIT_PERIOD_SECONDS: int = Field(default=60, ge=1)
    # API key authentication
    API_KEY_ENABLED: bool = Field(
        default=False,
        description="Enable API key authentication via X-API-Key header.",
    )
    API_KEY: SecretStr | None = Field(
        default=None,
        description="API key required when API_KEY_ENABLED=True.",
    )

    # =========================================================================
    # SEC EDGAR Ingestion
    # =========================================================================
    EDGAR_USER_AGENT: str = Field(
        default="financial-rag-agent contact@example.com",
        description="Required by EDGAR. Must identify your app and contact email.",
    )
    EDGAR_RATE_LIMIT_RPS: int = Field(
        default=8,
        ge=1,
        le=10,
        description="Requests per second. EDGAR hard limit is 10.",
    )
    EDGAR_REQUEST_TIMEOUT_SECONDS: int = Field(default=30, ge=5)
    EDGAR_MAX_RETRIES: int = Field(default=3, ge=1)

    # =========================================================================
    # Validators
    # =========================================================================
    @model_validator(mode="after")
    def validate_chunk_settings(self) -> Settings:
        """Chunk overlap must be strictly less than chunk size."""
        if self.CHUNK_OVERLAP_TOKENS >= self.CHUNK_SIZE_TOKENS:
            raise ValueError(
                f"CHUNK_OVERLAP_TOKENS ({self.CHUNK_OVERLAP_TOKENS}) must be less than "
                f"CHUNK_SIZE_TOKENS ({self.CHUNK_SIZE_TOKENS})."
            )
        return self

    @model_validator(mode="after")
    def validate_startup(self) -> Settings:
        """
        Single consolidated startup validator.

        Collects ALL configuration issues before raising so operators
        see every problem at once, not one at a time.

        Covers:
          - Provider API key presence  (skipped in testing)
          - Production hardening guards (production only)
        """
        issues: list[str] = []
        is_testing = self.APP_ENV == "testing" or self.MOCK_EXTERNAL_APIS

        # ── Provider key checks (skip in testing) ────────────────────────────
        if not is_testing:
            if self.EMBEDDING_PROVIDER == "openai" and not self.OPENAI_API_KEY:
                issues.append(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER='openai'. Set it in .env."
                )
            if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
                issues.append(
                    "OPENAI_API_KEY is required when LLM_PROVIDER='openai'. Set it in .env."
                )
            if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
                issues.append(
                    "ANTHROPIC_API_KEY is required when LLM_PROVIDER='anthropic'. Set it in .env."
                )

        # ── Production hardening (production environment only) ────────────────
        if self.APP_ENV == "production":
            if self.CORS_ORIGINS.strip() in ("*", ""):
                issues.append(
                    "CORS_ORIGINS is '*' or empty — must be specific domains in production."
                )
            if self.DEBUG:
                issues.append("DEBUG=True in production.")
            if not self.POSTGRES_HOST or self.POSTGRES_HOST == "localhost":
                issues.append("POSTGRES_HOST is 'localhost' — use a real host in production.")

        if issues:
            raise ValueError(
                "Configuration errors detected:\n" + "\n".join(f"  • {issue}" for issue in issues)
            )

        return self


# =============================================================================
# Accessor — single cached instance per process
# =============================================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    The @lru_cache ensures this is instantiated exactly once per process.
    In tests, call get_settings.cache_clear() before injecting overrides.

    Example:
        from financial_rag.config import get_settings
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
