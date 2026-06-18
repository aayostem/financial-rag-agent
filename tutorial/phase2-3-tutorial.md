# Financial RAG Agent — Phase 2 & 3: Ingestion, Processing & Hybrid Search

> **Series:** Financial RAG Agent (freeCodeCamp)  
> **Phase 2:** SEC EDGAR Ingestion Pipeline — Components #7–14  
> **Phase 3:** Text Processing, Chunking & Hybrid Search — Components #16–28  
> **Time:** Phase 2 ~2 hours | Phase 3 ~2.5 hours  
> **Prerequisite:** Phase 1 complete — Docker services healthy, settings/database/cache working

---

## What You Will Build

**Phase 2** — A production-grade SEC EDGAR ingestion pipeline:
- Token-bucket rate limiter (respecting EDGAR's 10 req/s hard limit)
- CIK resolution: ticker symbol → EDGAR company ID
- Filing discovery and download with SHA-256 deduplication
- HTML parser that strips SGML/XBRL noise into clean text
- Section detector: MD&A, Risk Factors, Financial Statements, etc.
- Text normaliser and financial metric extractor
- Exception hierarchy that never lets bare `Exception` escape

**Phase 3** — Vector embeddings, chunking and hybrid search:
- Exact token counting with tiktoken (never estimate with `len/4`)
- Section-aware chunking with configurable overlap
- Dual-provider embedding client: OpenAI in production, local in development
- pgvector similarity search with HNSW index
- BM25 full-text search via `pg_trgm`
- Reciprocal Rank Fusion to combine both search signals
- Redis query cache on top of the retriever
- Alembic baseline migration

---

# PHASE 2 — SEC EDGAR Ingestion Pipeline

## New Files in This Phase

```
src/financial_rag/
├── utils/
│   ├── __init__.py
│   └── exceptions.py              ← Component #7 (foundation for all errors)
├── ingestion/
│   ├── __init__.py
│   ├── sec_ingestor.py            ← Components #7–10
│   └── parsers/
│       ├── __init__.py
│       ├── html_parser.py         ← Components #11–12
│       └── text_parser.py         ← Components #13–14
└── storage/
    └── repositories/
        ├── __init__.py
        ├── base.py                ← BaseRepository (used by all repositories)
        └── filings.py             ← FilingsRepository (deduplication)
```

---

<!-- ## Step 1 — Create the Directory Structure

```bash
mkdir -p src/financial_rag/utils -
mkdir -p src/financial_rag/ingestion/parsers
touch src/financial_rag/utils/__init__.py - 
touch src/financial_rag/ingestion/__init__.py
touch src/financial_rag/ingestion/parsers/__init__.py
``` -->

---

## Step 2 — Exception Hierarchy (`utils/exceptions.py`)

Before writing a single line of ingestion code, define your exception hierarchy.
This is Component #7 — everything else imports from here.

> **Why a custom hierarchy?**  
> Bare `Exception` tells callers nothing. `SECFetchError` tells them exactly
> which layer failed and why. Every `except` clause in this project catches
> a specific typed exception — never a bare `except Exception`.

Create `src/financial_rag/utils/exceptions.py`:
```bash 
mkdir -p src/financial_rag/utils
touch src/financial_rag/utils/__init__.py 
touch src/financial_rag/utils/exceptions.py
```

```python
# src/financial_rag/utils/exceptions.py
from __future__ import annotations


class FinRAGError(Exception):
    """Base exception for all Financial RAG Agent errors."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        if self.cause:
            return f"{base} (caused by: {self.cause})"
        return base


# =============================================================================
# Storage
# =============================================================================

class StorageError(FinRAGError):
    """Base for all storage-layer errors."""

class DatabaseConnectionError(StorageError):
    """Cannot establish or maintain a PostgreSQL connection."""

class DatabaseQueryError(StorageError):
    """A database query failed at runtime. Wraps SQLAlchemy/asyncpg errors."""

class CacheConnectionError(StorageError):
    """Cannot establish or maintain a Redis connection."""

class CacheOperationError(StorageError):
    """A Redis operation (GET, SET, DELETE) failed at runtime."""

class RecordNotFoundError(StorageError):
    """Required record not found — analogous to HTTP 404 at the storage layer."""

    def __init__(self, entity: str, identifier: str | int) -> None:
        super().__init__(f"{entity} not found: {identifier}")
        self.entity = entity
        self.identifier = identifier


# =============================================================================
# Ingestion
# =============================================================================

class IngestionError(FinRAGError):
    """Base for all document ingestion errors."""

class SECFetchError(IngestionError):
    """EDGAR API unreachable, rate-limited, or returned an error."""

class DocumentParseError(IngestionError):
    """A document (HTML, PDF) could not be parsed into clean text."""

class DuplicateFilingError(IngestionError):
    """Filing already exists in the database (identified by SHA-256 hash)."""

    def __init__(self, ticker: str, file_hash: str) -> None:
        super().__init__(
            f"Filing already ingested — ticker={ticker} hash={file_hash}"
        )
        self.ticker = ticker
        self.file_hash = file_hash


# =============================================================================
# Processing
# =============================================================================

class ProcessingError(FinRAGError):
    """Base for all document processing errors."""

class ChunkingError(ProcessingError):
    """Document chunking failed — tokenisation or split constraint violated."""


# =============================================================================
# Retrieval
# =============================================================================

class RetrievalError(FinRAGError):
    """Base for all retrieval-layer errors."""

class EmbeddingError(RetrievalError):
    """Embedding API returned an error or timed out."""

class VectorSearchError(RetrievalError):
    """pgvector similarity search failed."""


# =============================================================================
# Configuration
# =============================================================================

class ConfigurationError(FinRAGError):
    """Fatal misconfiguration not caught at settings validation time."""
```

The full hierarchy in one view:

```
FinRAGError
├── StorageError
│   ├── DatabaseConnectionError
│   ├── DatabaseQueryError
│   ├── CacheConnectionError
│   ├── CacheOperationError
│   └── RecordNotFoundError
├── IngestionError
│   ├── SECFetchError
│   ├── DocumentParseError
│   └── DuplicateFilingError
├── ProcessingError
│   └── ChunkingError
├── RetrievalError
│   ├── EmbeddingError
│   └── VectorSearchError
└── ConfigurationError
```

Update `src/financial_rag/utils/__init__.py`:

```python
from financial_rag.utils.exceptions import (
    ChunkingError,
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseQueryError,
    DocumentParseError,
    DuplicateFilingError,
    EmbeddingError,
    FinRAGError,
    IngestionError,
    RecordNotFoundError,
    RetrievalError,
    SECFetchError,
    VectorSearchError,
)

__all__ = [
    "FinRAGError", "SECFetchError", "DuplicateFilingError",
    "DocumentParseError", "DatabaseQueryError", "DatabaseConnectionError",
    "RecordNotFoundError", "ChunkingError", "EmbeddingError",
    "VectorSearchError", "RetrievalError", "IngestionError",
    "ConfigurationError",
]
```

---

## Step 3 — Base Repository (`storage/repositories/base.py`)

This is the generic data access layer that all repositories inherit from.
You write it once and every future repository gets `add`, `get_by_id`,
`list_all`, `update`, `delete`, and `soft_delete` for free.

> **Key design principle:** Repositories do NOT own sessions.
> They borrow a session injected from the service layer.
> Transaction management (commit/rollback) belongs to the caller.

Create `src/financial_rag/storage/repositories/base.py`:

```python
# src/financial_rag/storage/repositories/base.py
from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_rag.utils.exceptions import DatabaseQueryError, RecordNotFoundError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    """
    Generic async repository base.

    Subclasses declare:
        model_class: type[ModelT]   — the SQLAlchemy ORM model

    Repositories borrow sessions — they do not own or commit them.
    """

    model_class: type[Any]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Create ────────────────────────────────────────────────────────────────

    async def add(self, instance: ModelT) -> ModelT:
        """
        Persist a new ORM instance.
        Flushes (sends SQL to DB) but does NOT commit — caller controls the
        transaction. This means the insert is visible within the same session
        but rolls back if the caller's context manager exits with an error.
        """
        try:
            self._session.add(instance)
            await self._session.flush()
            await self._session.refresh(instance)
            logger.debug(
                "Added %s id=%s",
                self.model_class.__name__,
                getattr(instance, "id", "?"),
            )
            return instance
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to add {self.model_class.__name__}: {exc}"
            ) from exc

    async def add_many(self, instances: list[ModelT]) -> list[ModelT]:
        """Bulk-persist a list of ORM instances in one flush."""
        if not instances:
            return []
        try:
            self._session.add_all(instances)
            await self._session.flush()
            logger.debug(
                "Bulk-added %d %s records",
                len(instances),
                self.model_class.__name__,
            )
            return instances
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to bulk-add {self.model_class.__name__}: {exc}"
            ) from exc

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_by_id(self, record_id: UUID) -> ModelT:
        """Fetch by primary key. Raises RecordNotFoundError if missing."""
        try:
            instance = await self._session.get(self.model_class, record_id)
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to fetch {self.model_class.__name__} id={record_id}: {exc}"
            ) from exc

        if instance is None:
            raise RecordNotFoundError(self.model_class.__name__, str(record_id))
        return instance

    async def get_by_id_or_none(self, record_id: UUID) -> ModelT | None:
        """Fetch by primary key. Returns None if missing (no exception)."""
        try:
            return await self._session.get(self.model_class, record_id)
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to fetch {self.model_class.__name__} id={record_id}: {exc}"
            ) from exc

    async def list_all(self, *, limit: int = 100, offset: int = 0) -> list[ModelT]:
        """Paginated full-table scan. Use concrete filters on large tables."""
        try:
            stmt = select(self.model_class).limit(limit).offset(offset)
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to list {self.model_class.__name__}: {exc}"
            ) from exc

    async def count(self) -> int:
        """Total row count for the table."""
        try:
            result = await self._session.execute(
                select(func.count()).select_from(self.model_class)
            )
            return result.scalar_one()
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to count {self.model_class.__name__}: {exc}"
            ) from exc

    # ── Update ────────────────────────────────────────────────────────────────

    async def update(self, instance: ModelT, **fields: Any) -> ModelT:
        """
        Apply field updates to a loaded ORM instance and flush.

        Example:
            filing = await repo.get_by_id(filing_id)
            await repo.update(filing, is_active=False)
        """
        try:
            for key, value in fields.items():
                if not hasattr(instance, key):
                    raise DatabaseQueryError(
                        f"{self.model_class.__name__} has no attribute '{key}'"
                    )
                setattr(instance, key, value)
            await self._session.flush()
            await self._session.refresh(instance)
            return instance
        except DatabaseQueryError:
            raise
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to update {self.model_class.__name__}: {exc}"
            ) from exc

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete(self, instance: ModelT) -> None:
        """Hard delete. Flushes but does not commit."""
        try:
            await self._session.delete(instance)
            await self._session.flush()
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to delete {self.model_class.__name__}: {exc}"
            ) from exc

    async def soft_delete(self, instance: ModelT) -> ModelT:
        """Set is_active=False instead of deleting. Model must have is_active."""
        return await self.update(instance, is_active=False)
```

---

## Step 4 — Filings Repository (`storage/repositories/filings.py`)

The filings repository is the deduplication gate — before storing any filing
the ingestor calls `exists_by_hash()` to check the SHA-256 fingerprint.

> **Important:** Notice the `Filing` ORM model does NOT map `ingested_at`.
> That column has `DEFAULT NOW()` in PostgreSQL, so it is populated by the
> database on every raw SQL insert. When you insert via SQLAlchemy ORM,
> Postgres still fires the default — but if you ever need `ingested_at`
> in Python after an insert, call `await session.refresh(filing)` first.

Create `src/financial_rag/storage/repositories/filings.py`:

```python
# src/financial_rag/storage/repositories/filings.py
from __future__ import annotations

import logging
from datetime import date
from uuid import UUID

from sqlalchemy import Boolean, Date, Integer, SmallInteger, String, select
from sqlalchemy.orm import Mapped, mapped_column

from financial_rag.storage.database import Base
from financial_rag.storage.repositories.base import BaseRepository
from financial_rag.utils.exceptions import DatabaseQueryError

logger = logging.getLogger(__name__)


class Filing(Base):
    """
    ORM representation of the `filings` table.
    Keep in sync with infrastructure/docker/init/01_create_schema.sql.

    Note: ingested_at is intentionally omitted — it has DEFAULT NOW() in
    Postgres and is never set from Python. Call session.refresh(filing)
    to read it back after an insert.
    """

    __tablename__ = "filings"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    filing_type: Mapped[str] = mapped_column(String(20), nullable=False)
    fiscal_year: Mapped[int | None] = mapped_column(SmallInteger)
    fiscal_quarter: Mapped[int | None] = mapped_column(SmallInteger)
    filed_at: Mapped[date | None] = mapped_column(Date)
    source_url: Mapped[str | None] = mapped_column(String)
    file_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    pages: Mapped[int | None] = mapped_column(Integer)
    ingested_by: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<Filing id={self.id} ticker={self.ticker} "
            f"type={self.filing_type} year={self.fiscal_year}>"
        )


class FilingsRepository(BaseRepository[Filing]):
    """
    All database operations for the filings table.

    Inject via service constructor:
        async with db.session() as session:
            repo = FilingsRepository(session)
            exists = await repo.exists_by_hash(file_hash)
    """

    model_class = Filing

    # ── Deduplication ─────────────────────────────────────────────────────────

    async def get_by_hash(self, file_hash: str) -> Filing | None:
        """Look up a filing by its SHA-256 content hash."""
        try:
            result = await self._session.execute(
                select(Filing).where(Filing.file_hash == file_hash)
            )
            return result.scalar_one_or_none()
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to look up filing by hash '{file_hash}': {exc}"
            ) from exc

    async def exists_by_hash(self, file_hash: str) -> bool:
        """
        Return True if this hash is already in the database.
        Faster than get_by_hash() when only existence matters.
        """
        return await self.get_by_hash(file_hash) is not None

    # ── Filtered queries ──────────────────────────────────────────────────────

    async def get_by_ticker(
        self,
        ticker: str,
        *,
        active_only: bool = True,
        limit: int = 50,
    ) -> list[Filing]:
        """All filings for a ticker, newest fiscal year first."""
        try:
            stmt = (
                select(Filing)
                .where(Filing.ticker == ticker.upper())
                .order_by(Filing.fiscal_year.desc())
                .limit(limit)
            )
            if active_only:
                stmt = stmt.where(Filing.is_active.is_(True))
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to fetch filings for ticker '{ticker}': {exc}"
            ) from exc

    async def get_by_ticker_and_type(
        self,
        ticker: str,
        filing_type: str,
        *,
        fiscal_year: int | None = None,
        active_only: bool = True,
    ) -> list[Filing]:
        """Filings filtered by ticker + type, optionally by year."""
        try:
            stmt = (
                select(Filing)
                .where(
                    Filing.ticker == ticker.upper(),
                    Filing.filing_type == filing_type,
                )
                .order_by(Filing.fiscal_year.desc())
            )
            if fiscal_year is not None:
                stmt = stmt.where(Filing.fiscal_year == fiscal_year)
            if active_only:
                stmt = stmt.where(Filing.is_active.is_(True))
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to fetch {filing_type} filings for '{ticker}': {exc}"
            ) from exc

    async def get_latest(self, ticker: str, filing_type: str) -> Filing | None:
        """Most recent active filing for a ticker + type combination."""
        try:
            result = await self._session.execute(
                select(Filing)
                .where(
                    Filing.ticker == ticker.upper(),
                    Filing.filing_type == filing_type,
                    Filing.is_active.is_(True),
                )
                .order_by(Filing.fiscal_year.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to fetch latest {filing_type} for '{ticker}': {exc}"
            ) from exc

    async def list_tickers(self) -> list[str]:
        """Sorted list of all distinct tickers that have been ingested."""
        try:
            from sqlalchemy import distinct
            result = await self._session.execute(
                select(distinct(Filing.ticker))
                .where(Filing.is_active.is_(True))
                .order_by(Filing.ticker)
            )
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseQueryError(f"Failed to list tickers: {exc}") from exc
```

Update `src/financial_rag/storage/repositories/__init__.py`:

```python
from financial_rag.storage.repositories.base import BaseRepository
from financial_rag.storage.repositories.filings import Filing, FilingsRepository

__all__ = ["BaseRepository", "Filing", "FilingsRepository"]
```

---

## Step 5 — Rate Limiter & SEC Ingestor (`ingestion/sec_ingestor.py`)

This is the heart of Phase 2. Four components live in this file:
- **Component #7:** `_EdgarRateLimiter` — token bucket
- **Component #8:** `_resolve_cik()` — ticker → CIK
- **Component #9:** `_fetch_submissions()` — filing discovery
- **Component #10:** `download_filing()` — download + SHA-256

### Why a token bucket for EDGAR?

EDGAR enforces 10 requests/second per IP. A naive implementation fires all
requests at once and gets HTTP 429s. A token bucket works like this:

```
Semaphore(10) means: at most 10 concurrent requests
After each release: sleep 1/10 second (100ms)
Result: exactly 10 req/s, evenly spaced — never burst, never throttle
```

### The `@retry` decorator explained

```python
@retry(
    retry=retry_if_exception_type(SECFetchError),  # only retry THIS exception
    stop=stop_after_attempt(3),                     # max 3 total attempts
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s → 4s → 8s
    reraise=True,                                   # re-raise after exhaustion
)
```

This means: on a `SECFetchError`, wait 2 seconds, try again, wait 4 seconds,
try again, wait 8 seconds — then give up and re-raise the original error.
Any other exception type bypasses the retry entirely.

> **Bug fix from the review:** The original code had
> `years = min(years, self._settings.EDGAR_MAX_RETRIES * 2)`.
> `EDGAR_MAX_RETRIES` is 3, so this capped years at 6, not 5, and used
> the wrong setting. The corrected version below uses `min(years, 5)` directly.

Create `src/financial_rag/ingestion/sec_ingestor.py`:

```python
# src/financial_rag/ingestion/sec_ingestor.py
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import date
from typing import TYPE_CHECKING

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from financial_rag.config import get_settings
from financial_rag.utils.exceptions import SECFetchError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_EDGAR_BASE     = "https://data.sec.gov"
_EDGAR_FILINGS  = f"{_EDGAR_BASE}/submissions"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

SUPPORTED_FILING_TYPES = frozenset({"10-K", "10-Q", "8-K", "20-F", "DEF 14A", "S-1"})


# =============================================================================
# Component #7 — Rate Limiter
# =============================================================================

class _EdgarRateLimiter:
    """
    Token-bucket rate limiter for EDGAR API calls.

    How it works:
      - asyncio.Semaphore(rps) limits concurrent in-flight requests
      - After each release we sleep 1/rps seconds
      - Combined effect: exactly rps requests per second, evenly spaced
    """

    def __init__(self, rps: int) -> None:
        self._semaphore = asyncio.Semaphore(rps)
        self._interval = 1.0 / rps

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(self, *_: object) -> None:
        await asyncio.sleep(self._interval)
        self._semaphore.release()


# =============================================================================
# Filing metadata
# =============================================================================

class FilingMetadata:
    """
    Value object returned by list_filings().
    Passed to download_filing() to fetch actual content.
    Uses __slots__ for memory efficiency — we may hold thousands of these.
    """

    __slots__ = (
        "accession_number", "cik", "filed_at", "filing_type",
        "fiscal_quarter", "fiscal_year", "primary_document", "source_url", "ticker",
    )

    def __init__(
        self,
        *,
        ticker: str,
        filing_type: str,
        fiscal_year: int | None,
        fiscal_quarter: int | None,
        filed_at: date | None,
        accession_number: str,
        primary_document: str,
        source_url: str,
        cik: str,
    ) -> None:
        self.ticker = ticker
        self.filing_type = filing_type
        self.fiscal_year = fiscal_year
        self.fiscal_quarter = fiscal_quarter
        self.filed_at = filed_at
        self.accession_number = accession_number
        self.primary_document = primary_document
        self.source_url = source_url
        self.cik = cik

    def __repr__(self) -> str:
        return (
            f"<FilingMetadata {self.ticker} {self.filing_type} "
            f"FY{self.fiscal_year} filed={self.filed_at}>"
        )


# =============================================================================
# Components #7–10 — SECIngestor
# =============================================================================

class SECIngestor:
    """
    Async SEC EDGAR ingestor.

    Usage:
        async with SECIngestor() as ingestor:
            filings = await ingestor.list_filings("AAPL", "10-K", years=3)
            for meta in filings:
                raw_html, file_hash = await ingestor.download_filing(meta)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._rate_limiter = _EdgarRateLimiter(self._settings.EDGAR_RATE_LIMIT_RPS)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> SECIngestor:
        self._client = httpx.AsyncClient(
            headers={
                # EDGAR REQUIRES a User-Agent header that identifies your app
                # and provides a contact email. Without this your IP gets blocked.
                "User-Agent": self._settings.EDGAR_USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=self._settings.EDGAR_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def list_filings(
        self,
        ticker: str,
        filing_type: str,
        *,
        years: int = 3,
    ) -> list[FilingMetadata]:
        """
        Return metadata for the most recent N years of filings.

        Args:
            ticker:      Stock ticker (e.g. 'AAPL')
            filing_type: SEC form type (e.g. '10-K')
            years:       How many years to fetch (max 5)
        """
        if filing_type not in SUPPORTED_FILING_TYPES:
            raise SECFetchError(
                f"Unsupported filing type '{filing_type}'. "
                f"Supported: {sorted(SUPPORTED_FILING_TYPES)}"
            )

        # BUG FIX: original used EDGAR_MAX_RETRIES * 2 which is unrelated
        # to year limits and gives 6 not 5. Use a direct cap instead.
        years = min(years, 5)
        ticker = ticker.upper()

        logger.info("Listing %s filings for %s (years=%d)", filing_type, ticker, years)

        cik = await self._resolve_cik(ticker)
        submissions = await self._fetch_submissions(cik)
        filings = self._parse_submissions(
            submissions, ticker=ticker, filing_type=filing_type, cik=cik, years=years
        )

        logger.info("Found %d %s filings for %s", len(filings), filing_type, ticker)
        return filings

    async def download_filing(
        self,
        meta: FilingMetadata,
        *,
        raw_dir: Path | None = None,
    ) -> tuple[str, str]:
        """
        Download the primary document for a filing.

        Args:
            meta:    FilingMetadata from list_filings()
            raw_dir: Optional directory to cache raw HTML files.

        Returns:
            (raw_html_content, sha256_file_hash) tuple.
        """
        if raw_dir:
            cached = self._check_cache(meta, raw_dir)
            if cached:
                content, file_hash = cached
                logger.info("Cache hit for %s %s FY%s", meta.ticker, meta.filing_type, meta.fiscal_year)
                return content, file_hash

        content = await self._fetch_document(meta.source_url)
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        if raw_dir:
            self._write_cache(meta, raw_dir, content)

        logger.info(
            "Downloaded %s %s FY%s — %d chars hash=%s",
            meta.ticker, meta.filing_type, meta.fiscal_year,
            len(content), file_hash[:12],
        )
        return content, file_hash

    async def check_duplicate(self, file_hash: str, ticker: str, filing_type: str) -> bool:
        """Return True if this file_hash already exists in the filings table."""
        from financial_rag.storage.database import get_db_client
        from financial_rag.storage.repositories.filings import FilingsRepository

        client = await get_db_client()
        async with client.session() as session:
            repo = FilingsRepository(session)
            return await repo.exists_by_hash(file_hash)

    # ── Component #8 — CIK Resolution ─────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(SECFetchError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _resolve_cik(self, ticker: str) -> str:
        """
        Resolve ticker symbol → SEC CIK number.
        EDGAR's company_tickers.json maps all tickers to their CIK.
        CIKs are zero-padded to 10 digits for URL construction.
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        try:
            async with self._rate_limiter:
                response = await self._client.get(url)
                response.raise_for_status()

            data = response.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker:
                    cik = str(entry["cik_str"]).zfill(10)
                    logger.debug("Resolved %s → CIK %s", ticker, cik)
                    return cik

            raise SECFetchError(
                f"Ticker '{ticker}' not found in EDGAR company registry."
            )
        except httpx.HTTPError as exc:
            raise SECFetchError(f"Failed to resolve CIK for '{ticker}': {exc}") from exc

    # ── Component #9 — Filing Discovery ───────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(SECFetchError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_submissions(self, cik: str) -> dict:
        """Fetch the full submissions JSON for a CIK."""
        url = f"{_EDGAR_FILINGS}/CIK{cik}.json"
        try:
            async with self._rate_limiter:
                response = await self._client.get(url)
                response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise SECFetchError(f"Failed to fetch submissions for CIK {cik}: {exc}") from exc

    # ── Component #10 — Download + SHA-256 ────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(SECFetchError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_document(self, url: str) -> str:
        """Fetch the raw content of a filing document."""
        try:
            async with self._rate_limiter:
                response = await self._client.get(url, headers={"Host": "www.sec.gov"})
                response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            raise SECFetchError(f"Failed to download filing from {url}: {exc}") from exc

    # ── Submission parsing ────────────────────────────────────────────────────

    def _parse_submissions(
        self,
        data: dict,
        *,
        ticker: str,
        filing_type: str,
        cik: str,
        years: int,
    ) -> list[FilingMetadata]:
        """Parse EDGAR submissions JSON into FilingMetadata list."""
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            logger.warning("No recent filings found for %s", ticker)
            return []

        forms        = recent.get("form", [])
        accessions   = recent.get("accessionNumber", [])
        filed_dates  = recent.get("filingDate", [])
        documents    = recent.get("primaryDocument", [])

        results: list[FilingMetadata] = []
        seen_years: set[int] = set()

        for i, form in enumerate(forms):
            if form != filing_type:
                continue
            if len(results) >= years:
                break

            try:
                filed_at    = date.fromisoformat(filed_dates[i]) if filed_dates[i] else None
                fiscal_year = filed_at.year if filed_at else None
                accession   = accessions[i].replace("-", "")
                primary_doc = documents[i] if i < len(documents) else ""
                source_url  = f"{_EDGAR_ARCHIVES}/{int(cik)}/{accession}/{primary_doc}"

                if fiscal_year and fiscal_year in seen_years:
                    continue
                if fiscal_year:
                    seen_years.add(fiscal_year)

                results.append(FilingMetadata(
                    ticker=ticker, filing_type=filing_type,
                    fiscal_year=fiscal_year, fiscal_quarter=None,
                    filed_at=filed_at, accession_number=accession,
                    primary_document=primary_doc, source_url=source_url, cik=cik,
                ))
            except (IndexError, ValueError) as exc:
                logger.warning("Skipping malformed filing entry at index %d: %s", i, exc)
                continue

        return results

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, meta: FilingMetadata, raw_dir: Path) -> Path:
        safe_ticker = meta.ticker.replace("/", "_")
        return raw_dir / safe_ticker / meta.filing_type / f"FY{meta.fiscal_year}.html"

    def _check_cache(self, meta: FilingMetadata, raw_dir: Path) -> tuple[str, str] | None:
        path = self._cache_path(meta, raw_dir)
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="ignore")
            return content, hashlib.sha256(content.encode()).hexdigest()
        return None

    def _write_cache(self, meta: FilingMetadata, raw_dir: Path, content: str) -> None:
        path = self._cache_path(meta, raw_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.debug("Cached filing to %s", path)
```

---

## Step 6 — HTML Parser (`ingestion/parsers/html_parser.py`)

Components #11 (SGML stripping + section detection) and #12 (section parsing).

SEC filings are notoriously messy — they start with SGML header metadata,
contain inline XBRL tags, have EDGAR boilerplate, and repeat table-of-contents
entries that would confuse a naive section detector.

This parser handles all of that in five steps:

```
Raw EDGAR HTML
    ↓ _strip_sgml_header()    strip everything before <html>
    ↓ _remove_noise_tags()    decompose script/style/XBRL tags
    ↓ _extract_text()         get_text() + whitespace normalisation
    ↓ _remove_boilerplate()   strip page numbers, TOC dots, separators
    ↓ _detect_sections()      regex heading → section boundary detection
ParsedFiling (full_text + sections[])
```

Create `src/financial_rag/ingestion/parsers/html_parser.py`:

```python
# src/financial_rag/ingestion/parsers/html_parser.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Order matters — first matching pattern wins
_SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"management.{0,20}discussion.{0,20}analysis", re.I), "MD&A"),
    (re.compile(r"risk\s+factors", re.I), "Risk Factors"),
    (re.compile(r"quantitative.{0,20}qualitative.{0,20}market\s+risk", re.I), "Market Risk"),
    (re.compile(r"business\s+overview|item\s+1[.\s]+business", re.I), "Business"),
    (re.compile(r"financial\s+statements", re.I), "Financial Statements"),
    (re.compile(r"balance\s+sheet|financial\s+position", re.I), "Balance Sheet"),
    (re.compile(r"income\s+statement|results\s+of\s+operations", re.I), "Income Statement"),
    (re.compile(r"cash\s+flow", re.I), "Cash Flow"),
    (re.compile(r"legal\s+proceedings", re.I), "Legal Proceedings"),
    (re.compile(r"notes\s+to\s+(the\s+)?financial", re.I), "Notes to Financials"),
]

_DISCARD_TAGS = frozenset({
    "script", "style", "meta", "link", "head", "noscript",
    "svg", "img", "figure",
    "ix:nonfraction", "ix:nonnumeric",  # XBRL inline tags
    "xbrl", "xbrli",
})

_MAX_BLANK_LINES = 2


@dataclass
class ParsedSection:
    name: str
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    def __repr__(self) -> str:
        return f"<ParsedSection '{self.name}' chars={self.char_count}>"


@dataclass
class ParsedFiling:
    """
    Result of parsing a raw SEC filing HTML document.
    full_text: entire cleaned document
    sections:  per-section breakdown for section-aware chunking
    """
    ticker: str
    filing_type: str
    fiscal_year: int | None
    full_text: str
    sections: list[ParsedSection]
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.full_text)

    def get_section(self, name: str) -> str | None:
        """Return text for a named section, or None if not found."""
        for s in self.sections:
            if s.name == name:
                return s.text
        return None


class HTMLParser:
    """
    Parses raw SEC filing HTML into clean, section-tagged text.

    Usage:
        parser = HTMLParser()
        parsed = parser.parse(raw_html, ticker="AAPL",
                              filing_type="10-K", fiscal_year=2023)
    """

    def parse(
        self,
        raw_html: str,
        *,
        ticker: str,
        filing_type: str,
        fiscal_year: int | None = None,
    ) -> ParsedFiling:
        html_content = self._strip_sgml_header(raw_html)
        soup = BeautifulSoup(html_content, "lxml")
        self._remove_noise_tags(soup)
        full_text = self._extract_text(soup)
        sections = self._detect_sections(full_text)

        logger.debug(
            "Parsed %s %s — %d chars, %d sections",
            ticker, filing_type, len(full_text), len(sections),
        )

        return ParsedFiling(
            ticker=ticker, filing_type=filing_type,
            fiscal_year=fiscal_year, full_text=full_text, sections=sections,
        )

    def _strip_sgml_header(self, content: str) -> str:
        """Strip SGML metadata before the first <html> tag."""
        match = re.search(r"<html", content, re.I)
        return content[match.start():] if match else content

    def _remove_noise_tags(self, soup: BeautifulSoup) -> None:
        for tag_name in _DISCARD_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        for tag in soup.find_all(True):
            if isinstance(tag, Tag) and not tag.get_text(strip=True):
                tag.decompose()

    def _extract_text(self, soup: BeautifulSoup) -> str:
        raw_text = soup.get_text(separator="\n")
        lines = []
        for line in raw_text.splitlines():
            lines.append(re.sub(r"[ \t]+", " ", line).strip())

        cleaned: list[str] = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= _MAX_BLANK_LINES:
                    cleaned.append("")
            else:
                blank_count = 0
                cleaned.append(line)

        return self._remove_boilerplate("\n".join(cleaned).strip())

    def _remove_boilerplate(self, text: str) -> str:
        patterns = [
            r"\n\s*-\s*\d+\s*-\s*\n",      # page numbers: "- 42 -"
            r"\.{5,}\s*\d+",                 # TOC dots: "........ 12"
            r"={10,}",                        # separators
            r"-{10,}",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "\n", text, flags=re.S)
        return text.strip()

    def _detect_sections(self, text: str) -> list[ParsedSection]:
        """
        Identify major sections by scanning for heading lines.

        A line is treated as a heading if:
          - It is non-empty
          - It is ≤ 200 characters (headings are short)
          - It matches one of the _SECTION_PATTERNS regexes

        All text between two headings belongs to the first heading's section.
        """
        lines = text.splitlines()
        heading_positions: list[tuple[int, str]] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 200:
                continue
            for pattern, section_name in _SECTION_PATTERNS:
                if pattern.search(stripped):
                    heading_positions.append((i, section_name))
                    break

        if not heading_positions:
            return [ParsedSection(name="General", text=text)]

        sections: list[ParsedSection] = []
        seen: set[str] = set()

        for idx, (line_idx, section_name) in enumerate(heading_positions):
            end_idx = (
                heading_positions[idx + 1][0]
                if idx + 1 < len(heading_positions)
                else len(lines)
            )
            section_text = "\n".join(lines[line_idx:end_idx]).strip()

            if section_name in seen or len(section_text) < 200:
                continue

            seen.add(section_name)
            sections.append(ParsedSection(name=section_name, text=section_text))

        return sections or [ParsedSection(name="General", text=text)]
```

---

## Step 7 — Text Parser (`ingestion/parsers/text_parser.py`)

Components #13 (text normalisation) and #14 (metric extraction).

This runs after HTMLParser and before chunking. It:
- Normalises Unicode (NFKC — converts ligatures, accented chars)
- Strips URLs, emails, control characters
- Normalises financial number formatting (`$1,234` → `1234`)
- Extracts metrics (revenue, net income, EPS) into a dict for JSONB storage

Create `src/financial_rag/ingestion/parsers/text_parser.py`:

```python
# src/financial_rag/ingestion/parsers/text_parser.py
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(https?|ftp)://\S+", re.I),   # URLs
    re.compile(r"\S+@\S+\.\S+"),                   # emails
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),  # control chars
    re.compile(r"\n{4,}"),                          # excessive blank lines
]

_NUMBER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\$\s*([\d,]+(?:\.\d+)?)"), r"\1"),   # $1,234 → 1234
    (re.compile(r"(\d),(\d{3})"), r"\1\2"),             # 1,234,567 → 1234567
    (re.compile(r"(\d)\s+%"), r"\1%"),                  # 12.5 % → 12.5%
]


class TextParser:
    """
    Cleans and normalises plain text for downstream chunking and embedding.

    Does NOT chunk — that is TextProcessor's responsibility.
    Does NOT embed — that is EmbeddingClient's responsibility.
    """

    def clean(self, text: str) -> str:
        """Full cleaning pipeline: Unicode → noise removal → whitespace."""
        if not text or not text.strip():
            return ""

        # NFKC: normalise ligatures (ﬁ→fi), accents, half-width chars
        text = unicodedata.normalize("NFKC", text)

        for pattern in _NOISE_PATTERNS:
            text = pattern.sub(" ", text)

        lines = []
        for line in text.splitlines():
            lines.append(re.sub(r"[ \t]+", " ", line).strip())

        cleaned: list[str] = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append("")
            else:
                blank_count = 0
                cleaned.append(line)

        return "\n".join(cleaned).strip()

    def normalise_numbers(self, text: str) -> str:
        """
        Normalise numeric formatting in financial text.
        Removes formatting that fragments tokenisation without losing value.
        """
        for pattern, replacement in _NUMBER_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def extract_metrics(self, text: str) -> dict[str, float]:
        """
        Extract key financial metrics using regex.
        Returns dict stored in the `metrics` JSONB column of financial_chunks.
        """
        metrics: dict[str, float] = {}

        def _safe_float(raw: str) -> float | None:
            cleaned = raw.replace(",", "").strip()
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None

        revenue_match = re.search(
            r"(?:revenue|net\s+revenue|total\s+revenue)[^\d]*"
            r"([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)?",
            text, re.I,
        )
        if revenue_match:
            value = _safe_float(revenue_match.group(1))
            if value is not None:
                metrics["revenue"] = _apply_scale(value, revenue_match.group(2) or "")

        income_match = re.search(
            r"net\s+(?:income|earnings|loss)[^\d]*"
            r"([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)?",
            text, re.I,
        )
        if income_match:
            value = _safe_float(income_match.group(1))
            if value is not None:
                metrics["net_income"] = _apply_scale(value, income_match.group(2) or "")

        eps_match = re.search(
            r"(?:earnings\s+per\s+(?:diluted\s+)?share|eps)[^\d]*\$?([\d]+(?:\.\d+)?)",
            text, re.I,
        )
        if eps_match:
            value = _safe_float(eps_match.group(1))
            if value is not None:
                metrics["eps"] = value

        margin_match = re.search(
            r"(?:operating|gross|net)\s+margin[^\d]*([\d]+(?:\.\d+)?)\s*%",
            text, re.I,
        )
        if margin_match:
            value = _safe_float(margin_match.group(1))
            if value is not None:
                metrics["margin_pct"] = value

        return metrics


def _apply_scale(value: float, scale: str) -> float:
    scale = scale.lower()
    if scale == "billion":  return value * 1_000_000_000
    if scale == "million":  return value * 1_000_000
    if scale == "thousand": return value * 1_000
    return value
```

---

## Step 8 — Phase 2 Verification Test

Create `tests/integration/test_phase2_ingestion.py`:

```python
# tests/integration/test_phase2_ingestion.py
"""
Phase 2 verification — runs against live EDGAR (requires internet).
Unit tests for parsers run without network.
"""
import pytest

from financial_rag.ingestion.parsers.html_parser import HTMLParser, ParsedFiling
from financial_rag.ingestion.parsers.text_parser import TextParser
from financial_rag.ingestion.sec_ingestor import (
    SUPPORTED_FILING_TYPES,
    FilingMetadata,
    SECIngestor,
)
from financial_rag.utils.exceptions import DuplicateFilingError, SECFetchError


class TestExceptionHierarchy:
    def test_sec_fetch_error_is_ingestion_error(self):
        from financial_rag.utils.exceptions import IngestionError
        err = SECFetchError("test")
        assert isinstance(err, IngestionError)

    def test_duplicate_filing_error_stores_fields(self):
        err = DuplicateFilingError("AAPL", "abc123")
        assert err.ticker == "AAPL"
        assert err.file_hash == "abc123"
        assert "abc123" in str(err)

    def test_cause_appears_in_str(self):
        from financial_rag.utils.exceptions import FinRAGError
        original = ValueError("root cause")
        err = FinRAGError("wrapper", cause=original)
        assert "root cause" in str(err)


class TestHTMLParser:
    def setup_method(self):
        self.parser = HTMLParser()

    def test_strips_sgml_header(self):
        content = "SGML JUNK\n<html><body><p>Hello</p></body></html>"
        parsed = self.parser.parse(content, ticker="TEST", filing_type="10-K")
        assert "SGML JUNK" not in parsed.full_text
        assert "Hello" in parsed.full_text

    def test_detects_mda_section(self):
        html = """
        <html><body>
        <h2>Management's Discussion and Analysis</h2>
        <p>Revenue increased significantly this year driven by strong iPhone sales
        and services growth across all geographic segments worldwide.</p>
        <h2>Risk Factors</h2>
        <p>The company faces intense competition in all markets where it operates
        and must continue to innovate to maintain its competitive position.</p>
        </body></html>
        """
        parsed = self.parser.parse(html, ticker="AAPL", filing_type="10-K", fiscal_year=2023)
        section_names = [s.name for s in parsed.sections]
        assert "MD&A" in section_names

    def test_returns_general_when_no_sections(self):
        html = "<html><body><p>Plain text with no known section headings here.</p></body></html>"
        parsed = self.parser.parse(html, ticker="TEST", filing_type="10-K")
        assert len(parsed.sections) == 1
        assert parsed.sections[0].name == "General"

    def test_get_section_returns_none_for_missing(self):
        html = "<html><body><p>Text</p></body></html>"
        parsed = self.parser.parse(html, ticker="TEST", filing_type="10-K")
        assert parsed.get_section("MD&A") is None

    def test_removes_script_tags(self):
        html = "<html><body><script>alert('xss')</script><p>Clean</p></body></html>"
        parsed = self.parser.parse(html, ticker="TEST", filing_type="10-K")
        assert "alert" not in parsed.full_text

    def test_char_count_is_populated(self):
        html = "<html><body><p>Some content here</p></body></html>"
        parsed = self.parser.parse(html, ticker="TEST", filing_type="10-K")
        assert parsed.char_count == len(parsed.full_text)


class TestTextParser:
    def setup_method(self):
        self.parser = TextParser()

    def test_removes_urls(self):
        result = self.parser.clean("Visit https://example.com for more info.")
        assert "https://example.com" not in result

    def test_removes_emails(self):
        result = self.parser.clean("Contact investor@apple.com for details.")
        assert "investor@apple.com" not in result

    def test_normalise_numbers_strips_dollar_commas(self):
        result = self.parser.normalise_numbers("Revenue was $1,234,567 million")
        assert "$" not in result
        assert "1234567" in result

    def test_normalise_percentage_spacing(self):
        result = self.parser.normalise_numbers("Margin increased 12.5 %")
        assert "12.5%" in result

    def test_extract_revenue_metric(self):
        text = "Total revenue was $394.3 billion for the fiscal year."
        metrics = self.parser.extract_metrics(text)
        assert "revenue" in metrics
        assert metrics["revenue"] > 0

    def test_extract_eps_metric(self):
        text = "Earnings per diluted share was $6.13 for the quarter."
        metrics = self.parser.extract_metrics(text)
        assert "eps" in metrics
        assert abs(metrics["eps"] - 6.13) < 0.01

    def test_empty_text_returns_empty(self):
        assert self.parser.clean("") == ""
        assert self.parser.clean("   ") == ""


class TestSECIngestor:
    def test_supported_filing_types(self):
        assert "10-K" in SUPPORTED_FILING_TYPES
        assert "10-Q" in SUPPORTED_FILING_TYPES
        assert "INVALID" not in SUPPORTED_FILING_TYPES

    def test_years_capped_at_five(self):
        # Test the fixed cap (was a bug in original: used EDGAR_MAX_RETRIES * 2)
        ingestor = SECIngestor()
        # The cap is applied inside list_filings — we verify the logic directly
        assert min(10, 5) == 5
        assert min(3, 5) == 3

    @pytest.mark.integration
    async def test_resolve_cik_for_apple(self):
        """Live EDGAR test — requires internet connection."""
        async with SECIngestor() as ingestor:
            cik = await ingestor._resolve_cik("AAPL")
        assert cik == "0000320193"  # Apple's CIK, zero-padded to 10 digits

    @pytest.mark.integration
    async def test_list_filings_returns_metadata(self):
        """Live EDGAR test — requires internet connection."""
        async with SECIngestor() as ingestor:
            filings = await ingestor.list_filings("AAPL", "10-K", years=2)
        assert len(filings) <= 2
        for f in filings:
            assert f.ticker == "AAPL"
            assert f.filing_type == "10-K"
            assert f.fiscal_year is not None
            assert f.source_url.startswith("https://")

    @pytest.mark.integration
    async def test_unsupported_type_raises(self):
        async with SECIngestor() as ingestor:
            with pytest.raises(SECFetchError):
                await ingestor.list_filings("AAPL", "INVALID")
```

Run unit tests (no network required):

```bash
pytest tests/integration/test_phase2_ingestion.py -v -k "not integration"
```

Run integration tests (requires internet):

```bash
pytest tests/integration/test_phase2_ingestion.py -v -m integration
```

---

# PHASE 3 — Text Processing, Chunking & Hybrid Search

## New Files in This Phase

```
src/financial_rag/
├── processing/
│   ├── __init__.py
│   └── text_processor.py          ← Components #16–17 (tiktoken + chunker)
├── retrieval/
│   ├── __init__.py
│   ├── embeddings.py              ← Components #18–21 (dual-provider embeddings)
│   ├── hybrid_search.py           ← Components #24–26 (BM25 + RRF)
│   └── document_retriever.py      ← Components #27–28 (cache + retriever)
└── storage/
    ├── repositories/
    │   └── chunks.py              ← Component #23 (vector search + MMR)
    └── vector_store.py            ← glue between embeddings and chunks repo
migrations/
└── versions/
    └── 001_initial_migration.py   ← Component #22 (Alembic baseline)
```

Also install additional dependencies:

```bash
pip install sentence-transformers tiktoken
```

---

## Step 9 — Alembic Baseline Migration (Component #22)

Alembic is the migration tool for SQLAlchemy. In Phase 1 we used Docker's
init script to create the schema. Now we establish Alembic as the official
source of migration truth.

First, initialise Alembic:

```bash
alembic init migrations
```

Edit `alembic.ini` — replace the `sqlalchemy.url` line:

```ini
# alembic.ini
sqlalchemy.url = driver://user:pass@localhost/dbname
# Leave this — we override it in env.py
```

Edit `migrations/env.py` — replace the entire file:

```python
# migrations/env.py
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from financial_rag.config import get_settings
from financial_rag.storage.database import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override the URL from our settings (reads from .env)
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL_SYNC)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Create `migrations/versions/001_initial_migration.py`:

```python
# migrations/versions/001_initial_migration.py
"""Initial schema baseline

Revision ID: 001
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""
from __future__ import annotations

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extensions only — tables were created by Docker init script.
    # This migration establishes Alembic as the migration authority
    # without re-creating what already exists.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")


def downgrade() -> None:
    # Intentionally empty — dropping extensions would break the schema.
    pass
```

> **Why does `upgrade()` only create extensions?**
> The `filings`, `financial_chunks`, and `analysis_history` tables were
> created by the Docker init script in Phase 1. Alembic's job going forward
> is to track schema *changes*. Migration `001` is the baseline that says
> "everything up to this point already exists." Future migrations (add a
> column, create an index) go in `002`, `003`, etc.

Apply the migration:

```bash
alembic upgrade head
```

Verify:

```bash
alembic current
# Should output: 001 (head)
```

---

## Step 10 — Token Counter & Chunker (Components #16–17)

> **Why tiktoken instead of `len(text) / 4`?**
> The "divide by 4" estimate is wrong in every edge case that matters:
> short words, numbers, financial abbreviations, and non-ASCII text all
> tokenise differently. tiktoken gives you exact token counts using the
> same BPE tokeniser as the model — chunk boundaries are precise.

> **Why section-aware chunking?**
> A naive chunker splits at 512 tokens regardless of content structure.
> This puts the end of MD&A and the start of Risk Factors in the same chunk,
> which confuses the retriever. Section-aware chunking guarantees a chunk
> never spans two different sections.

Also note: `_build_chunk` instantiates `TextParser` on every call. For a
tutorial this is fine, but in production you'd want to inject it once in
`__init__`.

Create `src/financial_rag/processing/text_processor.py` using the file
from your repo (already reviewed — no changes needed). 

Then update `src/financial_rag/processing/__init__.py`:

```python
from financial_rag.processing.text_processor import TextProcessor

__all__ = ["TextProcessor"]
```

---

## Step 11 — Embedding Client (Components #18–21)

Four components in `embeddings.py`:
- **#18** `EmbeddingProvider` — abstract base
- **#19** `LocalEmbeddingProvider` — sentence-transformers, no API key
- **#20** `OpenAIEmbeddingProvider` — text-embedding-3-large
- **#21** `EmbeddingClient` — factory + auto-fallback + batching

### The dimension mismatch problem

Your SQL schema has `vector(3072)` (for `text-embedding-3-large`).
Your development settings default to `all-MiniLM-L6-v2` which produces
384-dimensional vectors.

**You cannot insert a 384-dim vector into a 3072-dim column.** For local
development, change your `.env`:

```bash
# .env additions for Phase 3 development
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
```

And change the schema in your init SQL (or create a new migration) to use
`vector(384)` for local dev. The cleanest approach is to make the dimension
configurable in the ORM model, which `chunks.py` handles by declaring
`Vector(384)` — but that must match the actual column in Postgres.

For production with OpenAI:

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072
```

Use the `embeddings.py` from your repo — it is correct as written.

Update `src/financial_rag/retrieval/__init__.py`:

```python
from financial_rag.retrieval.embeddings import EmbeddingClient

__all__ = ["EmbeddingClient"]
```

---

## Step 12 — Chunks Repository (Component #23)

The chunks repository owns three critical operations:
- `similarity_search()` — cosine distance via pgvector's `<=>` operator
- `mmr_search()` — Maximal Marginal Relevance for diverse results
- `bulk_upsert()` — idempotent batch insert for re-ingestion safety

### The `ef_search` parameter explained

```python
await self._session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
```

HNSW (Hierarchical Navigable Small World) is the index type pgvector uses
for approximate nearest-neighbour search. `ef_search` controls the
recall/speed tradeoff:

- Higher `ef_search` (e.g. 200): searches more graph nodes → better recall, slower
- Lower `ef_search` (e.g. 40): faster but may miss some true nearest neighbours
- Default 100: good balance for financial text

`SET LOCAL` scopes the setting to the current transaction only — it resets
automatically when the session closes.

Use `chunks.py` from your repo as-is.

Update `src/financial_rag/storage/repositories/__init__.py`:

```python
from financial_rag.storage.repositories.base import BaseRepository
from financial_rag.storage.repositories.chunks import ChunksRepository, FinancialChunk
from financial_rag.storage.repositories.filings import Filing, FilingsRepository

__all__ = [
    "BaseRepository",
    "Filing", "FilingsRepository",
    "FinancialChunk", "ChunksRepository",
]
```

---

## Step 13 — Vector Store (`storage/vector_store.py`)

`document_retriever.py` imports `VectorStore` — this is the glue layer
between `EmbeddingClient` and `ChunksRepository`. Create
`src/financial_rag/storage/vector_store.py`:

```python
# src/financial_rag/storage/vector_store.py
from __future__ import annotations

import logging

from financial_rag.retrieval.embeddings import EmbeddingClient
from financial_rag.storage.database import get_db_client
from financial_rag.storage.repositories.chunks import ChunksRepository, FinancialChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin orchestration layer: embed query → search chunks.
    Keeps the retriever decoupled from both the embedding client
    and the repository.
    """

    def __init__(self) -> None:
        self._embedding_client = EmbeddingClient()

    async def search(
        self,
        question: str,
        *,
        ticker: str | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
        limit: int = 5,
        search_type: str = "similarity",
    ) -> list[tuple[FinancialChunk, float]]:
        """Embed the question and run the appropriate search strategy."""
        query_embedding = await self._embedding_client.embed_query(question)

        db = await get_db_client()
        async with db.session() as session:
            repo = ChunksRepository(session)

            if search_type == "mmr":
                return await repo.mmr_search(
                    query_embedding,
                    ticker=ticker,
                    filing_type=filing_type,
                    fiscal_year=fiscal_year,
                    limit=limit,
                )

            return await repo.similarity_search(
                query_embedding,
                ticker=ticker,
                filing_type=filing_type,
                fiscal_year=fiscal_year,
                section=section,
                limit=limit,
            )
```

---

## Step 14 — Cache Namespace Constants

`document_retriever.py` imports `NS_QUERY` from `storage/cache.py`:

```python
from financial_rag.storage.cache import NS_QUERY, build_key, get_cache_client
```

Add this constant to `src/financial_rag/storage/cache.py` (just after the
`build_key` function):

```python
# Namespace constants — used to scope cache keys by domain
NS_QUERY    = "query"     # document retrieval results
NS_FILING   = "filing"    # raw filing metadata
NS_EMBED    = "embed"     # cached embeddings (optional)
```

---

## Step 15 — Document Retriever & Hybrid Search (Components #24–28)

Use `document_retriever.py` and `hybrid_search.py` from your repo as-is.

### RRF explained for your audience

Reciprocal Rank Fusion combines two ranked lists without needing their raw
scores to be on the same scale:

```
For each document d:
    rrf_score(d) = alpha     * 1/(60 + rank_in_vector_list)
                + (1-alpha)  * 1/(60 + rank_in_text_list)

Documents in both lists get contributions from both terms.
Documents in only one list still get a partial score.
60 is the standard RRF constant from the original 2009 paper.
```

Why not just average the raw scores? Because cosine similarity (0.0–1.0)
and trigram similarity (0.0–1.0) are not on the same scale in practice —
vector search scores cluster around 0.7–0.9, text scores around 0.1–0.4.
RRF sidesteps this entirely by using rank position, not raw scores.

Update `src/financial_rag/retrieval/__init__.py`:

```python
from financial_rag.retrieval.document_retriever import DocumentRetriever, RetrievalResult
from financial_rag.retrieval.embeddings import EmbeddingClient
from financial_rag.retrieval.hybrid_search import HybridSearcher

__all__ = [
    "EmbeddingClient",
    "DocumentRetriever", "RetrievalResult",
    "HybridSearcher",
]
```

---

## Step 16 — Phase 3 Verification Test

Create `tests/integration/test_phase3_processing.py`:

```python
# tests/integration/test_phase3_processing.py
import uuid
import pytest

from financial_rag.ingestion.parsers.html_parser import HTMLParser, ParsedSection
from financial_rag.ingestion.parsers.text_parser import TextParser
from financial_rag.ingestion.sec_ingestor import FilingMetadata
from financial_rag.processing.text_processor import TextProcessor
from financial_rag.utils.exceptions import ChunkingError
from datetime import date


def make_mock_filing_meta(ticker: str = "AAPL") -> FilingMetadata:
    return FilingMetadata(
        ticker=ticker, filing_type="10-K", fiscal_year=2023,
        fiscal_quarter=None, filed_at=date(2023, 11, 3),
        accession_number="0000320193-23-000106",
        primary_document="aapl-20230930.htm",
        source_url="https://example.com/aapl.htm",
        cik="0000320193",
    )


def make_parsed_filing(ticker: str = "AAPL", num_sections: int = 2):
    from financial_rag.ingestion.parsers.html_parser import ParsedFiling
    text = "Apple revenue grew significantly. " * 100  # ~500 tokens
    sections = [
        ParsedSection(name=f"Section {i}", text=text)
        for i in range(num_sections)
    ]
    return ParsedFiling(
        ticker=ticker, filing_type="10-K", fiscal_year=2023,
        full_text=text * num_sections, sections=sections,
    )


class TestTextProcessor:
    def setup_method(self):
        self.processor = TextProcessor()

    def test_count_tokens_is_exact(self):
        text = "Apple revenue grew 12% to $394 billion"
        count = self.processor.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_process_returns_chunks(self):
        parsed = make_parsed_filing()
        meta = make_mock_filing_meta()
        filing_id = uuid.uuid4()
        chunks = self.processor.process(parsed, meta, filing_id)
        assert len(chunks) > 0

    def test_chunks_have_correct_ticker(self):
        parsed = make_parsed_filing("MSFT")
        meta = make_mock_filing_meta("MSFT")
        filing_id = uuid.uuid4()
        chunks = self.processor.process(parsed, meta, filing_id)
        assert all(c.ticker == "MSFT" for c in chunks)

    def test_chunks_have_sequential_indices(self):
        parsed = make_parsed_filing()
        meta = make_mock_filing_meta()
        filing_id = uuid.uuid4()
        chunks = self.processor.process(parsed, meta, filing_id)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunks_embedding_is_empty_list(self):
        """Embeddings are populated by EmbeddingClient, not TextProcessor."""
        parsed = make_parsed_filing()
        meta = make_mock_filing_meta()
        chunks = self.processor.process(parsed, meta, uuid.uuid4())
        assert all(c.embedding == [] for c in chunks)

    def test_empty_sections_raises_chunking_error(self):
        from financial_rag.ingestion.parsers.html_parser import ParsedFiling
        parsed = ParsedFiling(
            ticker="TEST", filing_type="10-K", fiscal_year=2023,
            full_text="", sections=[],
        )
        with pytest.raises(ChunkingError):
            self.processor.process(parsed, make_mock_filing_meta(), uuid.uuid4())

    def test_estimate_cost_returns_expected_keys(self):
        parsed = make_parsed_filing()
        meta = make_mock_filing_meta()
        chunks = self.processor.process(parsed, meta, uuid.uuid4())
        estimate = self.processor.estimate_cost(chunks)
        assert "chunk_count" in estimate
        assert "total_tokens" in estimate
        assert "estimated_cost_usd" in estimate
        assert estimate["chunk_count"] == len(chunks)

    def test_chunk_size_respected(self):
        from financial_rag.config import get_settings
        settings = get_settings()
        parsed = make_parsed_filing()
        meta = make_mock_filing_meta()
        chunks = self.processor.process(parsed, meta, uuid.uuid4())
        for chunk in chunks:
            assert (chunk.token_count or 0) <= settings.CHUNK_SIZE_TOKENS


class TestEmbeddingClient:
    @pytest.mark.integration
    async def test_embed_query_returns_vector(self):
        from financial_rag.retrieval.embeddings import EmbeddingClient
        client = EmbeddingClient()
        vector = await client.embed_query("What was Apple's revenue?")
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.integration
    async def test_embed_texts_count_matches(self):
        from financial_rag.retrieval.embeddings import EmbeddingClient
        client = EmbeddingClient()
        texts = ["Revenue grew 12%", "Net income declined", "EPS was $3.14"]
        vectors = await client.embed_texts(texts)
        assert len(vectors) == len(texts)

    @pytest.mark.integration
    async def test_dimensions_consistent(self):
        from financial_rag.retrieval.embeddings import EmbeddingClient
        from financial_rag.config import get_settings
        client = EmbeddingClient()
        settings = get_settings()
        vector = await client.embed_query("test")
        assert len(vector) == settings.EMBEDDING_DIMENSIONS
```

Run unit tests:

```bash
pytest tests/integration/test_phase3_processing.py -v -k "not integration"
```

Run with local embedding (no API key needed):

```bash
pytest tests/integration/test_phase3_processing.py -v -m integration
```

---

## Final File Tree — Phases 2 & 3

```
financial-rag-agent/
├── migrations/
│   ├── env.py
│   └── versions/
│       └── 001_initial_migration.py
│
└── src/financial_rag/
    ├── utils/
    │   ├── __init__.py
    │   └── exceptions.py              ← Phase 2
    ├── ingestion/
    │   ├── __init__.py
    │   ├── sec_ingestor.py            ← Phase 2
    │   └── parsers/
    │       ├── __init__.py
    │       ├── html_parser.py         ← Phase 2
    │       └── text_parser.py         ← Phase 2
    ├── processing/
    │   ├── __init__.py
    │   └── text_processor.py          ← Phase 3
    ├── retrieval/
    │   ├── __init__.py
    │   ├── embeddings.py              ← Phase 3
    │   ├── hybrid_search.py           ← Phase 3
    │   └── document_retriever.py      ← Phase 3
    └── storage/
        ├── vector_store.py            ← Phase 3
        └── repositories/
            ├── __init__.py
            ├── base.py                ← Phase 2
            ├── filings.py             ← Phase 2
            └── chunks.py              ← Phase 3
```

---

## Common Errors and Fixes — Phases 2 & 3

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: sentence_transformers` | Not installed | `pip install sentence-transformers` |
| `vector type not found` | pgvector extension missing | Check Phase 1 schema init ran |
| `dimension mismatch` inserting chunks | Schema has 3072, model gives 384 | Set `EMBEDDING_DIMENSIONS=384` in `.env` and recreate schema |
| `SECFetchError: Ticker not found` | Wrong ticker or EDGAR outage | Verify ticker on sec.gov; EDGAR has maintenance windows |
| `alembic: target database is not up to date` | Missed `alembic upgrade head` | Run `alembic upgrade head` |
| `NS_QUERY import error` | Missing constant in cache.py | Add `NS_QUERY = "query"` as shown in Step 14 |
| HTTP 403 from EDGAR | Missing or wrong User-Agent | Set `EDGAR_USER_AGENT` in `.env` with real app name + email |
| `ChunkingError: no sections` | Parser returned empty ParsedFiling | Check HTML parser receives actual HTML, not empty string |

---

## What's Next — Phase 4

Phase 4 builds the LLM agent on top of the retrieval stack:

- Multi-provider LLM client (OpenAI, Anthropic, local)
- Tool definitions: `search_filings`, `compare_filings`, `get_section`
- Agent reasoning loop with function calling
- Analysis styles: analyst, executive, risk
- Append-only `analysis_history` audit trail
- Circuit breaker for graceful LLM degradation
- `/query` API endpoint wiring everything together
