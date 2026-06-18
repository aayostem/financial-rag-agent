# Financial RAG Agent — Phase 4 & 5: LLM Agent + Full API Layer

> **Series:** Financial RAG Agent (freeCodeCamp)  
> **Phase 4:** LLM Agent, Function Calling & Audit Trail — Components #29–39  
> **Phase 5:** FastAPI Layer, Auth, Rate Limiting & Prometheus — Components #40–51  
> **Time:** Phase 4 ~2 hours | Phase 5 ~2 hours  
> **Prerequisite:** Phases 1–3 complete — database, embeddings, hybrid search working

---

## What You Will Build

**Phase 4** — The intelligent reasoning layer:
- Multi-provider LLM client (OpenAI, Groq, Anthropic, local fallback)
- Tool definitions for OpenAI function calling: `search_filings`, `compare_filings`
- Agent reasoning loop — the model decides when to search and what to search for
- Three analysis styles: analyst, executive, risk
- Append-only `analysis_history` audit trail in PostgreSQL
- Circuit breaker — graceful fallback to `QueryEngine` when LLM is down
- `/query` endpoint wiring everything together

**Phase 5** — The production API layer:
- FastAPI application with lifespan startup/shutdown pattern
- Dependency injection for `QueryEngine`, `VectorStore`, `DatabaseClient`
- API key authentication middleware (X-API-Key header)
- SlowAPI rate limiting with Redis backend in production
- Request logging middleware with request ID tracing
- RFC 7807 error responses (machine-readable error format)
- Prometheus metrics: query counts, latency histograms, error rates
- Health check endpoint with per-service status

---

# PHASE 4 — LLM Agent, Function Calling & Audit Trail

## New Files in This Phase

```
src/financial_rag/
├── retrieval/
│   └── query_engine.py            ← Components #29, #38 (LLM client + circuit breaker)
├── agents/
│   ├── __init__.py
│   └── financial_agent.py         ← Components #30–35 (tools + agent loop + styles)
└── storage/
    └── repositories/
        └── analysis.py            ← Components #36–37 (audit trail)
```

Also: extend `storage/vector_store.py` with `ingest()` and `stats()` methods.

---

## Step 1 — Extend `VectorStore` with `ingest()` and `stats()`

`routes.py` calls `store.ingest()` and `store.stats()` — these don't exist yet.
Add them to `src/financial_rag/storage/vector_store.py`:

```python
# src/financial_rag/storage/vector_store.py
# Add these methods to the existing VectorStore class

    async def ingest(
        self,
        chunks: list,                  # list[FinancialChunk]
        meta: FilingMetadata,
        file_hash: str,
    ) -> tuple[str, int]:
        """
        Persist chunks to the database.

        Steps:
          1. Check for duplicate via file_hash — raise DuplicateFilingError if found
          2. Create a Filing record
          3. Embed all chunks
          4. Bulk upsert chunks

        Returns (filing_id_str, chunks_stored).
        """
        import uuid

        from financial_rag.storage.database import get_db_client
        from financial_rag.storage.repositories.chunks import ChunksRepository
        from financial_rag.storage.repositories.filings import Filing, FilingsRepository
        from financial_rag.utils.exceptions import DuplicateFilingError

        db = await get_db_client()

        # Deduplication check
        async with db.session() as session:
            filings_repo = FilingsRepository(session)
            if await filings_repo.exists_by_hash(file_hash):
                raise DuplicateFilingError(meta.ticker, file_hash)

            # Create filing record
            filing = Filing(
                id=uuid.uuid4(),
                ticker=meta.ticker,
                filing_type=meta.filing_type,
                fiscal_year=meta.fiscal_year,
                fiscal_quarter=meta.fiscal_quarter,
                filed_at=meta.filed_at,
                source_url=meta.source_url,
                file_hash=file_hash,
            )
            await filings_repo.add(filing)
            filing_id = filing.id

        # Set correct filing_id on all chunks
        for chunk in chunks:
            chunk.filing_id = filing_id

        # Embed chunks
        await self._embedding_client.embed_chunks(chunks)

        # Bulk upsert
        async with db.session() as session:
            chunks_repo = ChunksRepository(session)
            stored = await chunks_repo.bulk_upsert(chunks)

        return str(filing_id), stored

    async def stats(self, ticker: str | None = None) -> dict:
        """
        Return aggregate statistics about the vector store.
        Used by /stats and /stats/{ticker} endpoints.
        """
        from sqlalchemy import func, select

        from financial_rag.storage.database import get_db_client
        from financial_rag.storage.repositories.chunks import FinancialChunk
        from financial_rag.storage.repositories.filings import Filing

        db = await get_db_client()
        async with db.session() as session:
            chunk_stmt = select(func.count()).select_from(FinancialChunk)
            filing_stmt = select(func.count()).select_from(Filing)

            if ticker:
                chunk_stmt = chunk_stmt.where(FinancialChunk.ticker == ticker.upper())
                filing_stmt = filing_stmt.where(Filing.ticker == ticker.upper())

            total_chunks = (await session.execute(chunk_stmt)).scalar_one()
            total_filings = (await session.execute(filing_stmt)).scalar_one()

        return {
            "ticker": ticker,
            "total_chunks": total_chunks,
            "total_filings": total_filings,
            "provider": self._embedding_client.provider_name,
            "dimensions": self._embedding_client.dimensions,
        }
```

---

## Step 2 — Extend `DatabaseClient` with `health_check()` and `verify_pgvector()`

`routes.py` calls `db.health_check()` and `dependencies.py` calls
`db.verify_pgvector()`. Add both methods to
`src/financial_rag/storage/database.py`:

```python
# Add to DatabaseClient class in database.py

    async def health_check(self) -> dict:
        """
        Lightweight health probe — returns pool and DB version info.
        Called by /health endpoint on every request.
        """
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar_one()
        pool = self._engine.pool
        return {
            "status": "ok",
            "pg_version": version.split(" ")[1] if version else "unknown",
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
        }

    async def verify_pgvector(self) -> None:
        """
        Confirm the pgvector extension is installed.
        Raises RuntimeError at startup if missing — fail fast, not silently.
        """
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            row = result.fetchone()
        if row is None:
            raise RuntimeError(
                "pgvector extension not found. "
                "Run: CREATE EXTENSION IF NOT EXISTS vector;"
            )
        logger.info("pgvector extension verified")
```

---

## Step 3 — Extend `CacheClient` with `health_check()`

`routes.py` calls `cache.health_check()`. Add to `cache.py`:

```python
# Add to CacheClient class in cache.py

    async def health_check(self) -> dict:
        """
        Lightweight health probe — calls PING and returns memory info.
        Called by /health endpoint.
        """
        self._assert_connected()
        try:
            await self._redis.ping()
            info = await self._redis.info("memory")
            return {
                "status": "ok",
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except RedisError as exc:
            return {"status": "error", "detail": str(exc)}
```

---

## Step 4 — Query Engine (Components #29, #38)

The `QueryEngine` is the public interface for the entire retrieval layer.
Everything above it (API routes, agents) talks only to `QueryEngine` —
never directly to `DocumentRetriever` or `HybridSearcher`.

**Component #29:** Multi-provider LLM client  
**Component #38:** Graceful degradation (circuit breaker pattern)

Key design decisions to explain to your audience:

### The routing table

```
search_type='similarity'  → DocumentRetriever (pgvector cosine distance)
search_type='mmr'         → DocumentRetriever with Maximal Marginal Relevance
search_type='hybrid'      → HybridSearcher (vector + BM25 RRF fusion)

Auto-upgrade: if similarity score < VECTOR_SEARCH_THRESHOLD → hybrid
```

### The circuit breaker

```python
if self._llm is None:
    return f"[LLM unavailable — returning raw context]\n\n{context}"
```

This is the simplest possible circuit breaker: if the LLM client could not
be built (missing API key, or `MOCK_EXTERNAL_APIS=True`), the engine
gracefully returns the raw retrieved context instead of crashing. The API
still responds with a 200 and useful content.

A production circuit breaker would also trip on repeated failures and reset
after a timeout — that's Tenacity's `retry` combined with a state flag.
For this phase, the key teaching point is: **never let the LLM being down
take down the entire service**.

### The `LLM_BASE_URL` / Groq note

Your `.env` defaults set `LLM_BASE_URL=https://api.groq.com/openai/v1` but
`LLM_PROVIDER=openai`. This is intentional — Groq exposes an OpenAI-compatible
API so you can use the OpenAI SDK with Groq's faster inference by just
pointing `base_url` at Groq. In development this saves money significantly.
For production with GPT-4o, remove `LLM_BASE_URL` from `.env`.

```bash
# .env — development with Groq (fast, cheap)
LLM_PROVIDER=openai
LLM_MODEL=llama3-70b-8192
LLM_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...  # your Groq API key

# .env — production with OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_BASE_URL=           # leave empty
OPENAI_API_KEY=sk-...
```

Use `query_engine.py` from your repo as-is. Update
`src/financial_rag/retrieval/__init__.py`:

```python
from financial_rag.retrieval.document_retriever import DocumentRetriever, RetrievalResult
from financial_rag.retrieval.embeddings import EmbeddingClient
from financial_rag.retrieval.hybrid_search import HybridSearcher
from financial_rag.retrieval.query_engine import QueryEngine, QueryResult

__all__ = [
    "EmbeddingClient",
    "DocumentRetriever", "RetrievalResult",
    "HybridSearcher",
    "QueryEngine", "QueryResult",
]
```

---

## Step 5 — Analysis History Repository (Components #36–37)

The audit trail is append-only by design — no updates, no deletes.
Every query that goes through the system gets a permanent record.

> **Why append-only?**  
> Financial analysis systems need tamper-evident audit logs. If a user
> later disputes an answer, you can show exactly what question was asked,
> what documents were retrieved, and what the model said — with a timestamp.
> This is also essential for debugging model drift and retrieval quality over time.

> **`latency_ms` as integer, not float:**  
> Stored as `INTEGER` for efficient range queries and aggregations.
> `AVG(latency_ms)` on an integer column is faster than on a float.
> Convert to seconds only at the presentation layer.

Use `analysis.py` from your repo as-is. Update
`src/financial_rag/storage/repositories/__init__.py`:

```python
from financial_rag.storage.repositories.analysis import AnalysisRecord, AnalysisRepository
from financial_rag.storage.repositories.base import BaseRepository
from financial_rag.storage.repositories.chunks import ChunksRepository, FinancialChunk
from financial_rag.storage.repositories.filings import Filing, FilingsRepository

__all__ = [
    "BaseRepository",
    "Filing", "FilingsRepository",
    "FinancialChunk", "ChunksRepository",
    "AnalysisRecord", "AnalysisRepository",
]
```

---

## Step 6 — Financial Agent (Components #30–35)

The `FinancialAgent` adds a reasoning layer on top of `QueryEngine`.
Instead of a single retrieval call, the agent:

1. Sends the question to the LLM with tool definitions attached
2. The LLM decides what to search for (and how many times)
3. Tool results go back into the conversation
4. The LLM synthesises a final answer grounded in retrieved evidence

This is OpenAI's **function calling** pattern — the model never executes
tools itself, it just emits structured JSON describing what it wants called.
Your code executes the tool and returns the result.

### The agent loop in plain English

```
User question
    ↓
[LLM + tools available]
    ↓ model emits tool_calls
Execute search_filings("AAPL revenue 2023")
    ↓ context chunks returned
[LLM + tool results in conversation]
    ↓ model emits tool_calls again (or final answer)
Execute compare_filings("revenue trend", years=[2021,2022,2023])
    ↓ more context chunks
[LLM + all tool results]
    ↓ no more tool_calls → final text answer
AgentResult
```

### Component #30 — Tool definitions

```python
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": "Search SEC filings for relevant passages...",
            "parameters": { ... }
        }
    },
    ...
]
```

Tool definitions are plain Python dicts passed to `tools=` in the OpenAI
API call. The model sees the `description` and `parameters` schema and
decides when and how to call each tool. Good descriptions are critical —
vague descriptions lead to wrong tool choices.

### Component #34 — The `max_tool_calls` safety cap

```python
while tool_call_count < max_tool_calls:
    ...
# Hit cap — force final answer without tools
final_response = await self._llm.chat.completions.create(
    model=..., messages=messages  # no tools= parameter
)
```

Without this cap, a misbehaving model could loop indefinitely calling tools.
The cap defaults to 5 — enough for complex multi-year comparisons but
prevents runaway API costs. When the cap is hit, the model is called one
final time *without* the `tools=` parameter, forcing it to answer from
whatever context it has accumulated.

### Component #35 — Analysis styles

Three distinct system prompts drive very different response styles:

| Style | Persona | Output |
|---|---|---|
| `analyst` | Senior financial analyst | Detailed, cited, evidence-based |
| `executive` | CFO-level advisor | Concise, quantitative, decision-ready |
| `risk` | Chief Risk Officer | Risk-focused, regulatory, trend-aware |

Use `financial_agent.py` from your repo as-is. Update
`src/financial_rag/agents/__init__.py`:

```python
from financial_rag.agents.financial_agent import AgentResult, FinancialAgent

__all__ = ["FinancialAgent", "AgentResult"]
```

---

## Step 7 — Phase 4 Verification Test

Create `tests/integration/test_phase4_agent.py`:

```python
# tests/integration/test_phase4_agent.py
"""
Phase 4 verification.
Unit tests run without LLM API key (MOCK_EXTERNAL_APIS=True).
Integration tests require a live LLM API key.
"""
import uuid
import pytest

from financial_rag.agents.financial_agent import AgentResult, FinancialAgent
from financial_rag.retrieval.query_engine import QueryEngine, QueryResult
from financial_rag.storage.repositories.analysis import AnalysisRecord, AnalysisRepository


class TestQueryResult:
    def test_latency_seconds_converts_correctly(self):
        result = QueryResult(
            question="test", answer="answer", analysis_style="analyst",
            search_type="similarity", agent_type="query_engine",
            latency_ms=1500,
        )
        assert result.latency_seconds == 1.5

    def test_repr_includes_key_fields(self):
        result = QueryResult(
            question="test", answer="answer", analysis_style="analyst",
            search_type="similarity", agent_type="query_engine",
            latency_ms=250,
        )
        assert "analyst" in repr(result)
        assert "250ms" in repr(result)


class TestAgentResult:
    def test_to_query_result_converts(self):
        agent_result = AgentResult(
            question="test", answer="answer", analysis_style="analyst",
            agent_type="financial_agent", latency_ms=800,
        )
        qr = agent_result.to_query_result()
        assert isinstance(qr, QueryResult)
        assert qr.search_type == "agent"
        assert qr.latency_ms == 800

    def test_latency_seconds_property(self):
        result = AgentResult(
            question="test", answer="answer", analysis_style="analyst",
            agent_type="financial_agent", latency_ms=2000,
        )
        assert result.latency_seconds == 2.0


class TestAnalysisRepository:
    @pytest.mark.integration
    async def test_record_and_retrieve(self, db):
        async with db.session() as session:
            repo = AnalysisRepository(session)
            record = await repo.record(
                question="What was Apple's revenue?",
                answer="Apple's revenue was $394 billion.",
                agent_type="query_engine",
                latency_ms=320,
                ticker="AAPL",
                analysis_style="analyst",
                search_type="similarity",
            )
        assert record.id is not None
        assert record.ticker == "AAPL"
        assert record.latency_ms == 320

    @pytest.mark.integration
    async def test_get_by_ticker(self, db):
        async with db.session() as session:
            repo = AnalysisRepository(session)
            records = await repo.get_by_ticker("AAPL")
        assert isinstance(records, list)

    @pytest.mark.integration
    async def test_average_latency_returns_float(self, db):
        async with db.session() as session:
            repo = AnalysisRepository(session)
            # Insert one record so average is defined
            await repo.record(
                question="test", answer="test", agent_type="test",
                latency_ms=500, ticker="MSFT",
            )
        async with db.session() as session:
            repo = AnalysisRepository(session)
            avg = await repo.average_latency_ms(ticker="MSFT")
        assert avg is None or isinstance(avg, float)

    @pytest.mark.integration
    async def test_error_rate_zero_on_empty(self, db):
        async with db.session() as session:
            repo = AnalysisRepository(session)
            rate = await repo.error_rate(ticker="NONEXISTENT_XYZ")
        assert rate == 0.0


class TestQueryEngineMocked:
    """Tests that run without LLM API key using MOCK_EXTERNAL_APIS."""

    @pytest.mark.integration
    async def test_query_returns_query_result(self, monkeypatch):
        monkeypatch.setenv("MOCK_EXTERNAL_APIS", "true")
        from financial_rag.config import get_settings
        get_settings.cache_clear()

        engine = QueryEngine()
        result = await engine.query("What is Apple's revenue?", ticker="AAPL")
        assert isinstance(result, QueryResult)
        assert result.question == "What is Apple's revenue?"

    @pytest.mark.integration
    async def test_agent_falls_back_to_query_engine_without_key(self, monkeypatch):
        monkeypatch.setenv("MOCK_EXTERNAL_APIS", "true")
        from financial_rag.config import get_settings
        get_settings.cache_clear()

        agent = FinancialAgent()
        assert agent._llm is None  # LLM disabled in mock mode
        result = await agent.analyze("Test question?")
        assert result.agent_type == "query_engine_fallback"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def db():
    from financial_rag.storage.database import DatabaseClient
    client = DatabaseClient()
    await client.connect()
    yield client
    await client.disconnect()
```

Run:

```bash
pytest tests/integration/test_phase4_agent.py -v -k "not integration"
pytest tests/integration/test_phase4_agent.py -v -m integration
```

---

# PHASE 5 — FastAPI Layer, Auth, Rate Limiting & Prometheus

## New Files in This Phase

```
src/financial_rag/
├── api/
│   ├── __init__.py
│   ├── server.py                  ← Component #40 (lifespan + app factory)
│   ├── dependencies.py            ← Component #41 (DI + Annotated types)
│   ├── middleware.py              ← Components #42–45 (auth + rate limit + logging)
│   ├── models.py                  ← request/response Pydantic models
│   └── routes.py                  ← Component #51 (all endpoints)
└── monitoring/
    ├── __init__.py
    └── metrics.py                 ← Components #46–48 (Prometheus)
```

---

## Step 8 — Prometheus Metrics (Components #46–48)

Install the Prometheus client:

```bash
pip install prometheus-client
```

Use `metrics.py` from your repo as-is. A few things worth explaining to
your audience:

### Counter vs Gauge vs Histogram

| Type | Use case | Example |
|---|---|---|
| `Counter` | Monotonically increasing count | Total queries, total errors |
| `Gauge` | Value that goes up and down | Chunks in vector store |
| `Histogram` | Distribution of values | Query latency |

### Label cardinality

Labels on Prometheus metrics multiply the number of time series:

```python
QUERY_TOTAL = Counter(
    "finrag_query_total",
    "...",
    ["analysis_style", "search_type", "agent_type", "status"],
)
```

With 3 styles × 3 search types × 2 agent types × 2 statuses = 36 series.
This is fine. The danger is labels with unbounded cardinality — like
`ticker` on `QUERY_TOTAL`. If you queried 10,000 tickers, you'd have
10,000 × 36 = 360,000 series. That's why `CHUNKS_TOTAL` uses `ticker` as
a label (bounded by your ingested universe) but `QUERY_TOTAL` does not.

Update `src/financial_rag/monitoring/__init__.py`:

```python
from financial_rag.monitoring.metrics import (
    get_metrics_output,
    record_ingestion,
    record_query,
    update_store_stats,
)

__all__ = [
    "record_query", "record_ingestion",
    "update_store_stats", "get_metrics_output",
]
```

---

## Step 9 — API Models (`api/models.py`)

Use `models.py` from your repo as-is.

Two patterns worth highlighting for your audience:

### `StrEnum` for API enumerations

```python
class AnalysisStyle(StrEnum):
    ANALYST = "analyst"
    EXECUTIVE = "executive"
    RISK = "risk"
```

`StrEnum` (Python 3.11+) means the enum value IS the string — no `.value`
needed in comparisons, and it serialises to the plain string in JSON
automatically. Your routes use `request.analysis_style.value` — with
`StrEnum` this is redundant but harmless.

### Required fields with `Field(...)`

```python
question: str = Field(..., description="The financial question to analyse")
```

`...` (Ellipsis) means the field is required with no default. Pydantic
raises a `ValidationError` immediately if the request body omits it —
before your handler function even runs.

---

## Step 10 — Dependency Injection (`api/dependencies.py`)

Use `dependencies.py` from your repo with these fixes:

The `initialise_dependencies()` function calls `db.connect()` but
`DatabaseClient` is already connected inside `get_db_client()` on first
call. Calling `connect()` a second time creates a second engine. Fix:

```python
# dependencies.py — corrected initialise_dependencies()
async def initialise_dependencies() -> None:
    """
    Initialise all application dependencies at startup.
    get_db_client() and get_cache_client() call connect() internally
    on first access — we just call them to trigger that.
    """
    # These calls trigger connect() internally via their singleton logic
    db = await get_db_client()
    logger.info("Database client connected")

    await db.verify_pgvector()

    await get_cache_client()
    logger.info("Cache client connected")

    # Pre-warm singletons so first request isn't slow
    get_vector_store()
    get_query_engine()
    logger.info("QueryEngine and VectorStore initialised")
```

The `shutdown_dependencies()` function calls `db.connect()` and
`cache.connect()` — these should be `disconnect()`. Fix:

```python
async def shutdown_dependencies() -> None:
    db = await get_db_client()
    await db.disconnect()
    logger.info("Database client disconnected")

    cache = await get_cache_client()
    await cache.disconnect()
    logger.info("Cache client disconnected")
```

The full corrected file:

```python
# src/financial_rag/api/dependencies.py
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends

from financial_rag.retrieval.query_engine import QueryEngine
from financial_rag.storage.cache import CacheClient, get_cache_client
from financial_rag.storage.database import DatabaseClient, get_db_client
from financial_rag.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

_query_engine: QueryEngine | None = None
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_query_engine() -> QueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine(vector_store=get_vector_store())
    return _query_engine


async def db_client() -> DatabaseClient:
    return await get_db_client()


async def cache_client() -> CacheClient:
    return await get_cache_client()


async def query_engine() -> QueryEngine:
    return get_query_engine()


async def vector_store() -> VectorStore:
    return get_vector_store()


DBClient = Annotated[DatabaseClient, Depends(db_client)]
CacheClient_ = Annotated[CacheClient, Depends(cache_client)]
Engine = Annotated[QueryEngine, Depends(query_engine)]
Store = Annotated[VectorStore, Depends(vector_store)]


async def initialise_dependencies() -> None:
    """Trigger connect() on all singletons at startup."""
    db = await get_db_client()
    logger.info("Database client connected")
    await db.verify_pgvector()

    await get_cache_client()
    logger.info("Cache client connected")

    get_vector_store()
    get_query_engine()
    logger.info("QueryEngine and VectorStore initialised")


async def shutdown_dependencies() -> None:
    """Gracefully close all connections at shutdown."""
    db = await get_db_client()
    await db.disconnect()
    logger.info("Database client disconnected")

    cache = await get_cache_client()
    await cache.disconnect()
    logger.info("Cache client disconnected")
```

---

## Step 11 — Middleware (Components #42–45)

Use `middleware.py` from your repo as-is.

### Component #42 — API Key Authentication

```python
class APIKeyMiddleware(BaseHTTPMiddleware):
    EXEMPT_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json", "/"})

    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.API_KEY_ENABLED or request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        ...
```

Three things worth teaching here:

1. **Middleware order matters.** FastAPI applies middleware in reverse
   registration order. `RequestLoggingMiddleware` is registered first in
   `server.py` but `APIKeyMiddleware` is added last — so auth runs before
   logging. Check `server.py`'s `add_middleware` calls to verify your
   intended order.

2. **Exempt paths are essential.** Without exempting `/health`, your
   load balancer's health probes would get 401s and mark the instance
   unhealthy.

3. **RFC 7807 error format.** The 401 response uses the machine-readable
   problem detail format — `type`, `title`, `status`, `detail`, `instance`.
   This is the standard that production APIs use so clients can handle
   errors programmatically rather than parsing error strings.

### Component #43 — Rate Limiting

```python
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
```

In development: in-memory (resets when the process restarts).  
In production: Redis backend (shared across multiple API replicas).

To apply a stricter limit to a specific route:

```python
from financial_rag.api.middleware import limiter

@router.post("/query")
@limiter.limit("10/minute")   # 10 queries/min per IP, overrides 100/min default
async def query(request: Request, ...):
    ...
```

`get_remote_address` extracts the client IP. Behind a load balancer,
set `X-Forwarded-For` headers and use `get_ipaddr` instead.

### Component #44 — Structured Logging

Each request gets a `request_id` UUID injected into `request.state`:

```python
request_id = str(uuid.uuid4())
request.state.request_id = request_id
response.headers["X-Request-ID"] = request_id
```

Every log line in the request lifecycle can include `request_id` for
end-to-end tracing — you can grep your logs for a single request ID and
see exactly what happened.

---

## Step 12 — Fix `routes.py` (Duplicate Write Bug)

Before using `routes.py`, fix the duplicate `analysis_history` write.
The file writes to `analysis_history` twice — the second block uses
`repo.create()` which does not exist on `AnalysisRepository`:

```python
# REMOVE this entire second block from the query() handler (lines ~115-130):

# Write to analysis_history (fire and forget — non-fatal)  ← DELETE FROM HERE
try:
    from financial_rag.storage.repositories.analysis import AnalysisRepository
    db = await get_db_client()
    async with db.session() as session:
        repo = AnalysisRepository(session)
        await repo.create(                                  # ← method doesn't exist
            ticker=request.ticker,
            ...
        )
except Exception as exc:
    logger.warning("Failed to write analysis_history (non-fatal): %s", exc)
                                                            # ← DELETE TO HERE
```

The correct write (using `repo.record()`) already exists earlier in the
same handler and should be kept. Here is the cleaned `query` handler:

```python
@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query(request: QueryRequest, engine: Engine) -> QueryResponse:
    """Run a financial question through the full RAG pipeline."""
    import time as _time

    _t0 = _time.monotonic()

    result = await engine.query(
        request.question,
        ticker=request.ticker,
        filing_type=request.filing_type,
        fiscal_year=request.fiscal_year,
        analysis_style=request.analysis_style.value,
        search_type=request.search_type.value,
        limit=request.limit,
    )

    # ── Audit trail (non-fatal) ───────────────────────────────────────────────
    try:
        from uuid import UUID
        from financial_rag.storage.repositories.analysis import AnalysisRepository

        db = await get_db_client()
        async with db.session() as session:
            repo = AnalysisRepository(session)
            await repo.record(
                question=request.question,
                answer=result.answer,
                agent_type=result.agent_type,
                latency_ms=result.latency_ms,
                ticker=request.ticker,
                analysis_style=result.analysis_style,
                search_type=result.search_type,
                source_chunk_ids=[UUID(r.chunk_id) for r in result.source_documents],
                error=result.error,
            )
    except Exception as exc:
        logger.warning("Failed to write analysis_history (non-fatal): %s", exc)

    # ── Prometheus metrics (non-fatal) ────────────────────────────────────────
    try:
        from financial_rag.monitoring.metrics import record_query
        record_query(
            analysis_style=request.analysis_style.value,
            search_type=request.search_type.value,
            agent_type=result.agent_type,
            latency_seconds=_time.monotonic() - _t0,
            success=result.error is None,
            error_type=type(result.error).__name__ if result.error else None,
        )
    except Exception:
        pass  # metrics failure never fails a request

    if result.error and not result.answer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
        )

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        analysis_style=result.analysis_style,
        search_type=result.search_type,
        agent_type=result.agent_type,
        latency_seconds=result.latency_seconds,
        source_documents=[
            DocumentResponse(
                chunk_id=r.chunk_id,
                content=r.chunk_text,
                ticker=r.ticker,
                filing_type=r.filing_type,
                fiscal_year=r.fiscal_year,
                section=r.section,
                score=r.score,
                metrics=r.metrics,
            )
            for r in result.source_documents
        ],
        error=result.error,
    )
```

---

## Step 13 — Application Server (Component #40)

Use `server.py` from your repo as-is.

### The lifespan pattern

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await initialise_dependencies()    # startup
    yield                              # app runs here
    await shutdown_dependencies()      # shutdown
```

This replaces the deprecated `@app.on_event("startup")` decorator.
The `yield` is the key — everything before it runs at startup, everything
after runs at shutdown, and any exception in startup prevents the app from
starting (which is the correct behaviour — fail fast).

### Docs disabled in production

```python
docs_url="/docs" if settings.DEBUG else None,
redoc_url="/redoc" if settings.DEBUG else None,
openapi_url="/openapi.json" if settings.DEBUG else None,
```

In production (`APP_ENV=production`), `DEBUG=False` so Swagger UI,
ReDoc, and the OpenAPI schema endpoint are all disabled. This prevents
attackers from enumerating your API surface. In development they're
available at `localhost:8000/docs`.

---

## Step 14 — Start the API Server

```bash
# Development
uvicorn financial_rag.api.server:app --reload --port 8000
```

Expected startup output:
```
INFO: Starting financial-rag-agent v0.1.0 [development]
INFO: Database client connected
INFO: pgvector extension verified
INFO: Cache client connected
INFO: QueryEngine and VectorStore initialised
INFO: Application ready — listening on 0.0.0.0:8000
INFO: Uvicorn running on http://0.0.0.0:8000
```

Test the health endpoint:

```bash
curl http://localhost:8000/health | python -m json.tool
```

Expected:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "database": { "healthy": true, "details": { "status": "ok", ... } },
    "cache":    { "healthy": true, "details": { "status": "ok", ... } }
  }
}
```

Test a query (requires ingested data):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were Apple'\''s main revenue segments in FY2023?",
    "ticker": "AAPL",
    "analysis_style": "analyst",
    "search_type": "hybrid"
  }' | python -m json.tool
```

Test ingestion (background task — returns 202 immediately):

```bash
curl -X POST http://localhost:8000/ingest/sec \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "filing_type": "10-K", "years": 2}'
```

Check ingestion progress:

```bash
curl http://localhost:8000/stats/AAPL | python -m json.tool
```

View Prometheus metrics:

```bash
curl http://localhost:8000/metrics
```

---

## Step 15 — Phase 5 Verification Test

Create `tests/integration/test_phase5_api.py`:

```python
# tests/integration/test_phase5_api.py
"""
Phase 5 end-to-end API tests.
Uses FastAPI TestClient — no real server needed.
"""
import pytest
from fastapi.testclient import TestClient

from financial_rag.api.server import create_app
from financial_rag.config import get_settings


@pytest.fixture
def client():
    """
    Create a TestClient with the full app.
    Uses development settings (MOCK_EXTERNAL_APIS=False unless overridden).
    """
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_has_services(self, client):
        data = client.get("/health").json()
        assert "services" in data
        assert "database" in data["services"]
        assert "cache" in data["services"]

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        settings = get_settings()
        assert data["version"] == settings.APP_VERSION


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_service_name(self, client):
        data = client.get("/").json()
        assert "service" in data
        assert "financial-rag-agent" in data["service"]


class TestQueryEndpoint:
    def test_query_requires_question(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_rejects_invalid_style(self, client):
        response = client.post("/query", json={
            "question": "test", "analysis_style": "invalid"
        })
        assert response.status_code == 422

    def test_query_accepts_valid_request(self, client):
        response = client.post("/query", json={
            "question": "What is Apple's revenue?",
            "ticker": "AAPL",
            "analysis_style": "analyst",
            "search_type": "similarity",
        })
        # 200 or 500 depending on whether data is ingested
        # But 422 (validation error) must not occur
        assert response.status_code != 422

    def test_query_response_has_required_fields(self, client):
        response = client.post("/query", json={"question": "test"})
        if response.status_code == 200:
            data = response.json()
            assert "question" in data
            assert "answer" in data
            assert "source_documents" in data
            assert "latency_seconds" in data


class TestIngestionEndpoint:
    def test_ingest_returns_202(self, client):
        response = client.post("/ingest/sec", json={
            "ticker": "AAPL", "filing_type": "10-K", "years": 1
        })
        assert response.status_code == 202

    def test_ingest_requires_ticker(self, client):
        response = client.post("/ingest/sec", json={"filing_type": "10-K"})
        assert response.status_code == 422

    def test_ingest_rejects_years_above_5(self, client):
        response = client.post("/ingest/sec", json={
            "ticker": "AAPL", "years": 10
        })
        assert response.status_code == 422


class TestStatsEndpoints:
    def test_global_stats_returns_200(self, client):
        response = client.get("/stats")
        assert response.status_code == 200

    def test_ticker_stats_returns_200(self, client):
        response = client.get("/stats/AAPL")
        assert response.status_code == 200

    def test_stats_has_required_fields(self, client):
        data = client.get("/stats").json()
        assert "total_chunks" in data
        assert "total_filings" in data


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "finrag_query_total" in response.text
        assert "finrag_query_latency_seconds" in response.text


class TestAPIKeyMiddleware:
    def test_health_exempt_without_key(self, client):
        """Health endpoint must not require API key."""
        response = client.get("/health")
        assert response.status_code != 401

    def test_query_allowed_without_key_when_disabled(self, client):
        """When API_KEY_ENABLED=False (default), no key required."""
        response = client.post("/query", json={"question": "test"})
        assert response.status_code != 401


class TestRequestIDs:
    def test_response_has_request_id_header(self, client):
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_response_has_process_time_header(self, client):
        response = client.get("/health")
        assert "X-Process-Time" in response.headers
```

Run:

```bash
pytest tests/integration/test_phase5_api.py -v
```

---

## Final File Tree — Phases 4 & 5

```
src/financial_rag/
├── agents/
│   ├── __init__.py
│   └── financial_agent.py         ← Phase 4
├── retrieval/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── hybrid_search.py
│   ├── document_retriever.py
│   └── query_engine.py            ← Phase 4
├── storage/
│   ├── vector_store.py            ← extended: ingest() + stats()
│   └── repositories/
│       ├── __init__.py
│       ├── base.py
│       ├── filings.py
│       ├── chunks.py
│       └── analysis.py            ← Phase 4
├── api/
│   ├── __init__.py
│   ├── server.py                  ← Phase 5
│   ├── dependencies.py            ← Phase 5 (corrected)
│   ├── middleware.py              ← Phase 5
│   ├── models.py                  ← Phase 5
│   └── routes.py                  ← Phase 5 (duplicate write removed)
└── monitoring/
    ├── __init__.py
    └── metrics.py                 ← Phase 5
```

---

## Common Errors and Fixes — Phases 4 & 5

| Error | Cause | Fix |
|---|---|---|
| `AttributeError: repo.create` | Duplicate write block still in routes.py | Remove second analysis_history write block |
| `RuntimeError: pgvector not found` at startup | Extension missing | Run `CREATE EXTENSION IF NOT EXISTS vector` in psql |
| `connect() already called` | `initialise_dependencies` calling connect twice | Use corrected version in Step 10 |
| `RateLimitExceeded` in tests | SlowAPI counting test requests | Use `TestClient` which bypasses real rate limiting |
| `401 Unauthorized` on health | Health path not in EXEMPT_PATHS | Check `/health` is in `APIKeyMiddleware.EXEMPT_PATHS` |
| `LLM_BASE_URL` mismatch | Groq URL set but OpenAI key used | Match key type to base URL (Groq key starts with `gsk_`) |
| `max_tool_calls` hit immediately | LLM model not supporting function calling | Ensure model supports tools (gpt-3.5-turbo+ or llama3 on Groq) |
| `CORS error` in browser | Origin not in CORS_ORIGINS | Add frontend URL to `.env` CORS_ORIGINS |

---

## End-to-End Smoke Test

With all five phases complete, run this full pipeline test:

```bash
# 1. Start services
docker compose up -d

# 2. Start API
uvicorn financial_rag.api.server:app --reload --port 8000

# 3. Ingest Apple filings
curl -X POST http://localhost:8000/ingest/sec \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "filing_type": "10-K", "years": 2}'

# 4. Wait 60–120 seconds for background ingestion

# 5. Check ingestion completed
curl http://localhost:8000/stats/AAPL

# 6. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were Apple main sources of revenue growth in FY2023?",
    "ticker": "AAPL",
    "analysis_style": "analyst",
    "search_type": "hybrid"
  }' | python -m json.tool

# 7. Check audit trail (via psql)
docker exec finrag-postgres psql -U finrag -d financial_rag \
  -c "SELECT ticker, analysis_style, latency_ms, created_at FROM analysis_history ORDER BY created_at DESC LIMIT 5;"

# 8. Check Prometheus metrics
curl http://localhost:8000/metrics | grep finrag_query
```

## What's Next — Phase 6

Phase 6 moves into Kubernetes security with HashiCorp Vault:
- Dynamic database credentials (rotated automatically)
- PKI secrets engine for mTLS certificates
- Kubernetes auth method — pods authenticate with their service account
- Vault Agent sidecar pattern — secrets injected as files, never in env vars
- Production HA with Raft storage and AWS KMS auto-unseal
