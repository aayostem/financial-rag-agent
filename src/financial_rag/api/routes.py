# =============================================================================
# Financial RAG Agent — API Routes
# src/financial_rag/api/routes.py
#
# Phase 2 endpoints:
#   GET  /health          — liveness + readiness probe
#   POST /query           — RAG query pipeline
#   POST /ingest/sec      — SEC filing ingestion (background task)
#   GET  /stats           — vector store statistics
#   GET  /stats/{ticker}  — per-ticker statistics
# =============================================================================

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import Response

from financial_rag.api.dependencies import Engine, Store
from financial_rag.api.models import (
    DocumentResponse,
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    ServiceStatus,
    StatsResponse,
)
from financial_rag.ingestion.parsers.html_parser import HTMLParser
from financial_rag.ingestion.parsers.text_parser import TextParser
from financial_rag.ingestion.sec_ingestor import SECIngestor
from financial_rag.processing.text_processor import TextProcessor
from financial_rag.storage.cache import get_cache_client
from financial_rag.storage.database import get_db_client
from financial_rag.storage.vector_store import VectorStore
from financial_rag.utils.exceptions import DuplicateFilingError

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Health
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["ops"],
)
async def health_check() -> HealthResponse:
    """
    Liveness and readiness probe.

    Returns the health status of all backing services.
    Status is 'healthy' only when all services are reachable.
    Status is 'degraded' when cache is down but DB is up.
    Status is 'unhealthy' when the database is unreachable.
    """
    from financial_rag.config import get_settings

    settings = get_settings()

    db_status = ServiceStatus(healthy=False)
    cache_status = ServiceStatus(healthy=False)

    # Database probe
    try:
        db = await get_db_client()
        db_info = await db.health_check()
        db_status = ServiceStatus(healthy=True, details=db_info)
    except Exception as exc:
        db_status = ServiceStatus(healthy=False, details={"error": str(exc)})

    # Cache probe
    try:
        cache = await get_cache_client()
        cache_info = await cache.health_check()
        cache_status = ServiceStatus(healthy=True, details=cache_info)
    except Exception as exc:
        cache_status = ServiceStatus(healthy=False, details={"error": str(exc)})

    # Determine overall status
    if not db_status.healthy:
        overall = "unhealthy"
    elif not cache_status.healthy:
        overall = "degraded"
    else:
        overall = "healthy"

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        services={
            "database": db_status,
            "cache": cache_status,
        },
    )


# =============================================================================
# Query
# =============================================================================


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Financial RAG query",
    tags=["query"],
)
async def query(
    request: QueryRequest,
    engine: Engine,
) -> QueryResponse:
    """
    Run a financial question through the full RAG pipeline.

    Retrieves relevant chunks from ingested SEC filings and generates
    a grounded answer using the configured LLM.

    - **question**: Natural language financial question
    - **ticker**: Optional company filter (e.g. `AAPL`)
    - **analysis_style**: `analyst` | `executive` | `risk`
    - **search_type**: `similarity` | `mmr` | `hybrid`
    """
    result = await engine.query(
        request.question,
        ticker=request.ticker,
        filing_type=request.filing_type,
        fiscal_year=request.fiscal_year,
        analysis_style=request.analysis_style.value,
        search_type=request.search_type.value,
        limit=request.limit,
    )
    # ── Write to analysis_history (non-fatal) ─────────────────────────────────
    try:
        import time as _time
        from uuid import UUID

        _t0 = _time.monotonic()

        from financial_rag.storage.repositories.analysis import AnalysisRepository

        _db = await get_db_client()
        async with _db.session() as _session:
            _repo = AnalysisRepository(_session)
            await _repo.record(
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
    except Exception as _exc:
        logger.warning("Failed to write analysis_history (non-fatal): %s", _exc)

    # Record Prometheus metrics
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
        pass

    if result.error and not result.answer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
        )
    # Write to analysis_history (fire and forget — non-fatal)
    try:
        from financial_rag.storage.repositories.analysis import AnalysisRepository

        db = await get_db_client()
        async with db.session() as session:
            repo = AnalysisRepository(session)
            await repo.create(
                ticker=request.ticker,
                question=request.question,
                answer=result.answer,
                analysis_style=result.analysis_style,
                agent_type=result.agent_type,
                search_type=result.search_type,
                latency_ms=result.latency_ms,
                source_chunk_ids=[r.chunk_id for r in result.source_documents],
                error=result.error,
            )
    except Exception as exc:
        logger.warning("Failed to write analysis_history (non-fatal): %s", exc)

    source_docs = [
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
    ]

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        analysis_style=result.analysis_style,
        search_type=result.search_type,
        agent_type=result.agent_type,
        latency_seconds=result.latency_seconds,
        source_documents=source_docs,
        error=result.error,
    )


# =============================================================================
# Ingestion
# =============================================================================


@router.post(
    "/ingest/sec",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest SEC filings",
    tags=["ingestion"],
)
async def ingest_sec(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    store: Store,
) -> IngestionResponse:
    """
    Trigger ingestion of SEC filings for a ticker.

    Downloads filings from EDGAR, parses, chunks, embeds, and stores them
    in pgvector. Runs as a background task — returns immediately with 202.

    Check `/stats/{ticker}` to monitor ingestion progress.
    """
    background_tasks.add_task(
        _ingest_background,
        ticker=request.ticker,
        filing_type=request.filing_type,
        years=request.years,
        store=store,
    )

    return IngestionResponse(
        ticker=request.ticker.upper(),
        filing_type=request.filing_type,
        filings_found=0,  # updated async
        chunks_stored=0,  # updated async
        success=True,
        error=None,
    )


async def _ingest_background(
    *,
    ticker: str,
    filing_type: str,
    years: int,
    store: VectorStore,  # type: ignore[name-defined]
) -> None:
    """
    Full ingestion pipeline run as a FastAPI background task.

    Steps:
      1. List available filings from EDGAR
      2. Download each filing (with local cache)
      3. Parse HTML → clean text + sections
      4. Chunk with TextProcessor
      5. Embed + upsert via VectorStore (skips duplicates)
    """
    from financial_rag.config import get_settings

    settings = get_settings()

    ticker = ticker.upper()
    html_parser = HTMLParser()
    text_parser = TextParser()
    processor = TextProcessor()

    logger.info(
        "Background ingestion started — ticker=%s type=%s years=%d",
        ticker,
        filing_type,
        years,
    )

    total_chunks = 0
    total_filings = 0
    skipped = 0

    try:
        async with SECIngestor() as ingestor:
            filings = await ingestor.list_filings(ticker, filing_type, years=years)
            total_filings = len(filings)

            raw_dir = settings.RAW_DATA_DIR / ticker / filing_type
            raw_dir.mkdir(parents=True, exist_ok=True)

            for meta in filings:
                try:
                    raw_html, file_hash = await ingestor.download_filing(meta, raw_dir=raw_dir)
                except Exception as exc:
                    logger.error(
                        "Failed to download %s FY%s: %s",
                        meta.filing_type,
                        meta.fiscal_year,
                        exc,
                    )
                    continue

                # Parse
                parsed = html_parser.parse(
                    raw_html,
                    ticker=ticker,
                    filing_type=filing_type,
                    fiscal_year=meta.fiscal_year,
                )

                # Clean text in each section
                for section in parsed.sections:
                    section.text = text_parser.clean(section.text)

                # Chunk
                import uuid

                placeholder_filing_id = uuid.uuid4()
                chunks = processor.process(parsed, meta, placeholder_filing_id)

                # Embed + store (handles dedup internally)
                try:
                    _, stored = await store.ingest(chunks, meta, file_hash)
                    total_chunks += stored
                    logger.info(
                        "Ingested %s FY%s — %d chunks",
                        ticker,
                        meta.fiscal_year,
                        stored,
                    )
                except DuplicateFilingError:
                    skipped += 1
                    logger.info(
                        "Skipped duplicate — %s FY%s hash=%s",
                        ticker,
                        meta.fiscal_year,
                        file_hash[:12],
                    )

        logger.info(
            "Background ingestion complete — ticker=%s filings=%d chunks=%d skipped=%d",
            ticker,
            total_filings,
            total_chunks,
            skipped,
        )

    except Exception as exc:
        logger.error(
            "Background ingestion failed — ticker=%s error=%s",
            ticker,
            exc,
            exc_info=True,
        )


# =============================================================================
# Stats
# =============================================================================


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Global vector store statistics",
    tags=["ops"],
)
async def global_stats(store: Store) -> StatsResponse:
    """Return aggregate statistics across all ingested filings."""
    stats = await store.stats()
    return StatsResponse(**stats)


@router.get(
    "/stats/{ticker}",
    response_model=StatsResponse,
    summary="Per-ticker statistics",
    tags=["ops"],
)
async def ticker_stats(ticker: str, store: Store) -> StatsResponse:
    """Return statistics for a specific ticker."""
    stats = await store.stats(ticker=ticker.upper())
    return StatsResponse(**stats)


# ── Metrics ───────────────────────────────────────────────────────────────────


@router.get(
    "/metrics",
    include_in_schema=False,
    tags=["ops"],
)
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response as FastAPIResponse

    from financial_rag.monitoring.metrics import get_metrics_output

    data, content_type = get_metrics_output()
    return FastAPIResponse(content=data, media_type=content_type)
