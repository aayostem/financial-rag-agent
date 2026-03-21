# =============================================================================
# Financial RAG Agent — Prometheus Metrics
# src/financial_rag/monitoring/metrics.py
#
# Exposes application metrics in Prometheus format.
# Metrics:
#   - finrag_query_total: total queries by style, search_type, agent_type
#   - finrag_query_latency_seconds: query latency histogram
#   - finrag_query_errors_total: total errors by type
#   - finrag_chunks_total: total chunks in vector store
#   - finrag_ingestion_total: total ingestion jobs by ticker
#   - finrag_ingestion_latency_seconds: ingestion latency histogram
# =============================================================================

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# =============================================================================
# Metric definitions
# =============================================================================

QUERY_TOTAL = Counter(
    "finrag_query_total",
    "Total number of RAG queries",
    ["analysis_style", "search_type", "agent_type", "status"],
)

QUERY_LATENCY = Histogram(
    "finrag_query_latency_seconds",
    "Query latency in seconds",
    ["analysis_style", "search_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

QUERY_ERRORS = Counter(
    "finrag_query_errors_total",
    "Total number of query errors",
    ["error_type"],
)

CHUNKS_TOTAL = Gauge(
    "finrag_chunks_total",
    "Total number of chunks in vector store",
    ["ticker"],
)

FILINGS_TOTAL = Gauge(
    "finrag_filings_total",
    "Total number of filings ingested",
    ["ticker", "filing_type"],
)

INGESTION_TOTAL = Counter(
    "finrag_ingestion_total",
    "Total number of ingestion jobs",
    ["ticker", "filing_type", "status"],
)

INGESTION_LATENCY = Histogram(
    "finrag_ingestion_latency_seconds",
    "Ingestion job latency in seconds",
    ["ticker"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

CACHE_HITS = Counter(
    "finrag_cache_hits_total",
    "Total Redis cache hits",
)

CACHE_MISSES = Counter(
    "finrag_cache_misses_total",
    "Total Redis cache misses",
)

# =============================================================================
# Helper functions
# =============================================================================


def record_query(
    *,
    analysis_style: str,
    search_type: str,
    agent_type: str,
    latency_seconds: float,
    success: bool,
    error_type: str | None = None,
) -> None:
    """Record metrics for a completed query."""
    status = "success" if success else "error"

    QUERY_TOTAL.labels(
        analysis_style=analysis_style,
        search_type=search_type,
        agent_type=agent_type,
        status=status,
    ).inc()

    QUERY_LATENCY.labels(
        analysis_style=analysis_style,
        search_type=search_type,
    ).observe(latency_seconds)

    if not success and error_type:
        QUERY_ERRORS.labels(error_type=error_type).inc()


def record_ingestion(
    *,
    ticker: str,
    filing_type: str,
    latency_seconds: float,
    success: bool,
) -> None:
    """Record metrics for a completed ingestion job."""
    status = "success" if success else "error"
    INGESTION_TOTAL.labels(
        ticker=ticker,
        filing_type=filing_type,
        status=status,
    ).inc()
    INGESTION_LATENCY.labels(ticker=ticker).observe(latency_seconds)


def update_store_stats(*, ticker: str, chunks: int, filings: int, filing_type: str = "all") -> None:
    """Update gauge metrics for vector store size."""
    CHUNKS_TOTAL.labels(ticker=ticker).set(chunks)
    FILINGS_TOTAL.labels(ticker=ticker, filing_type=filing_type).set(filings)


def get_metrics_output() -> tuple[bytes, str]:
    """Return Prometheus metrics in text format."""
    return generate_latest(), CONTENT_TYPE_LATEST
