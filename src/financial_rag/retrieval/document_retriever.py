# =============================================================================
# Financial RAG Agent — Document Retriever
# src/financial_rag/retrieval/document_retriever.py
#
# Retrieves relevant chunks for a query using vector similarity search.
# Adds a Redis cache layer on top of VectorStore so repeated queries
# against the same ticker/question don't hit pgvector on every request.
#
# Cache key: finrag:query:{hash(question+filters)}
# Cache TTL:  settings.REDIS_DEFAULT_TTL_SECONDS (default 1 hour)
# =============================================================================

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from financial_rag.config import get_settings
from financial_rag.storage.cache import NS_QUERY, build_key, get_cache_client
from financial_rag.storage.vector_store import VectorStore
from financial_rag.utils.exceptions import RetrievalError

if TYPE_CHECKING:
    from financial_rag.storage.repositories.chunks import FinancialChunk

logger = logging.getLogger(__name__)


# =============================================================================
# Retrieval result
# =============================================================================


@dataclass
class RetrievalResult:
    """
    A single retrieved chunk with its similarity score and provenance.
    Returned by DocumentRetriever.retrieve().
    """

    chunk_id: str
    chunk_text: str
    ticker: str
    filing_type: str
    fiscal_year: int | None
    section: str | None
    score: float
    metrics: dict[str, object] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """
        Format this result as a context block for the LLM prompt.
        Includes source attribution so the model can cite its sources.
        """
        header = (
            f"[Source: {self.ticker} {self.filing_type} "
            f"FY{self.fiscal_year} — {self.section or 'General'} "
            f"(score={self.score:.3f})]"
        )
        return f"{header}\n{self.chunk_text}"

    @classmethod
    def from_chunk(cls, chunk: FinancialChunk, score: float) -> RetrievalResult:
        return cls(
            chunk_id=str(chunk.id),
            chunk_text=chunk.chunk_text,
            ticker=chunk.ticker,
            filing_type=chunk.filing_type,
            fiscal_year=chunk.fiscal_year,
            section=chunk.section,
            score=score,
            metrics=chunk.metrics or {},
        )


# =============================================================================
# DocumentRetriever
# =============================================================================


class DocumentRetriever:
    """
    Retrieves relevant document chunks for a query.

    Adds Redis caching on top of VectorStore — repeated queries with
    the same question and filters return cached results without hitting
    the database or embedding API.

    Usage:
        retriever = DocumentRetriever()
        results = await retriever.retrieve(
            question="What was Apple's revenue in FY2023?",
            ticker="AAPL",
            limit=5,
        )
        context = retriever.build_context(results)
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._settings = get_settings()
        self._vector_store = vector_store or VectorStore()

    async def retrieve(
        self,
        question: str,
        *,
        ticker: str | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
        limit: int | None = None,
        search_type: str = "similarity",
        use_cache: bool = True,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a question.

        Args:
            question:        Natural language question
            ticker:          Optional company filter
            filing_type:     Optional form type filter
            fiscal_year:     Optional year filter
            section:         Optional section filter
            limit:           Max results (defaults to settings.TOP_K_RESULTS)
            search_type:     'similarity' or 'mmr'
            use_cache:       Whether to check/populate Redis cache
            score_threshold: Minimum similarity score (0.0-1.0).
                             Defaults to settings.VECTOR_SEARCH_THRESHOLD.

        Returns:
            List of RetrievalResult, best match first.
            Empty list if no results meet the threshold.
        """
        effective_limit = limit or self._settings.TOP_K_RESULTS
        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else self._settings.VECTOR_SEARCH_THRESHOLD
        )

        # ── Cache check ───────────────────────────────────────────────────────
        cache_key = None
        if use_cache and not self._settings.MOCK_EXTERNAL_APIS:
            cache_key = self._build_cache_key(
                question,
                ticker,
                filing_type,
                fiscal_year,
                section,
                effective_limit,
                search_type,
            )
            cached = await self._get_cached(cache_key)
            if cached is not None:
                logger.debug("Cache hit for query hash=%s", cache_key[-8:])
                return cached

        # ── Vector search ─────────────────────────────────────────────────────
        try:
            raw_results = await self._vector_store.search(
                question,
                ticker=ticker,
                filing_type=filing_type,
                fiscal_year=fiscal_year,
                section=section,
                limit=effective_limit,
                search_type=search_type,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Vector search failed for question='{question[:50]}...': {exc}"
            ) from exc

        # ── Score filtering ───────────────────────────────────────────────────
        results = [
            RetrievalResult.from_chunk(chunk, score)
            for chunk, score in raw_results
            if score >= effective_threshold
        ]

        logger.info(
            "Retrieved %d/%d results above threshold=%.2f for '%s...'",
            len(results),
            len(raw_results),
            effective_threshold,
            question[:50],
        )

        # ── Cache population ──────────────────────────────────────────────────
        if use_cache and cache_key and results:
            await self._set_cached(cache_key, results)

        return results

    def build_context(
        self,
        results: list[RetrievalResult],
        *,
        max_tokens: int | None = None,
    ) -> str:
        """
        Assemble retrieved chunks into a single context string for the LLM.

        Respects max_tokens budget — stops adding chunks when the context
        would exceed the limit. Uses settings.MAX_CONTEXT_TOKENS if not
        specified.

        Args:
            results:    Retrieved chunks from retrieve()
            max_tokens: Token budget for context window

        Returns:
            Formatted context string with source headers.
        """
        import tiktoken

        budget = max_tokens or self._settings.MAX_CONTEXT_TOKENS
        enc = tiktoken.get_encoding("cl100k_base")

        context_parts: list[str] = []
        tokens_used = 0

        for result in results:
            block = result.to_context_string()
            block_tokens = len(enc.encode(block))

            if tokens_used + block_tokens > budget:
                logger.debug(
                    "Context budget reached at %d/%d tokens after %d chunks",
                    tokens_used,
                    budget,
                    len(context_parts),
                )
                break

            context_parts.append(block)
            tokens_used += block_tokens

        return "\n\n---\n\n".join(context_parts)

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _build_cache_key(
        self,
        question: str,
        ticker: str | None,
        filing_type: str | None,
        fiscal_year: int | None,
        section: str | None,
        limit: int,
        search_type: str,
    ) -> str:
        """Build a deterministic cache key from query parameters."""
        payload = json.dumps(
            {
                "q": question,
                "t": ticker,
                "ft": filing_type,
                "fy": fiscal_year,
                "s": section,
                "l": limit,
                "st": search_type,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return build_key(NS_QUERY, digest)

    async def _get_cached(self, cache_key: str) -> list[RetrievalResult] | None:
        try:
            client = await get_cache_client()
            data = await client.get(cache_key)
            if data is None:
                return None
            return [RetrievalResult(**item) for item in data]
        except Exception as exc:
            # Cache failures are non-fatal — fall through to vector search
            logger.warning("Cache GET failed (non-fatal): %s", exc)
            return None

    async def _set_cached(
        self,
        cache_key: str,
        results: list[RetrievalResult],
    ) -> None:
        try:
            client = await get_cache_client()
            serialisable = [
                {
                    "chunk_id": r.chunk_id,
                    "chunk_text": r.chunk_text,
                    "ticker": r.ticker,
                    "filing_type": r.filing_type,
                    "fiscal_year": r.fiscal_year,
                    "section": r.section,
                    "score": r.score,
                    "metrics": r.metrics,
                }
                for r in results
            ]
            await client.set(cache_key, serialisable)
        except Exception as exc:
            logger.warning("Cache SET failed (non-fatal): %s", exc)
