# =============================================================================
# Financial RAG Agent — Hybrid Search
# src/financial_rag/retrieval/hybrid_search.py
#
# Combines pgvector similarity search with PostgreSQL full-text search (trgm)
# using Reciprocal Rank Fusion (RRF) scoring.
#
# Used as a fallback when vector-only search confidence is low, or when
# the query contains specific financial terms (tickers, numbers, acronyms)
# that benefit from exact keyword matching.
#
# RRF score: 1 / (k + rank) where k=60 (standard RRF constant)
# Final score: alpha * vector_rrf + (1-alpha) * text_rrf
# =============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import text

from financial_rag.config import get_settings
from financial_rag.retrieval.document_retriever import RetrievalResult
from financial_rag.retrieval.embeddings import EmbeddingClient
from financial_rag.storage.database import get_db_client
from financial_rag.storage.repositories.chunks import ChunksRepository
from financial_rag.utils.exceptions import RetrievalError

if TYPE_CHECKING:
    from financial_rag.storage.repositories.chunks import FinancialChunk

logger = logging.getLogger(__name__)

# Standard RRF constant — 60 is well-established in the literature
_RRF_K = 60


class HybridSearcher:
    """
    Combines vector similarity search and PostgreSQL trigram full-text search
    using Reciprocal Rank Fusion.

    Use when:
      - Vector search returns low confidence scores (< VECTOR_SEARCH_THRESHOLD)
      - Query contains specific financial identifiers (ticker symbols, dollar
        amounts, specific dates, product names)

    Usage:
        searcher = HybridSearcher()
        results = await searcher.search(
            question="AAPL revenue 2023 iPhone segment",
            ticker="AAPL",
            limit=5,
        )
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedding_client = EmbeddingClient()

    async def search(
        self,
        question: str,
        *,
        ticker: str | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        limit: int | None = None,
        alpha: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid vector + full-text search with RRF score fusion.

        Args:
            question:    Natural language question or keyword query
            ticker:      Optional company filter
            filing_type: Optional form type filter
            fiscal_year: Optional year filter
            limit:       Max results to return
            alpha:       Weight for vector vs text scores.
                         1.0 = pure vector, 0.0 = pure text.
                         Defaults to settings.HYBRID_SEARCH_ALPHA (0.7)

        Returns:
            List of RetrievalResult sorted by fused RRF score, best first.
        """
        effective_limit = limit or self._settings.TOP_K_RESULTS
        effective_alpha = alpha if alpha is not None else self._settings.HYBRID_SEARCH_ALPHA

        # Fetch more candidates than needed — fusion narrows the list
        fetch_limit = effective_limit * 3

        # Run both searches concurrently
        import asyncio

        vector_task = self._vector_search(
            question,
            ticker=ticker,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
            limit=fetch_limit,
        )
        text_task = self._text_search(
            question,
            ticker=ticker,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
            limit=fetch_limit,
        )

        try:
            vector_results, text_results = await asyncio.gather(vector_task, text_task)
        except Exception as exc:
            raise RetrievalError(f"Hybrid search failed for '{question[:50]}': {exc}") from exc

        # Fuse results using RRF
        fused = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            text_results=text_results,
            alpha=effective_alpha,
            limit=effective_limit,
        )

        logger.info(
            "Hybrid search — vector=%d text=%d fused=%d alpha=%.2f",
            len(vector_results),
            len(text_results),
            len(fused),
            effective_alpha,
        )

        return fused

    # ── Vector search ─────────────────────────────────────────────────────────

    async def _vector_search(
        self,
        question: str,
        *,
        ticker: str | None,
        filing_type: str | None,
        fiscal_year: int | None,
        limit: int,
    ) -> list[tuple[FinancialChunk, float]]:
        """Embed the question and run pgvector similarity search."""
        query_embedding = await self._embedding_client.embed_query(question)

        db = await get_db_client()
        async with db.session() as session:
            repo = ChunksRepository(session)
            return await repo.similarity_search(
                query_embedding,
                ticker=ticker,
                filing_type=filing_type,
                fiscal_year=fiscal_year,
                limit=limit,
            )

    # ── Full-text search ──────────────────────────────────────────────────────

    async def _text_search(
        self,
        question: str,
        *,
        ticker: str | None,
        filing_type: str | None,
        fiscal_year: int | None,
        limit: int,
    ) -> list[tuple[FinancialChunk, float]]:
        """
        PostgreSQL trigram similarity search using pg_trgm.
        Requires the idx_chunks_text_trgm GIN index from create_schema.sql.
        """
        db = await get_db_client()
        async with db.session() as session:
            # Build parameterised query
            filters = []
            params: dict[str, object] = {
                "query": question,
                "limit": limit,
            }

            if ticker:
                filters.append("ticker = :ticker")
                params["ticker"] = ticker.upper()
            if filing_type:
                filters.append("filing_type = :filing_type")
                params["filing_type"] = filing_type
            if fiscal_year:
                filters.append("fiscal_year = :fiscal_year")
                params["fiscal_year"] = fiscal_year

            where_clause = "WHERE " + " AND ".join(filters) if filters else ""

            sql = text(f"""
                SELECT
                    id,
                    similarity(chunk_text, :query) AS trgm_score
                FROM financial_chunks
                {where_clause}
                ORDER BY trgm_score DESC
                LIMIT :limit
            """)

            try:
                result = await session.execute(sql, params)
                rows = result.fetchall()
            except Exception as exc:
                logger.warning(
                    "Full-text search failed (non-fatal, falling back to vector-only): %s", exc
                )
                return []

            if not rows:
                return []

            # Fetch the actual chunk objects for matched IDs
            chunk_ids = [row[0] for row in rows]
            score_map = {row[0]: float(row[1]) for row in rows}

            from sqlalchemy import select

            from financial_rag.storage.repositories.chunks import FinancialChunk as ChunkModel

            chunks_result = await session.execute(
                select(ChunkModel).where(ChunkModel.id.in_(chunk_ids))
            )
            chunks = {c.id: c for c in chunks_result.scalars().all()}

            return [(chunks[cid], score_map[cid]) for cid in chunk_ids if cid in chunks]

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        *,
        vector_results: list[tuple[FinancialChunk, float]],
        text_results: list[tuple[FinancialChunk, float]],
        alpha: float,
        limit: int,
    ) -> list[RetrievalResult]:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.

        RRF score for document d:
            rrf(d) = alpha * sum(1/(k + rank_vector(d)))
                   + (1-alpha) * sum(1/(k + rank_text(d)))

        Documents that appear in both lists receive a higher fused score.
        Documents that appear in only one list are still included but ranked
        lower than those appearing in both.
        """
        # Build rank maps (1-indexed)
        vector_ranks: dict[str, int] = {
            str(chunk.id): rank + 1 for rank, (chunk, _) in enumerate(vector_results)
        }
        text_ranks: dict[str, int] = {
            str(chunk.id): rank + 1 for rank, (chunk, _) in enumerate(text_results)
        }

        # All unique chunk IDs across both lists
        all_ids: set[str] = set(vector_ranks) | set(text_ranks)

        # Build a lookup from id → chunk
        chunk_lookup: dict[str, FinancialChunk] = {}
        for chunk, _ in vector_results:
            chunk_lookup[str(chunk.id)] = chunk
        for chunk, _ in text_results:
            chunk_lookup[str(chunk.id)] = chunk

        # Compute fused RRF scores
        fused_scores: list[tuple[str, float]] = []
        for chunk_id in all_ids:
            vector_rrf = (
                1.0 / (_RRF_K + vector_ranks[chunk_id]) if chunk_id in vector_ranks else 0.0
            )
            text_rrf = 1.0 / (_RRF_K + text_ranks[chunk_id]) if chunk_id in text_ranks else 0.0
            fused_score = alpha * vector_rrf + (1 - alpha) * text_rrf
            fused_scores.append((chunk_id, fused_score))

        # Sort descending by fused score and take top-limit
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        top = fused_scores[:limit]

        return [
            RetrievalResult.from_chunk(chunk_lookup[cid], score)
            for cid, score in top
            if cid in chunk_lookup
        ]
