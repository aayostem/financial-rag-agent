# =============================================================================
# Financial RAG Agent — Vector Store
# src/financial_rag/storage/vector_store.py
#
# Orchestrates the full chunk → embed → store pipeline.
# Sits between the processing layer (TextProcessor) and the
# retrieval layer (DocumentRetriever).
#
# Responsibilities:
#   - Accept FinancialChunk lists from TextProcessor
#   - Embed via EmbeddingClient
#   - Upsert into PostgreSQL via ChunksRepository
#   - Register filings in FilingsRepository
# =============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from financial_rag.retrieval.embeddings import EmbeddingClient
from financial_rag.storage.database import get_db_client
from financial_rag.storage.repositories.chunks import ChunksRepository
from financial_rag.storage.repositories.filings import Filing, FilingsRepository

if TYPE_CHECKING:
    import uuid

    from financial_rag.ingestion.sec_ingestor import FilingMetadata
    from financial_rag.storage.repositories.chunks import FinancialChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Orchestrates the pipeline from chunks → embeddings → pgvector storage.

    Usage:
        store = VectorStore()
        filing_id, chunk_count = await store.ingest(chunks, meta, file_hash)

        results = await store.search(query_embedding, ticker="AAPL", limit=5)
    """

    def __init__(self) -> None:
        self._embedding_client = EmbeddingClient()
        logger.info(
            "VectorStore initialised — provider=%s dims=%d",
            self._embedding_client.provider_name,
            self._embedding_client.dimensions,
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def ingest(
        self,
        chunks: list[FinancialChunk],
        meta: FilingMetadata,
        file_hash: str,
    ) -> tuple[uuid.UUID, int]:
        """
        Full ingestion pipeline for a single filing.

        Steps:
          1. Check for duplicate via file_hash — skip if already ingested
          2. Embed all chunks (batched)
          3. Register filing in filings table
          4. Bulk upsert chunks into financial_chunks table

        Args:
            chunks:    FinancialChunk instances from TextProcessor
                       (embedding field must be empty list)
            meta:      FilingMetadata from SECIngestor
            file_hash: SHA-256 of the raw filing content

        Returns:
            (filing_id, chunks_stored) tuple

        Raises:
            DuplicateFilingError if this hash was already ingested.
            EmbeddingError on embedding failure.
            DatabaseQueryError on storage failure.
        """
        from financial_rag.utils.exceptions import DuplicateFilingError

        db = await get_db_client()

        # ── Step 1: Deduplication check ───────────────────────────────────────
        async with db.session() as session:
            filings_repo = FilingsRepository(session)
            if await filings_repo.exists_by_hash(file_hash):
                raise DuplicateFilingError(
                    ticker=meta.ticker,
                    file_hash=file_hash,
                )

        logger.info(
            "Ingesting %s %s FY%s — %d chunks",
            meta.ticker,
            meta.filing_type,
            meta.fiscal_year,
            len(chunks),
        )

        # ── Step 2: Embed all chunks ──────────────────────────────────────────
        chunks_with_embeddings = await self._embedding_client.embed_chunks(chunks)

        # ── Step 3: Register filing ───────────────────────────────────────────
        import uuid as uuid_module

        filing_id = uuid_module.uuid4()

        async with db.session() as session:
            filings_repo = FilingsRepository(session)
            filing = Filing(
                id=filing_id,
                ticker=meta.ticker,
                filing_type=meta.filing_type,
                fiscal_year=meta.fiscal_year,
                fiscal_quarter=meta.fiscal_quarter,
                filed_at=meta.filed_at,
                source_url=meta.source_url,
                file_hash=file_hash,
                ingested_by="VectorStore",
                is_active=True,
            )
            await filings_repo.add(filing)

        # ── Step 4: Bulk upsert chunks ────────────────────────────────────────
        # Set filing_id on all chunks (was uuid4() placeholder before)
        for chunk in chunks_with_embeddings:
            chunk.filing_id = filing_id

        async with db.session() as session:
            chunks_repo = ChunksRepository(session)
            stored = await chunks_repo.bulk_upsert(chunks_with_embeddings)

        logger.info(
            "Ingestion complete — %s %s FY%s: filing_id=%s chunks=%d",
            meta.ticker,
            meta.filing_type,
            meta.fiscal_year,
            str(filing_id)[:8],
            stored,
        )
        return filing_id, stored

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
        limit: int = 5,
        search_type: str = "similarity",
        ef_search: int = 100,
    ) -> list[tuple[FinancialChunk, float]]:
        """
        Embed a query and retrieve the most relevant chunks.

        Args:
            query:       Natural language question
            ticker:      Optional company filter
            filing_type: Optional form type filter ('10-K', '10-Q', etc.)
            fiscal_year: Optional year filter
            section:     Optional section filter ('MD&A', 'Risk Factors', etc.)
            limit:       Max chunks to return
            search_type: 'similarity' (default) or 'mmr' (diversity)
            ef_search:   HNSW recall parameter (higher = better recall, slower)

        Returns:
            List of (chunk, similarity_score) tuples, best match first.
        """
        query_embedding = await self._embedding_client.embed_query(query)

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
            else:
                return await repo.similarity_search(
                    query_embedding,
                    ticker=ticker,
                    filing_type=filing_type,
                    fiscal_year=fiscal_year,
                    section=section,
                    limit=limit,
                    ef_search=ef_search,
                )

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def stats(self, ticker: str | None = None) -> dict[str, object]:
        """
        Return storage statistics for monitoring.
        """
        db = await get_db_client()
        async with db.session() as session:
            chunks_repo = ChunksRepository(session)
            filings_repo = FilingsRepository(session)

            if ticker:
                chunk_count = await chunks_repo.count_by_ticker(ticker)
                filing_count = len(await filings_repo.get_by_ticker(ticker))
            else:
                chunk_count = await chunks_repo.count()
                filing_count = await filings_repo.count()

        return {
            "ticker": ticker or "all",
            "total_chunks": chunk_count,
            "total_filings": filing_count,
            "provider": self._embedding_client.provider_name,
            "dimensions": self._embedding_client.dimensions,
        }
