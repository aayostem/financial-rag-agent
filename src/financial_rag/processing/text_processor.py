# =============================================================================
# Financial RAG Agent — Text Processor
# src/financial_rag/processing/text_processor.py
#
# Converts ParsedFiling objects into FinancialChunk ORM instances
# ready for vector embedding and database storage.
#
# Design decisions:
#   - tiktoken for exact token counting (never estimate with len/4)
#   - Section-aware chunking: never split across section boundaries
#   - Overlap at token level, not character level
#   - Each chunk carries full provenance metadata
#   - Returns FinancialChunk ORM objects — not dicts, not LangChain Documents
# =============================================================================

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import tiktoken

from financial_rag.config import get_settings
from financial_rag.storage.repositories.chunks import FinancialChunk
from financial_rag.utils.exceptions import ChunkingError

if TYPE_CHECKING:
    from financial_rag.ingestion.parsers.html_parser import ParsedFiling
    from financial_rag.ingestion.sec_ingestor import FilingMetadata

logger = logging.getLogger(__name__)

# tiktoken encoding — cl100k_base covers GPT-4, text-embedding-3-*
# Use this encoding regardless of which model is active
_ENCODING_NAME = "cl100k_base"


# =============================================================================
# Chunk specification
# =============================================================================


@dataclass
class ChunkSpec:
    """
    Specifies how a single chunk should be created.
    Returned by _plan_chunks(), consumed by _build_chunk().
    """

    text: str
    section: str
    chunk_index: int
    token_count: int
    filing_id: uuid.UUID
    ticker: str
    filing_type: str
    fiscal_year: int | None
    metrics: dict[str, object] = field(default_factory=dict)
    entities: dict[str, object] = field(default_factory=dict)
    sentiment_score: float | None = None


# =============================================================================
# TextProcessor
# =============================================================================


class TextProcessor:
    """
    Converts ParsedFiling objects into FinancialChunk ORM instances.

    Usage:
        processor = TextProcessor()
        chunks = processor.process(parsed_filing, filing_meta, filing_id)

    The returned chunks have no embeddings yet — those are added by the
    embeddings module (Commit 5).
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        try:
            self._encoding = tiktoken.get_encoding(_ENCODING_NAME)
        except Exception as exc:
            raise ChunkingError(
                f"Failed to load tiktoken encoding '{_ENCODING_NAME}': {exc}"
            ) from exc

        self._chunk_size = self._settings.CHUNK_SIZE_TOKENS
        self._chunk_overlap = self._settings.CHUNK_OVERLAP_TOKENS

        logger.info(
            "TextProcessor initialised — chunk_size=%d overlap=%d encoding=%s",
            self._chunk_size,
            self._chunk_overlap,
            _ENCODING_NAME,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        parsed: ParsedFiling,
        meta: FilingMetadata,
        filing_id: uuid.UUID,
    ) -> list[FinancialChunk]:
        """
        Convert a ParsedFiling into FinancialChunk ORM instances.

        Args:
            parsed:     ParsedFiling from HTMLParser
            meta:       FilingMetadata from SECIngestor
            filing_id:  UUID of the filing row in the filings table

        Returns:
            List of FinancialChunk instances with all fields populated
            except `embedding` — that is set by the embeddings module.

        Raises:
            ChunkingError if any section cannot be chunked.
        """
        if not parsed.sections:
            raise ChunkingError(
                f"ParsedFiling for {meta.ticker} {meta.filing_type} "
                f"FY{meta.fiscal_year} has no sections to chunk."
            )

        all_chunks: list[FinancialChunk] = []
        global_index = 0

        for section in parsed.sections:
            if not section.text or not section.text.strip():
                continue

            try:
                section_chunks = self._chunk_section(
                    text=section.text,
                    section_name=section.name,
                    filing_id=filing_id,
                    meta=meta,
                    start_index=global_index,
                )
            except Exception as exc:
                raise ChunkingError(
                    f"Failed to chunk section '{section.name}' for "
                    f"{meta.ticker} {meta.filing_type}: {exc}"
                ) from exc

            all_chunks.extend(section_chunks)
            global_index += len(section_chunks)

        logger.info(
            "Processed %s %s FY%s — %d sections → %d chunks",
            meta.ticker,
            meta.filing_type,
            meta.fiscal_year,
            len(parsed.sections),
            len(all_chunks),
        )
        return all_chunks

    def count_tokens(self, text: str) -> int:
        """Return exact token count for a string using tiktoken."""
        return len(self._encoding.encode(text))

    def estimate_cost(
        self,
        chunks: list[FinancialChunk],
        *,
        cost_per_million_tokens: float = 0.13,  # text-embedding-3-large as of 2024
    ) -> dict[str, float]:
        """
        Estimate the OpenAI embedding cost for a list of chunks.

        Args:
            chunks:                   Chunks to embed
            cost_per_million_tokens:  OpenAI pricing per 1M tokens

        Returns:
            dict with total_tokens, estimated_cost_usd
        """
        total_tokens = sum(c.token_count or 0 for c in chunks)
        cost = (total_tokens / 1_000_000) * cost_per_million_tokens
        return {
            "chunk_count": len(chunks),
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost, 4),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _chunk_section(
        self,
        text: str,
        section_name: str,
        filing_id: uuid.UUID,
        meta: FilingMetadata,
        start_index: int,
    ) -> list[FinancialChunk]:
        """
        Split a single section's text into overlapping token-bounded chunks.

        Algorithm:
          1. Tokenise the full section text
          2. Slide a window of chunk_size tokens with overlap
          3. Decode each window back to text
          4. Build a FinancialChunk ORM instance per window
        """
        tokens = self._encoding.encode(text)

        if not tokens:
            return []

        # If the entire section fits in one chunk — no splitting needed
        if len(tokens) <= self._chunk_size:
            return [
                self._build_chunk(
                    text=text,
                    token_count=len(tokens),
                    section=section_name,
                    chunk_index=start_index,
                    filing_id=filing_id,
                    meta=meta,
                )
            ]

        # Sliding window chunking
        chunks: list[FinancialChunk] = []
        start = 0
        chunk_index = start_index

        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text — tiktoken handles this exactly
            chunk_text = self._encoding.decode(chunk_tokens)

            # Skip near-empty chunks (can happen at end of section)
            if len(chunk_text.strip()) < 50:
                break

            chunks.append(
                self._build_chunk(
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    section=section_name,
                    chunk_index=chunk_index,
                    filing_id=filing_id,
                    meta=meta,
                )
            )

            chunk_index += 1

            # Advance by (chunk_size - overlap) tokens
            step = self._chunk_size - self._chunk_overlap
            start += step

            # Safety: avoid infinite loop if step is 0
            if step <= 0:
                logger.warning("chunk_overlap >= chunk_size — processing single chunk only")
                break

        return chunks

    def _build_chunk(
        self,
        *,
        text: str,
        token_count: int,
        section: str,
        chunk_index: int,
        filing_id: uuid.UUID,
        meta: FilingMetadata,
    ) -> FinancialChunk:
        """
        Construct a FinancialChunk ORM instance from chunk parameters.

        The `embedding` field is intentionally left as an empty list —
        it will be populated by the embeddings module before DB insertion.
        """
        from financial_rag.ingestion.parsers.text_parser import TextParser

        text_parser = TextParser()

        # Extract metrics from this chunk's text
        metrics = text_parser.extract_metrics(text)

        return FinancialChunk(
            id=uuid.uuid4(),
            filing_id=filing_id,
            ticker=meta.ticker,
            filing_type=meta.filing_type,
            fiscal_year=meta.fiscal_year,
            section=section,
            chunk_index=chunk_index,
            chunk_text=text,
            token_count=token_count,
            embedding=[],  # populated by embeddings module
            metrics=metrics,
            entities={},  # populated by NER pipeline (future phase)
            sentiment_score=None,
            model_version=self._settings.EMBEDDING_MODEL,
        )
