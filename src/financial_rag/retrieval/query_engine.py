# =============================================================================
# Financial RAG Agent — Query Engine
# src/financial_rag/retrieval/query_engine.py
#
# Orchestrates the full RAG pipeline:
#   1. Retrieve relevant chunks (vector or hybrid)
#   2. Assemble context respecting token budget
#   3. Call LLM with system prompt + context + question
#   4. Return typed QueryResult with sources and latency
#
# The query engine is the public interface for the retrieval layer.
# Everything above this (API, agents) calls query_engine, nothing else.
# =============================================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from openai import AsyncOpenAI

from financial_rag.config import get_settings
from financial_rag.retrieval.document_retriever import DocumentRetriever, RetrievalResult
from financial_rag.retrieval.hybrid_search import HybridSearcher
from financial_rag.storage.vector_store import VectorStore
from financial_rag.utils.exceptions import RetrievalError

logger = logging.getLogger(__name__)

# =============================================================================
# System prompts — one per analysis style
# =============================================================================

_SYSTEM_PROMPTS: dict[str, str] = {
    "analyst": """You are a senior financial analyst with deep expertise in \
SEC filings, financial statements, and corporate strategy. Provide detailed, \
precise analysis grounded strictly in the provided context. Cite specific \
figures, dates, and sections. If the context does not contain sufficient \
information to answer the question, say so explicitly rather than speculating.""",
    "executive": """You are a CFO-level advisor providing concise, \
decision-ready financial insights. Synthesise the key points from the \
provided context into clear, actionable intelligence. Lead with the \
most important finding. Be direct and quantitative.""",
    "risk": """You are a chief risk officer conducting a thorough risk \
assessment. Analyse the provided context for financial risks, regulatory \
exposures, operational vulnerabilities, and forward-looking risk factors. \
Quantify risks where possible. Flag any material concerns explicitly.""",
}

_DEFAULT_STYLE = "analyst"


# =============================================================================
# Query result
# =============================================================================


@dataclass
class QueryResult:
    """
    Complete result from a RAG query.
    Returned by QueryEngine.query().
    Aligned with api/models.py QueryResponse.
    """

    question: str
    answer: str
    analysis_style: str
    search_type: str
    agent_type: str
    latency_ms: int
    source_documents: list[RetrievalResult] = field(default_factory=list)
    error: str | None = None

    @property
    def latency_seconds(self) -> float:
        return self.latency_ms / 1000.0

    def __repr__(self) -> str:
        return (
            f"<QueryResult style={self.analysis_style} "
            f"sources={len(self.source_documents)} "
            f"latency={self.latency_ms}ms>"
        )


# =============================================================================
# QueryEngine
# =============================================================================


class QueryEngine:
    """
    Full RAG pipeline from question → grounded answer.

    Routing logic:
      - search_type='similarity': DocumentRetriever (vector only)
      - search_type='mmr':        DocumentRetriever with MMR diversity
      - search_type='hybrid':     HybridSearcher (vector + full-text RRF)
      - If vector search returns < threshold scores, auto-upgrades to hybrid

    Usage:
        engine = QueryEngine()
        result = await engine.query(
            "What were Apple's main risk factors in FY2023?",
            ticker="AAPL",
            analysis_style="risk",
        )
        print(result.answer)
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._settings = get_settings()
        vs = vector_store or VectorStore()
        self._retriever = DocumentRetriever(vs)
        self._hybrid = HybridSearcher()
        self._llm = self._build_llm_client()

    def _build_llm_client(self) -> AsyncOpenAI | None:
        """
        Build async OpenAI client if API key is available.
        Returns None in testing/mock mode — query() returns a stub response.
        """
        if self._settings.MOCK_EXTERNAL_APIS:
            logger.info("QueryEngine: LLM calls mocked (MOCK_EXTERNAL_APIS=True)")
            return None

        if not self._settings.OPENAI_API_KEY:
            logger.warning(
                "QueryEngine: OPENAI_API_KEY not set — LLM calls disabled. "
                "Returning context only."
            )
            return None

        return AsyncOpenAI(
            api_key=self._settings.OPENAI_API_KEY.get_secret_value(),
            base_url=self._settings.LLM_BASE_URL or None,
            timeout=self._settings.LLM_REQUEST_TIMEOUT,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def query(
        self,
        question: str,
        *,
        ticker: str | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
        analysis_style: str = _DEFAULT_STYLE,
        search_type: Literal["similarity", "mmr", "hybrid"] = "similarity",
        limit: int | None = None,
    ) -> QueryResult:
        """
        Run the full RAG pipeline for a question.

        Args:
            question:       Natural language financial question
            ticker:         Optional company filter (e.g. 'AAPL')
            filing_type:    Optional form filter ('10-K', '10-Q', etc.)
            fiscal_year:    Optional year filter
            analysis_style: 'analyst' | 'executive' | 'risk'
            search_type:    'similarity' | 'mmr' | 'hybrid'
            limit:          Max source chunks (defaults to TOP_K_RESULTS)

        Returns:
            QueryResult with answer, sources, latency.
        """
        t0 = time.monotonic()

        if analysis_style not in _SYSTEM_PROMPTS:
            analysis_style = _DEFAULT_STYLE

        try:
            # ── Step 1: Retrieve ──────────────────────────────────────────────
            results = await self._retrieve(
                question,
                ticker=ticker,
                filing_type=filing_type,
                fiscal_year=fiscal_year,
                section=section,
                search_type=search_type,
                limit=limit,
            )

            # ── Step 2: Auto-upgrade to hybrid if vector confidence is low ────
            if (
                search_type == "similarity"
                and results
                and results[0].score < self._settings.VECTOR_SEARCH_THRESHOLD
            ):
                logger.info(
                    "Low vector confidence (%.3f < %.3f) — upgrading to hybrid",
                    results[0].score,
                    self._settings.VECTOR_SEARCH_THRESHOLD,
                )
                results = await self._hybrid.search(
                    question,
                    ticker=ticker,
                    filing_type=filing_type,
                    fiscal_year=fiscal_year,
                    limit=limit or self._settings.TOP_K_RESULTS,
                )
                search_type = "hybrid"  # update for logging

            if not results:
                return QueryResult(
                    question=question,
                    answer=(
                        "I could not find relevant information in the available "
                        "financial documents to answer this question. Please ensure "
                        "the relevant filings have been ingested."
                    ),
                    analysis_style=analysis_style,
                    search_type=search_type,
                    agent_type="query_engine",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    source_documents=[],
                )

            # ── Step 3: Build context ─────────────────────────────────────────
            context = self._retriever.build_context(results)

            # ── Step 4: Generate answer ───────────────────────────────────────
            answer = await self._generate_answer(
                question=question,
                context=context,
                analysis_style=analysis_style,
            )

            latency_ms = int((time.monotonic() - t0) * 1000)

            logger.info(
                "Query complete — style=%s search=%s sources=%d latency=%dms",
                analysis_style,
                search_type,
                len(results),
                latency_ms,
            )

            return QueryResult(
                question=question,
                answer=answer,
                analysis_style=analysis_style,
                search_type=search_type,
                agent_type="query_engine",
                latency_ms=latency_ms,
                source_documents=results,
            )

        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.error("Query failed after %dms: %s", latency_ms, exc)
            return QueryResult(
                question=question,
                answer="",
                analysis_style=analysis_style,
                search_type=search_type,
                agent_type="query_engine",
                latency_ms=latency_ms,
                error=str(exc),
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _retrieve(
        self,
        question: str,
        *,
        ticker: str | None,
        filing_type: str | None,
        fiscal_year: int | None,
        section: str | None = None,
        search_type: str,
        limit: int | None,
    ) -> list[RetrievalResult]:
        """Route to the appropriate retrieval strategy."""
        if search_type == "hybrid":
            return await self._hybrid.search(
                question,
                ticker=ticker,
                filing_type=filing_type,
                fiscal_year=fiscal_year,
                limit=limit or self._settings.TOP_K_RESULTS,
            )

        return await self._retriever.retrieve(
            question,
            ticker=ticker,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
            section=section,
            limit=limit,
            search_type=search_type,
        )

    async def _generate_answer(
        self,
        *,
        question: str,
        context: str,
        analysis_style: str,
    ) -> str:
        """
        Call the LLM to generate a grounded answer.
        Returns context-only string if LLM client is unavailable.
        """
        if self._llm is None:
            # No LLM — return the raw context for testing/inspection
            return f"[LLM unavailable — returning raw context]\n\n{context}"

        system_prompt = _SYSTEM_PROMPTS.get(analysis_style, _SYSTEM_PROMPTS[_DEFAULT_STYLE])

        user_message = (
            f"Using only the following financial document excerpts, "
            f"answer this question:\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )

        try:
            response = await self._llm.chat.completions.create(
                model=self._settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self._settings.LLM_TEMPERATURE,
                max_tokens=self._settings.LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content or ""

        except Exception as exc:
            raise RetrievalError(f"LLM generation failed: {exc}") from exc
