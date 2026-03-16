# =============================================================================
# Financial RAG Agent — Analysis History Repository
# src/financial_rag/storage/repositories/analysis.py
#
# Data access layer for the `analysis_history` table.
# Immutable audit log of every query and response.
# =============================================================================

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, Integer, String, Text, func, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TIMESTAMP

from financial_rag.storage.database import Base
from financial_rag.storage.repositories.base import BaseRepository
from financial_rag.utils.exceptions import DatabaseQueryError

logger = logging.getLogger(__name__)


# =============================================================================
# ORM Model
# =============================================================================


class AnalysisRecord(Base):
    """
    ORM representation of the `analysis_history` table.
    Mirrors create_schema.sql exactly — keep in sync.
    """

    __tablename__ = "analysis_history"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    ticker: Mapped[str | None] = mapped_column(String(10))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_style: Mapped[str] = mapped_column(String(20), nullable=False, default="analyst")
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    search_type: Mapped[str] = mapped_column(String(20), nullable=False, default="similarity")
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    source_chunk_ids: Mapped[list[Any]] = mapped_column(ARRAY(PG_UUID(as_uuid=True)), default=list)
    real_time_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    session_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), index=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisRecord id={self.id} ticker={self.ticker} "
            f"agent={self.agent_type} latency={self.latency_ms}ms>"
        )


# =============================================================================
# Repository
# =============================================================================


class AnalysisRepository(BaseRepository[AnalysisRecord]):
    """
    All database operations for the analysis_history table.

    Records are append-only — never update or delete history.
    Use soft patterns (error field, session grouping) instead.
    """

    model_class = AnalysisRecord

    # ── Write ─────────────────────────────────────────────────────────────────

    async def record(
        self,
        *,
        question: str,
        answer: str,
        agent_type: str,
        latency_ms: int,
        ticker: str | None = None,
        analysis_style: str = "analyst",
        search_type: str = "similarity",
        source_chunk_ids: list[UUID] | None = None,
        real_time_used: bool = False,
        error: str | None = None,
        session_id: UUID | None = None,
    ) -> AnalysisRecord:
        """
        Append a completed analysis to the history table.
        This is the primary write method — all analysis must be recorded.

        Note: latency is stored as milliseconds (int), not seconds (float),
        for efficient range queries and aggregations.
        """
        import uuid

        record = AnalysisRecord(
            id=uuid.uuid4(),
            ticker=ticker.upper() if ticker else None,
            question=question,
            answer=answer,
            analysis_style=analysis_style,
            agent_type=agent_type,
            search_type=search_type,
            latency_ms=latency_ms,
            source_chunk_ids=source_chunk_ids or [],
            real_time_used=real_time_used,
            error=error,
            session_id=session_id,
        )
        return await self.add(record)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_by_ticker(
        self,
        ticker: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AnalysisRecord]:
        """Return analysis history for a ticker, newest first."""
        try:
            result = await self._session.execute(
                select(AnalysisRecord)
                .where(AnalysisRecord.ticker == ticker.upper())
                .order_by(AnalysisRecord.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to fetch analysis history for '{ticker}': {exc}"
            ) from exc

    async def get_by_session(self, session_id: UUID) -> list[AnalysisRecord]:
        """Return all records belonging to a session, in chronological order."""
        try:
            result = await self._session.execute(
                select(AnalysisRecord)
                .where(AnalysisRecord.session_id == session_id)
                .order_by(AnalysisRecord.created_at)
            )
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to fetch session history {session_id}: {exc}"
            ) from exc

    async def get_recent(
        self,
        *,
        limit: int = 20,
        agent_type: str | None = None,
        errors_only: bool = False,
    ) -> list[AnalysisRecord]:
        """
        Return the most recent analyses globally.
        Useful for monitoring dashboards and debugging.
        """
        try:
            stmt = select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).limit(limit)
            if agent_type:
                stmt = stmt.where(AnalysisRecord.agent_type == agent_type)
            if errors_only:
                stmt = stmt.where(AnalysisRecord.error.isnot(None))

            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseQueryError(f"Failed to fetch recent analyses: {exc}") from exc

    # ── Aggregations ──────────────────────────────────────────────────────────

    async def average_latency_ms(
        self,
        *,
        ticker: str | None = None,
        agent_type: str | None = None,
    ) -> float | None:
        """
        Return average query latency in milliseconds.
        Returns None if no matching records exist.
        """
        try:
            stmt = select(func.avg(AnalysisRecord.latency_ms))
            if ticker:
                stmt = stmt.where(AnalysisRecord.ticker == ticker.upper())
            if agent_type:
                stmt = stmt.where(AnalysisRecord.agent_type == agent_type)

            result = await self._session.execute(stmt)
            avg = result.scalar_one_or_none()
            return float(avg) if avg is not None else None
        except Exception as exc:
            raise DatabaseQueryError(f"Failed to compute average latency: {exc}") from exc

    async def error_rate(
        self,
        *,
        ticker: str | None = None,
    ) -> float:
        """
        Return the fraction of analyses that resulted in an error.
        Returns 0.0 if no records exist.
        """
        try:
            base_stmt = select(func.count()).select_from(AnalysisRecord)
            error_stmt = (
                select(func.count())
                .select_from(AnalysisRecord)
                .where(AnalysisRecord.error.isnot(None))
            )

            if ticker:
                base_stmt = base_stmt.where(AnalysisRecord.ticker == ticker.upper())
                error_stmt = error_stmt.where(AnalysisRecord.ticker == ticker.upper())

            total = (await self._session.execute(base_stmt)).scalar_one()
            errors = (await self._session.execute(error_stmt)).scalar_one()

            return errors / total if total > 0 else 0.0
        except Exception as exc:
            raise DatabaseQueryError(f"Failed to compute error rate: {exc}") from exc
