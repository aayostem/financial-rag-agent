# =============================================================================
# Financial RAG Agent — Base Repository
# src/financial_rag/storage/repositories/base.py
#
# Abstract base providing common query patterns shared by all repositories.
# Concrete repositories inherit from BaseRepository[ModelT].
# =============================================================================

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

    All methods receive an AsyncSession injected from the caller.
    Repositories do NOT own sessions — they borrow them.
    Transaction management belongs to the service layer.
    """

    model_class: type[Any]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Create ────────────────────────────────────────────────────────────────

    async def add(self, instance: ModelT) -> ModelT:
        """
        Persist a new ORM instance.
        The session is flushed but NOT committed — caller controls the transaction.
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
        """
        Bulk-persist a list of ORM instances.
        More efficient than calling add() in a loop for large batches.
        """
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
        """
        Fetch a single record by primary key.
        Raises RecordNotFoundError if not found.
        """
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
        """
        Fetch a single record by primary key.
        Returns None if not found (no exception).
        """
        try:
            return await self._session.get(self.model_class, record_id)
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to fetch {self.model_class.__name__} id={record_id}: {exc}"
            ) from exc

    async def list_all(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelT]:
        """
        Fetch all records with pagination.
        Do not use without a WHERE clause on large tables — use concrete
        repository methods with filters instead.
        """
        try:
            stmt = select(self.model_class).limit(limit).offset(offset)
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to list {self.model_class.__name__}: {exc}"
            ) from exc

    async def count(self) -> int:
        """Return the total number of records in the table."""
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
        Apply field updates to an already-loaded ORM instance and flush.

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
        """
        Delete an ORM instance. Flushes but does not commit.
        """
        try:
            await self._session.delete(instance)
            await self._session.flush()
        except Exception as exc:
            raise DatabaseQueryError(
                f"Failed to delete {self.model_class.__name__}: {exc}"
            ) from exc

    async def soft_delete(self, instance: ModelT) -> ModelT:
        """
        Set is_active=False rather than hard-deleting.
        The model must have an is_active column.
        """
        return await self.update(instance, is_active=False)
