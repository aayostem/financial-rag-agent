# =============================================================================
# Financial RAG Agent — Exception Hierarchy
# src/financial_rag/utils/exceptions.py
#
# All application exceptions inherit from FinRAGError.
# Never raise bare Exception() anywhere in the codebase.
#
# Hierarchy:
#   FinRAGError
#   ├── StorageError
#   │   ├── DatabaseConnectionError
#   │   ├── DatabaseQueryError
#   │   ├── CacheConnectionError
#   │   └── CacheOperationError
#   ├── IngestionError
#   │   ├── SECFetchError
#   │   └── DocumentParseError
#   ├── ProcessingError
#   │   └── ChunkingError
#   ├── RetrievalError
#   │   ├── EmbeddingError
#   │   └── VectorSearchError
#   └── ConfigurationError
# =============================================================================

from __future__ import annotations


class FinRAGError(Exception):
    """Base exception for all Financial RAG Agent errors."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        if self.cause:
            return f"{base} (caused by: {self.cause})"
        return base


# =============================================================================
# Storage
# =============================================================================


class StorageError(FinRAGError):
    """Base for all storage-layer errors."""


class DatabaseConnectionError(StorageError):
    """
    Raised when the application cannot establish or maintain a
    connection to PostgreSQL.
    """


class DatabaseQueryError(StorageError):
    """
    Raised when a database query fails at runtime.
    Wraps SQLAlchemy / asyncpg exceptions.
    """


class CacheConnectionError(StorageError):
    """
    Raised when the application cannot establish or maintain a
    connection to Redis.
    """


class CacheOperationError(StorageError):
    """
    Raised when a Redis operation (GET, SET, DELETE) fails at runtime.
    """


class RecordNotFoundError(StorageError):
    """
    Raised when a repository lookup returns no results for a required record.
    Analogous to HTTP 404 at the storage layer.
    """

    def __init__(self, entity: str, identifier: str | int) -> None:
        super().__init__(f"{entity} not found: {identifier}")
        self.entity = entity
        self.identifier = identifier


# =============================================================================
# Ingestion
# =============================================================================


class IngestionError(FinRAGError):
    """Base for all document ingestion errors."""


class SECFetchError(IngestionError):
    """
    Raised when the EDGAR API cannot be reached or returns an error.
    Includes rate-limit violations and network timeouts.
    """


class DocumentParseError(IngestionError):
    """
    Raised when a document (HTML, PDF) cannot be parsed into clean text.
    """


class DuplicateFilingError(IngestionError):
    """
    Raised when an attempt is made to ingest a filing that already
    exists in the database (identified by file_hash).
    """

    def __init__(self, ticker: str, file_hash: str) -> None:
        super().__init__(f"Filing already ingested — ticker={ticker} hash={file_hash}")
        self.ticker = ticker
        self.file_hash = file_hash


# =============================================================================
# Processing
# =============================================================================


class ProcessingError(FinRAGError):
    """Base for all document processing errors."""


class ChunkingError(ProcessingError):
    """
    Raised when document chunking fails — e.g. a document cannot be
    tokenised or split within configured constraints.
    """


# =============================================================================
# Retrieval
# =============================================================================


class RetrievalError(FinRAGError):
    """Base for all retrieval-layer errors."""


class EmbeddingError(RetrievalError):
    """
    Raised when the embedding API returns an error or times out.
    Includes OpenAI rate-limit errors.
    """


class VectorSearchError(RetrievalError):
    """
    Raised when a pgvector similarity search fails.
    """


# =============================================================================
# Configuration
# =============================================================================


class ConfigurationError(FinRAGError):
    """
    Raised when the application detects a fatal misconfiguration
    that was not caught at settings validation time.
    """
