# =============================================================================
# Financial RAG Agent — Ingestion Package
# src/financial_rag/ingestion/__init__.py
# =============================================================================

from .sec_ingestor import SUPPORTED_FILING_TYPES, FilingMetadata, SECIngestor

__all__ = [
    "SUPPORTED_FILING_TYPES",
    "FilingMetadata",
    "SECIngestor",
]
