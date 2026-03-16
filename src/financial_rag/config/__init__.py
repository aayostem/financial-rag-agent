# =============================================================================
# Financial RAG Agent — Config Package
# src/financial_rag/config/__init__.py
#
# Single export surface. Everything outside this package imports from here.
# The environments/ and features/ subdirectories are deleted — no longer needed.
# =============================================================================

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
