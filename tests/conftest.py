# =============================================================================
# Financial RAG Agent — Shared Test Fixtures
# tests/conftest.py
# =============================================================================

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from financial_rag.config import get_settings

# Minimum env vars for all tests
VALID_SECRETS = {
    "POSTGRES_PASSWORD": "test-pg-password-32-chars-minimum",
    "REDIS_PASSWORD": "test-redis-password-32-chars-min",
    "APP_ENV": "testing",
}


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_env():
    """Patch environment with testing defaults."""
    with patch.dict(os.environ, VALID_SECRETS, clear=False):
        yield


@pytest.fixture
def mock_db_session():
    """Mock async SQLAlchemy session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def mock_cache_client():
    """Mock Redis cache client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=True)
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    return client


@pytest.fixture
def sample_embedding():
    """384-dim zero embedding for local model tests."""
    return [0.0] * 384


@pytest.fixture
def sample_chunk_text():
    return (
        "Apple Inc. reported total net sales of $391.0 billion for fiscal year 2024, "
        "representing a 2% increase compared to fiscal year 2023. The Company's "
        "Services segment achieved record revenue of $96.2 billion."
    )
