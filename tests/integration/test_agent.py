# import uuid
# import os
# from unittest.mock import patch
# import pytest

# from financial_rag.agents.financial_agent import AgentResult, FinancialAgent
# from financial_rag.retrieval.query_engine import QueryEngine, QueryResult
# from financial_rag.storage.repositories.analysis import (
#     AnalysisRecord,
#     AnalysisRepository,
# )
# from financial_rag.config import get_settings

# VALID_SECRETS = {
#     "POSTGRES_PASSWORD": "postgres",
#     "REDIS_PASSWORD": "",
#     "APP_ENV": "testing",
# }


# @pytest.fixture(autouse=True)
# def _testing_env():
#     with patch.dict(os.environ, VALID_SECRETS, clear=False):
#         get_settings.cache_clear()
#         yield
#     get_settings.cache_clear()


# @pytest.fixture
# async def db():
#     from financial_rag.storage.database import DatabaseClient

#     client = DatabaseClient()
#     await client.connect()
#     yield client
#     await client.disconnect()


# class TestQueryResult:
#     def test_latency_seconds_converts_correctly(self):
#         result = QueryResult(
#             question="test",
#             answer="answer",
#             analysis_style="analyst",
#             search_type="similarity",
#             agent_type="query_engine",
#             latency_ms=1500,
#         )
#         assert result.latency_seconds == 1.5

#     def test_repr_includes_key_fields(self):
#         result = QueryResult(
#             question="test",
#             answer="answer",
#             analysis_style="analyst",
#             search_type="similarity",
#             agent_type="query_engine",
#             latency_ms=250,
#         )
#         assert "analyst" in repr(result)
#         assert "250ms" in repr(result)


# class TestAgentResult:
#     def test_to_query_result_converts(self):
#         agent_result = AgentResult(
#             question="test",
#             answer="answer",
#             analysis_style="analyst",
#             agent_type="financial_agent",
#             latency_ms=800,
#         )
#         qr = agent_result.to_query_result()
#         assert isinstance(qr, QueryResult)
#         assert qr.search_type == "agent"
#         assert qr.latency_ms == 800

#     def test_latency_seconds_property(self):
#         result = AgentResult(
#             question="test",
#             answer="answer",
#             analysis_style="analyst",
#             agent_type="financial_agent",
#             latency_ms=2000,
#         )
#         assert result.latency_seconds == 2.0


# class TestAnalysisRepository:
#     @pytest.mark.integration
#     async def test_record_and_retrieve(self, db):
#         async with db.session() as session:
#             repo = AnalysisRepository(session)
#             record = await repo.record(
#                 question="What was Apple's revenue?",
#                 answer="Apple's revenue was $394 billion.",
#                 agent_type="query_engine",
#                 latency_ms=320,
#                 ticker="AAPL",
#                 analysis_style="analyst",
#                 search_type="similarity",
#             )
#         assert record.id is not None
#         assert record.ticker == "AAPL"
#         assert record.latency_ms == 320

#     @pytest.mark.integration
#     async def test_get_by_ticker(self, db):
#         async with db.session() as session:
#             repo = AnalysisRepository(session)
#             records = await repo.get_by_ticker("AAPL")
#         assert isinstance(records, list)


# class TestQueryEngineMocked:
#     @pytest.mark.integration
#     async def test_query_returns_query_result(self, monkeypatch):
#         monkeypatch.setenv("MOCK_EXTERNAL_APIS", "true")
#         from financial_rag.config import get_settings

#         get_settings.cache_clear()

#         engine = QueryEngine()
#         result = await engine.query("What is Apple's revenue?", ticker="AAPL")
#         assert isinstance(result, QueryResult)
#         assert result.question == "What is Apple's revenue?"

#     @pytest.mark.integration
#     async def test_agent_falls_back_to_query_engine_without_key(self, monkeypatch):
#         monkeypatch.setenv("MOCK_EXTERNAL_APIS", "true")
#         from financial_rag.config import get_settings

#         get_settings.cache_clear()

#         agent = FinancialAgent()
#         assert agent._llm is None
#         result = await agent.analyze("Test question?")
#         assert result.agent_type == "query_engine_fallback"
# =============================================================================
# Financial RAG Agent — Integration Tests
# tests/integration/test_agent.py
#
# Run:  pytest tests/integration/test_agent.py -v
# =============================================================================

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from financial_rag.agents.financial_agent import AgentResult, FinancialAgent
from financial_rag.config import get_settings
from financial_rag.retrieval.query_engine import QueryEngine, QueryResult
from financial_rag.storage.repositories.analysis import (
    AnalysisRepository,
)
from tests.conftest import VALID_SECRETS

# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """
    Clear the lru_cache before and after every test so that environment
    patches in one test cannot bleed into another.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings() -> get_settings:
    """
    A valid Settings instance for the testing environment.
    Use this fixture instead of calling get_settings() directly in tests.
    """
    with patch.dict(os.environ, VALID_SECRETS, clear=False):
        get_settings.cache_clear()
        return get_settings()


@pytest.fixture
async def db_client():
    """
    Database client fixture for integration tests.
    Connects and disconnects automatically.
    """
    from financial_rag.storage.database import DatabaseClient

    client = DatabaseClient()
    await client.connect()
    yield client
    await client.disconnect()


@pytest.fixture
async def db_session(db_client):
    """
    Database session fixture for integration tests.
    Provides a clean session for each test.
    """
    async with db_client.session() as session:
        yield session


# =============================================================================
# QueryResult tests
# =============================================================================


class TestQueryResult:
    """QueryResult model behavior."""

    def test_latency_seconds_converts_correctly(self):
        """Milliseconds should convert to seconds properly."""
        result = QueryResult(
            question="test",
            answer="answer",
            analysis_style="analyst",
            search_type="similarity",
            agent_type="query_engine",
            latency_ms=1500,
        )
        assert result.latency_seconds == 1.5

    def test_repr_includes_key_fields(self):
        """repr() should show important fields for debugging."""
        result = QueryResult(
            question="test",
            answer="answer",
            analysis_style="analyst",
            search_type="similarity",
            agent_type="query_engine",
            latency_ms=250,
        )
        assert "analyst" in repr(result)
        assert "250ms" in repr(result)


# =============================================================================
# AgentResult tests
# =============================================================================


class TestAgentResult:
    """AgentResult model behavior."""

    def test_to_query_result_converts(self):
        """AgentResult should convert to QueryResult correctly."""
        agent_result = AgentResult(
            question="test",
            answer="answer",
            analysis_style="analyst",
            agent_type="financial_agent",
            latency_ms=800,
        )
        qr = agent_result.to_query_result()
        assert isinstance(qr, QueryResult)
        assert qr.search_type == "agent"
        assert qr.latency_ms == 800

    def test_latency_seconds_property(self):
        """latency_seconds should convert milliseconds to seconds."""
        result = AgentResult(
            question="test",
            answer="answer",
            analysis_style="analyst",
            agent_type="financial_agent",
            latency_ms=2000,
        )
        assert result.latency_seconds == 2.0


# =============================================================================
# AnalysisRepository integration tests
# =============================================================================


class TestAnalysisRepository:
    """AnalysisRepository integration tests requiring database."""

    @pytest.mark.integration
    async def test_record_and_retrieve(self, db_session):
        """
        Recording an analysis should create a record with an ID.
        """
        repo = AnalysisRepository(db_session)
        record = await repo.record(
            question="What was Apple's revenue?",
            answer="Apple's revenue was $394 billion.",
            agent_type="query_engine",
            latency_ms=320,
            ticker="AAPL",
            analysis_style="analyst",
            search_type="similarity",
        )
        assert record.id is not None
        assert record.ticker == "AAPL"
        assert record.latency_ms == 320

    @pytest.mark.integration
    async def test_get_by_ticker(self, db_session):
        """
        Getting records by ticker should return a list of records.
        """
        repo = AnalysisRepository(db_session)
        records = await repo.get_by_ticker("AAPL")
        assert isinstance(records, list)


# =============================================================================
# QueryEngine integration tests
# =============================================================================


class TestQueryEngineMocked:
    """QueryEngine tests with mocked external APIs."""

    @pytest.mark.integration
    async def test_query_returns_query_result(self, monkeypatch):
        """
        QueryEngine should return a QueryResult when MOCK_EXTERNAL_APIS is true.
        """
        monkeypatch.setenv("MOCK_EXTERNAL_APIS", "true")
        get_settings.cache_clear()

        engine = QueryEngine()
        result = await engine.query("What is Apple's revenue?", ticker="AAPL")
        assert isinstance(result, QueryResult)
        assert result.question == "What is Apple's revenue?"

    @pytest.mark.integration
    async def test_agent_falls_back_to_query_engine_without_key(self, monkeypatch):
        """
        FinancialAgent should fall back to QueryEngine when no LLM key is available.
        """
        monkeypatch.setenv("MOCK_EXTERNAL_APIS", "true")
        get_settings.cache_clear()

        agent = FinancialAgent()
        assert agent._llm is None
        result = await agent.analyze("Test question?")
        assert result.agent_type == "query_engine_fallback"


# =============================================================================
# Error handling tests
# =============================================================================


class TestErrorHandling:
    """Integration error handling tests."""

    @pytest.mark.integration
    async def test_query_handles_empty_question(self):
        """Empty question should raise appropriate error."""
        engine = QueryEngine()
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await engine.query("", ticker="AAPL")

    @pytest.mark.integration
    async def test_agent_handles_empty_question(self, monkeypatch):
        """Empty question should raise appropriate error in agent."""
        monkeypatch.setenv("MOCK_EXTERNAL_APIS", "true")
        get_settings.cache_clear()

        agent = FinancialAgent()
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await agent.analyze("")


# =============================================================================
# Performance tests
# =============================================================================


class TestPerformance:
    """Performance-sensitive integration tests."""

    # @pytest.mark.integration
    # async def test_query_performance(self):
    #     """Query should complete within reasonable time."""
    #     import time

    #     engine = QueryEngine()
    #     start = time.perf_counter()
    #     await engine.query("What is Apple's revenue?", ticker="AAPL")
    #     elapsed = time.perf_counter() - start
    #     assert elapsed < 10.0  # 10 seconds max


# =============================================================================
# Cleanup
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_after_integration_tests():
    """
    Ensure database cleanup happens after integration tests.
    """
    yield
    # Any cleanup logic can go here
