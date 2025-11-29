import pytest
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
from typing import Dict, Any, Generator

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.main import app
from src.database.models import Base
from src.core.config import get_settings

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test_financial_rag.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Test configuration settings."""
    settings = get_settings()
    settings.DATABASE_URL = TEST_DATABASE_URL
    settings.TESTING = True
    settings.DEBUG = True
    return settings


@pytest.fixture(scope="session")
def test_engine(test_settings):
    """Test database engine."""
    engine = create_engine(test_settings.DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create a fresh database session for each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def test_client(test_session):
    """Test client for FastAPI application."""

    def override_get_db():
        try:
            yield test_session
        finally:
            pass

    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def mock_openai():
    """Mock OpenAI API calls."""
    with patch("src.agents.llm_client.openai.ChatCompletion.create") as mock:
        mock.return_value = {
            "choices": [
                {
                    "message": {"content": "Mocked AI response", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        }
        yield mock


@pytest.fixture
def mock_sec_edgar():
    """Mock SEC EDGAR API calls."""
    with patch("src.data.sec_edgar_client.SECClient.get_filing") as mock:
        mock.return_value = {
            "company": "TEST COMPANY",
            "filing_type": "10-K",
            "filing_date": "2023-12-31",
            "content": "This is a test filing content for financial analysis.",
        }
        yield mock


@pytest.fixture
def mock_yahoo_finance():
    """Mock Yahoo Finance API calls."""
    with patch("src.data.yahoo_finance_client.YahooFinanceClient.get_stock_data") as mock:
        mock.return_value = {
            "symbol": "AAPL",
            "price": 150.0,
            "change": 2.5,
            "change_percent": 1.67,
            "volume": 1000000,
            "market_cap": 2500000000000,
        }
        yield mock


@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing."""
    return {
        "balance_sheet": {
            "total_assets": 1000000,
            "total_liabilities": 600000,
            "shareholders_equity": 400000,
        },
        "income_statement": {"revenue": 500000, "net_income": 75000, "eps": 5.0},
        "cash_flow": {
            "operating_cash_flow": 100000,
            "investing_cash_flow": -50000,
            "financing_cash_flow": -20000,
        },
    }


@pytest.fixture
def sample_company_data():
    """Sample company data for testing."""
    return {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 2500000000000,
        "employees": 164000,
    }


@pytest.fixture
def mock_vector_store():
    """Mock vector store operations."""
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [
        MagicMock(
            page_content="Test document content about financial metrics",
            metadata={"source": "10-K", "company": "TEST"},
        )
    ]
    mock_store.add_documents.return_value = ["doc1", "doc2"]
    return mock_store


@pytest.fixture
def sample_rag_query():
    """Sample RAG query for testing."""
    return {
        "query": "What are the key financial ratios for Apple?",
        "company": "AAPL",
        "analysis_type": "financial_ratios",
    }


@pytest.fixture
def mock_agent_response():
    """Mock agent response for testing."""
    return {
        "analysis": "Comprehensive financial analysis",
        "recommendations": ["Buy", "Hold", "Sell"],
        "confidence": 0.85,
        "supporting_data": ["Revenue growth", "Profit margins"],
    }
