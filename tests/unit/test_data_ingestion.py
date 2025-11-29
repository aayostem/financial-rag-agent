import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
from datetime import datetime

from src.data.sec_edgar_client import SECEdgarClient
from src.data.yahoo_finance_client import YahooFinanceClient
from src.data.news_ingestor import NewsIngestor


class TestSECEdgarClient:
    """Test SEC EDGAR client functionality."""

    @pytest.mark.unit
    def test_sec_client_initialization(self):
        """Test SEC client initialization."""
        client = SECEdgarClient()

        assert client.base_url == "https://www.sec.gov/archives/edgar/data"
        assert hasattr(client, "get_filing")
        assert hasattr(client, "get_company_filings")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_filing_success(self, mock_sec_edgar):
        """Test successful filing retrieval."""
        client = SECEdgarClient()

        filing = await client.get_filing(
            cik="0000320193", filing_type="10-K", year=2023  # Apple CIK
        )

        assert filing["company"] == "TEST COMPANY"
        assert filing["filing_type"] == "10-K"
        assert "content" in filing

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_company_filings(self, mock_sec_edgar):
        """Test company filings list retrieval."""
        client = SECEdgarClient()

        filings = await client.get_company_filings(
            cik="0000320193", start_date="2023-01-01", end_date="2023-12-31"
        )

        assert isinstance(filings, list)
        assert len(filings) > 0

    @pytest.mark.unit
    def test_parse_filing_content(self):
        """Test filing content parsing."""
        client = SECEdgarClient()

        sample_filing = """
        <DOCUMENT>
        <TYPE>10-K
        <SEQUENCE>1
        <FILENAME>test.txt
        <DESCRIPTION>ANNUAL REPORT
        <TEXT>
        ITEM 1. BUSINESS
        Company overview...
        ITEM 7. MD&A
        Management discussion...
        </TEXT>
        </DOCUMENT>
        """

        parsed = client._parse_filing_content(sample_filing)

        assert "business" in parsed
        assert "management_discussion" in parsed
        assert len(parsed["business"]) > 0


class TestYahooFinanceClient:
    """Test Yahoo Finance client functionality."""

    @pytest.mark.unit
    def test_yahoo_client_initialization(self):
        """Test Yahoo Finance client initialization."""
        client = YahooFinanceClient()

        assert hasattr(client, "get_stock_data")
        assert hasattr(client, "get_historical_data")
        assert hasattr(client, "get_company_info")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_stock_data(self, mock_yahoo_finance):
        """Test current stock data retrieval."""
        client = YahooFinanceClient()

        stock_data = await client.get_stock_data("AAPL")

        assert stock_data["symbol"] == "AAPL"
        assert stock_data["price"] == 150.0
        assert "change_percent" in stock_data

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_historical_data(self):
        """Test historical data retrieval."""
        client = YahooFinanceClient()

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame(
                {"Close": [150, 155, 160], "Volume": [1000000, 1200000, 1100000]},
                index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            )

            historical_data = await client.get_historical_data(
                symbol="AAPL", start_date="2023-01-01", end_date="2023-01-31"
            )

            assert "prices" in historical_data
            assert "volumes" in historical_data
            assert len(historical_data["prices"]) == 3

    @pytest.mark.unit
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculations."""
        client = YahooFinanceClient()

        prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]

        indicators = client._calculate_technical_indicators(prices)

        assert "sma_20" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators
        assert all(isinstance(val, (int, float)) for val in indicators.values())


class TestNewsIngestor:
    """Test News Ingestor functionality."""

    @pytest.mark.unit
    def test_news_ingestor_initialization(self):
        """Test news ingestor initialization."""
        ingestor = NewsIngestor()

        assert hasattr(ingestor, "fetch_news")
        assert hasattr(ingestor, "analyze_sentiment")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fetch_company_news(self):
        """Test company news fetching."""
        ingestor = NewsIngestor()

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "articles": [
                    {
                        "title": "Apple Reports Strong Earnings",
                        "description": "Apple exceeded Q4 expectations",
                        "publishedAt": "2023-11-01T10:00:00Z",
                    }
                ]
            }

            news = await ingestor.fetch_news(company="Apple", days_back=7)

            assert len(news) > 0
            assert "title" in news[0]
            assert "sentiment" in news[0]

    @pytest.mark.unit
    def test_analyze_sentiment(self):
        """Test news sentiment analysis."""
        ingestor = NewsIngestor()

        news_text = "Apple reported outstanding quarterly results with record revenue growth."

        sentiment = ingestor.analyze_sentiment(news_text)

        assert "score" in sentiment
        assert "label" in sentiment
        assert -1 <= sentiment["score"] <= 1
        assert sentiment["label"] in ["positive", "negative", "neutral"]

    @pytest.mark.unit
    def test_extract_key_financial_events(self):
        """Test financial event extraction from news."""
        ingestor = NewsIngestor()

        news_articles = [
            {
                "title": "Apple Q4 Earnings Beat Estimates",
                "content": "Apple reported EPS of $1.50 vs $1.35 expected. Revenue was $100B.",
            }
        ]

        events = ingestor._extract_key_financial_events(news_articles)

        assert len(events) > 0
        assert any("earnings" in event["type"].lower() for event in events)
        assert any("revenue" in event["metrics"] for event in events)
