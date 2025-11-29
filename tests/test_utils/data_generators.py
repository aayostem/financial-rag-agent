"""Test data generators for financial RAG agent tests."""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import pandas as pd
import numpy as np


class FinancialDataGenerator:
    """Generate realistic financial test data."""

    @staticmethod
    def generate_company_profile(symbol: str) -> Dict[str, Any]:
        """Generate realistic company profile."""
        sectors = ["Technology", "Healthcare", "Financial", "Consumer", "Industrial"]
        industries = {
            "Technology": ["Software", "Hardware", "Semiconductors", "Internet"],
            "Healthcare": ["Pharmaceuticals", "Biotech", "Medical Devices"],
            "Financial": ["Banking", "Insurance", "Investment"],
            "Consumer": ["Retail", "Automotive", "Apparel"],
            "Industrial": ["Manufacturing", "Construction", "Transportation"],
        }

        sector = random.choice(sectors)
        industry = random.choice(industries[sector])

        return {
            "symbol": symbol,
            "name": f"{symbol} Corporation",
            "sector": sector,
            "industry": industry,
            "market_cap": random.randint(10_000_000, 1_000_000_000_000),
            "employees": random.randint(100, 500000),
            "founded_year": random.randint(1950, 2020),
        }

    @staticmethod
    def generate_income_statement(periods: int = 4) -> List[Dict[str, Any]]:
        """Generate quarterly income statements."""
        statements = []
        base_revenue = random.randint(1_000_000, 10_000_000)

        for i in range(periods):
            revenue = base_revenue * (1 + random.uniform(-0.1, 0.2))
            cogs = revenue * random.uniform(0.4, 0.7)
            gross_profit = revenue - cogs
            operating_expenses = revenue * random.uniform(0.1, 0.3)
            operating_income = gross_profit - operating_expenses
            net_income = operating_income * random.uniform(0.7, 0.9)

            statements.append(
                {
                    "period": f"Q{4-i}-2023",
                    "revenue": round(revenue, 2),
                    "cost_of_goods_sold": round(cogs, 2),
                    "gross_profit": round(gross_profit, 2),
                    "operating_expenses": round(operating_expenses, 2),
                    "operating_income": round(operating_income, 2),
                    "net_income": round(net_income, 2),
                    "eps": round(net_income / random.randint(100, 1000), 2),
                }
            )

        return statements

    @staticmethod
    def generate_balance_sheet() -> Dict[str, Any]:
        """Generate balance sheet data."""
        total_assets = random.randint(1_000_000, 10_000_000)
        current_assets = total_assets * random.uniform(0.3, 0.6)
        non_current_assets = total_assets - current_assets

        total_liabilities = total_assets * random.uniform(0.4, 0.7)
        current_liabilities = total_liabilities * random.uniform(0.3, 0.6)
        long_term_debt = total_liabilities - current_liabilities

        shareholders_equity = total_assets - total_liabilities

        return {
            "total_assets": round(total_assets, 2),
            "current_assets": round(current_assets, 2),
            "non_current_assets": round(non_current_assets, 2),
            "total_liabilities": round(total_liabilities, 2),
            "current_liabilities": round(current_liabilities, 2),
            "long_term_debt": round(long_term_debt, 2),
            "shareholders_equity": round(shareholders_equity, 2),
        }

    @staticmethod
    def generate_stock_prices(days: int = 30) -> pd.DataFrame:
        """Generate realistic stock price data."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        base_price = random.randint(50, 500)

        prices = []
        current_price = base_price

        for _ in range(days):
            change_percent = random.uniform(-0.05, 0.05)
            current_price = current_price * (1 + change_percent)
            volume = random.randint(1000000, 10000000)
            prices.append(
                {
                    "close": round(current_price, 2),
                    "volume": volume,
                    "high": round(current_price * random.uniform(1.01, 1.05), 2),
                    "low": round(current_price * random.uniform(0.95, 0.99), 2),
                    "open": round(current_price * random.uniform(0.98, 1.02), 2),
                }
            )

        df = pd.DataFrame(prices, index=dates)
        return df

    @staticmethod
    def generate_news_articles(company: str, count: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic news articles."""
        sentiments = ["positive", "negative", "neutral"]
        topics = [
            "earnings",
            "merger",
            "product_launch",
            "regulation",
            "management",
            "market_trends",
            "competition",
        ]

        articles = []
        for i in range(count):
            sentiment = random.choice(sentiments)
            topic = random.choice(topics)

            articles.append(
                {
                    "title": f"{company} {topic.replace('_', ' ').title()} News",
                    "content": f"Detailed analysis of {company}'s recent {topic} developments.",
                    "published_at": datetime.now() - timedelta(days=random.randint(1, 30)),
                    "sentiment": sentiment,
                    "sentiment_score": random.uniform(-1, 1) if sentiment != "neutral" else 0,
                    "source": random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ"]),
                    "url": f"https://example.com/news/{company}-{topic}-{i}",
                }
            )

        return articles


class RAGTestDataGenerator:
    """Generate test data for RAG system testing."""

    @staticmethod
    def generate_financial_documents(company: str, count: int = 20) -> List[Dict[str, Any]]:
        """Generate financial documents for vector store testing."""
        doc_types = ["10-K", "10-Q", "8-K", "Press Release", "Earnings Call Transcript"]
        sections = [
            "Business Overview",
            "Risk Factors",
            "MD&A",
            "Financial Statements",
            "Market Analysis",
            "Competition",
        ]

        documents = []
        for i in range(count):
            doc_type = random.choice(doc_types)
            section = random.choice(sections)

            documents.append(
                {
                    "id": f"doc_{company}_{i}",
                    "content": f"""
                {company} {doc_type} - {section}
                
                This section contains important financial information about {company}.
                Key metrics include revenue growth of {random.randint(5, 25)}%,
                profit margins of {random.randint(10, 40)}%, and market share of {random.randint(5, 30)}%.
                
                Management discusses the company's strategy and future outlook in this section.
                Important risks include market competition and regulatory changes.
                """,
                    "metadata": {
                        "company": company,
                        "document_type": doc_type,
                        "section": section,
                        "filing_date": (
                            datetime.now() - timedelta(days=random.randint(1, 365))
                        ).strftime("%Y-%m-%d"),
                        "source": (
                            "SEC" if doc_type in ["10-K", "10-Q", "8-K"] else "Company Website"
                        ),
                    },
                }
            )

        return documents

    @staticmethod
    def generate_test_queries(company: str) -> List[Dict[str, Any]]:
        """Generate test queries for RAG evaluation."""
        return [
            {
                "query": f"What are {company}'s main revenue drivers?",
                "expected_topics": ["revenue", "business segments", "growth"],
                "difficulty": "easy",
            },
            {
                "query": f"Analyze {company}'s debt levels and liquidity position",
                "expected_topics": ["debt", "liquidity", "balance sheet"],
                "difficulty": "medium",
            },
            {
                "query": f"What risks does {company} face in current market conditions?",
                "expected_topics": ["risk factors", "market conditions", "competition"],
                "difficulty": "hard",
            },
            {
                "query": f"Compare {company}'s profit margins with industry averages",
                "expected_topics": ["profit margins", "industry comparison", "financial ratios"],
                "difficulty": "hard",
            },
        ]


class PerformanceDataGenerator:
    """Generate data for performance testing."""

    @staticmethod
    def generate_large_document_set(num_documents: int = 1000) -> List[Dict[str, Any]]:
        """Generate large document set for performance testing."""
        companies = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NFLX", "NVDA"]

        documents = []
        for i in range(num_documents):
            company = random.choice(companies)
            documents.append(
                {
                    "content": f"Financial document {i} for {company}. " * 50,  # Large document
                    "metadata": {"company": company, "document_id": f"doc_{i}", "size": "large"},
                }
            )

        return documents

    @staticmethod
    def generate_concurrent_queries(num_queries: int = 100) -> List[str]:
        """Generate queries for concurrent testing."""
        base_queries = [
            "financial performance",
            "revenue growth",
            "profit margins",
            "debt levels",
            "cash flow",
            "market share",
            "competitive analysis",
            "risk factors",
            "investment thesis",
            "valuation metrics",
        ]

        queries = []
        for _ in range(num_queries):
            base = random.choice(base_queries)
            company = random.choice(["AAPL", "GOOGL", "MSFT", "AMZN"])
            queries.append(f"What is {company}'s {base}?")

        return queries
