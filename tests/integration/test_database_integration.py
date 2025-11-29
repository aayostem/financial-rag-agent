import pytest
import asyncio
from sqlalchemy import text
from datetime import datetime, timedelta

from src.database.models import AnalysisResult, Company, FinancialData
from src.database.session import get_db
from src.services.analysis_service import AnalysisService


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.integration
    @pytest.mark.database
    def test_database_connection(self, test_session):
        """Test database connection and basic operations."""
        # Test connection by executing a simple query
        result = test_session.execute(text("SELECT 1"))
        assert result.scalar() == 1

    @pytest.mark.integration
    @pytest.mark.database
    def test_analysis_result_crud(self, test_session):
        """Test CRUD operations for AnalysisResult model."""
        # Create
        analysis = AnalysisResult(
            company_symbol="AAPL",
            analysis_type="fundamental",
            result_data={"recommendation": "BUY", "confidence": 0.85},
            created_at=datetime.utcnow(),
        )

        test_session.add(analysis)
        test_session.commit()

        # Read
        saved_analysis = test_session.query(AnalysisResult).filter_by(company_symbol="AAPL").first()

        assert saved_analysis is not None
        assert saved_analysis.result_data["recommendation"] == "BUY"

        # Update
        saved_analysis.result_data["confidence"] = 0.90
        test_session.commit()

        updated_analysis = (
            test_session.query(AnalysisResult).filter_by(company_symbol="AAPL").first()
        )

        assert updated_analysis.result_data["confidence"] == 0.90

        # Delete
        test_session.delete(updated_analysis)
        test_session.commit()

        deleted_analysis = (
            test_session.query(AnalysisResult).filter_by(company_symbol="AAPL").first()
        )

        assert deleted_analysis is None

    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_analysis_service_integration(self, test_session):
        """Test AnalysisService with real database."""
        analysis_service = AnalysisService(db=test_session)

        # Store analysis result
        analysis_data = {
            "recommendation": "BUY",
            "confidence": 0.88,
            "price_target": 180.0,
            "risk_level": "MEDIUM",
        }

        result_id = await analysis_service.store_analysis_result(
            company_symbol="AAPL", analysis_type="comprehensive", result_data=analysis_data
        )

        assert result_id is not None

        # Retrieve analysis result
        retrieved = await analysis_service.get_analysis_result(result_id)

        assert retrieved.company_symbol == "AAPL"
        assert retrieved.result_data["recommendation"] == "BUY"
        assert retrieved.result_data["confidence"] == 0.88

    @pytest.mark.integration
    @pytest.mark.database
    def test_financial_data_relationships(self, test_session):
        """Test relationships between Company and FinancialData models."""
        # Create company
        company = Company(
            symbol="AAPL", name="Apple Inc.", sector="Technology", industry="Consumer Electronics"
        )
        test_session.add(company)
        test_session.commit()

        # Create financial data
        financial_data = FinancialData(
            company_symbol="AAPL",
            period="Q4-2023",
            data_type="income_statement",
            data={"revenue": 1000000000, "net_income": 250000000},
            reported_date=datetime.utcnow(),
        )
        test_session.add(financial_data)
        test_session.commit()

        # Test relationship
        company_with_data = test_session.query(Company).filter_by(symbol="AAPL").first()
        assert len(company_with_data.financial_data) == 1
        assert company_with_data.financial_data[0].data_type == "income_statement"

    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, test_session):
        """Test concurrent database operations."""
        import asyncio

        analysis_service = AnalysisService(db=test_session)

        async def store_analysis(symbol, analysis_type):
            return await analysis_service.store_analysis_result(
                company_symbol=symbol,
                analysis_type=analysis_type,
                result_data={"recommendation": "BUY", "confidence": 0.8},
            )

        # Run concurrent operations
        tasks = [
            store_analysis("AAPL", "fundamental"),
            store_analysis("GOOGL", "technical"),
            store_analysis("MSFT", "risk"),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(result is not None for result in results)

        # Verify all records were created
        analysis_count = test_session.query(AnalysisResult).count()
        assert analysis_count == 3
