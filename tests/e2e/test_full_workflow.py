import pytest
import asyncio
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

from src.api.main import app
from fastapi.testclient import TestClient


class TestFullWorkflow:
    """End-to-end tests for complete workflow scenarios."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_investment_analysis_workflow(self):
        """Test complete investment analysis workflow from API call to result."""
        with TestClient(app) as client:
            # Mock external dependencies
            with patch("src.data.sec_edgar_client.SECEdgarClient.get_filing") as mock_sec:
                with patch(
                    "src.data.yahoo_finance_client.YahooFinanceClient.get_stock_data"
                ) as mock_yahoo:
                    with patch("src.rag.engine.RAGEngine.process_query") as mock_rag:
                        with patch(
                            "src.agents.coordinator.AgentCoordinator.orchestrate_analysis"
                        ) as mock_coordinator:

                            # Setup mocks
                            mock_sec.return_value = {
                                "company": "Apple Inc.",
                                "filing_type": "10-K",
                                "content": "Comprehensive financial data...",
                            }

                            mock_yahoo.return_value = {
                                "symbol": "AAPL",
                                "price": 150.0,
                                "change_percent": 2.5,
                            }

                            mock_rag.return_value = {
                                "answer": "Apple shows strong financial performance...",
                                "sources": ["10-K", "Q1 Earnings"],
                                "confidence": 0.85,
                            }

                            mock_coordinator.return_value = {
                                "consensus": "BUY",
                                "final_recommendation": "BUY",
                                "confidence_score": 0.82,
                                "agent_responses": [
                                    {
                                        "agent": "Research",
                                        "recommendation": "BUY",
                                        "confidence": 0.85,
                                    },
                                    {"agent": "Quant", "recommendation": "BUY", "confidence": 0.78},
                                    {"agent": "Risk", "recommendation": "HOLD", "confidence": 0.65},
                                ],
                            }

                            # Execute complete workflow
                            response = client.post(
                                "/api/v1/analyze/company",
                                json={
                                    "company_symbol": "AAPL",
                                    "analysis_type": "comprehensive",
                                    "include_news": True,
                                    "include_technical": True,
                                },
                            )

                            # Verify response
                            assert response.status_code == 200
                            data = response.json()

                            assert "analysis_id" in data
                            assert "recommendation" in data
                            assert "confidence" in data
                            assert "supporting_data" in data
                            assert "timestamp" in data

                            # Verify all components were called
                            mock_sec.assert_called()
                            mock_yahoo.assert_called()
                            mock_rag.assert_called()
                            mock_coordinator.assert_called()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_risk_assessment_workflow(self):
        """Test complete risk assessment workflow."""
        with TestClient(app) as client:
            with patch("src.agents.risk_officer.RiskOfficerAgent.assess_risk_factors") as mock_risk:
                with patch(
                    "src.data.yahoo_finance_client.YahooFinanceClient.get_historical_data"
                ) as mock_historical:

                    mock_risk.return_value = {
                        "overall_risk": 0.35,
                        "risk_factors": [
                            {"factor": "market_volatility", "score": 0.6},
                            {"factor": "liquidity", "score": 0.3},
                            {"factor": "competition", "score": 0.4},
                        ],
                        "mitigation_strategies": ["Diversify portfolio", "Set stop-loss orders"],
                    }

                    mock_historical.return_value = {
                        "prices": [100, 105, 110, 115, 120],
                        "volumes": [1000000, 1200000, 1100000, 1300000, 1400000],
                        "dates": [
                            "2023-01-01",
                            "2023-01-02",
                            "2023-01-03",
                            "2023-01-04",
                            "2023-01-05",
                        ],
                    }

                    response = client.post(
                        "/api/v1/analyze/risk",
                        json={
                            "company_symbol": "AAPL",
                            "time_period": "1y",
                            "confidence_level": 0.95,
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()

                    assert "risk_assessment" in data
                    assert "var_95" in data
                    assert "stress_test_results" in data
                    assert "recommendations" in data

    @pytest.mark.e2e
    def test_api_error_handling_workflow(self):
        """Test error handling throughout the workflow."""
        with TestClient(app) as client:
            # Test with invalid company symbol
            response = client.post(
                "/api/v1/analyze/company",
                json={"company_symbol": "INVALID_SYMBOL_123", "analysis_type": "comprehensive"},
            )

            # Should handle gracefully, not crash
            assert response.status_code in [400, 404, 422]

            # Test with missing required fields
            response = client.post(
                "/api/v1/analyze/company",
                json={
                    "analysis_type": "comprehensive"
                    # Missing company_symbol
                },
            )

            assert response.status_code == 422  # Validation error

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_data_persistence_workflow(self, test_session):
        """Test complete workflow with data persistence."""
        with TestClient(app) as client:
            with patch("src.database.session.get_db") as mock_get_db:
                mock_get_db.return_value = test_session

                response = client.post(
                    "/api/v1/analyze/company",
                    json={"company_symbol": "TEST", "analysis_type": "quick", "store_result": True},
                )

                assert response.status_code == 200
                data = response.json()

                # Verify analysis was stored in database
                from src.database.models import AnalysisResult

                stored_analysis = (
                    test_session.query(AnalysisResult).filter_by(company_symbol="TEST").first()
                )

                assert stored_analysis is not None
                assert stored_analysis.analysis_type == "quick"
                assert "recommendation" in stored_analysis.result_data


class TestUserScenarios:
    """End-to-end tests for specific user scenarios."""

    @pytest.mark.e2e
    def test_investment_researcher_scenario(self):
        """Test scenario for an investment researcher."""
        with TestClient(app) as client:
            # Researcher wants comprehensive analysis of multiple companies
            companies = ["AAPL", "GOOGL", "MSFT"]

            for symbol in companies:
                response = client.post(
                    "/api/v1/analyze/company",
                    json={
                        "company_symbol": symbol,
                        "analysis_type": "comprehensive",
                        "include_news": True,
                        "include_technical": True,
                        "include_forecasting": True,
                    },
                )

                assert response.status_code == 200
                data = response.json()

                # Verify research-grade output
                assert "detailed_analysis" in data
                assert "financial_metrics" in data
                assert "growth_projections" in data
                assert "risk_assessment" in data
                assert "investment_recommendation" in data

    @pytest.mark.e2e
    def test_risk_manager_scenario(self):
        """Test scenario for a risk manager."""
        with TestClient(app) as client:
            # Risk manager wants portfolio risk analysis
            response = client.post(
                "/api/v1/analyze/portfolio-risk",
                json={
                    "portfolio": [
                        {"symbol": "AAPL", "weight": 0.4},
                        {"symbol": "GOOGL", "weight": 0.3},
                        {"symbol": "MSFT", "weight": 0.3},
                    ],
                    "time_horizon": "1y",
                    "confidence_level": 0.99,
                    "stress_scenarios": ["market_crash", "interest_rate_hike"],
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Verify risk management output
            assert "portfolio_var" in data
            assert "component_risks" in data
            assert "stress_test_results" in data
            assert "correlation_matrix" in data
            assert "risk_mitigation_strategies" in data
