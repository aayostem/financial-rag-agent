import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import json

from src.agents.research_analyst import ResearchAnalystAgent
from src.agents.quantitative_analyst import QuantitativeAnalystAgent
from src.agents.risk_officer import RiskOfficerAgent
from src.agents.coordinator import AgentCoordinator


class TestResearchAnalystAgent:
    """Test Research Analyst Agent functionality."""

    @pytest.mark.unit
    def test_agent_initialization(self, mock_openai):
        """Test agent initialization with proper configuration."""
        agent = ResearchAnalystAgent()

        assert agent.name == "Research Analyst"
        assert agent.role == "fundamental_analysis"
        assert agent.temperature == 0.1
        assert agent.max_tokens == 2000

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_fundamentals(self, mock_openai, sample_financial_data):
        """Test fundamental analysis generation."""
        agent = ResearchAnalystAgent()

        result = await agent.analyze_fundamentals(
            company_data=sample_financial_data, market_context={"sector": "Technology"}
        )

        assert "analysis" in result
        assert "recommendations" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_investment_thesis(self, mock_openai, sample_company_data):
        """Test investment thesis generation."""
        agent = ResearchAnalystAgent()

        thesis = await agent.generate_investment_thesis(
            company_data=sample_company_data, historical_data={"revenue_growth": 0.15}
        )

        assert isinstance(thesis, str)
        assert len(thesis) > 0
        assert any(
            keyword in thesis.lower() for keyword in ["growth", "risk", "opportunity", "valuation"]
        )


class TestQuantitativeAnalystAgent:
    """Test Quantitative Analyst Agent functionality."""

    @pytest.mark.unit
    def test_quant_agent_initialization(self):
        """Test quantitative agent initialization."""
        agent = QuantitativeAnalystAgent()

        assert agent.name == "Quantitative Analyst"
        assert agent.role == "statistical_analysis"
        assert hasattr(agent, "calculate_metrics")

    @pytest.mark.unit
    def test_calculate_financial_ratios(self, sample_financial_data):
        """Test financial ratio calculations."""
        agent = QuantitativeAnalystAgent()

        ratios = agent.calculate_financial_ratios(sample_financial_data)

        expected_ratios = [
            "current_ratio",
            "debt_to_equity",
            "return_on_equity",
            "gross_margin",
            "operating_margin",
            "net_margin",
        ]

        for ratio in expected_ratios:
            assert ratio in ratios
            assert isinstance(ratios[ratio], (int, float))

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_time_series(self, mock_openai):
        """Test time series analysis."""
        agent = QuantitativeAnalystAgent()

        time_series_data = {
            "dates": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "prices": [100, 105, 110],
            "volumes": [1000000, 1200000, 1100000],
        }

        analysis = await agent.analyze_time_series(time_series_data)

        assert "trend" in analysis
        assert "volatility" in analysis
        assert "seasonality" in analysis
        assert "predictions" in analysis


class TestRiskOfficerAgent:
    """Test Risk Officer Agent functionality."""

    @pytest.mark.unit
    def test_risk_agent_initialization(self):
        """Test risk officer agent initialization."""
        agent = RiskOfficerAgent()

        assert agent.name == "Risk Officer"
        assert agent.role == "risk_assessment"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_assess_risk_factors(self, mock_openai, sample_financial_data):
        """Test risk factor assessment."""
        agent = RiskOfficerAgent()

        risk_assessment = await agent.assess_risk_factors(
            financial_data=sample_financial_data, market_conditions={"volatility": "high"}
        )

        assert "overall_risk" in risk_assessment
        assert "risk_factors" in risk_assessment
        assert "mitigation_strategies" in risk_assessment
        assert isinstance(risk_assessment["overall_risk"], (int, float))

    @pytest.mark.unit
    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        agent = RiskOfficerAgent()

        returns = [0.01, -0.02, 0.015, -0.01, 0.02]
        var_95 = agent.calculate_var(returns, confidence_level=0.95)

        assert isinstance(var_95, float)
        assert var_95 <= 0  # VaR is typically negative for losses


class TestAgentCoordinator:
    """Test Agent Coordinator functionality."""

    @pytest.mark.unit
    def test_coordinator_initialization(self):
        """Test coordinator initialization with multiple agents."""
        coordinator = AgentCoordinator()

        assert len(coordinator.agents) >= 3
        assert any(agent.name == "Research Analyst" for agent in coordinator.agents)
        assert any(agent.name == "Quantitative Analyst" for agent in coordinator.agents)
        assert any(agent.name == "Risk Officer" for agent in coordinator.agents)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_orchestrate_analysis(self, mock_openai, sample_company_data):
        """Test multi-agent analysis orchestration."""
        coordinator = AgentCoordinator()

        analysis_result = await coordinator.orchestrate_analysis(
            company_data=sample_company_data, analysis_type="comprehensive"
        )

        assert "consensus" in analysis_result
        assert "agent_responses" in analysis_result
        assert "final_recommendation" in analysis_result
        assert "confidence_score" in analysis_result

        assert len(analysis_result["agent_responses"]) == len(coordinator.agents)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_resolve_conflicts(self, mock_openai):
        """Test conflict resolution between agents."""
        coordinator = AgentCoordinator()

        conflicting_responses = [
            {"recommendation": "BUY", "confidence": 0.8},
            {"recommendation": "SELL", "confidence": 0.7},
            {"recommendation": "HOLD", "confidence": 0.6},
        ]

        resolution = await coordinator.resolve_conflicts(conflicting_responses)

        assert "final_decision" in resolution
        assert "resolution_reasoning" in resolution
        assert "weighted_votes" in resolution
