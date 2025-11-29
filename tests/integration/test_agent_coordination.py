import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import json

from src.agents.coordinator import AgentCoordinator
from src.agents.research_analyst import ResearchAnalystAgent
from src.agents.quantitative_analyst import QuantitativeAnalystAgent
from src.agents.risk_officer import RiskOfficerAgent


class TestAgentIntegration:
    """Integration tests for agent coordination."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_agent_analysis_integration(self, sample_company_data):
        """Test integrated multi-agent analysis workflow."""
        coordinator = AgentCoordinator()

        # Mock individual agent responses
        with patch.object(ResearchAnalystAgent, "analyze_fundamentals") as mock_research:
            with patch.object(QuantitativeAnalystAgent, "calculate_financial_ratios") as mock_quant:
                with patch.object(RiskOfficerAgent, "assess_risk_factors") as mock_risk:

                    mock_research.return_value = {
                        "analysis": "Strong fundamentals with growth potential",
                        "recommendation": "BUY",
                        "confidence": 0.85,
                    }

                    mock_quant.return_value = {
                        "pe_ratio": 25.0,
                        "roe": 0.15,
                        "debt_to_equity": 0.3,
                        "recommendation": "BUY",
                        "confidence": 0.78,
                    }

                    mock_risk.return_value = {
                        "overall_risk": 0.3,
                        "risk_factors": ["market_volatility", "competition"],
                        "recommendation": "HOLD",
                        "confidence": 0.65,
                    }

                    result = await coordinator.orchestrate_analysis(
                        company_data=sample_company_data, analysis_type="comprehensive"
                    )

                    assert "consensus" in result
                    assert "final_recommendation" in result
                    assert "confidence_score" in result
                    assert len(result["agent_responses"]) == 3

                    # Verify all agents were called
                    mock_research.assert_called_once()
                    mock_quant.assert_called_once()
                    mock_risk.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_conflict_resolution(self):
        """Test conflict resolution between disagreeing agents."""
        coordinator = AgentCoordinator()

        conflicting_responses = [
            {"recommendation": "BUY", "confidence": 0.9, "reasoning": "Strong growth"},
            {"recommendation": "SELL", "confidence": 0.8, "reasoning": "Overvalued"},
            {"recommendation": "HOLD", "confidence": 0.7, "reasoning": "Market uncertainty"},
        ]

        with patch(
            "src.agents.coordinator.AgentCoordinator._get_consensus_analysis"
        ) as mock_consensus:
            mock_consensus.return_value = {
                "final_decision": "HOLD",
                "resolution_reasoning": "Balanced risk-reward profile",
                "weighted_votes": {"BUY": 0.3, "SELL": 0.4, "HOLD": 0.3},
            }

            resolution = await coordinator.resolve_conflicts(conflicting_responses)

            assert resolution["final_decision"] in ["BUY", "SELL", "HOLD"]
            assert "resolution_reasoning" in resolution
            assert "weighted_votes" in resolution

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_time_data_integration(self, mock_yahoo_finance, mock_sec_edgar):
        """Test integration with real-time data sources."""
        coordinator = AgentCoordinator()

        # This test uses actual mocks to simulate real API calls
        result = await coordinator.orchestrate_analysis(
            company_data={"symbol": "AAPL", "name": "Apple Inc."}, analysis_type="real_time"
        )

        assert result is not None
        assert "final_recommendation" in result
        assert "data_sources" in result


class TestRAGIntegration:
    """Integration tests for RAG system."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_pipeline_integration(self, mock_vector_store, mock_openai):
        """Test complete RAG pipeline integration."""
        from src.rag.engine import RAGEngine

        rag_engine = RAGEngine(vector_store=mock_vector_store)

        # Test full RAG workflow
        query = "What is Apple's current debt-to-equity ratio?"

        # Mock document retrieval
        mock_vector_store.similarity_search.return_value = [
            MagicMock(
                page_content="Apple's debt-to-equity ratio is 1.5 for FY2023",
                metadata={"source": "10-K", "company": "AAPL"},
            )
        ]

        response = await rag_engine.process_query(query)

        assert "answer" in response
        assert "sources" in response
        assert "confidence" in response
        assert "debt-to-equity" in response["answer"].lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self):
        """Test document processing pipeline integration."""
        from src.rag.document_processor import DocumentProcessor
        from src.data.sec_edgar_client import SECEdgarClient

        processor = DocumentProcessor()
        sec_client = SECEdgarClient()

        # Mock SEC filing
        with patch.object(SECEdgarClient, "get_filing") as mock_filing:
            mock_filing.return_value = {
                "content": "ITEM 7. MD&A\nRevenue growth was 15%...\nITEM 8. Financial Statements\nTotal assets: $100M",
                "filing_type": "10-K",
                "company": "TEST CORP",
            }

            filing = await sec_client.get_filing("000001", "10-K", 2023)
            chunks = processor.chunk_documents([filing["content"]])

            assert len(chunks) > 0
            assert all(hasattr(chunk, "page_content") for chunk in chunks)
            assert all(hasattr(chunk, "metadata") for chunk in chunks)
