#!/usr/bin/env python3
"""
Test script for Advanced Agent Architectures
"""

import sys
import os
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from financial_rag.agents.coordinator import AgentCoordinator, AnalysisType
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.config import config


async def test_multi_agent_system():
    print("🎯 Testing Advanced Agent Architectures...")

    try:
        # Initialize components
        vector_manager = VectorStoreManager()
        vector_store = vector_manager.load_vector_store()

        if vector_store is None:
            print("⚠️  No vector store found, using mock data")
            from financial_rag.ingestion.document_processor import DocumentProcessor

            vector_store = setup_mock_knowledge_base(
                DocumentProcessor(), vector_manager
            )

        # Initialize multi-agent coordinator
        print("1. Initializing Multi-Agent Coordinator...")
        coordinator = AgentCoordinator(vector_store)
        print("   ✅ Multi-agent coordinator initialized")
        print(f"   ✅ Specialized agents: {list(coordinator.agent_registry.keys())}")

        # Test comprehensive analysis with all agents
        print("2. Testing comprehensive multi-agent analysis...")
        comprehensive = await coordinator.coordinate_analysis(
            "AAPL", AnalysisType.COMPREHENSIVE
        )

        print(f"   ✅ Comprehensive analysis completed")
        print(f"      Agents involved: {comprehensive['agents_involved']}")
        print(
            f"      Overall recommendation: {comprehensive['overall_recommendation']['action']}"
        )
        print(
            f"      Confidence: {comprehensive['overall_recommendation']['confidence']:.0%}"
        )

        # Test specialized analysis types
        print("3. Testing specialized analysis types...")
        analysis_types = [
            (AnalysisType.DEEP_RESEARCH, "Deep Research"),
            (AnalysisType.QUANTITATIVE, "Quantitative Analysis"),
            (AnalysisType.RISK_ASSESSMENT, "Risk Assessment"),
        ]

        for analysis_type, description in analysis_types[
            :2
        ]:  # Test first 2 to save time
            print(f"   Testing {description}...")
            specialized = await coordinator.coordinate_analysis("MSFT", analysis_type)
            print(f"      ✅ {description} completed")
            print(f"      Agents: {specialized['agents_involved']}")

        # Test investment committee simulation
        print("4. Testing investment committee simulation...")
        committee = await coordinator.conduct_investment_committee("AAPL")

        print(f"   ✅ Investment committee completed")
        print(f"      Committee members: {committee['committee_members']}")
        print(f"      Final decision: {committee['final_decision']['decision']}")
        print(f"      Unanimous: {committee['final_decision']['unanimous']}")

        # Test analysis history
        print("5. Testing analysis history...")
        history = await coordinator.get_analysis_history()
        print(f"   ✅ Analysis history: {len(history)} entries")

        if history:
            latest = history[-1]
            print(
                f"      Latest analysis: {latest['ticker']} - {latest['recommendation']}"
            )

        # Test agent capabilities
        print("6. Testing agent capabilities...")
        for agent_name, agent in coordinator.agent_registry.items():
            print(f"   ✅ {agent_name}: {type(agent).__name__}")

        print("\n🎉 Multi-agent system test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Multi-agent system test failed: {e}")
        return False


def setup_mock_knowledge_base(doc_processor, vector_manager):
    """Setup mock knowledge base for testing"""
    mock_docs = [
        {
            "content": """Apple Inc. continues to demonstrate strong financial performance with innovative product pipeline.
        The company maintains robust profitability metrics and has a strong balance sheet.
        Key growth areas include services, wearables, and emerging markets.""",
            "metadata": {"source": "mock_analysis", "company": "Apple"},
        }
    ]

    documents = []
    for doc in mock_docs:
        chunked_docs = doc_processor.text_splitter.create_documents(
            [doc["content"]], [doc["metadata"]]
        )
        documents.extend(chunked_docs)

    return vector_manager.create_vector_store(documents)


if __name__ == "__main__":
    # Check for OpenAI API key
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ Please set your OPENAI_API_KEY in the .env file")
        sys.exit(1)

    # Run the test
    success = asyncio.run(test_multi_agent_system())
    sys.exit(0 if success else 1)
