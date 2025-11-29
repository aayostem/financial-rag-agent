#!/usr/bin/env python3
"""
Test script for Real-Time Market Intelligence Features
"""

import sys
import os
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from financial_rag.agents.real_time_analyst import RealTimeAnalystAgent
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.config import config


async def test_real_time_features():
    print("🎯 Testing Real-Time Market Intelligence...")

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

        # Initialize real-time agent
        print("1. Initializing Real-Time Analyst Agent...")
        real_time_agent = RealTimeAnalystAgent(vector_store)
        print("   ✅ Real-time agent initialized")

        # Test real-time market data
        print("2. Testing real-time market data...")
        market_data = await real_time_agent.real_time_data.get_live_market_data(
            ["AAPL", "MSFT"]
        )
        print(f"   ✅ Real-time data retrieved for {len(market_data)} tickers")
        for ticker, data in market_data.items():
            print(f"      {ticker}: ${data['price']} ({data['change_pct']:+.2f}%)")

        # Test market summary
        print("3. Testing market summary...")
        market_summary = await real_time_agent.real_time_data.get_market_summary()
        print(f"   ✅ Market summary: {market_summary.get('market_summary', 'N/A')}")

        # Test real-time analysis
        print("4. Testing real-time analysis...")
        test_questions = [
            "What are Apple's main risk factors given current market conditions?",
            "How is Microsoft performing today and what are their growth prospects?",
            "What's the overall market sentiment and how might it affect tech stocks?",
        ]

        for i, question in enumerate(test_questions[:2]):  # Test first 2
            print(f"   Question {i+1}: {question}")
            result = await real_time_agent.analyze_with_market_context(question)

            print(f"      Answer: {result['answer'][:200]}...")
            if result.get("real_time_insights"):
                print(f"      Insights: {result['real_time_insights']}")
            if result.get("alerts"):
                print(f"      Alerts: {len(result['alerts'])} alerts generated")

        # Test alert system
        print("5. Testing alert system...")
        alerts = await real_time_agent.market_alerts.check_alerts(
            ["AAPL", "MSFT"],
            await real_time_agent.get_real_time_context(["AAPL", "MSFT"]),
        )
        print(f"   ✅ Alert system: {len(alerts)} potential alerts")

        print("\n🎉 Real-time features test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Real-time features test failed: {e}")
        return False


def setup_mock_knowledge_base(doc_processor, vector_manager):
    """Setup mock knowledge base for testing"""
    mock_docs = [
        {
            "content": """Apple Inc. is a technology company known for iPhone, iPad, and Mac.
        Recent performance shows strong services growth and continued iPhone demand.
        Risk factors include supply chain dependencies and intense competition.""",
            "metadata": {"source": "mock_apple", "company": "Apple"},
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
    success = asyncio.run(test_real_time_features())
    sys.exit(0 if success else 1)
