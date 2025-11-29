#!/usr/bin/env python3
"""
Test script for Predictive Analytics & Forecasting Features
"""

import sys
import os
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from financial_rag.agents.predictive_analyst import PredictiveAnalystAgent
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.config import config


async def test_predictive_analytics():
    print("🎯 Testing Predictive Analytics & Forecasting...")

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

        # Initialize predictive analytics agent
        print("1. Initializing Predictive Analytics Agent...")
        predictive_agent = PredictiveAnalystAgent(vector_store)
        print("   ✅ Predictive analytics agent initialized")

        # Test trend analysis
        print("2. Testing trend analysis...")
        trend_result = await predictive_agent.time_series_analyzer.analyze_stock_trends(
            "AAPL", "1y"
        )
        print(f"   ✅ Trend analysis completed")
        print(
            f"      Consensus: {trend_result['trend_analysis']['consensus']['direction']}"
        )
        print(
            f"      Confidence: {trend_result['trend_analysis']['consensus']['confidence']:.0%}"
        )

        # Test momentum analysis
        print("3. Testing momentum analysis...")
        momentum_result = await predictive_agent.analyze_momentum("AAPL")
        print(f"   ✅ Momentum analysis completed")
        print(f"      Momentum score: {momentum_result['momentum_score']:.2f}")
        print(f"      Strength: {momentum_result['momentum_strength']}")

        # Test technical outlook
        print("4. Testing technical outlook...")
        technical_result = await predictive_agent.assess_technical_outlook("AAPL")
        print(f"   ✅ Technical outlook completed")
        print(f"      Technical bias: {technical_result['technical_bias']}")
        print(f"      Outlook score: {technical_result['outlook_score']:.2f}")

        # Test comprehensive predictive analysis
        print("5. Testing comprehensive predictive analysis...")
        predictive_result = await predictive_agent.predictive_analysis("AAPL", "30d")

        print(f"   ✅ Predictive analysis completed")
        print(f"      Composite score: {predictive_result['composite_score']:.2f}")
        print(f"      Overall bias: {predictive_result['composite_score']}")
        print(
            f"      Confidence: {predictive_result['price_forecast'].get('ensemble', {}).get('confidence_level', 'N/A')}"
        )

        # Test earnings prediction
        print("6. Testing earnings prediction...")
        try:
            earnings_result = await predictive_agent.forecaster.predict_earnings("AAPL")
            print(f"   ✅ Earnings prediction completed")
            print(f"      Predicted EPS: {earnings_result['predicted_eps']}")
            print(
                f"      Surprise probability: {earnings_result['surprise_probability']:.0%}"
            )
        except Exception as e:
            print(f"   ⚠️  Earnings prediction skipped: {e}")

        # Test forecasting models
        print("7. Testing forecasting models...")
        models = list(predictive_agent.forecaster.model_registry.keys())
        print(f"   ✅ Available models: {models}")

        print("\n🎉 Predictive analytics test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Predictive analytics test failed: {e}")
        return False


def setup_mock_knowledge_base(doc_processor, vector_manager):
    """Setup mock knowledge base for testing"""
    mock_docs = [
        {
            "content": """Apple Inc. demonstrates strong financial performance with consistent innovation.
        The company's ecosystem strategy continues to drive revenue growth and customer loyalty.
        Key risk factors include supply chain dependencies and intense competition.""",
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
    success = asyncio.run(test_predictive_analytics())
    sys.exit(0 if success else 1)
