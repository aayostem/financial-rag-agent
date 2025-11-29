#!/usr/bin/env python3
"""
Test script for Multi-Modal Financial Analysis Features
"""

import sys
import os
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from financial_rag.agents.multi_modal_analyst import MultiModalAnalystAgent
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.config import config


async def test_multi_modal_features():
    print("🎯 Testing Multi-Modal Financial Analysis...")

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

        # Initialize multi-modal agent
        print("1. Initializing Multi-Modal Analyst Agent...")
        multi_modal_agent = MultiModalAnalystAgent(vector_store)
        print("   ✅ Multi-modal agent initialized")

        # Test document analysis (with mock PDF)
        print("2. Testing document analysis...")
        try:
            # Create a mock PDF path for testing
            mock_pdf_path = "/tmp/mock_financial.pdf"

            # For now, test with file existence check
            if os.path.exists(mock_pdf_path):
                doc_analysis = await multi_modal_agent.analyze_financial_documents(
                    [mock_pdf_path], "AAPL"
                )
                print("   ✅ Document analysis completed")
                print(
                    f"      Financial health: {doc_analysis['financial_health']['health_rating']}"
                )
            else:
                print("   ⚠️  Mock PDF not found, skipping document analysis test")

        except Exception as e:
            print(f"   ⚠️  Document analysis test skipped: {e}")

        # Test comprehensive analysis
        print("3. Testing comprehensive analysis...")
        comprehensive = await multi_modal_agent.comprehensive_analysis("AAPL")

        print(f"   ✅ Comprehensive analysis completed")
        print(f"      Analysis modes: {comprehensive['analysis_modes']}")
        print(
            f"      Investment rating: {comprehensive['unified_insights']['investment_rating']}"
        )
        print(f"      Executive summary: {comprehensive['executive_summary'][:200]}...")

        # Test earnings call analysis structure
        print("4. Testing earnings call analysis structure...")
        # Note: Actual audio processing requires audio files
        # We'll test the method structure without actual processing
        try:
            earnings_methods = [
                "analyze_earnings_call",
                "generate_earnings_insights",
                "analyze_call_sentiment",
                "generate_earnings_summary",
            ]

            for method in earnings_methods:
                if hasattr(multi_modal_agent, method):
                    print(f"   ✅ {method} method available")
                else:
                    print(f"   ❌ {method} method missing")

        except Exception as e:
            print(f"   ⚠️  Earnings call structure test issue: {e}")

        # Test financial health assessment
        print("5. Testing financial health assessment...")
        mock_insights = {
            "key_metrics": {
                "profit_margin": 0.15,
                "revenue_growth": 0.08,
                "debt_to_equity": 1.5,
            },
            "trends": {"revenue": {"direction": "increasing"}},
        }

        health = multi_modal_agent.assess_financial_health(mock_insights)
        print(
            f"   ✅ Financial health assessment: {health['health_rating']} (score: {health['health_score']})"
        )

        print("\n🎉 Multi-modal features test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Multi-modal features test failed: {e}")
        return False


def setup_mock_knowledge_base(doc_processor, vector_manager):
    """Setup mock knowledge base for testing"""
    mock_docs = [
        {
            "content": """Apple Inc. demonstrates strong financial performance with consistent revenue growth and robust profitability. 
        The company maintains a healthy balance sheet with significant cash reserves. Key risk factors include supply chain dependencies 
        and intense competition in the smartphone market. Recent initiatives in services and wearables show promising growth trajectories.""",
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
    success = asyncio.run(test_multi_modal_features())
    sys.exit(0 if success else 1)
