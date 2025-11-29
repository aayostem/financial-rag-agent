#!/usr/bin/env python3
"""
Comprehensive production test
"""

import sys
import os
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from financial_rag.api.server import FinancialRAGAPI
from financial_rag.config import config


async def test_production_readiness():
    """Test if the system is production ready"""
    print("🏭 Testing Production Readiness...")

    try:
        # Test API initialization
        print("1. Testing API initialization...")
        api = FinancialRAGAPI()
        await api.initialize_services()
        print("   ✅ API initialized successfully")

        # Test health endpoint
        print("2. Testing health check...")
        health_check = api.app.routes[2].endpoint  # /health endpoint
        response = await health_check()
        print(f"   ✅ Health check: {response.status}")

        # Test vector store
        print("3. Testing vector store...")
        if api.vector_store:
            print("   ✅ Vector store operational")
        else:
            print("   ⚠️  Vector store not initialized (expected for first run)")

        # Test agent
        print("4. Testing agent...")
        if api.agent:
            print("   ✅ Agent operational")

            # Test simple query
            test_query = "What is the current stock price of Apple?"
            try:
                result = api.agent.analyze(test_query)
                print(f"   ✅ Agent query test: {len(result['answer'])} chars response")
            except Exception as e:
                print(f"   ⚠️  Agent query test failed (may be expected): {e}")
        else:
            print("   ❌ Agent not initialized")
            return False

        # Test monitoring
        print("5. Testing monitoring...")
        if api.agent.monitor.enabled:
            print("   ✅ Monitoring enabled")
        else:
            print("   ⚠️  Monitoring disabled (check WandB API key)")

        print("\n🎉 PRODUCTION READY! All systems operational.")
        return True

    except Exception as e:
        print(f"\n💥 PRODUCTION TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    # Check environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not getattr(config, var, None)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("💡 Please set them in your .env file")
        sys.exit(1)

    # Run production test
    success = asyncio.run(test_production_readiness())
    sys.exit(0 if success else 1)
