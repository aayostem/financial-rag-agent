#!/usr/bin/env python3
"""
Production health check and test script
"""

import requests
import json
import sys
import time


def test_production_endpoints(base_url: str = "http://localhost:8000"):
    """Test all production endpoints"""

    print("🧪 Testing Production API Endpoints...")

    endpoints = [
        ("GET", "/health", None),
        ("GET", "/system/stats", None),
        (
            "POST",
            "/query",
            {
                "question": "What is the current stock price of Apple?",
                "use_agent": True,
                "analysis_style": "analyst",
            },
        ),
    ]

    for method, path, data in endpoints:
        try:
            url = f"{base_url}{path}"
            print(f"\nTesting {method} {path}...")

            if method == "GET":
                response = requests.get(url, timeout=30)
            else:
                response = requests.post(url, json=data, timeout=30)

            if response.status_code == 200:
                print(f"✅ SUCCESS: {response.status_code}")
                if path == "/health":
                    health_data = response.json()
                    print(f"   Status: {health_data['status']}")
                    print(f"   Vector Store: {health_data['vector_store_ready']}")
                    print(f"   LLM: {health_data['llm_ready']}")
            else:
                print(f"❌ FAILED: {response.status_code}")
                print(f"   Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: {e}")
            return False

    print("\n🎉 All production tests completed!")
    return True


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = test_production_endpoints(base_url)
    sys.exit(0 if success else 1)
