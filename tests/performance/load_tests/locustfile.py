from locust import HttpUser, task, between, TaskSet
import json
import random


class FinancialRAGUser(HttpUser):
    wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.companies = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NFLX", "NVDA"]
        self.analysis_types = ["quick", "comprehensive", "risk", "technical"]

    @task(3)
    def test_health_check(self):
        """Test health endpoint."""
        self.client.get("/health")

    @task(5)
    def test_company_analysis(self):
        """Test company analysis endpoint."""
        company = random.choice(self.companies)
        payload = {
            "company_symbol": company,
            "analysis_type": random.choice(self.analysis_types),
            "include_news": True,
            "include_technical": False,
        }

        with self.client.post(
            "/api/v1/analyze/company", json=payload, catch_response=True, name="Company Analysis"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def test_rag_query(self):
        """Test RAG query endpoint."""
        query = random.choice(
            [
                "What is the current P/E ratio?",
                "Analyze debt levels",
                "Revenue growth trends",
                "Cash flow analysis",
                "Profit margins",
            ]
        )

        payload = {"query": query, "company_filter": random.choice(self.companies), "top_k": 5}

        with self.client.post(
            "/api/v1/rag/query", json=payload, catch_response=True, name="RAG Query"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" in data and "sources" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def test_multi_agent_analysis(self):
        """Test multi-agent analysis endpoint (more resource intensive)."""
        payload = {
            "company_symbol": random.choice(self.companies),
            "analysis_types": ["fundamental", "technical", "risk"],
            "depth": "deep",
        }

        with self.client.post(
            "/api/v1/analyze/multi-agent",
            json=payload,
            catch_response=True,
            name="Multi-Agent Analysis",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "consensus" in data and "agent_responses" in data:
                    response.success()
                else:
                    response.failure("Invalid multi-agent response")
            else:
                response.failure(f"Status code: {response.status_code}")


class HighLoadUser(TaskSet):
    """High load user performing intensive operations."""

    @task
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis with all features."""
        payload = {
            "company_symbol": random.choice(["AAPL", "GOOGL", "MSFT"]),
            "analysis_type": "comprehensive",
            "include_news": True,
            "include_technical": True,
            "include_forecasting": True,
            "time_horizon": "1y",
        }

        self.client.post("/api/v1/analyze/company", json=payload)

    @task
    def test_batch_analysis(self):
        """Test batch analysis of multiple companies."""
        payload = {
            "companies": ["AAPL", "GOOGL", "MSFT", "AMZN"],
            "analysis_type": "quick",
            "parallel_processing": True,
        }

        self.client.post("/api/v1/analyze/batch", json=payload)


class SpikeUser(HttpUser):
    """User for testing sudden traffic spikes."""

    tasks = [HighLoadUser]
    wait_time = between(0.1, 0.5)  # Very aggressive
