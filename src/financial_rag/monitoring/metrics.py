from prometheus_client import Counter, Histogram, Gauge, generate_latest
from loguru import logger
import time

# Metrics for Prometheus
QUERY_COUNTER = Counter(
    "financial_rag_queries_total", "Total number of queries", ["status", "agent_type"]
)
QUERY_DURATION = Histogram(
    "financial_rag_query_duration_seconds", "Query duration in seconds"
)
AGENT_TOOL_USAGE = Counter(
    "financial_rag_agent_tool_usage_total", "Agent tool usage", ["tool_name", "status"]
)
VECTOR_STORE_SIZE = Gauge(
    "financial_rag_vector_store_documents", "Number of documents in vector store"
)
LLM_TOKEN_USAGE = Counter("financial_rag_llm_tokens_total", "LLM token usage", ["type"])


class MetricsCollector:
    """Collect and expose metrics for Prometheus"""

    def __init__(self):
        self.metrics_registry = {}

    def record_query(self, status: str, agent_type: str, duration: float):
        """Record query metrics"""
        QUERY_COUNTER.labels(status=status, agent_type=agent_type).inc()
        QUERY_DURATION.observe(duration)

    def record_tool_usage(self, tool_name: str, success: bool):
        """Record agent tool usage"""
        status = "success" if success else "failure"
        AGENT_TOOL_USAGE.labels(tool_name=tool_name, status=status).inc()

    def record_token_usage(self, token_type: str, count: int):
        """Record LLM token usage"""
        LLM_TOKEN_USAGE.labels(type=token_type).inc(count)

    def update_vector_store_size(self, size: int):
        """Update vector store document count"""
        VECTOR_STORE_SIZE.set(size)

    def get_metrics(self):
        """Get all metrics in Prometheus format"""
        return generate_latest()


# Global metrics collector
metrics_collector = MetricsCollector()
