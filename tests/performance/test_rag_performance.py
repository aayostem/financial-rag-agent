import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import MagicMock, patch
import statistics

from src.rag.engine import RAGEngine
from src.agents.coordinator import AgentCoordinator


class TestRAGPerformance:
    """Performance tests for RAG system."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_rag_query_response_time(self, mock_vector_store, mock_openai):
        """Test RAG query response time under normal load."""
        rag_engine = RAGEngine(vector_store=mock_vector_store)

        query = "What are the key financial metrics for technology companies?"

        response_times = []
        for i in range(10):  # Run 10 queries
            start_time = time.time()

            await rag_engine.process_query(query)

            end_time = time.time()
            response_times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)

        print(f"Average response time: {avg_response_time:.2f}ms")
        print(f"Max response time: {max_response_time:.2f}ms")

        # Assert performance requirements
        assert avg_response_time < 2000  # 2 seconds average
        assert max_response_time < 5000  # 5 seconds max

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_rag_queries(self, mock_vector_store, mock_openai):
        """Test RAG performance under concurrent load."""
        rag_engine = RAGEngine(vector_store=mock_vector_store)

        queries = [
            "What is the current P/E ratio?",
            "Analyze the debt-to-equity ratio",
            "What are the revenue growth trends?",
            "Explain the cash flow situation",
            "What is the return on equity?",
        ] * 2  # 10 total queries

        async def process_single_query(query):
            start_time = time.time()
            await rag_engine.process_query(query)
            return (time.time() - start_time) * 1000

        # Run queries concurrently
        tasks = [process_single_query(query) for query in queries]
        response_times = await asyncio.gather(*tasks)

        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile

        print(f"Concurrent average response time: {avg_response_time:.2f}ms")
        print(f"95th percentile response time: {p95_response_time:.2f}ms")

        assert avg_response_time < 3000  # 3 seconds average under load
        assert p95_response_time < 6000  # 6 seconds for 95% of requests

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_document_retrieval_performance(self, mock_vector_store):
        """Test document retrieval performance with large datasets."""
        rag_engine = RAGEngine(vector_store=mock_vector_store)

        # Mock a large number of documents
        large_document_set = [
            MagicMock(
                page_content=f"Financial document {i} with important data",
                metadata={"source": "10-K", "company": f"COMP{i}"},
            )
            for i in range(1000)
        ]

        mock_vector_store.similarity_search.return_value = large_document_set[:5]

        start_time = time.time()
        documents = await rag_engine.retrieve_documents(query="financial analysis", top_k=5)
        retrieval_time = (time.time() - start_time) * 1000

        print(f"Document retrieval time: {retrieval_time:.2f}ms")

        assert retrieval_time < 1000  # 1 second for retrieval
        assert len(documents) == 5

    @pytest.mark.performance
    def test_memory_usage_during_rag_processing(self):
        """Test memory usage during RAG processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate memory-intensive RAG processing
        large_documents = [
            f"Large financial document {i} with extensive content " * 1000 for i in range(100)
        ]

        # Process documents (simulated)
        processed_docs = []
        for doc in large_documents:
            processed_docs.append(doc.upper())  # Simulate processing

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory increase: {memory_increase:.2f}MB")

        assert memory_increase < 500  # Should not increase more than 500MB


class TestAgentPerformance:
    """Performance tests for AI agents."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multi_agent_analysis_performance(self, sample_company_data):
        """Test performance of multi-agent analysis."""
        coordinator = AgentCoordinator()

        analysis_times = []
        for i in range(5):  # Run 5 analyses
            start_time = time.time()

            await coordinator.orchestrate_analysis(
                company_data=sample_company_data, analysis_type="comprehensive"
            )

            analysis_time = (time.time() - start_time) * 1000
            analysis_times.append(analysis_time)

        avg_analysis_time = statistics.mean(analysis_times)
        max_analysis_time = max(analysis_times)

        print(f"Average multi-agent analysis time: {avg_analysis_time:.2f}ms")
        print(f"Max multi-agent analysis time: {max_analysis_time:.2f}ms")

        assert avg_analysis_time < 30000  # 30 seconds average
        assert max_analysis_time < 60000  # 60 seconds max

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_agent_concurrent_requests(self, sample_company_data):
        """Test agent performance under concurrent requests."""
        coordinator = AgentCoordinator()

        async def run_single_analysis():
            start_time = time.time()
            await coordinator.orchestrate_analysis(
                company_data=sample_company_data, analysis_type="quick"
            )
            return (time.time() - start_time) * 1000

        # Run multiple analyses concurrently
        tasks = [run_single_analysis() for _ in range(3)]
        response_times = await asyncio.gather(*tasks)

        avg_response_time = statistics.mean(response_times)

        print(f"Concurrent agent analysis average time: {avg_response_time:.2f}ms")

        assert avg_response_time < 45000  # 45 seconds average under concurrent load

    @pytest.mark.performance
    def test_agent_memory_efficiency(self):
        """Test memory efficiency of agent operations."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate multiple agent initializations
        agents = []
        for i in range(10):
            from src.agents.research_analyst import ResearchAnalystAgent

            agent = ResearchAnalystAgent()
            agents.append(agent)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory increase for 10 agents: {memory_increase:.2f}MB")

        assert memory_increase < 100  # Should not increase more than 100MB
