import pytest
import asyncio
from unittest.mock import MagicMock, patch
import numpy as np

from src.rag.engine import RAGEngine
from src.rag.document_processor import DocumentProcessor
from src.rag.vector_store import VectorStoreManager


class TestRAGEngine:
    """Test RAG Engine functionality."""

    @pytest.mark.unit
    def test_rag_engine_initialization(self, mock_vector_store):
        """Test RAG engine initialization."""
        rag_engine = RAGEngine(vector_store=mock_vector_store)

        assert rag_engine.vector_store is not None
        assert hasattr(rag_engine, "retrieve_documents")
        assert hasattr(rag_engine, "generate_response")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_documents(self, mock_vector_store):
        """Test document retrieval functionality."""
        rag_engine = RAGEngine(vector_store=mock_vector_store)

        query = "What are Apple's profit margins?"
        documents = await rag_engine.retrieve_documents(query=query, company_filter="AAPL", top_k=5)

        assert len(documents) > 0
        mock_vector_store.similarity_search.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_openai, mock_vector_store):
        """Test response generation with context."""
        rag_engine = RAGEngine(vector_store=mock_vector_store)

        query = "Analyze financial performance"
        context_docs = [
            MagicMock(page_content="Revenue increased by 15%", metadata={"source": "10-Q"}),
            MagicMock(page_content="Net income margin improved", metadata={"source": "10-K"}),
        ]

        response = await rag_engine.generate_response(query=query, context_documents=context_docs)

        assert "answer" in response
        assert "sources" in response
        assert "confidence" in response
        assert len(response["sources"]) == len(context_docs)

    @pytest.mark.unit
    def test_calculate_similarity_score(self):
        """Test similarity score calculation."""
        rag_engine = RAGEngine(vector_store=mock_vector_store)

        query_embedding = np.random.rand(384)
        doc_embedding = np.random.rand(384)

        similarity = rag_engine._calculate_similarity_score(query_embedding, doc_embedding)

        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1


class TestDocumentProcessor:
    """Test Document Processor functionality."""

    @pytest.mark.unit
    def test_document_processor_initialization(self):
        """Test document processor initialization."""
        processor = DocumentProcessor()

        assert hasattr(processor, "chunk_documents")
        assert hasattr(processor, "extract_metadata")

    @pytest.mark.unit
    def test_chunk_documents_financial(self):
        """Test financial document chunking."""
        processor = DocumentProcessor()

        sample_document = """
        APPLE INC.
        FORM 10-K
        FY 2023
        
        ITEM 1. BUSINESS
        Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories.
        
        ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
        Net sales increased 8% during fiscal 2023 compared to fiscal 2022.
        Gross margin was 43.3% compared to 41.8% in the prior year.
        
        ITEM 8. FINANCIAL STATEMENTS
        CONSOLIDATED BALANCE SHEETS
        Total assets: $100,000,000
        Total liabilities: $60,000,000
        """

        chunks = processor.chunk_documents(
            documents=[sample_document], chunk_size=500, chunk_overlap=50
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.page_content) <= 500
            assert "metadata" in chunk.__dict__

    @pytest.mark.unit
    def test_extract_financial_tables(self):
        """Test financial table extraction."""
        processor = DocumentProcessor()

        document_with_tables = """
        INCOME STATEMENT
        Revenue: $100,000,000
        Cost of Goods Sold: $60,000,000
        Gross Profit: $40,000,000
        
        BALANCE SHEET
        Cash: $10,000,000
        Accounts Receivable: $5,000,000
        """

        tables = processor.extract_financial_tables(document_with_tables)

        assert len(tables) > 0
        for table in tables:
            assert "data" in table
            assert "type" in table


class TestVectorStoreManager:
    """Test Vector Store Manager functionality."""

    @pytest.mark.unit
    def test_vector_store_initialization(self):
        """Test vector store manager initialization."""
        vector_store = VectorStoreManager()

        assert hasattr(vector_store, "add_documents")
        assert hasattr(vector_store, "similarity_search")
        assert hasattr(vector_store, "delete_documents")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_documents(self, mock_vector_store):
        """Test document addition to vector store."""
        vector_store = VectorStoreManager()
        vector_store.store = mock_vector_store

        documents = [
            MagicMock(page_content="Document 1", metadata={"source": "10-K"}),
            MagicMock(page_content="Document 2", metadata={"source": "10-Q"}),
        ]

        result = await vector_store.add_documents(documents)

        assert len(result) == len(documents)
        mock_vector_store.add_documents.assert_called_once_with(documents)

    @pytest.mark.unit
    def test_generate_embeddings(self):
        """Test embedding generation."""
        vector_store = VectorStoreManager()

        text = "Financial analysis and reporting"
        embedding = vector_store._generate_embeddings(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
