import sys
import os

sys.path.append("src")

from financial_rag.ingestion.sec_ingestor import SECIngestor
from financial_rag.ingestion.document_processor import DocumentProcessor
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.config import config


def test_foundation():
    print("🧪 Testing Financial RAG Foundation...")

    # 1. Initialize components
    sec_ingestor = SECIngestor()
    doc_processor = DocumentProcessor()
    vector_manager = VectorStoreManager()

    # 2. Download a small amount of test data
    print("📥 Downloading test SEC filings...")
    try:
        sec_ingestor.download_filings("AAPL", "10-K", years=1)
    except Exception as e:
        print(f"⚠️  SEC download failed (this might be expected): {e}")
        print("📝 Using mock data for testing...")
        # We'll create mock data for testing
        return create_mock_test(vector_manager, doc_processor)

    # 3. Process documents
    print("🔧 Processing documents...")
    filing_paths = sec_ingestor.get_filing_paths("AAPL", "10-K")

    if filing_paths:
        documents = []
        for filing_path in filing_paths[:1]:  # Just process one filing
            doc = doc_processor.process_sec_filing(filing_path)
            documents.append(doc)

        # 4. Chunk documents
        print("✂️  Chunking documents...")
        chunks = doc_processor.chunk_documents(documents)

        # 5. Create vector store
        print("🗄️  Creating vector store...")
        vector_store = vector_manager.create_vector_store(chunks)

        # 6. Test retrieval
        print("🔍 Testing retrieval...")
        retriever = vector_manager.get_retriever(vector_store)
        test_results = retriever.get_relevant_documents("What are the risk factors?")

        print(f"✅ Success! Retrieved {len(test_results)} relevant chunks")
        return True
    else:
        return create_mock_test(vector_manager, doc_processor)


def create_mock_test(vector_manager, doc_processor):
    """Create a test with mock financial data"""
    print("📝 Creating mock financial documents...")

    mock_documents = [
        "Apple Inc. reported revenue of $383 billion for fiscal year 2023, with iPhone sales contributing 52% of total revenue. The company's gross margin was 43% and operating margin was 30%. Major risk factors include supply chain disruptions, foreign exchange volatility, and intense competition in the smartphone market.",
        "Microsoft Corporation achieved $211 billion in revenue for FY2023, driven by cloud services growth. Azure revenue grew 27% year-over-year. The company maintains a strong balance sheet with $130 billion in cash and short-term investments. Key challenges include cybersecurity threats and regulatory compliance across multiple jurisdictions.",
        "Amazon.com Inc. reported net sales of $574 billion for 2023. AWS segment revenue was $90 billion with 29% operating margin. The company faces risks related to economic conditions affecting consumer spending, international expansion challenges, and increasing competition in cloud services and e-commerce.",
    ]

    documents = []
    for i, content in enumerate(mock_documents):
        documents.append(
            doc_processor.text_splitter.create_documents(
                [content],
                [{"source": f"mock_financial_{i}", "document_type": "MOCK_DATA"}],
            )[0]
        )

    # Create vector store
    vector_store = vector_manager.create_vector_store(documents)

    # Test retrieval
    retriever = vector_manager.get_retriever(vector_store)
    test_results = retriever.get_relevant_documents("revenue and risk factors")

    print(f"✅ Mock test successful! Retrieved {len(test_results)} relevant chunks")
    for i, result in enumerate(test_results):
        print(f"Chunk {i+1}: {result.page_content[:100]}...")

    return True


if __name__ == "__main__":
    test_foundation()
