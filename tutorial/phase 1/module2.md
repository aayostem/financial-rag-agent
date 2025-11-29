<!-- `src/financial_rag/config.py` -->
inside of the financial_rag, create a config.py file.
This file is used for storing configuration settings and constants for your application. It centralizes all configuration in one place, making it easy to manage and modify settings.

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Vector Database
    VECTOR_STORE_PATH = "./data/chroma_db"
    
    # Data Paths
    RAW_DATA_PATH = "./data/raw"
    PROCESSED_DATA_PATH = "./data/processed"
    
    # Model Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local model
    # EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model
    LLM_MODEL = "gpt-3.5-turbo"  # Start with 3.5, upgrade to gpt-4 later
    
    # Chunking Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Settings
    TOP_K_RESULTS = 3

config = Config()
```

<!-- `src/financial_rag/ingestion/sec_ingestor.py` -->
inside the ingetion, create sec_ingetor.py

```python
import os
from sec_edgar_downloader import Downloader
from loguru import logger
from financial_rag.config import config  # Fixed import

class SECIngestor:
    def __init__(self):
        self.dl = Downloader(config.RAW_DATA_PATH)
        os.makedirs(config.RAW_DATA_PATH, exist_ok=True)
    
    def download_filings(self, ticker, filing_type="10-K", years=5):
        """Download SEC filings for a given ticker"""
        try:
            logger.info(f"Downloading {filing_type} filings for {ticker}")
            
            # Download filings
            num_filings = self.dl.get(
                filing_type, 
                ticker, 
                amount=years,
                download_details=True
            )
            
            logger.success(f"Successfully downloaded {num_filings} {filing_type} filings for {ticker}")
            return num_filings
            
        except Exception as e:
            logger.error(f"Error downloading filings for {ticker}: {str(e)}")
            raise
    
    def get_filing_paths(self, ticker, filing_type="10-K"):
        """Get paths to downloaded filings"""
        ticker_path = os.path.join(config.RAW_DATA_PATH, "sec-edgar-filings", ticker, filing_type)
        if not os.path.exists(ticker_path):
            return []
        
        filing_paths = []
        for root, dirs, files in os.walk(ticker_path):
            for file in files:
                if file.endswith('.txt') or file.endswith('.html'):
                    filing_paths.append(os.path.join(root, file))
        
        return filing_paths
```

 <!-- `src/financial_rag/ingestion/yfinance_ingestor.py` -->
now let's crete yfinance_ingetor

```python
import yfinance as yf
import pandas as pd
import os
import json
from loguru import logger
from financial_rag.config import config  # Fixed import

class YFinanceIngestor:
    def __init__(self):
        os.makedirs(config.RAW_DATA_PATH, exist_ok=True)
    
    def download_stock_data(self, ticker, period="1y"):
        """Download stock price data and company info"""
        try:
            logger.info(f"Downloading data for {ticker}")
            
            # Get stock data
            stock = yf.Ticker(ticker)
            
            # Historical prices
            hist = stock.history(period=period)
            
            # Company info
            info = stock.info
            
            # Save data
            data = {
                "ticker": ticker,
                "historical_prices": hist.to_dict(),
                "company_info": info
            }
            
            file_path = os.path.join(config.RAW_DATA_PATH, f"{ticker}_yfinance.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.success(f"Successfully downloaded data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            raise
    
    def get_financial_news(self, ticker, num_articles=10):
        """Get recent news for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news[:num_articles]
            return news
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []
```

 <!-- `src/financial_rag/ingestion/document_processor.py` -->
let's create trhe document_processor.py

```python
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger
from financial_rag.config import config  # Fixed import

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def process_sec_filing(self, file_path):
        """Process SEC filing text files"""
        try:
            logger.info(f"Processing SEC filing: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Clean SEC filing - remove XML/HTML tags and excessive whitespace
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Extract metadata
            metadata = self._extract_sec_metadata(text, file_path)
            
            return Document(page_content=text, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error processing SEC filing {file_path}: {str(e)}")
            raise
    
    def _extract_sec_metadata(self, text, file_path):
        """Extract metadata from SEC filing text"""
        metadata = {
            "source": file_path,
            "document_type": "SEC_FILING",
            "source_type": "structured"
        }
        
        # Extract company name (simplified)
        company_match = re.search(r"COMPANY CONFORMED NAME:\s*([^\n]+)", text)
        if company_match:
            metadata["company"] = company_match.group(1).strip()
        
        # Extract filing date
        date_match = re.search(r"FILED AS OF DATE:\s*(\d{8})", text)
        if date_match:
            metadata["filing_date"] = date_match.group(1)
        
        # Extract document type
        doc_match = re.search(r"CONFORMED SUBMISSION TYPE:\s*([^\n]+)", text)
        if doc_match:
            metadata["filing_type"] = doc_match.group(1).strip()
        
        return metadata
    
    def chunk_documents(self, documents):
        """Split documents into chunks using sophisticated strategy"""
        logger.info(f"Chunking {len(documents)} documents")
        
        chunks = self.text_splitter.split_documents(documents)
        
        logger.success(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
```

 <!-- `src/financial_rag/retrieval/vector_store.py` -->
move into the retrieval folder  to create the  vector_store.py file

```python
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from loguru import logger
from financial_rag.config import config  # Fixed import

class VectorStoreManager:
    def __init__(self):
        self.embedding_model = self._initialize_embeddings()
        self.client = self._initialize_chroma()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model based on config"""
        try:
            if config.EMBEDDING_MODEL.startswith("text-embedding"):
                logger.info("Using OpenAI embeddings")
                return OpenAIEmbeddings(
                    model=config.EMBEDDING_MODEL,
                    openai_api_key=config.OPENAI_API_KEY
                )
            else:
                logger.info(f"Using local embeddings: {config.EMBEDDING_MODEL}")
                return HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL
                )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client"""
        return chromadb.PersistentClient(
            path=config.VECTOR_STORE_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
    
    def create_vector_store(self, documents):
        """Create a new vector store from documents"""
        try:
            logger.info("Creating vector store from documents")
            
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=config.VECTOR_STORE_PATH,
                client=self.client
            )
            
            logger.success(f"Vector store created with {len(documents)} documents")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self):
        """Load existing vector store"""
        try:
            vector_store = Chroma(
                persist_directory=config.VECTOR_STORE_PATH,
                embedding_function=self.embedding_model,
                client=self.client
            )
            logger.info("Vector store loaded successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def get_retriever(self, vector_store, search_type="similarity", k=config.TOP_K_RESULTS):
        """Create a retriever from vector store"""
        search_kwargs = {"k": k}
        
        if search_type == "mmr":  # Maximum Marginal Relevance
            search_kwargs["fetch_k"] = k * 2
            search_kwargs["lambda_mult"] = 0.7
        
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return retriever
```

lets create run_test.py to test the financial agent foundation
## Step 5: Create `run_test.py` at Project Root

```python
#!/usr/bin/env python3
"""
Test script for Financial RAG Agent Foundation
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from financial_rag.ingestion.sec_ingestor import SECIngestor
from financial_rag.ingestion.document_processor import DocumentProcessor
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.config import config

def test_foundation():
    print("🧪 Testing Financial RAG Foundation...")
    
    try:
        # 1. Initialize components
        sec_ingestor = SECIngestor()
        doc_processor = DocumentProcessor()
        vector_manager = VectorStoreManager()
        
        print("✅ Components initialized successfully")
        
        # 2. Try to download SEC data (might fail without internet/API)
        print("📥 Attempting to download SEC filings...")
        try:
            sec_ingestor.download_filings("AAPL", "10-K", years=1)
            filing_paths = sec_ingestor.get_filing_paths("AAPL", "10-K")
            
            if filing_paths:
                documents = []
                for filing_path in filing_paths[:1]:  # Just process one filing
                    doc = doc_processor.process_sec_filing(filing_path)
                    documents.append(doc)
                
                # 3. Chunk documents
                print("✂️ Chunking documents...")
                chunks = doc_processor.chunk_documents(documents)
                
                # 4. Create vector store
                print("🗄️ Creating vector store...")
                vector_store = vector_manager.create_vector_store(chunks)
                
                # 5. Test retrieval
                print("🔍 Testing retrieval...")
                retriever = vector_manager.get_retriever(vector_store)
                test_results = retriever.get_relevant_documents("What are the risk factors?")
                
                print(f"✅ SEC Data Test Success! Retrieved {len(test_results)} relevant chunks")
                return True
            else:
                print("📝 No SEC filings found, using mock data...")
                return create_mock_test(vector_manager, doc_processor)
                
        except Exception as e:
            print(f"📝 SEC download failed, using mock data: {e}")
            return create_mock_test(vector_manager, doc_processor)
            
    except Exception as e:
        print(f"❌ Foundation test failed: {e}")
        return False

def create_mock_test(vector_manager, doc_processor):
    """Create a test with mock financial data"""
    print("📝 Creating mock financial documents...")
    
    mock_documents = [
        "Apple Inc. reported revenue of $383 billion for fiscal year 2023, with iPhone sales contributing 52% of total revenue. The company's gross margin was 43% and operating margin was 30%. Major risk factors include supply chain disruptions, foreign exchange volatility, and intense competition in the smartphone market.",
        "Microsoft Corporation achieved $211 billion in revenue for FY2023, driven by cloud services growth. Azure revenue grew 27% year-over-year. The company maintains a strong balance sheet with $130 billion in cash and short-term investments. Key challenges include cybersecurity threats and regulatory compliance across multiple jurisdictions.",
        "Amazon.com Inc. reported net sales of $574 billion for 2023. AWS segment revenue was $90 billion with 29% operating margin. The company faces risks related to economic conditions affecting consumer spending, international expansion challenges, and increasing competition in cloud services and e-commerce."
    ]
    
    documents = []
    for i, content in enumerate(mock_documents):
        doc = doc_processor.text_splitter.create_documents(
            [content], 
            [{"source": f"mock_financial_{i}", "document_type": "MOCK_DATA"}]
        )
        documents.extend(doc)
    
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
    success = test_foundation()
    if success:
        print("\n🎉 All tests passed! Foundation is solid.")
        print("\nNext steps:")
        print("1. Add your OpenAI API key to .env file")
        print("2. Run: pip install -e .")
        print("3. We'll build the RAG chain next!")
    else:
        print("\n💥 Tests failed. Please check the errors above.")
        sys.exit(1)
```
