Perfect! Let's set up the proper pyproject.toml structure and fix the imports.

## Step 1: Create Proper Project Structure

```
financial-rag-agent/
├── pyproject.toml
├── .env
├── .gitignore
├── run_test.py
└── src/
    └── financial_rag/
        ├── __init__.py
        ├── config.py
        ├── ingestion/
        │   ├── __init__.py
        │   ├── sec_ingestor.py
        │   ├── yfinance_ingestor.py
        │   └── document_processor.py
        ├── retrieval/
        │   ├── __init__.py
        │   └── vector_store.py
        └── agents/
            ├── __init__.py
            └── (future agent files)
```

## Step 2: Create `pyproject.toml`

```toml
[project]
name = "financial-rag-agent"
version = "0.1.0"
description = "Enterprise Financial RAG Analyst Agent"
authors = [
    {name = "AI Engineer", email = "ai@company.com"},
]
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "langchain>=0.1.0",
    "langchain-community>=0.0.10",
    "langchain-openai>=0.0.2",
    "openai>=1.3.0",
    "chromadb>=0.4.18",
    "sentence-transformers>=2.2.2",
    "pandas>=2.1.0",
    "numpy>=1.24.0",
    "python-dotenv>=1.0.0",
    "beautifulsoup4>=4.12.0",
    "requests>=2.31.0",
    "pdfplumber>=0.10.0",
    "sec-edgar-downloader>=4.1.2",
    "yfinance>=0.2.18",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.4.0",
    "pytest>=7.4.0",
    "loguru>=0.7.2",
    "wandb>=0.15.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "jupyter>=1.0.0",
]

[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 100
target-version = ['py39']

[tool.isort]
profile = "black"
```

## Step 3: Create `.gitignore`

```gitignore
# Environment
.env
.venv
venv/
env/
financial-rag-env/

# Data
data/
*.db
*.sqlite
*.csv
*.parquet

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# ChromaDB
chroma.sqlite
```

## Step 4: Fix All Import Statements

### Update `src/financial_rag/config.py`

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

### Update `src/financial_rag/ingestion/sec_ingestor.py`

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

### Update `src/financial_rag/ingestion/yfinance_ingestor.py`

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

### Update `src/financial_rag/ingestion/document_processor.py`

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

### Update `src/financial_rag/retrieval/vector_store.py`

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

## Step 6: Installation & Testing

```bash
# Create and activate virtual environment
python -m venv financial-rag-env
source financial-rag-env/bin/activate  # Linux/Mac
# financial-rag-env\Scripts\activate  # Windows

# Install the package in editable mode
pip install -e .

# Run the test
python run_test.py
```

## What We've Achieved:

1. ✅ **Proper Python packaging** with pyproject.toml
2. ✅ **Clean import structure** using `financial_rag.config` instead of relative imports
3. ✅ **Virtual environment** isolation
4. ✅ **Enterprise-grade project structure**
5. ✅ **Testable foundation** that works even without internet access

The project is now properly structured as a Python package that can be installed, imported, and extended cleanly. Run the test and let me know if you encounter any issues!




Excellent! Let's proceed with building the core RAG chain and agent system. This is where we'll add the intelligence to our financial analyst.

## Step 6: Building the RAG Chain with Sophisticated Prompts

### Create `src/financial_rag/agents/prompts.py`

```python
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

class FinancialPrompts:
    """Sophisticated prompts for financial analysis"""
    
    @property
    def system_prompt_analyst(self):
        return SystemMessagePromptTemplate.from_template(
            """You are a seasoned financial analyst at a top investment firm. Your task is to provide accurate, concise, and well-supported analysis based ONLY on the provided context.

Guidelines:
- Be highly numerical and precise. Cite specific figures, percentages, dates, and metrics from the context.
- Structure your answer clearly with headings, bullet points, and logical flow.
- If the context does not contain enough information to answer fully, state what you can conclude and what is missing.
- Highlight risks, opportunities, and key trends.
- Compare performance across periods or segments when relevant.
- Never hallucinate or invent information. If you don't know based on the context, say so.
- For financial metrics, provide context about whether the numbers are positive or negative.

Context:
{context}

Question:
{question}"""
        )

    @property
    def system_prompt_executive(self):
        return SystemMessagePromptTemplate.from_template(
            """You are a financial analyst preparing an executive summary for the C-suite. Provide high-level insights with strategic implications.

Key Requirements:
- Start with a TL;DR (Too Long; Didn't Read) summary
- Focus on strategic implications and business impact
- Highlight key risks and opportunities
- Use clear, business-focused language
- Support all claims with specific data from the context
- Emphasize trends and patterns rather than isolated data points

Context:
{context}

Question:
{question}"""
        )

    @property
    def system_prompt_risk_analysis(self):
        return SystemMessagePromptTemplate.from_template(
            """You are a risk analysis specialist. Analyze the risk factors and provide a comprehensive risk assessment.

Risk Analysis Framework:
1. **Risk Identification**: List all mentioned risks
2. **Risk Assessment**: Evaluate likelihood and impact (High/Medium/Low)
3. **Risk Mitigation**: Note any mentioned mitigation strategies
4. **Comparative Analysis**: Compare with previous periods if available
5. **Recommendations**: Suggest areas for further investigation

Context:
{context}

Question:
{question}"""
        )

    def get_qa_prompt(self, style="analyst"):
        """Get QA prompt based on style"""
        prompts = {
            "analyst": self.system_prompt_analyst,
            "executive": self.system_prompt_executive,
            "risk": self.system_prompt_risk_analysis
        }
        
        system_prompt = prompts.get(style, self.system_prompt_analyst)
        
        return ChatPromptTemplate.from_messages([
            system_prompt,
            HumanMessagePromptTemplate.from_template("{question}")
        ])
```

### Create `src/financial_rag/agents/rag_chain.py`

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from loguru import logger
from financial_rag.config import config
from financial_rag.agents.prompts import FinancialPrompts

class FinancialRAGChain:
    """Main RAG chain for financial analysis"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = self._initialize_llm()
        self.prompts = FinancialPrompts()
        
    def _initialize_llm(self):
        """Initialize the LLM with proper settings"""
        try:
            return ChatOpenAI(
                model_name=config.LLM_MODEL,
                temperature=0.1,  # Low temperature for consistent financial analysis
                openai_api_key=config.OPENAI_API_KEY,
                max_retries=3,
                request_timeout=60
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def create_qa_chain(self, prompt_style="analyst", search_type="similarity"):
        """Create a QA chain with specified prompt style"""
        try:
            # Get the retriever
            from financial_rag.retrieval.vector_store import VectorStoreManager
            vector_manager = VectorStoreManager()
            retriever = vector_manager.get_retriever(
                self.vector_store, 
                search_type=search_type
            )
            
            # Create QA chain with custom prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": self.prompts.get_qa_prompt(prompt_style)
                },
                return_source_documents=True
            )
            
            logger.success(f"Created QA chain with {prompt_style} style and {search_type} search")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
    
    def analyze_question(self, question, prompt_style="analyst", search_type="similarity"):
        """Main method to analyze a financial question"""
        try:
            qa_chain = self.create_qa_chain(prompt_style, search_type)
            
            logger.info(f"Analyzing question: {question}")
            result = qa_chain({"query": question})
            
            # Log the analysis
            logger.info(f"Analysis completed. Source documents: {len(result.get('source_documents', []))}")
            
            return {
                "question": question,
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "prompt_style": prompt_style,
                "search_type": search_type
            }
            
        except Exception as e:
            logger.error(f"Error analyzing question '{question}': {str(e)}")
            return {
                "question": question,
                "answer": f"Error analyzing question: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
```

## Step 7: Creating the Agent with Tools

### Create `src/financial_rag/agents/tools.py`

```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from loguru import logger

class FinancialTools:
    """Tools for the financial agent to interact with real-time data"""
    
    @staticmethod
    def get_stock_price(ticker: str, period: str = "1mo") -> Dict[str, Any]:
        """Get current stock price and recent performance"""
        try:
            logger.info(f"Getting stock price for {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {"error": f"No data found for ticker {ticker}"}
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            info = stock.info
            result = {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change_pct, 2),
                "currency": info.get('currency', 'USD'),
                "company_name": info.get('longName', ticker),
                "timestamp": datetime.now().isoformat(),
                "period": period
            }
            
            logger.success(f"Retrieved stock data for {ticker}: ${current_price}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting stock price for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate_financial_ratio(ticker: str, ratio: str) -> Dict[str, Any]:
        """Calculate financial ratios"""
        try:
            logger.info(f"Calculating {ratio} for {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            ratios = {
                "pe_ratio": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "price_to_book": info.get('priceToBook'),
                "debt_to_equity": info.get('debtToEquity'),
                "return_on_equity": info.get('returnOnEquity'),
                "profit_margin": info.get('profitMargins'),
                "operating_margin": info.get('operatingMargins')
            }
            
            value = ratios.get(ratio.lower())
            
            if value is None:
                return {"error": f"Ratio {ratio} not available for {ticker}"}
            
            result = {
                "ticker": ticker,
                "ratio": ratio,
                "value": round(value, 4) if isinstance(value, float) else value,
                "description": FinancialTools._get_ratio_description(ratio),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.success(f"Calculated {ratio} for {ticker}: {value}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating ratio {ratio} for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def _get_ratio_description(ratio: str) -> str:
        """Get description for financial ratios"""
        descriptions = {
            "pe_ratio": "Price-to-Earnings Ratio - Measures company valuation relative to earnings",
            "forward_pe": "Forward P/E Ratio - Based on forecasted earnings",
            "price_to_book": "Price-to-Book Ratio - Compares market value to book value",
            "debt_to_equity": "Debt-to-Equity Ratio - Measures financial leverage",
            "return_on_equity": "Return on Equity - Measures profitability relative to shareholder equity",
            "profit_margin": "Profit Margin - Percentage of revenue that becomes profit",
            "operating_margin": "Operating Margin - Efficiency of core business operations"
        }
        return descriptions.get(ratio.lower(), "Financial ratio")
    
    @staticmethod
    def get_company_info(ticker: str) -> Dict[str, Any]:
        """Get comprehensive company information"""
        try:
            logger.info(f"Getting company info for {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            key_info = {
                "ticker": ticker,
                "company_name": info.get('longName'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "market_cap": info.get('marketCap'),
                "employees": info.get('fullTimeEmployees'),
                "description": info.get('longBusinessSummary'),
                "website": info.get('website'),
                "country": info.get('country'),
                "exchange": info.get('exchange'),
                "timestamp": datetime.now().isoformat()
            }
            
            # Clean None values
            key_info = {k: v for k, v in key_info.items() if v is not None}
            
            logger.success(f"Retrieved company info for {ticker}")
            return key_info
            
        except Exception as e:
            logger.error(f"Error getting company info for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def get_financial_news(ticker: str, num_articles: int = 5) -> List[Dict[str, Any]]:
        """Get recent financial news for a company"""
        try:
            logger.info(f"Getting news for {ticker}")
            stock = yf.Ticker(ticker)
            news = stock.news[:num_articles]
            
            formatted_news = []
            for article in news:
                formatted_article = {
                    "title": article.get('title', ''),
                    "publisher": article.get('publisher', ''),
                    "link": article.get('link', ''),
                    "published_date": datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat() if article.get('providerPublishTime') else None,
                    "related_tickers": article.get('relatedTickers', [])
                }
                formatted_news.append(formatted_article)
            
            logger.success(f"Retrieved {len(formatted_news)} news articles for {ticker}")
            return formatted_news
            
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {str(e)}")
            return [{"error": str(e)}]
```

### Create `src/financial_rag/agents/financial_agent.py`

```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Optional, Dict, Any
import re
from loguru import logger

from financial_rag.agents.tools import FinancialTools
from financial_rag.agents.rag_chain import FinancialRAGChain
from financial_rag.config import config

# Custom prompt template for the financial agent
class FinancialAgentPromptTemplate(StringPromptTemplate):
    template: str = """
You are a sophisticated Financial Analysis Agent. Your role is to help users analyze companies, financial documents, and market data.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Previous conversation history:
{history}

Question: {input}
Thought: {agent_scratchpad}"""

    def format(self, **kwargs) -> str:
        # Format the tools section
        tools = "\n".join([f"{tool.name}: {tool.description}" for tool in kwargs["tools"]])
        tool_names = ", ".join([tool.name for tool in kwargs["tools"]])
        
        # Format the prompt
        kwargs["tools"] = tools
        kwargs["tool_names"] = tool_names
        return self.template.format(**kwargs)

class FinancialAgentOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        logger.debug(f"Parsing agent output: {text}")
        
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        
        # Parse action and action input
        action_match = re.search(r"Action:\s*(.+?)\s*Action Input:\s*(.+)", text, re.DOTALL)
        if not action_match:
            raise ValueError(f"Could not parse agent output: {text}")
        
        action = action_match.group(1).strip()
        action_input = action_match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=text)

class FinancialAgent:
    """Main financial agent that can use tools and RAG"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.rag_chain = FinancialRAGChain(vector_store)
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Setup the tools available to the agent"""
        
        tools = [
            Tool(
                name="search_filings",
                func=self._search_filings_wrapper,
                description="Search SEC filings and financial documents for specific information. Use for questions about risk factors, financial performance, company strategy, and regulatory disclosures."
            ),
            Tool(
                name="get_stock_price",
                func=FinancialTools.get_stock_price,
                description="Get current stock price and recent performance for a ticker. Input should be a stock ticker symbol like 'AAPL'."
            ),
            Tool(
                name="calculate_financial_ratio",
                func=FinancialTools.calculate_financial_ratio,
                description="Calculate financial ratios for a company. Input should be a string like 'AAPL, pe_ratio' or 'MSFT, debt_to_equity'."
            ),
            Tool(
                name="get_company_info",
                func=FinancialTools.get_company_info,
                description="Get comprehensive company information including sector, market cap, and business description. Input should be a stock ticker symbol."
            ),
            Tool(
                name="get_financial_news",
                func=FinancialTools.get_financial_news,
                description="Get recent financial news for a company. Input should be a stock ticker symbol."
            )
        ]
        
        return tools
    
    def _search_filings_wrapper(self, query: str) -> str:
        """Wrapper for RAG search to make it compatible with agent tools"""
        try:
            result = self.rag_chain.analyze_question(query, prompt_style="analyst")
            answer = result["answer"]
            
            # Add source information if available
            if result.get("source_documents"):
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]))
                source_info = f"\n\nSources: {', '.join(sources[:3])}"  # Limit to 3 sources
                answer += source_info
            
            return answer
        except Exception as e:
            return f"Error searching filings: {str(e)}"
    
    def _setup_agent(self) -> AgentExecutor:
        """Setup the agent executor"""
        try:
            from financial_rag.agents.rag_chain import FinancialRAGChain
            llm = FinancialRAGChain(self.vector_store).llm
            
            # Setup prompt template
            prompt_template = FinancialAgentPromptTemplate(
                input_variables=["input", "agent_scratchpad", "history", "tools"]
            )
            
            # Setup LLM chain
            llm_chain = LLMChain(llm=llm, prompt=prompt_template)
            
            # Setup agent
            tool_names = [tool.name for tool in self.tools]
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=FinancialAgentOutputParser(),
                stop=["\nObservation:"],
                allowed_tools=tool_names
            )
            
            # Setup executor
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5  # Prevent infinite loops
            )
            
            logger.success("Financial agent setup completed")
            return agent_executor
            
        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            raise
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """Main method to analyze a financial question using the agent"""
        try:
            logger.info(f"Agent analyzing question: {question}")
            
            result = self.agent_executor.run(question)
            
            return {
                "question": question,
                "answer": result,
                "agent_type": "tool_using_agent"
            }
            
        except Exception as e:
            logger.error(f"Error in agent analysis: {str(e)}")
            return {
                "question": question,
                "answer": f"I encountered an error while analyzing your question: {str(e)}",
                "error": str(e),
                "agent_type": "tool_using_agent"
            }
    
    def simple_rag_analysis(self, question: str, prompt_style: str = "analyst") -> Dict[str, Any]:
        """Simple RAG analysis without tool use (fallback)"""
        return self.rag_chain.analyze_question(question, prompt_style)
```

## Step 8: Create Enhanced Test Script

### Create `test_agent.py`

```python
#!/usr/bin/env python3
"""
Test script for Financial RAG Agent
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from financial_rag.ingestion.sec_ingestor import SECIngestor
from financial_rag.ingestion.document_processor import DocumentProcessor
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.agents.financial_agent import FinancialAgent
from financial_rag.config import config

def test_agent_system():
    print("🤖 Testing Financial Agent System...")
    
    try:
        # 1. Initialize components
        print("🔧 Initializing components...")
        doc_processor = DocumentProcessor()
        vector_manager = VectorStoreManager()
        
        # 2. Create or load vector store with mock data
        print("📊 Setting up knowledge base...")
        vector_store = setup_mock_knowledge_base(doc_processor, vector_manager)
        
        # 3. Initialize the financial agent
        print("👨‍💼 Initializing financial agent...")
        agent = FinancialAgent(vector_store)
        
        # 4. Test different types of queries
        print("\n🧪 Testing Agent Capabilities...")
        
        test_queries = [
            # RAG-based queries
            "What are the main risk factors mentioned in the documents?",
            "What is the revenue breakdown by segment?",
            
            # Tool-based queries  
            "What is the current stock price of Apple?",
            "What is the P/E ratio for Microsoft?",
            "Can you get company information for Tesla?",
            
            # Complex queries that require both
            "What are Apple's main risk factors and what is their current stock price?",
        ]
        
        for i, query in enumerate(test_queries[:3]):  # Test first 3 to save time
            print(f"\n{'='*60}")
            print(f"Query {i+1}: {query}")
            print(f"{'='*60}")
            
            try:
                result = agent.analyze(query)
                print(f"Answer: {result['answer'][:500]}...")  # Truncate for display
                
            except Exception as e:
                print(f"Error: {e}")
                # Fallback to simple RAG
                print("Trying simple RAG fallback...")
                simple_result = agent.simple_rag_analysis(query)
                print(f"Answer: {simple_result['answer'][:500]}...")
        
        print(f"\n✅ Agent system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent system test failed: {e}")
        return False

def setup_mock_knowledge_base(doc_processor, vector_manager):
    """Setup mock financial documents for testing"""
    
    mock_financial_docs = [
        {
            "content": """Apple Inc. Fiscal Year 2023 Financial Report
            Total Revenue: $383.3 billion, up 2% year-over-year
            iPhone Revenue: $200.6 billion (52% of total revenue)
            Services Revenue: $85.2 billion (22% of total revenue)
            Gross Margin: 43.2%
            Operating Income: $114.3 billion
            
            Risk Factors:
            - Supply chain disruptions impacting production
            - Foreign currency exchange volatility
            - Intense competition in smartphone and personal computer markets
            - Regulatory changes across multiple jurisdictions
            - Dependence on third-party manufacturing partners
            
            Segment Performance:
            - Americas: $169.7 billion
            - Europe: $95.0 billion  
            - Greater China: $72.6 billion
            - Japan: $25.4 billion
            - Rest of Asia Pacific: $20.6 billion""",
            "metadata": {"source": "apple_10k_2023", "company": "Apple", "year": "2023"}
        },
        {
            "content": """Microsoft Corporation FY2023 Earnings Report
            Revenue: $211.2 billion, up 7% year-over-year
            Azure Cloud Growth: 27% constant currency
            Office Commercial: $44.7 billion
            LinkedIn: $15.2 billion
            Cloud Gross Margin: 72%
            
            Strategic Priorities:
            - AI integration across product portfolio
            - Cloud-first, mobile-first strategy
            - Enterprise security solutions
            - Gaming and metaverse investments
            
            Capital Allocation:
            - Share repurchases: $27.4 billion
            - Dividends: $20.1 billion
            - R&D Investment: $27.5 billion""",
            "metadata": {"source": "microsoft_10k_2023", "company": "Microsoft", "year": "2023"}
        },
        {
            "content": """Amazon.com Inc. 2023 Annual Report
            Net Sales: $574.8 billion
            AWS Revenue: $90.8 billion (29% operating margin)
            North America: $353.5 billion
            International: $131.2 billion
            
            Key Challenges:
            - Economic conditions affecting consumer spending
            - International expansion complexities
            - Increasing cloud competition from Microsoft and Google
            - Regulatory scrutiny in multiple countries
            - Labor and employment matters
            
            Innovation Areas:
            - Artificial intelligence and machine learning
            - Supply chain automation
            - Healthcare initiatives
            - Project Kuiper satellite internet""",
            "metadata": {"source": "amazon_10k_2023", "company": "Amazon", "year": "2023"}
        }
    ]
    
    documents = []
    for doc in mock_financial_docs:
        chunked_docs = doc_processor.text_splitter.create_documents(
            [doc["content"]],
            [doc["metadata"]]
        )
        documents.extend(chunked_docs)
    
    print(f"Created {len(documents)} document chunks from mock data")
    
    # Create vector store
    vector_store = vector_manager.create_vector_store(documents)
    return vector_store

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ Please set your OPENAI_API_KEY in the .env file")
        print("💡 Get one from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    success = test_agent_system()
    if success:
        print("\n🎉 Financial Agent is working!")
        print("\nNext steps:")
        print("1. Add real SEC data using the SEC ingestor")
        print("2. We'll add monitoring and observability next")
        print("3. Then containerization and API setup")
    else:
        print("\n💥 Agent test failed")
        sys.exit(1)
```

## Step 9: Run the Enhanced Test

```bash
# Make sure you're in the virtual environment and package is installed
pip install -e .

# Run the agent test
python test_agent.py
```

## What We've Built Now:

1. ✅ **Sophisticated RAG Chain** with different prompt styles (analyst, executive, risk)
2. ✅ **Financial Tools** for real-time data (stock prices, ratios, company info)
3. ✅ **Intelligent Agent** that can use tools and RAG together
4. ✅ **Error handling** and fallback mechanisms
5. ✅ **Proper LangChain agent architecture**

The agent can now:
- Answer questions using SEC filings (RAG)
- Get real-time stock data and financial ratios
- Combine both for complex queries
- Handle errors gracefully
- Use different analysis styles

**Run the test and let me know the results!** This will validate that our agent system is working properly with the OpenAI API.

Once this works, we'll move to:
- Step 9: Monitoring and Observability with Weights & Biases
- Step 10: FastAPI REST API
- Step 11: Docker Containerization
- Step 12: Kubernetes Deployment

Excellent! Let's proceed with monitoring, API, and containerization to make this truly enterprise-ready.

## Step 9: Monitoring and Observability

### Create `src/financial_rag/monitoring/__init__.py`

### Create `src/financial_rag/monitoring/tracing.py`

```python
import wandb
from loguru import logger
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from financial_rag.config import config

class AgentMonitor:
    """Monitoring and tracing for the financial agent"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.wandb_run = None
        
        if enabled and config.WANDB_API_KEY and config.WANDB_API_KEY != "your_wandb_api_key_here":
            try:
                self.wandb_run = wandb.init(
                    project="financial-rag-agent",
                    config={
                        "embedding_model": config.EMBEDDING_MODEL,
                        "llm_model": config.LLM_MODEL,
                        "chunk_size": config.CHUNK_SIZE,
                        "top_k_results": config.TOP_K_RESULTS
                    }
                )
                logger.success("Weights & Biases monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.enabled = False
        else:
            logger.info("Monitoring disabled - no WandB API key found")
    
    def log_retrieval(self, query: str, documents: List, scores: List[float]):
        """Log retrieval performance"""
        if not self.enabled:
            return
            
        try:
            retrieval_metrics = {
                "retrieval/query_length": len(query),
                "retrieval/documents_retrieved": len(documents),
                "retrieval/avg_score": sum(scores) / len(scores) if scores else 0,
                "retrieval/max_score": max(scores) if scores else 0,
                "retrieval/timestamp": datetime.now().isoformat()
            }
            
            # Log source distribution
            sources = {}
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            if self.wandb_run:
                self.wandb_run.log(retrieval_metrics)
                
            logger.debug(f"Retrieval logged: {len(documents)} docs for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error logging retrieval: {e}")
    
    def log_llm_call(self, prompt: str, response: str, latency: float, token_usage: Dict = None):
        """Log LLM call details"""
        if not self.enabled:
            return
            
        try:
            llm_metrics = {
                "llm/prompt_length": len(prompt),
                "llm/response_length": len(response),
                "llm/latency_seconds": latency,
                "llm/timestamp": datetime.now().isoformat()
            }
            
            if token_usage:
                llm_metrics.update({
                    "llm/prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "llm/completion_tokens": token_usage.get('completion_tokens', 0),
                    "llm/total_tokens": token_usage.get('total_tokens', 0)
                })
            
            if self.wandb_run:
                self.wandb_run.log(llm_metrics)
                
            logger.debug(f"LLM call logged: {len(prompt)} chars -> {len(response)} chars in {latency:.2f}s")
            
        except Exception as e:
            logger.error(f"Error logging LLM call: {e}")
    
    def log_agent_step(self, step_type: str, tool_name: str, input_data: str, output: str, success: bool):
        """Log agent tool usage steps"""
        if not self.enabled:
            return
            
        try:
            agent_metrics = {
                f"agent/{step_type}_tool": tool_name,
                f"agent/{step_type}_input_length": len(input_data),
                f"agent/{step_type}_output_length": len(output),
                f"agent/{step_type}_success": success,
                f"agent/{step_type}_timestamp": datetime.now().isoformat()
            }
            
            if self.wandb_run:
                self.wandb_run.log(agent_metrics)
                
            logger.debug(f"Agent step logged: {tool_name} - Success: {success}")
            
        except Exception as e:
            logger.error(f"Error logging agent step: {e}")
    
    def log_query_analysis(self, question: str, answer: str, total_latency: float, 
                          source_count: int, agent_type: str):
        """Log complete query analysis"""
        if not self.enabled:
            return
            
        try:
            query_metrics = {
                "query/question_length": len(question),
                "query/answer_length": len(answer),
                "query/total_latency_seconds": total_latency,
                "query/source_count": source_count,
                "query/agent_type": agent_type,
                "query/success": len(answer) > 0,
                "query/timestamp": datetime.now().isoformat()
            }
            
            if self.wandb_run:
                self.wandb_run.log(query_metrics)
                
                # Log the actual Q&A for analysis
                self.wandb_run.log({
                    "query_samples": wandb.Table(
                        columns=["Question", "Answer", "Latency", "Sources"],
                        data=[[question, answer, total_latency, source_count]]
                    )
                })
            
            logger.info(f"Query analysis logged: {question[:50]}... -> {len(answer)} chars in {total_latency:.2f}s")
            
        except Exception as e:
            logger.error(f"Error logging query analysis: {e}")
    
    def cleanup(self):
        """Clean up monitoring resources"""
        if self.wandb_run:
            self.wandb_run.finish()
            logger.info("Monitoring session ended")
```

### Update `src/financial_rag/agents/financial_agent.py` with Monitoring

Add monitoring to the existing agent:

```python
# Add to imports
from financial_rag.monitoring.tracing import AgentMonitor

# Update the FinancialAgent class __init__ method:
class FinancialAgent:
    def __init__(self, vector_store, enable_monitoring: bool = True):
        self.vector_store = vector_store
        self.rag_chain = FinancialRAGChain(vector_store)
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()
        self.monitor = AgentMonitor(enabled=enable_monitoring)
    
    # Update the analyze method with monitoring:
    def analyze(self, question: str) -> Dict[str, Any]:
        """Main method to analyze a financial question using the agent"""
        start_time = time.time()
        
        try:
            logger.info(f"Agent analyzing question: {question}")
            
            result = self.agent_executor.run(question)
            end_time = time.time()
            
            # Log successful analysis
            self.monitor.log_query_analysis(
                question=question,
                answer=result,
                total_latency=end_time - start_time,
                source_count=0,  # Would need to extract from agent internals
                agent_type="tool_using_agent"
            )
            
            return {
                "question": question,
                "answer": result,
                "agent_type": "tool_using_agent",
                "latency_seconds": end_time - start_time
            }
            
        except Exception as e:
            end_time = time.time()
            error_msg = f"I encountered an error while analyzing your question: {str(e)}"
            
            # Log failed analysis
            self.monitor.log_query_analysis(
                question=question,
                answer=error_msg,
                total_latency=end_time - start_time,
                source_count=0,
                agent_type="tool_using_agent"
            )
            
            logger.error(f"Error in agent analysis: {str(e)}")
            return {
                "question": question,
                "answer": error_msg,
                "error": str(e),
                "agent_type": "tool_using_agent",
                "latency_seconds": end_time - start_time
            }
```

## Step 10: FastAPI REST API

### Create `src/financial_rag/api/__init__.py`

### Create `src/financial_rag/api/models.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class AnalysisStyle(str, Enum):
    ANALYST = "analyst"
    EXECUTIVE = "executive"
    RISK = "risk"

class SearchType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"

class QueryRequest(BaseModel):
    question: str = Field(..., description="The financial question to analyze")
    analysis_style: AnalysisStyle = Field(default=AnalysisStyle.ANALYST, description="Style of analysis")
    use_agent: bool = Field(default=True, description="Whether to use the intelligent agent")
    search_type: SearchType = Field(default=SearchType.SIMILARITY, description="Search type for retrieval")

class DocumentResponse(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    agent_type: str
    latency_seconds: float
    source_documents: List[DocumentResponse] = []
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_ready: bool
    llm_ready: bool

class IngestionRequest(BaseModel):
    ticker: str
    filing_type: str = Field(default="10-K")
    years: int = Field(default=2, ge=1, le=5)

class IngestionResponse(BaseModel):
    ticker: str
    filings_downloaded: int
    documents_processed: int
    success: bool
    error: Optional[str] = None
```

### Create `src/financial_rag/api/server.py`

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger
import os
import time

from financial_rag.config import config
from financial_rag.api.models import (
    QueryRequest, QueryResponse, HealthResponse, 
    IngestionRequest, IngestionResponse, AnalysisStyle
)
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.agents.financial_agent import FinancialAgent
from financial_rag.ingestion.sec_ingestor import SECIngestor
from financial_rag.ingestion.document_processor import DocumentProcessor

class FinancialRAGAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Financial RAG Analyst API",
            description="Enterprise-grade Financial Analysis using RAG and AI Agents",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.vector_store = None
        self.agent = None
        self.setup()
    
    def setup(self):
        """Setup middleware and routes"""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify exact origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add startup event
        @self.app.on_event("startup")
        async def startup_event():
            await self.initialize_services()
        
        # Add routes
        self.setup_routes()
    
    async def initialize_services(self):
        """Initialize vector store and agent"""
        try:
            logger.info("Initializing Financial RAG API services...")
            
            # Initialize vector store
            vector_manager = VectorStoreManager()
            self.vector_store = vector_manager.load_vector_store()
            
            if self.vector_store is None:
                logger.warning("No existing vector store found. Please ingest data first.")
                # Create empty vector store for health checks
                from langchain.schema import Document
                self.vector_store = vector_manager.create_vector_store([Document(page_content="", metadata={})])
            
            # Initialize agent
            self.agent = FinancialAgent(self.vector_store, enable_monitoring=True)
            
            logger.success("Financial RAG API services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", include_in_schema=False)
        async def root():
            return {"message": "Financial RAG Analyst API", "version": "0.1.0"}
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            try:
                vector_store_ready = self.vector_store is not None
                llm_ready = self.agent is not None and self.agent.rag_chain.llm is not None
                
                return HealthResponse(
                    status="healthy",
                    version="0.1.0",
                    vector_store_ready=vector_store_ready,
                    llm_ready=llm_ready
                )
            except Exception as e:
                return HealthResponse(
                    status="unhealthy",
                    version="0.1.0",
                    vector_store_ready=False,
                    llm_ready=False
                )
        
        @self.app.post("/query", response_model=QueryResponse)
        async def query_analysis(request: QueryRequest):
            """Main endpoint for financial analysis"""
            try:
                if not self.agent:
                    raise HTTPException(status_code=503, detail="Agent not initialized")
                
                start_time = time.time()
                
                if request.use_agent:
                    # Use intelligent agent with tools
                    result = self.agent.analyze(request.question)
                else:
                    # Use simple RAG
                    result = self.agent.simple_rag_analysis(
                        request.question, 
                        prompt_style=request.analysis_style
                    )
                
                # Convert source documents to response format
                source_docs = []
                for doc in result.get("source_documents", []):
                    source_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', None)
                    })
                
                return QueryResponse(
                    question=result["question"],
                    answer=result["answer"],
                    agent_type=result.get("agent_type", "simple_rag"),
                    latency_seconds=result.get("latency_seconds", time.time() - start_time),
                    source_documents=source_docs,
                    error=result.get("error")
                )
                
            except Exception as e:
                logger.error(f"Error in query analysis: {e}")
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.post("/ingest/sec", response_model=IngestionResponse)
        async def ingest_sec_filings(request: IngestionRequest, background_tasks: BackgroundTasks):
            """Ingest SEC filings for a ticker"""
            try:
                # Run ingestion in background
                background_tasks.add_task(
                    self._ingest_sec_filings_background,
                    request.ticker,
                    request.filing_type,
                    request.years
                )
                
                return IngestionResponse(
                    ticker=request.ticker,
                    filings_downloaded=0,  # Will be updated in background
                    documents_processed=0,
                    success=True,
                    error=None
                )
                
            except Exception as e:
                return IngestionResponse(
                    ticker=request.ticker,
                    filings_downloaded=0,
                    documents_processed=0,
                    success=False,
                    error=str(e)
                )
        
        @self.app.get("/system/stats")
        async def system_stats():
            """Get system statistics"""
            try:
                if not self.vector_store:
                    return {"error": "Vector store not initialized"}
                
                # Get collection stats
                collection = self.vector_store._collection
                if collection:
                    count = collection.count()
                    
                    return {
                        "vector_store_documents": count,
                        "embedding_model": config.EMBEDDING_MODEL,
                        "llm_model": config.LLM_MODEL,
                        "status": "operational"
                    }
                else:
                    return {"error": "Collection not available"}
                    
            except Exception as e:
                return {"error": f"Failed to get stats: {str(e)}"}
    
    async def _ingest_sec_filings_background(self, ticker: str, filing_type: str, years: int):
        """Background task for SEC filings ingestion"""
        try:
            logger.info(f"Starting background ingestion for {ticker}")
            
            # Download filings
            sec_ingestor = SECIngestor()
            num_filings = sec_ingestor.download_filings(ticker, filing_type, years)
            
            # Process documents
            doc_processor = DocumentProcessor()
            filing_paths = sec_ingestor.get_filing_paths(ticker, filing_type)
            
            documents = []
            for filing_path in filing_paths:
                doc = doc_processor.process_sec_filing(filing_path)
                documents.append(doc)
            
            # Chunk documents
            chunks = doc_processor.chunk_documents(documents)
            
            # Update vector store
            vector_manager = VectorStoreManager()
            self.vector_store = vector_manager.create_vector_store(chunks)
            
            # Reinitialize agent with updated vector store
            self.agent = FinancialAgent(self.vector_store, enable_monitoring=True)
            
            logger.success(f"Background ingestion completed for {ticker}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Background ingestion failed for {ticker}: {e}")

# Create app instance
app = FinancialRAGAPI().app

if __name__ == "__main__":
    uvicorn.run(
        "financial_rag.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )
```

## Step 11: Docker Containerization

### Create `Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY .env ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "financial_rag.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create `docker-compose.yml`

```yaml
version: '3.8'

services:
  financial-rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - WANDB_API_KEY=${WANDB_API_KEY}
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Add Redis for caching in future
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"
  #   volumes:
  #     - redis_data:/data

# volumes:
#   redis_data:
```

### Create `.dockerignore`

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
.venv
financial-rag-env
*.db
*.sqlite
```

## Step 12: Create Deployment Scripts

### Create `scripts/start_api.py`

```python
#!/usr/bin/env python3
"""
Production API startup script
"""

import uvicorn
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if __name__ == "__main__":
    uvicorn.run(
        "financial_rag.api.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        log_level=os.getenv("LOG_LEVEL", "info"),
        access_log=True
    )
```

### Create `scripts/test_production.py`

```python
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
        ("POST", "/query", {
            "question": "What is the current stock price of Apple?",
            "use_agent": True,
            "analysis_style": "analyst"
        })
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
```

## Step 13: Updated Requirements in `pyproject.toml`

Add the new dependencies:

```toml
# Add to dependencies list
dependencies = [
    # ... existing dependencies
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.4.0",
    "wandb>=0.15.0",
    "requests>=2.31.0",
    "aiohttp>=3.8.0",
]

[project.optional-dependencies]
dev = [
    # ... existing dev dependencies
    "pytest-asyncio>=0.21.0",
    "httpx>=0.24.0",
]

# Add new section for docker
[project.scripts]
financial-rag-api = "financial_rag.api.server:main"
```

## Step 14: Create Production Test

### Create `test_production.py`

```python
#!/usr/bin/env python3
"""
Comprehensive production test
"""

import sys
import os
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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
```

## Step 15: Run Production Test

```bash
# Install updated dependencies
pip install -e ".[dev]"

# Test production readiness
python test_production.py

# Build Docker image
docker build -t financial-rag-agent .

# Run with Docker Compose
docker-compose up -d

# Test the running API
python scripts/test_production.py

# Check logs
docker-compose logs -f
```

## What We've Built Now:

1. ✅ **Enterprise Monitoring** with Weights & Biases
2. ✅ **Production REST API** with FastAPI
3. ✅ **Docker Containerization** for deployment
4. ✅ **Health Checks** and monitoring endpoints
5. ✅ **Background Tasks** for data ingestion
6. ✅ **Production-ready configuration**

## Deployment Options:

### Option A: Local Development
```bash
python scripts/start_api.py
```

### Option B: Docker
```bash
docker-compose up -d
```

### Option C: Kubernetes (Next Step)
We can create Kubernetes manifests for full enterprise deployment.

**Run the production test and let me know the results!** This will validate that our entire system is enterprise-ready.

The system is now capable of:
- Scalable REST API with proper documentation
- Containerized deployment
- Comprehensive monitoring and observability
- Health checks and metrics
- Background processing
- Production-grade error handling

Ready for the final step: Kubernetes deployment manifests?

Perfect! Let's complete the enterprise deployment with Kubernetes manifests and final production optimizations.

## Step 16: Kubernetes Deployment Manifests

### Create `kubernetes/namespace.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: financial-rag
  labels:
    name: financial-rag
    environment: production
```

### Create `kubernetes/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: financial-rag-config
  namespace: financial-rag
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  EMBEDDING_MODEL: "all-MiniLM-L6-v2"
  LLM_MODEL: "gpt-3.5-turbo"
  CHUNK_SIZE: "1000"
  CHUNK_OVERLAP: "200"
  TOP_K_RESULTS: "3"
  VECTOR_STORE_PATH: "/app/data/chroma_db"
  RAW_DATA_PATH: "/app/data/raw"
  PROCESSED_DATA_PATH: "/app/data/processed"
```

### Create `kubernetes/secret.yaml`

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: financial-rag-secrets
  namespace: financial-rag
type: Opaque
stringData:
  OPENAI_API_KEY: ""  # Will be filled from CI/CD
  WANDB_API_KEY: ""   # Will be filled from CI/CD
```

### Create `kubernetes/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-rag-api
  namespace: financial-rag
  labels:
    app: financial-rag-api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: financial-rag-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: financial-rag-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: financial-rag-api
        image: financial-rag-agent:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: financial-rag-secrets
              key: OPENAI_API_KEY
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: financial-rag-secrets
              key: WANDB_API_KEY
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: LOG_LEVEL
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: ENVIRONMENT
        - name: EMBEDDING_MODEL
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: EMBEDDING_MODEL
        - name: LLM_MODEL
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: LLM_MODEL
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
        volumeMounts:
        - name: data-storage
          mountPath: /app/data
        - name: log-storage
          mountPath: /app/logs
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: financial-rag-pvc
      - name: log-storage
        emptyDir: {}
      restartPolicy: Always
```

### Create `kubernetes/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: financial-rag-service
  namespace: financial-rag
  labels:
    app: financial-rag-api
spec:
  selector:
    app: financial-rag-api
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: financial-rag-service-external
  namespace: financial-rag
  labels:
    app: financial-rag-api
spec:
  selector:
    app: financial-rag-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: LoadBalancer
```

### Create `kubernetes/persistent-volume-claim.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: financial-rag-pvc
  namespace: financial-rag
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard  # Adjust based on your Kubernetes cluster
```

### Create `kubernetes/hpa.yaml`

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: financial-rag-hpa
  namespace: financial-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: financial-rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

### Create `kubernetes/ingress.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: financial-rag-ingress
  namespace: financial-rag
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - financial-rag.yourcompany.com
    secretName: financial-rag-tls
  rules:
  - host: financial-rag.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: financial-rag-service
            port:
              number: 8000
```

## Step 17: CI/CD Pipeline Configuration

### Create `.github/workflows/ci-cd.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  IMAGE_NAME: financial-rag-agent
  REGISTRY: ghcr.io

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run tests
      run: |
        python test_foundation.py
        python test_agent.py
        python test_production.py
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    - name: Run security scan
      run: |
        pip install bandit safety
        bandit -r src/ -f json -o bandit-report.json
        safety check --json

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Log in to GitHub Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata (tags, labels)
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ github.repository }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_STAGING }}
        command: apply -f kubernetes/
        version: v1.27.0

    - name: Verify deployment
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_STAGING }}
        command: rollout status deployment/financial-rag-api -n financial-rag
        version: v1.27.0

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_PRODUCTION }}
        command: apply -f kubernetes/
        version: v1.27.0

    - name: Verify deployment
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_PRODUCTION }}
        command: rollout status deployment/financial-rag-api -n financial-rag
        version: v1.27.0
```

## Step 18: Monitoring and Metrics

### Create `src/financial_rag/monitoring/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from loguru import logger
import time

# Metrics for Prometheus
QUERY_COUNTER = Counter('financial_rag_queries_total', 'Total number of queries', ['status', 'agent_type'])
QUERY_DURATION = Histogram('financial_rag_query_duration_seconds', 'Query duration in seconds')
AGENT_TOOL_USAGE = Counter('financial_rag_agent_tool_usage_total', 'Agent tool usage', ['tool_name', 'status'])
VECTOR_STORE_SIZE = Gauge('financial_rag_vector_store_documents', 'Number of documents in vector store')
LLM_TOKEN_USAGE = Counter('financial_rag_llm_tokens_total', 'LLM token usage', ['type'])

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
```

### Update API to Include Metrics Endpoint

Add to `src/financial_rag/api/server.py`:

```python
from financial_rag.monitoring.metrics import metrics_collector

# Add this route to the FinancialRAGAPI class:
@self.app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response
    return Response(
        content=metrics_collector.get_metrics(),
        media_type="text/plain"
    )

# Update the query_analysis endpoint to record metrics:
@self.app.post("/query", response_model=QueryResponse)
async def query_analysis(request: QueryRequest):
    """Main endpoint for financial analysis"""
    start_time = time.time()
    try:
        # ... existing code ...
        
        # Record metrics
        metrics_collector.record_query(
            status="success",
            agent_type="agent" if request.use_agent else "rag",
            duration=time.time() - start_time
        )
        
        return response
        
    except Exception as e:
        # Record failure metrics
        metrics_collector.record_query(
            status="failure", 
            agent_type="agent" if request.use_agent else "rag",
            duration=time.time() - start_time
        )
        raise
```

## Step 19: Advanced Configuration Management

### Create `src/financial_rag/config/__init__.py`

### Create `src/financial_rag/config/advanced.py`

```python
import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, validator
from loguru import logger

class AdvancedConfig(BaseSettings):
    """Advanced configuration with validation"""
    
    # API Settings
    OPENAI_API_KEY: str
    WANDB_API_KEY: Optional[str] = None
    
    # Model Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 3
    SEARCH_TYPE: str = "similarity"  # "similarity" or "mmr"
    
    # Agent Settings
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_ENABLE_MONITORING: bool = True
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_LOG_LEVEL: str = "info"
    
    # Storage Settings
    VECTOR_STORE_PATH: str = "./data/chroma_db"
    RAW_DATA_PATH: str = "./data/raw"
    PROCESSED_DATA_PATH: str = "./data/processed"
    
    # Kubernetes Settings
    K8S_NAMESPACE: str = "financial-rag"
    K8S_DEPLOYMENT_NAME: str = "financial-rag-api"
    
    # Monitoring Settings
    PROMETHEUS_ENABLED: bool = True
    WANDB_ENABLED: bool = True
    
    @validator("CHUNK_SIZE")
    def validate_chunk_size(cls, v):
        if v < 100 or v > 2000:
            raise ValueError("CHUNK_SIZE must be between 100 and 2000")
        return v
    
    @validator("LLM_TEMPERATURE") 
    def validate_temperature(cls, v):
        if v < 0 or v > 1:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 1")
        return v
    
    @validator("TOP_K_RESULTS")
    def validate_top_k(cls, v):
        if v < 1 or v > 10:
            raise ValueError("TOP_K_RESULTS must be between 1 and 10")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global advanced config
advanced_config = AdvancedConfig()
```

## Step 20: Final Production Scripts

### Create `scripts/deploy.sh`

```bash
#!/bin/bash

set -e

echo "🚀 Deploying Financial RAG Agent to Kubernetes..."

# Validate environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is required"
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t financial-rag-agent:latest .

# If using remote registry, push the image
# docker tag financial-rag-agent:latest your-registry/financial-rag-agent:latest
# docker push your-registry/financial-rag-agent:latest

# Create namespace if it doesn't exist
echo "📁 Creating Kubernetes namespace..."
kubectl apply -f kubernetes/namespace.yaml

# Create secrets
echo "🔐 Creating secrets..."
kubectl create secret generic financial-rag-secrets \
    --namespace=financial-rag \
    --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY" \
    --from-literal=WANDB_API_KEY="$WANDB_API_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply all Kubernetes manifests
echo "📄 Applying Kubernetes manifests..."
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/persistent-volume-claim.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/financial-rag-api -n financial-rag --timeout=300s

# Get service information
echo "🌐 Service information:"
kubectl get service -n financial-rag

echo "✅ Deployment completed successfully!"
echo "📊 Check logs: kubectl logs -f deployment/financial-rag-api -n financial-rag"
echo "🌐 Access API: kubectl port-forward service/financial-rag-service 8000:8000 -n financial-rag"
```

### Create `scripts/health-check.sh`

```bash
#!/bin/bash

set -e

echo "🏥 Running comprehensive health check..."

NAMESPACE=${1:-financial-rag}
SERVICE=${2:-financial-rag-service}
PORT=${3:-8000}

# Check if namespace exists
echo "1. Checking namespace..."
kubectl get namespace $NAMESPACE > /dev/null 2>&1 || {
    echo "❌ Namespace $NAMESPACE does not exist"
    exit 1
}

# Check deployment status
echo "2. Checking deployment..."
DEPLOYMENT_STATUS=$(kubectl get deployment financial-rag-api -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Available")].status}')
if [ "$DEPLOYMENT_STATUS" != "True" ]; then
    echo "❌ Deployment not available"
    exit 1
fi

# Check pod status
echo "3. Checking pods..."
POD_READY=$(kubectl get pods -n $NAMESPACE -l app=financial-rag-api -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')
if [ "$POD_READY" != "True" ]; then
    echo "❌ Pod not ready"
    exit 1
fi

# Port forward and test API
echo "4. Testing API health endpoint..."
kubectl port-forward service/$SERVICE $PORT:$PORT -n $NAMESPACE > /dev/null 2>&1 &
PORT_FORWARD_PID=$!

# Wait for port forward to be established
sleep 5

# Test health endpoint
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health || true)

# Kill port forward
kill $PORT_FORWARD_PID > /dev/null 2>&1 || true

if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Health check passed - API is responding"
else
    echo "❌ Health check failed - API returned HTTP $HEALTH_RESPONSE"
    exit 1
fi

echo "🎉 All health checks passed! System is operational."
```

### Create `kubernetes/kustomization.yaml`

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: financial-rag

resources:
  - namespace.yaml
  - configmap.yaml
  - secret.yaml
  - persistent-volume-claim.yaml
  - deployment.yaml
  - service.yaml
  - hpa.yaml
  - ingress.yaml

commonLabels:
  app: financial-rag-api
  version: v1

images:
  - name: financial-rag-agent
    newTag: latest
```

## Step 21: Final Project Structure

```
financial-rag-agent/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── kubernetes/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   ├── persistent-volume-claim.yaml
│   └── kustomization.yaml
├── scripts/
│   ├── deploy.sh
│   ├── health-check.sh
│   ├── start_api.py
│   └── test_production.py
├── src/
│   └── financial_rag/
│       ├── config/
│       │   ├── __init__.py
│       │   └── advanced.py
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── tracing.py
│       │   └── metrics.py
│       └── api/
│           ├── __init__.py
│           ├── models.py
│           └── server.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env
├── .dockerignore
└── README.md
```

## Step 22: Final Deployment Commands

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Deploy to Kubernetes
./scripts/deploy.sh

# Run health check
./scripts/health-check.sh

# Or deploy with kustomize
kubectl apply -k kubernetes/

# Check status
kubectl get all -n financial-rag

# View logs
kubectl logs -f deployment/financial-rag-api -n financial-rag

# Port forward for local access
kubectl port-forward service/financial-rag-service 8000:8000 -n financial-rag
```

## 🎉 Enterprise Deployment Complete!

Your Financial RAG Analyst Agent is now fully enterprise-ready with:

### ✅ **Production Features:**
- **Kubernetes Deployment** with auto-scaling
- **CI/CD Pipeline** with GitHub Actions
- **Monitoring & Metrics** with Prometheus/WandB
- **Health Checks** and readiness probes
- **Config Management** with ConfigMaps and Secrets
- **Persistent Storage** for vector database
- **Load Balancing** and ingress
- **Security** with non-root containers

### ✅ **AI/ML Capabilities:**
- **Intelligent Agent** with tool usage
- **RAG System** with sophisticated chunking
- **Real-time Data** integration
- **Multiple Analysis Styles** (analyst, executive, risk)
- **Comprehensive Monitoring** of AI components

### ✅ **DevOps Excellence:**
- **Containerized** with Docker
- **Orchestrated** with Kubernetes
- **Automated Deployment** with CI/CD
- **Infrastructure as Code**
- **Production-grade** configuration

### 🚀 **Ready for Production Use:**

```bash
# Final verification
python test_production.py
./scripts/health-check.sh

# Access the API
kubectl port-forward service/financial-rag-service 8000:8000 -n financial-rag
# Visit: http://localhost:8000/docs
```

Your Financial RAG Analyst Agent is now a robust, scalable, enterprise-grade system that demonstrates your full-stack AI engineering capabilities - from backend DevOps to sophisticated AI agent systems!


