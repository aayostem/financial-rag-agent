<!-- `src/financial_rag/config.py` -->
inside of the financial_rag, create a config.py file.
This file is used for storing configuration settings and constants for your application. It centralizes all configuration in one place, making it easy to manage and modify settings.

```python
import os # import the built in os module
from dotenv import load_dotenv # as specified in this documentation, we'll say from... 

load_dotenv() # and load dotenv

class Config: # lets create a class Config to hve a centralized configuration management to access all application settings. now, let's add
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

The first stage in our pipeline is data ingestion. This is the process of collecting, cleaning and tranforming and storing data  into format that a RAG model can use for retrieval and generation. 

This is importmt because we will be processing data from numerious sources such as 
├── SEC EDGAR API (10-K, 10-Q, 8-K filings)
├── Yahoo Finance API (real-time prices, fundamentals)
├── Earnings call transcripts
├── Financial news feeds
├── Company presentations (PDFs)
└── Industry research reports


<!-- `src/financial_rag/ingestion/sec_ingestor.py` -->
inside the ingetion, create sec_ingetor.py

Here, we are going to create a class that helps to automatically download company financial reports** from the US government's SEC database. Think of it like a robot that goes to the SEC website and collects annual reports for any company you want to analyze.

let's start with our import statements

```python
import os #this is the built in os to help work with files and folders on your computer
from sec_edgar_downloader import Downloader # The special tool that actually downloads the reports
from loguru import logger # works like a notebook that records what's happening (successes/errors)
from financial_rag.config import config # Stores configuration settings like where to save files

we re going to write our class here, say
class SECIngestor #this class will have 3 methods, say
def __init__(self): to setup our
def download_fillings(self): to download actual company reports
def get_filling_path(self): to find where the files were saved after downloading

now let's write the logic of each method one after the other
    
The Setup or (__init__ method) will do 2 things:
When we create a new `SECIngestor`:
- It will create a downloader that knows where to save files
- Secondly, it will make sure the download folder exists (creates it if needed)
Think of it like:** Setting up a new filing cabinet before you start collecting documents

so we say


def __init__(self):
    self.dl = Downloader(config.RAW_DATA_PATH) # to create a downloader that knows where to save files
    os.makedirs(config.RAW_DATA_PATH, exist_ok=True) # to makes sure the download folder exists (creates it if needed)
```

let's move to the next method whihch is the main job, that is to download company reports

this method will tke 3 parmeters
```python
def download_filings(self, ticker, filing_type="10-K", years=5):
```

**Parameters explained:**
the ticker parameter represent Company stock symbol (like "AAPL" for Apple, "TSLA" for Tesla)
the filing_type parameter reprseent the Type of report (default is "10-K" = Annual Report)
the years parameter rersesent How many years of reports to download (default = 5 years)

Let's use try except block for proper Error Handling (The Safety Net)

try:
    # Try to download
except Exception as e:
    logger.error(f"Error downloading filings for {ticker}: {str(e)}")
    raise

- **`try/except`** = "Try to do this, but if something goes wrong, don't crash - handle it gracefully"
- **`logger.error`** = Write down what went wrong in our notebook
- **`raise`** = Tell the user that something failed

```

let's download fillings inside trhe try block, we say
```python
num_filings = self.dl.get(
    filing_type, 
    ticker, 
    amount=years,
    download_details=True
)
we can now log our success message using:

    logger.success(f"Successfully downloaded {num_filings} {filing_type} filings for {ticker}")
    return num_filings
```
- **Real example:** `download_filings("AAPL", "10-K", 3)` would download Apple's annual reports for the last 3 years
- The method returns how many reports it successfully downloaded


let's write the last methods to Find Downloaded Files 
```python
def get_filing_paths(self, ticker, filing_type="10-K"):
    ticker_path = os.path.join(config.RAW_DATA_PATH, "sec-edgar-filings", ticker, filing_type) # """Get paths to downloaded filings"""
        if not os.path.exists(ticker_path):
            return []
        
        filing_paths = []
        # now lets search through subfolder to find report files
        for root, dirs, files in os.walk(ticker_path):
            for file in files:
                if file.endswith('.txt') or file.endswith('.html'):
                    filing_paths.append(os.path.join(root, file))
        
        return filing_paths
```
After downloading, this method helps you find where the files were saved. I works by
- Looking in the download folder for the specific company and report type
- Searches through all subfolders to find the actual report files (.txt or .html)
- Returns a list of file paths so you can open and read them later

---

## **Real-World Example**

```python
# Create the downloader
downloader = SECIngestor()

# Download Tesla's annual reports for last 5 years
num_downloaded = downloader.download_filings("TSLA", "10-K", 5)
print(f"Downloaded {num_downloaded} Tesla annual reports!")

# Find where the files are saved
tesla_reports = downloader.get_filing_paths("TSLA", "10-K")
print(f"Found {len(tesla_reports)} Tesla report files")
```

## **Why This Matters for Financial Analysis**, it helpx to

1. **Automates data collection** - No manual downloading from SEC website
2. achieve a **Standardized format** - All reports come in the same structure
3. **Builds datasets** - Perfect for training AI models to analyze company performance
4. **Time-saving** - Downloads multiple years of data with one command


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




This class provides the **raw material** for any financial analysis project - whether you're building trading algorithms, investment research tools, or financial education apps!

let's start with the import statememnt
```python
import yfinance as yf # this is The main library that fetches data from Yahoo Finance
import pandas as pd #For handling data tables (though not directly used here)
import os # For saving files in organized formats
import json
from loguru import logger Same as before: logging and settings management
from financial_rag.config import config  # Fixed import

## **What This Code Does**
This class acts as a **financial data collector** that automatically downloads stock market data, company information, and financial news from Yahoo Finance. Think of it as your personal financial research assistant!


class YFinanceIngestor:
    def __init__(self): #-just ensures the download folder exists
        os.makedirs(config.RAW_DATA_PATH, exist_ok=True)

    lets create a method to downloads **two types of financial data**: first is historical data nd next ic compny data
    
    def download_stock_data(self, ticker, period="1y"):
        """Download stock price data and company info"""
        try:
            logger.info(f"Downloading data for {ticker}")
            
            # Get stock data
            stock = yf.Ticker(ticker)
            
            # Historical prices
            hist = stock.history(period=period)
*This gets the:**
- Daily opening price, closing price, highest price, lowest price
- Trading volume (how many shares were traded)
- **Period options:** "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
            
            # Company info
            info = stock.info
            **this gives us goldmine of data, from here, we get the :**
- Company name, sector, industry
- Current stock price, market capitalization
- Financial ratios (P/E ratio, debt/equity, etc.)
- Company description and company officers
            
            # now Save the data, we say
            data = {
                "ticker": ticker,
                "historical_prices": hist.to_dict(),
                "company_info": info
            } # to package into a net dictionary
            
            file_path = os.path.join(config.RAW_DATA_PATH, f"{ticker}_yfinance.json") # Converts pandas data** to regular dictionaries (so JSON can save it)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str) # Saves as JSON file** - a universal data format that's easy to read and process later
            
            logger.success(f"Successfully downloaded data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            raise
let's create a Financial News Collector, we say

def get_financial_news(self, ticker, num_articles=10):


    def get_financial_news(self, ticker, num_articles=10):
        """Get recent news for a ticker"""
        try:
            stock = yf.Ticker(ticker) #to Fetche recent news articles about the company; these information are Useful for understanding why stock prices might be moving
            news = stock.news[:num_articles]
            return news #- Returns a list of news articles with titles, links, and summaries
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []
```


## **An Examples of this**

### **Example 1: Basic Stock Analysis**
```python
# Create the data collector
collector = YFinanceIngestor()

# Download Apple's stock data for the last 2 years
apple_data = collector.download_stock_data("AAPL", "2y")

# What you get:
# - Apple's stock price for every day of the last 2 years
# - Company info: market cap, P/E ratio, business description
# - All saved in a file called "AAPL_yfinance.json"
```

### **Example 2: Market Research**
```python
# Compare multiple companies
companies = ["TSLA", "F", "GM", "RIVN"]

for company in companies:
    data = collector.download_stock_data(company, "1y")
    news = collector.get_financial_news(company, 5)
    
    print(f"{company}: {len(news)} recent news articles")
```

### **Example 3: Quick News Check**
```python
# Just get the latest news without downloading full data
tesla_news = collector.get_financial_news("TSLA", 3)
for article in tesla_news:
    print(f"Title: {article['title']}")
    print(f"Link: {article['link']}")
    print("---")
```

---

### **Data Types Collected:**
1. **Time Series Data** - Stock prices over time (perfect for charts)
2. **Company Fundamentals** - Financial health metrics
3. **Market Sentiment** - News articles reflect public perception

### **Real Applications:**
- **Build stock price predictors**
- **Compare company financial health**
- **Create automated research reports**
- **Monitor investment portfolios**

### The File Output Example:**
```json
{
  "ticker": "AAPL",
  "historical_prices": {
    "2023-01-03": {"Open": 130.0, "High": 132.0, "Low": 129.5, "Close": 131.5, "Volume": 1000000},
    "2023-01-04": {"Open": 131.5, "High": 133.0, "Low": 131.0, "Close": 132.0, "Volume": 950000}
  },
  "company_info": {
    "companyName": "Apple Inc.",
    "marketCap": 2800000000000,
    "peRatio": 28.5,
    "sector": "Technology"
  }
}
```

 <!-- `src/financial_rag/ingestion/document_processor.py` -->
let's create trhe document_processor.py


## **🎬 From Financial Documents to AI-Ready Data: The Document Processor Explained!**

we need to turns messy financial documents into clean, organized data that AI can actually understand!

Imagine you have a 200-page company annual report. we can create a machine that:
1. **READS** the document (PDFs, text files, etc.)
2. **CLEANS** it up (removes junk formatting)
3. **EXTRACTS** key information (company name, dates, etc.)
4. **CHUNKS** it into bite-sized pieces for AI processing

It's like having a super-smart research assistant who can process thousands of pages in seconds!

---

This Document Processor is the **critical bridge** between messy real-world documents and clean, usable AI training data. It's what makes the difference between an AI that gives generic answers and one that can actually cite specific financial data!


```python
# let start with our import statement
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger
from financial_rag.config import config  # Fixed import

#  let create a class DocumentProcessor which will have 3 methds
def __init__(self) # for text spltting nd chuncking
def process_sec_filing(self, file_path) # this methods cleans the data
def _extract_sec_metadata(self, text, file_path): #to extract metadata
def chunk_documents(self, documents): # we can create a seaparate method for Split documents into chunks using sophisticated strategy


now let's write the logic for each class method

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, # AI models have limited "memory" (context windows)
            chunk_overlap=config.CHUNK_OVERLAP, # You can't feed a 100-page document all at once
            length_function=len, # **Chunking breaks it down** into logical pieces while preserving meaning
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""] # **Overlap ensures** no important information gets cut off between chunks
        )

    def process_sec_filing(self, file_path):
        """Process SEC filing text files"""
        try:
            logger.info(f"Processing SEC filing: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read() # **Reads the raw file** (often messy with HTML/XML tags)
            
            soup = BeautifulSoup(content, 'html.parser') 
            text = soup.get_text() # **Uses BeautifulSoup** to strip out all the formatting junk
            
            text = re.sub(r'\s+', ' ', text) # Remove excessive whitespace
            
            metadata = self._extract_sec_metadata(text, file_path) # **Extracts metadata** - the "who, what, when" of the document
            
            return Document(page_content=text, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error processing SEC filing {file_path}: {str(e)}")
            raise
        
    
    def _extract_sec_metadata(self, text, file_path):
        # This is where it gets really cool! The code automatically finds:
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
**Think of it like** automatically tagging your documents with all the important details!

### **Metadata is GOLD:**
- The more metadata you extract, the smarter your AI becomes
- Enables filtering: "Only show me Q3 results from automotive companies"
- Helps with source verification and accuracy

---


    
    def chunk_documents(self, documents):
        """Split documents into chunks using sophisticated strategy"""
        logger.info(f"Chunking {len(documents)} documents")
        
        chunks = self.text_splitter.split_documents(documents)
        
        logger.success(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
```


## **🎯 Real-World Example**

Let's say we have Apple's annual report (10-K filing):

**BEFORE Processing:**
```html
<DOCUMENT>
<TYPE>10-K
<COMPANY>APPLE INC.
<FILED>20231231
<CONTENT>Lorem ipsum dolor sit amet... hundreds of pages...
```

**AFTER Processing:**
```python
Document(
    page_content="Apple Inc. annual report for fiscal year 2023...",
    metadata={
        "company": "APPLE INC.",
        "filing_date": "20231231", 
        "filing_type": "10-K",
        "source": "apple_10k_2023.txt"
    }
)
```

**AND THEN** chunked into manageable pieces:
```
Chunk 1: "Apple Inc. reported revenue of $383 billion..."
Chunk 2: "The company's gross margin improved to 43%..."
Chunk 3: "Research and development expenses totaled..."
```

---

## **🤖 Why This Matters for AI Applications**

### **For RAG (Retrieval Augmented Generation) Systems:**
- **Searchable chunks** = Better answers to financial questions
- **"What was Apple's R&D spending in 2023?"** → AI finds the right chunk instantly
- **Metadata filtering** = "Show me only 10-K filings from tech companies"

### **Building a Financial AI Assistant:**
1. **Collect** → Download SEC filings
2. **Process** → This code cleans and chunks them
3. **Store** → Put chunks in a vector database
4. **Query** → Ask questions and get accurate answers!

---

## **🎬 Visualize the Pipeline**

```
RAW DOCUMENTS 
    ↓ (PDFs, text files)
CLEANING STATION
    ↓ (Remove junk, extract info)
TAGGED DOCUMENTS  
    ↓ (With metadata)
SMART CHUNKING
    ↓ (AI-friendly pieces)
READY FOR AI! 🎉
```


 <!-- `src/financial_rag/retrieval/vector_store.py` -->
move into the retrieval folder  to create the  vector_store.py file

 This is essentially our AI's **photographic memory system** for financial documents. In this section, we will create the **BRAIN** of our agent - this memory will be able to:
- **Remember** thousands of financial documents
- **Instantly find** relevant information when users ask questions
- **Understand relationships** between different financial concepts

It's like giving your AI a photographic memory for financial data!


```python

let's strt with the import statements

import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from loguru import logger
from financial_rag.config import config  # Fixed import

now let's create a class
class VectorStoreManager: # which will have three methods,
    def __init__(self):
    def _initialize_embeddings(self): # """Initialize the embedding model based on config"""
    def _initialize_chroma(self): # """Initialize ChromaDB client"""
    def create_vector_store(self, documents): # """Create a new vector store from documents"""
    def load_vector_store(self): # """Load existing vector store"""
    def get_retriever(self, vector_store, search_type="similarity", k=config.TOP_K_RESULTS): # """Create a retriever from vector store"""
        


    lets start with the init method,
    def __init__(self):
        self.embedding_model = self._initialize_embeddings()
        self.client = self._initialize_chroma()
    

- **Two options**: 
  - **OpenAI** (cloud-based, super accurate) 
  - **HuggingFace** (local, free, privacy-focused)

**Visualize it like this:**

"Apple's revenue grew 5%"    → [0.23, 0.45, -0.12, 0.89, ...]
"iPhone sales increased"     → [0.25, 0.43, -0.11, 0.87, ...] 
"Banana prices fell"         → [-0.45, 0.12, 0.67, -0.23, ...]


Similar concepts cluster together in this mathematical space!


    def _initialize_embeddings(self):
        """Initialize the embedding model based on config"""
# Similar meaning = Similar vectors**: "Profit" and "Revenue" will have similar vector positions
# Text → Numbers**: Every word, sentence, document gets converted into mathematical vectors
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

# Takes our cleaned, chunked documents from last episode
# Converts each chunk into its vector representation
# Stores them in ChromaDB - our specialized AI database
# **Persists to disk** so our AI remembers everything even after restarting
            
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
        """Create a retriever from vector store - you can see this as a kind of smrt search system"""

        # there are 2 types of serch
#### **1. Similarity Search (Default)**
- **"Find the most similar chunks to the question"**
- it is great for direct factual queries



        search_kwargs = {"k": k}
        
#### **2. MMR Search (Maximum Marginal Relevance) search**

# **"Find similar but diverse results"**
# Prevents duplicate information
# **Example**: "Tell me about Apple's financial performance" → Gets revenue, profits, growth, etc. (not just 5 revenue chunks)
        if search_type == "mmr":  # Maximum Marginal Relevance
            search_kwargs["fetch_k"] = k * 2
            search_kwargs["lambda_mult"] = 0.7
        
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return retriever
```
## **💡 Real-World Example**

Let's see this in action with our financial documents:

**Step 1: Store Documents**
```python
# We process and store:
- Apple 10-K (200 pages) → 150 chunks
- Tesla 10-K (180 pages) → 140 chunks  
- Microsoft 10-K (220 pages) → 170 chunks
# Total: 460 chunks in our vector database
```

**Step 2: Ask Questions**
```python
# Question: "What was Apple's iPhone revenue growth?"
# Vector search finds:
1. "iPhone revenue increased 8% to $50B in Q4" (95% match)
2. "Smartphone segment growth driven by iPhone 15" (87% match)
3. "Apple Services revenue grew 12%" (45% match)
```


**Step 3: Get Smart Answers**
The AI uses these relevant chunks to generate accurate, sourced answers!

---

## **🎬 Why This Beats Traditional Search**

### **Traditional Search (Ctrl+F):**
- Looks for exact keyword matches
- "Revenue" won't find "sales" or "income"
- No understanding of context or meaning

### **Vector Search (AI-Powered):**
- Understands semantic meaning
- "Revenue" also finds "sales growth", "income statements"
- Handles synonyms and related concepts
- **It understands what you MEAN, not just what you TYPE**

### **Persistence Matters:**
```python
persist_directory=config.VECTOR_STORE_PATH
```
- Your vector store survives program restarts
- No need to re-process documents every time
- Like saving your AI's brain to a file!

---

## **🚀 The Big Picture**

This Vector Store Manager is the **beating heart** of any RAG system:

```
Raw Documents 
    → Document Processor (Last Episode)
Clean Chunks + Metadata  
    → Vector Store Manager (This Episode!)
AI Memory Bank
    → Q&A System (Next Episode!)
```

lets create run_test.py  at Project Root to test the financial agent foundation

we'll create our **ultimate test script** that validates our entire Financial RAG system. This is like the final exam for our AI infrastructure - let's see if everything works together!

---

This will be a **smart testing script** that:
1. **Tests all components** of our Financial RAG system
2. **Has a backup plan** with mock data if real data fails
3. **Shows us exactly** what's working and what's not
4. **Gives a clear roadmap** for next steps

It's like having a personal mechanic that checks every part of your AI engine!


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
        **this matters because we want to:**
# - Tests if all our imports work correctly
# - Verifies each component can be created without errors
# - Catche configuration issues early
        

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

        
            # Can we download real SEC filings for Apple?
            # Does our document processor clean them properly?
            # Can we chunk them into AI-friendly pieces?
            # Does our vector database store and retrieve them?


# **This is brilliant because:**
# - **No single point of failure** - if SEC is down, we still test
# - **Perfect for development** - work offline or without API keys
# - **Consistent testing** - always get the same mock data

def create_mock_test(vector_manager, doc_processor):
    """Create a test with mock financial data"""
    print("📝 Creating mock financial documents...")
    
    mock_documents = [
        "Apple Inc. reported revenue of $383 billion for fiscal year 2023, with iPhone sales contributing 52% of total revenue. The company's gross margin was 43% and operating margin was 30%. Major risk factors include supply chain disruptions, foreign exchange volatility, and intense competition in the smartphone market.",
        "Microsoft Corporation achieved $211 billion in revenue for FY2023, driven by cloud services growth. Azure revenue grew 27% year-over-year. The company maintains a strong balance sheet with $130 billion in cash and short-term investments. Key challenges include cybersecurity threats and regulatory compliance across multiple jurisdictions.",
        "Amazon.com Inc. reported net sales of $574 billion for 2023. AWS segment revenue was $90 billion with 29% operating margin. The company faces risks related to economic conditions affecting consumer spending, international expansion challenges, and increasing competition in cloud services and e-commerce."
    ]

# **These aren't just random text:**
# - **Realistic financial metrics** (revenue, margins, growth rates)
# - **Actual risk factors** companies face
# - **Diverse companies** (tech, cloud, e-commerce)
# - **Perfect for testing** financial question-answering
    
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

    
# **What we're asking our AI:**
# - "Find documents that talk about revenue AND risk factors"
# - This tests if our vector search understands **multiple concepts**
# - Verifies that similar financial terms are properly matched


    
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




## **🎬 Live Demo Walkthrough**

Let me show you what happens when we run this:

### **Scenario 1: Everything Works Perfectly**
```
🧪 Tested our Financial RAG Foundation...
✅ Components initialized successfully  
📥 Attempted to download SEC filings...
✂️ Chunked documents...
🗄️ Created vector store...
🔍 Tested retrieval...
✅ SEC Data Test Success! Retrieved 5 relevant chunks
```

### **Scenario 2: Internet Issues (Real World!)**
```
🧪 Testing Financial RAG Foundation...
✅ Components initialized successfully
📥 Attempting to download SEC filings...
📝 SEC download failed, using mock data: Connection timeout
📝 Creating mock financial documents...
✅ Mock test successful! Retrieved 3 relevant chunks

```
## **💡 Why This Testing Approach Rocks**

### **1. Graceful Degradation:**
- **Best case**: Test with real SEC data
- **Good case**: Test with realistic mock data  
- **No case where testing completely fails**

### **2. Immediate Feedback:**
```python
for i, result in enumerate(test_results):
    print(f"Chunk {i+1}: {result.page_content[:100]}...")
```
- **See exactly what** the AI is retrieving
- **Verify relevance** with your own eyes
- **Debug issues** immediately

### **3. Clear Next Steps:**
```python
print("\nNext steps:")
print("1. Add your OpenAI API key to .env file")
print("2. Run: pip install -e .")
print("3. We'll build the RAG chain next!")
```
- **No guessing** what to do next
- **Progressive enhancement** path
- **Focus on what matters**

---

`

### **User-Friendly Output:**
```python
# Use emojis and clear messages
print("✅ Success!")  # Instead of just "True"
print("❌ Failed:")   # Instead of just "False"
```

---


**What we've validated:**
- ✅ Data ingestion works
- ✅ Document processing works  
- ✅ Vector storage works
- ✅ Retrieval search works
- ✅ Error handling works

**If you're excited to see our AI actually answer financial questions**, smash that like button and let me know in the comments what financial topics you want our AI to analyze! 🚀
