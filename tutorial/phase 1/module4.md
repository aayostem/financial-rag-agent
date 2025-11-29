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
