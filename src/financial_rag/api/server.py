from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, metrics_collector
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger
import os
import time

from financial_rag.config import config
from financial_rag.api.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    AnalysisStyle,
)
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.agents.financial_agent import FinancialAgent
from financial_rag.ingestion.sec_ingestor import SECIngestor
from financial_rag.ingestion.document_processor import DocumentProcessor

from financial_rag.agents.multi_modal_analyst import MultiModalAnalystAgent
from financial_rag.api.models import (
    EarningsCallRequest,
    DocumentAnalysisRequest,
    MultiModalAnalysisRequest,
    EarningsCallResponse,
    DocumentAnalysisResponse,
    MultiModalResponse,
)

# Add new imports
from financial_rag.agents.coordinator import AgentCoordinator, AnalysisType
from financial_rag.api.models import (
    MultiAgentAnalysisRequest,
    InvestmentCommitteeRequest,
    AgentAnalysisResponse,
    InvestmentCommitteeResponse,
    AnalysisHistoryResponse,
)

# Add new imports
from financial_rag.agents.predictive_analyst import PredictiveAnalystAgent
from financial_rag.api.models import (
    PredictiveAnalysisRequest,
    EarningsPredictionRequest,
    TrendAnalysisRequest,
    PredictiveAnalysisResponse,
    EarningsPredictionResponse,
    TrendAnalysisResponse,
)


class FinancialRAGAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Financial RAG Analyst API",
            description="Enterprise-grade Financial Analysis using RAG and AI Agents",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        self.vector_store = None
        self.agent = None
        self.real_time_agent = None
        self.multi_modal_agent = None
        self.agent_coordinator = None
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
                logger.warning(
                    "No existing vector store found. Please ingest data first."
                )
                # Create empty vector store for health checks
                from langchain.schema import Document

                self.vector_store = vector_manager.create_vector_store(
                    [Document(page_content="", metadata={})]
                )

            # Initialize agent
            self.agent = FinancialAgent(self.vector_store, enable_monitoring=True)
            if self.vector_store:
                self.real_time_agent = RealTimeAnalystAgent(
                    self.vector_store, enable_monitoring=True
                )
                logger.success("Real-time analyst agent initialized")

            if self.vector_store:
                self.multi_modal_agent = MultiModalAnalystAgent(
                    self.vector_store, enable_monitoring=True
                )
                logger.success("Multi-modal analyst agent initialized")

            # Initialize multi-agent coordinator
            if self.vector_store:
                self.agent_coordinator = AgentCoordinator(
                    self.vector_store, enable_monitoring=True
                )
                logger.success("Multi-agent coordinator initialized")

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
                llm_ready = (
                    self.agent is not None and self.agent.rag_chain.llm is not None
                )

                return HealthResponse(
                    status="healthy",
                    version="0.1.0",
                    vector_store_ready=vector_store_ready,
                    llm_ready=llm_ready,
                )
            except Exception as e:
                return HealthResponse(
                    status="unhealthy",
                    version="0.1.0",
                    vector_store_ready=False,
                    llm_ready=False,
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
                        request.question, prompt_style=request.analysis_style
                    )

                # Convert source documents to response format
                source_docs = []
                for doc in result.get("source_documents", []):
                    source_docs.append(
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": getattr(doc, "score", None),
                        }
                    )

                # Record metrics
                metrics_collector.record_query(
                    status="success",
                    agent_type="agent" if request.use_agent else "rag",
                    duration=time.time() - start_time,
                )

                return QueryResponse(
                    question=result["question"],
                    answer=result["answer"],
                    agent_type=result.get("agent_type", "simple_rag"),
                    latency_seconds=result.get(
                        "latency_seconds", time.time() - start_time
                    ),
                    source_documents=source_docs,
                    error=result.get("error"),
                )

            except Exception as e:

                # Record failure metrics
                metrics_collector.record_query(
                    status="failure",
                    agent_type="agent" if request.use_agent else "rag",
                    duration=time.time() - start_time,
                )

                logger.error(f"Error in query analysis: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Analysis failed: {str(e)}"
                )

        @self.app.post("/ingest/sec", response_model=IngestionResponse)
        async def ingest_sec_filings(
            request: IngestionRequest, background_tasks: BackgroundTasks
        ):
            """Ingest SEC filings for a ticker"""
            try:
                # Run ingestion in background
                background_tasks.add_task(
                    self._ingest_sec_filings_background,
                    request.ticker,
                    request.filing_type,
                    request.years,
                )

                return IngestionResponse(
                    ticker=request.ticker,
                    filings_downloaded=0,  # Will be updated in background
                    documents_processed=0,
                    success=True,
                    error=None,
                )

            except Exception as e:
                return IngestionResponse(
                    ticker=request.ticker,
                    filings_downloaded=0,
                    documents_processed=0,
                    success=False,
                    error=str(e),
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
                        "status": "operational",
                    }
                else:
                    return {"error": "Collection not available"}

            except Exception as e:
                return {"error": f"Failed to get stats: {str(e)}"}

        @self.app.get("/metrics")
        async def metrics_endpoint():
            """Prometheus metrics endpoint"""
            from fastapi.responses import Response

            return Response(
                content=metrics_collector.get_metrics(), media_type="text/plain"
            )

        @self.app.post("/query/real-time", response_model=RealTimeResponse)
        async def real_time_query_analysis(request: RealTimeQueryRequest):
            """Real-time financial analysis with market context"""
            try:
                if not self.real_time_agent:
                    raise HTTPException(
                        status_code=503, detail="Real-time agent not initialized"
                    )

                result = await self.real_time_agent.analyze_with_market_context(
                    question=request.question, tickers=request.tickers
                )

                return RealTimeResponse(
                    question=result["question"],
                    answer=result["answer"],
                    agent_type=result.get("agent_type", "real_time_analyst"),
                    latency_seconds=result.get("latency_seconds", 0),
                    source_documents=result.get("source_documents", []),
                    real_time_insights=result.get("real_time_insights", []),
                    alerts=result.get("alerts", []),
                    market_context=result.get("real_time_context", {}),
                )

            except Exception as e:
                logger.error(f"Error in real-time analysis: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Real-time analysis failed: {str(e)}"
                )

        @self.app.websocket("/ws/market-updates")
        async def websocket_market_updates(websocket: WebSocket):
            """WebSocket for real-time market updates"""
            await websocket.accept()
            try:
                # Receive tickers to monitor
                data = await websocket.receive_text()
                tickers = json.loads(data).get("tickers", [])

                async def send_update(context):
                    await websocket.send_json(
                        {
                            "type": "market_update",
                            "data": context,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # Start streaming updates
                await self.real_time_agent.stream_market_updates(tickers, send_update)

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.close()

        @self.app.get("/market/summary")
        async def get_market_summary():
            """Get current market summary"""
            try:
                if not self.real_time_agent:
                    raise HTTPException(
                        status_code=503, detail="Real-time agent not initialized"
                    )

                summary = await self.real_time_agent.real_time_data.get_market_summary()
                return summary

            except Exception as e:
                logger.error(f"Error getting market summary: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/market/stock/{ticker}")
        async def get_stock_data(ticker: str):
            """Get real-time data for specific stock"""
            try:
                if not self.real_time_agent:
                    raise HTTPException(
                        status_code=503, detail="Real-time agent not initialized"
                    )

                data = await self.real_time_agent.real_time_data.get_live_market_data(
                    [ticker]
                )
                return data.get(ticker, {})

            except Exception as e:
                logger.error(f"Error getting stock data for {ticker}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/analysis/earnings-call", response_model=EarningsCallResponse)
        async def analyze_earnings_call(request: EarningsCallRequest):
            """Analyze earnings call audio"""
            try:
                if not self.multi_modal_agent:
                    raise HTTPException(
                        status_code=503, detail="Multi-modal agent not initialized"
                    )

                # Determine audio source
                audio_path = None
                if request.audio_file:
                    audio_path = request.audio_file
                elif request.audio_url:
                    # Download audio from URL (implementation needed)
                    audio_path = await self.download_audio(request.audio_url)

                if not audio_path:
                    raise HTTPException(
                        status_code=400, detail="No audio source provided"
                    )

                result = await self.multi_modal_agent.analyze_earnings_call(
                    audio_path, request.ticker
                )

                return EarningsCallResponse(
                    ticker=result["ticker"],
                    call_analysis=result["call_analysis"],
                    insights=result["insights"],
                    summary=result["summary"],
                    timestamp=result.get("timestamp", datetime.now().isoformat()),
                )

            except Exception as e:
                logger.error(f"Error analyzing earnings call: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Earnings call analysis failed: {str(e)}"
                )

        @self.app.post("/analysis/documents", response_model=DocumentAnalysisResponse)
        async def analyze_financial_documents(request: DocumentAnalysisRequest):
            """Analyze financial documents"""
            try:
                if not self.multi_modal_agent:
                    raise HTTPException(
                        status_code=503, detail="Multi-modal agent not initialized"
                    )

                # Determine document sources
                document_paths = []
                if request.local_paths:
                    document_paths.extend(request.local_paths)
                if request.document_urls:
                    # Download documents from URLs (implementation needed)
                    downloaded_paths = await self.download_documents(
                        request.document_urls
                    )
                    document_paths.extend(downloaded_paths)

                if not document_paths:
                    raise HTTPException(status_code=400, detail="No documents provided")

                result = await self.multi_modal_agent.analyze_financial_documents(
                    document_paths, request.ticker
                )

                return DocumentAnalysisResponse(
                    ticker=request.ticker,
                    document_insights=result["document_insights"],
                    financial_health=result["financial_health"],
                    recommendation=result["investment_recommendation"],
                    timestamp=datetime.now().isoformat(),
                )

            except Exception as e:
                logger.error(f"Error analyzing documents: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Document analysis failed: {str(e)}"
                )

        @self.app.post("/analysis/comprehensive", response_model=MultiModalResponse)
        async def comprehensive_analysis(request: MultiModalAnalysisRequest):
            """Comprehensive multi-modal analysis"""
            try:
                if not self.multi_modal_agent:
                    raise HTTPException(
                        status_code=503, detail="Multi-modal agent not initialized"
                    )

                # Prepare analysis inputs
                audio_path = None
                if request.earnings_call:
                    if request.earnings_call.audio_file:
                        audio_path = request.earnings_call.audio_file
                    elif request.earnings_call.audio_url:
                        audio_path = await self.download_audio(
                            request.earnings_call.audio_url
                        )

                document_paths = []
                if request.documents:
                    if request.documents.local_paths:
                        document_paths.extend(request.documents.local_paths)
                    if request.documents.document_urls:
                        downloaded_paths = await self.download_documents(
                            request.documents.document_urls
                        )
                        document_paths.extend(downloaded_paths)

                result = await self.multi_modal_agent.comprehensive_analysis(
                    ticker=request.ticker,
                    audio_path=audio_path,
                    document_paths=document_paths,
                )

                return MultiModalResponse(**result)

            except Exception as e:
                logger.error(f"Error in comprehensive analysis: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Comprehensive analysis failed: {str(e)}"
                )

        async def download_audio(self, audio_url: str) -> str:
            """Download audio from URL (placeholder implementation)"""
            # In production, implement actual download logic
            return f"/tmp/audio_{hash(audio_url)}.mp3"

        async def download_documents(self, document_urls: List[str]) -> List[str]:
            """Download documents from URLs (placeholder implementation)"""
            # In production, implement actual download logic
            return [f"/tmp/doc_{hash(url)}.pdf" for url in document_urls]

        @self.app.post("/agents/analyze", response_model=AgentAnalysisResponse)
        async def multi_agent_analysis(request: MultiAgentAnalysisRequest):
            """Coordinate analysis across multiple specialized agents"""
            try:
                if not self.agent_coordinator:
                    raise HTTPException(
                        status_code=503, detail="Agent coordinator not initialized"
                    )

                # Convert analysis type string to enum
                try:
                    analysis_type = AnalysisType(request.analysis_type)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid analysis type: {request.analysis_type}",
                    )

                result = await self.agent_coordinator.coordinate_analysis(
                    ticker=request.ticker,
                    analysis_type=analysis_type,
                    research_focus=request.research_focus,
                )

                return AgentAnalysisResponse(**result)

            except Exception as e:
                logger.error(f"Error in multi-agent analysis: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Multi-agent analysis failed: {str(e)}"
                )

        @self.app.post("/agents/committee", response_model=InvestmentCommitteeResponse)
        async def investment_committee_analysis(request: InvestmentCommitteeRequest):
            """Simulate investment committee meeting with all agents"""
            try:
                if not self.agent_coordinator:
                    raise HTTPException(
                        status_code=503, detail="Agent coordinator not initialized"
                    )

                result = await self.agent_coordinator.conduct_investment_committee(
                    ticker=request.ticker
                )

                return InvestmentCommitteeResponse(**result)

            except Exception as e:
                logger.error(f"Error in investment committee: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Investment committee failed: {str(e)}"
                )

        @self.app.get("/agents/history", response_model=AnalysisHistoryResponse)
        async def get_analysis_history(ticker: Optional[str] = None):
            """Get analysis history from agent coordinator"""
            try:
                if not self.agent_coordinator:
                    raise HTTPException(
                        status_code=503, detail="Agent coordinator not initialized"
                    )

                history = await self.agent_coordinator.get_analysis_history(ticker)

                return AnalysisHistoryResponse(
                    ticker=ticker, history=history, total_analyses=len(history)
                )

            except Exception as e:
                logger.error(f"Error getting analysis history: {e}")
                raise HTTPException(
                    status_code=500, detail=f"History retrieval failed: {str(e)}"
                )

        @self.app.get("/agents/status")
        async def get_agent_status():
            """Get status of all specialized agents"""
            try:
                if not self.agent_coordinator:
                    raise HTTPException(
                        status_code=503, detail="Agent coordinator not initialized"
                    )

                status = {
                    "coordinator": "active",
                    "specialized_agents": list(
                        self.agent_coordinator.agent_registry.keys()
                    ),
                    "total_analyses_performed": len(
                        self.agent_coordinator.analysis_history
                    ),
                    "agent_capabilities": self.get_agent_capabilities(),
                }

                return status

            except Exception as e:
                logger.error(f"Error getting agent status: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Status check failed: {str(e)}"
                )

        def get_agent_capabilities(self) -> Dict[str, Any]:
            """Get capabilities of each specialized agent"""
            return {
                "research_analyst": {
                    "description": "Deep fundamental research and analysis",
                    "capabilities": [
                        "Business model analysis",
                        "Industry and competitive analysis",
                        "Financial statement deep dive",
                        "Growth prospect evaluation",
                        "Risk assessment",
                    ],
                },
                "quantitative_analyst": {
                    "description": "Quantitative analysis and financial modeling",
                    "capabilities": [
                        "Financial ratio analysis",
                        "Risk modeling and metrics",
                        "Valuation modeling (DCF, Comps)",
                        "Statistical analysis",
                        "Portfolio optimization",
                    ],
                },
                "risk_officer": {
                    "description": "Risk management and compliance",
                    "capabilities": [
                        "Enterprise risk assessment",
                        "Regulatory compliance monitoring",
                        "Risk mitigation strategies",
                        "Stress testing",
                        "Internal controls evaluation",
                    ],
                },
            }

    async def _ingest_sec_filings_background(
        self, ticker: str, filing_type: str, years: int
    ):
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

            logger.success(
                f"Background ingestion completed for {ticker}: {len(chunks)} chunks"
            )

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
        log_level="info",
    )
