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
    analysis_style: AnalysisStyle = Field(
        default=AnalysisStyle.ANALYST, description="Style of analysis"
    )
    use_agent: bool = Field(
        default=True, description="Whether to use the intelligent agent"
    )
    search_type: SearchType = Field(
        default=SearchType.SIMILARITY, description="Search type for retrieval"
    )


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


# Add new models for real-time features
class RealTimeQueryRequest(QueryRequest):
    include_real_time: bool = Field(
        default=True, description="Include real-time market data"
    )
    tickers: Optional[List[str]] = Field(
        default=None, description="Specific tickers to analyze"
    )
    stream_updates: bool = Field(default=False, description="Stream real-time updates")


class MarketAlert(BaseModel):
    ticker: str
    type: str
    message: str
    timestamp: str
    data: Dict[str, Any]


class RealTimeResponse(QueryResponse):
    real_time_insights: List[str] = []
    alerts: List[MarketAlert] = []
    market_context: Optional[Dict[str, Any]] = None


class StreamUpdate(BaseModel):
    type: str = Field(..., description="Type of update: price, news, alert")
    data: Dict[str, Any]
    timestamp: str


# Add new models for multi-modal features
class EarningsCallRequest(BaseModel):
    ticker: str
    audio_url: Optional[str] = Field(None, description="URL to earnings call audio")
    audio_file: Optional[str] = Field(None, description="Path to local audio file")


class DocumentAnalysisRequest(BaseModel):
    ticker: str
    document_urls: List[str] = Field(..., description="URLs to financial documents")
    local_paths: Optional[List[str]] = Field(
        None, description="Local paths to documents"
    )


class MultiModalAnalysisRequest(BaseModel):
    ticker: str
    earnings_call: Optional[EarningsCallRequest] = None
    documents: Optional[DocumentAnalysisRequest] = None
    include_real_time: bool = True


class EarningsCallResponse(BaseModel):
    ticker: str
    call_analysis: Dict[str, Any]
    insights: Dict[str, Any]
    summary: str
    timestamp: str


class DocumentAnalysisResponse(BaseModel):
    ticker: str
    document_insights: Dict[str, Any]
    financial_health: Dict[str, Any]
    recommendation: Dict[str, Any]
    timestamp: str


class MultiModalResponse(BaseModel):
    ticker: str
    analysis_modes: List[str]
    unified_insights: Dict[str, Any]
    executive_summary: str
    detailed_analysis: Dict[str, Any]
    timestamp: str


# Add new models for multi-agent system
class MultiAgentAnalysisRequest(BaseModel):
    ticker: str
    analysis_type: str = Field(
        default="comprehensive", description="Type of analysis to perform"
    )
    research_focus: str = Field(
        default="comprehensive", description="Research focus area"
    )
    involve_agents: Optional[List[str]] = Field(
        None, description="Specific agents to involve"
    )


class InvestmentCommitteeRequest(BaseModel):
    ticker: str
    include_historical: bool = Field(
        default=True, description="Include historical analysis context"
    )


class AgentAnalysisResponse(BaseModel):
    ticker: str
    analysis_type: str
    agents_involved: List[str]
    consensus_analysis: Dict[str, Any]
    conflicting_viewpoints: List[Dict[str, Any]]
    overall_recommendation: Dict[str, Any]
    synthesis_timestamp: str


class InvestmentCommitteeResponse(BaseModel):
    ticker: str
    committee_members: List[str]
    final_decision: Dict[str, Any]
    meeting_minutes: str
    analyses_presented: Dict[str, Any]
    timestamp: str


class AnalysisHistoryResponse(BaseModel):
    ticker: Optional[str]
    history: List[Dict[str, Any]]
    total_analyses: int


# Add new models for predictive analytics
class PredictiveAnalysisRequest(BaseModel):
    ticker: str
    analysis_horizon: str = Field(
        default="30d", description="Analysis horizon (e.g., 30d, 3m, 1y)"
    )
    include_forecasts: bool = Field(default=True, description="Include price forecasts")
    confidence_level: float = Field(
        default=0.95, ge=0.5, le=0.99, description="Confidence level for forecasts"
    )


class EarningsPredictionRequest(BaseModel):
    ticker: str
    next_quarter: bool = Field(
        default=True, description="Predict next quarter earnings"
    )
    include_analyst_consensus: bool = Field(
        default=True, description="Include analyst consensus"
    )


class TrendAnalysisRequest(BaseModel):
    ticker: str
    period: str = Field(default="2y", description="Analysis period")
    include_technical: bool = Field(
        default=True, description="Include technical analysis"
    )


class PredictiveAnalysisResponse(BaseModel):
    ticker: str
    analysis_horizon: str
    price_forecast: Dict[str, Any]
    momentum_analysis: Dict[str, Any]
    technical_outlook: Dict[str, Any]
    fundamental_forecast: Dict[str, Any]
    composite_score: float
    investment_thesis: str
    key_risks: List[Dict[str, str]]
    monitoring_recommendations: List[str]
    timestamp: str


class EarningsPredictionResponse(BaseModel):
    ticker: str
    period: str
    predicted_eps: float
    confidence: float
    analyst_consensus: Dict[str, Any]
    deviation_from_consensus: float
    surprise_probability: float
    key_drivers: List[str]
    timestamp: str


class TrendAnalysisResponse(BaseModel):
    ticker: str
    analysis_period: str
    trend_analysis: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    seasonality_analysis: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    insights: List[str]
    timestamp: str
