from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL')")
    task: str = Field(default="Comprehensive stock analysis")
    account_size: float = Field(default=100000.0, ge=1000)
    risk_tolerance: RiskTolerance = Field(default=RiskTolerance.MODERATE)


class QuickAnalysisRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")


class PortfolioAnalysisRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    weights: Optional[List[float]] = Field(default=None)


class PositionSizeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    account_size: float = Field(default=100000.0, ge=1000)
    risk_per_trade_pct: float = Field(default=2.0, ge=0.1, le=10.0)
    stop_loss_pct: Optional[float] = Field(default=None, ge=0.5, le=50.0)


class AgentMessage(BaseModel):
    role: str
    agent: str
    content: str


class AnalysisResponse(BaseModel):
    success: bool
    ticker: str
    final_recommendation: str
    analysis_results: Dict[str, Any]
    realtime_quote: Optional[Dict[str, Any]] = None
    data_source: Optional[str] = None
    messages: List[Dict[str, Any]]
    errors: List[str]


class MarketDataResponse(BaseModel):
    success: bool
    ticker: str
    stock_data: Optional[Dict[str, Any]] = None
    technical_indicators: Optional[Dict[str, Any]] = None
    fundamental_data: Optional[Dict[str, Any]] = None
    returns_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TechnicalResponse(BaseModel):
    success: bool
    ticker: str
    regime_analysis: Optional[Dict[str, Any]] = None
    support_resistance: Optional[Dict[str, Any]] = None
    chart_patterns: Optional[Dict[str, Any]] = None
    trend_structure: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RiskResponse(BaseModel):
    success: bool
    ticker: str
    position_sizing: Optional[Dict[str, Any]] = None
    var_analysis: Optional[Dict[str, Any]] = None
    drawdown_analysis: Optional[Dict[str, Any]] = None
    risk_limits: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PortfolioRiskResponse(BaseModel):
    success: bool
    tickers: List[str]
    portfolio_metrics: Optional[Dict[str, Any]] = None
    correlation_matrix: Optional[Dict[str, Any]] = None
    risk_contribution: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    agents: List[str]
