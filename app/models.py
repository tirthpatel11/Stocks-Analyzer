"""
Pydantic models for API requests and responses
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class RiskTolerance(str, Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AnalysisRequest(BaseModel):
    """Request model for stock analysis"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL')")
    task: str = Field(
        default="Comprehensive stock analysis",
        description="Specific analysis task or question"
    )
    account_size: float = Field(
        default=100000.0,
        ge=1000,
        description="Account size for position sizing calculations"
    )
    risk_tolerance: RiskTolerance = Field(
        default=RiskTolerance.MODERATE,
        description="Risk tolerance level"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "task": "Should I buy this stock? What's the risk/reward?",
                "account_size": 50000.0,
                "risk_tolerance": "moderate"
            }
        }


class QuickAnalysisRequest(BaseModel):
    """Request model for quick single-agent analysis"""
    ticker: str = Field(..., description="Stock ticker symbol")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "TSLA"
            }
        }


class PortfolioAnalysisRequest(BaseModel):
    """Request model for portfolio analysis"""
    tickers: List[str] = Field(..., min_length=2, max_length=20, description="List of ticker symbols")
    weights: Optional[List[float]] = Field(
        default=None,
        description="Portfolio weights (must sum to 1.0, uses equal weight if not provided)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                "weights": [0.25, 0.25, 0.25, 0.25]
            }
        }


class PositionSizeRequest(BaseModel):
    """Request model for position sizing"""
    ticker: str = Field(..., description="Stock ticker symbol")
    account_size: float = Field(default=100000.0, ge=1000, description="Account size")
    risk_per_trade_pct: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Maximum percentage of account to risk per trade"
    )
    stop_loss_pct: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=50.0,
        description="Fixed stop loss percentage (uses ATR-based if not provided)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "NVDA",
                "account_size": 25000.0,
                "risk_per_trade_pct": 1.5,
                "stop_loss_pct": 5.0
            }
        }


class AgentMessage(BaseModel):
    """Model for agent messages in response"""
    role: str
    agent: str
    content: str


class AnalysisResponse(BaseModel):
    """Response model for stock analysis"""
    success: bool
    ticker: str
    final_recommendation: str
    analysis_results: Dict[str, Any]
    realtime_quote: Optional[Dict[str, Any]] = None  # Real-time price data
    data_source: Optional[str] = None  # Data source info
    messages: List[Dict[str, Any]]
    errors: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "ticker": "RELIANCE.NS",
                "final_recommendation": "BUY with moderate confidence...",
                "analysis_results": {},
                "realtime_quote": {
                    "last_price": 2890.50,
                    "change_pct": 1.25,
                    "source": "NSE_REALTIME"
                },
                "data_source": "NSE India (Real-time) + Yahoo Finance (Historical)",
                "messages": [],
                "errors": []
            }
        }


class MarketDataResponse(BaseModel):
    """Response model for market data analysis"""
    success: bool
    ticker: str
    stock_data: Optional[Dict[str, Any]] = None
    technical_indicators: Optional[Dict[str, Any]] = None
    fundamental_data: Optional[Dict[str, Any]] = None
    returns_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TechnicalResponse(BaseModel):
    """Response model for technical analysis"""
    success: bool
    ticker: str
    regime_analysis: Optional[Dict[str, Any]] = None
    support_resistance: Optional[Dict[str, Any]] = None
    chart_patterns: Optional[Dict[str, Any]] = None
    trend_structure: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RiskResponse(BaseModel):
    """Response model for risk analysis"""
    success: bool
    ticker: str
    position_sizing: Optional[Dict[str, Any]] = None
    var_analysis: Optional[Dict[str, Any]] = None
    drawdown_analysis: Optional[Dict[str, Any]] = None
    risk_limits: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PortfolioRiskResponse(BaseModel):
    """Response model for portfolio risk analysis"""
    success: bool
    tickers: List[str]
    portfolio_metrics: Optional[Dict[str, Any]] = None
    correlation_matrix: Optional[Dict[str, Any]] = None
    risk_contribution: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    agents: List[str]

