"""
Multi-Agent Stock Analysis System - FastAPI Backend

A sophisticated multi-agent system for comprehensive stock analysis using:
- LangChain & LangGraph for agent orchestration
- Grok API for LLM capabilities
- FastAPI for the REST API backend
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict
import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import settings
from app.models import (
    AnalysisRequest,
    AnalysisResponse,
    QuickAnalysisRequest,
    MarketDataResponse,
    TechnicalResponse,
    RiskResponse,
    PortfolioAnalysisRequest,
    PortfolioRiskResponse,
    PositionSizeRequest,
    HealthResponse,
)
from app.agents.market_data_agent import MarketDataAgent
from app.agents.technical_agent import TechnicalAgent
from app.agents.risk_agent import RiskAgent
from app.agents.supervisor import run_analysis
from app.agents.screener_agent import ScreenerAgent, INDIAN_STOCK_UNIVERSE
from app.agents.signals_agent import SignalsAgent
from app.data.provider import get_data_provider, DataProvider

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("🚀 Starting Multi-Agent Stock Analysis System...")
    logger.info(f"📊 Version: {__version__}")
    logger.info(f"🤖 Using Grok model: {settings.grok_model}")
    
    # Validate API key
    if not settings.grok_api_key:
        logger.warning("⚠️ GROK_API_KEY not set - LLM features will not work")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Multi-Agent Stock Analysis System...")


# Create FastAPI application
app = FastAPI(
    title="Multi-Agent Stock Analysis System",
    description="""
A sophisticated multi-agent system for comprehensive stock analysis.

## Agents

- **Market Data Agent**: Fetches market data, computes technical indicators, feature engineering
- **Technical Agent**: Chart patterns, market structure, regime detection, support/resistance
- **Risk Agent**: Position sizing, VaR calculation, drawdown analysis, portfolio guardrails
- **Supervisor Agent**: Orchestrates all agents and synthesizes recommendations

## Features

- Real-time market data from Yahoo Finance
- Comprehensive technical indicator calculation
- Multiple risk metrics (VaR, Sharpe, Sortino, Calmar)
- Chart pattern recognition
- Market regime classification
- Position sizing recommendations
- Portfolio-level analysis

## Usage

1. Use `/analyze` for full multi-agent analysis
2. Use `/quick/*` endpoints for single-agent analysis
3. Use `/portfolio/*` endpoints for portfolio-level analysis
    """,
    version=__version__,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Health & Info Endpoints ==============

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Multi-Agent Stock Analysis System",
        "version": __version__,
        "description": "A multi-agent system for comprehensive stock analysis",
        "docs": "/docs",
        "agents": [
            "MarketDataAgent",
            "TechnicalAgent", 
            "RiskAgent",
            "SupervisorAgent"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=__version__,
        agents=[
            "MarketDataAgent",
            "TechnicalAgent",
            "RiskAgent",
            "SupervisorAgent"
        ]
    )


# ============== Full Analysis Endpoint ==============

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_stock(request: AnalysisRequest):
    """
    Run comprehensive multi-agent stock analysis with REAL-TIME data
    
    This endpoint orchestrates all agents in sequence:
    1. Market Data Agent - fetches and processes data
    2. Technical Agent - analyzes charts and patterns
    3. Risk Agent - calculates risk metrics
    4. Supervisor Agent - synthesizes final recommendation
    
    Data Sources:
    - Real-time quote: NSE India / Yahoo Finance
    - Historical data: Yahoo Finance
    - Fundamentals: Yahoo Finance
    
    Returns a comprehensive analysis with actionable recommendations.
    """
    try:
        logger.info(f"Starting analysis for {request.ticker}")
        
        # Get real-time quote first
        data_provider = get_data_provider(prefer_realtime=True)
        realtime_quote = data_provider.get_quote(request.ticker, force_realtime=True)
        logger.info(f"Real-time quote for {request.ticker}: ₹{realtime_quote.get('last_price')} [{realtime_quote.get('source')}]")
        
        result = await run_analysis(
            ticker=request.ticker,
            task=request.task,
            account_size=request.account_size,
            risk_tolerance=request.risk_tolerance.value
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        # Determine data source description
        data_source = f"{realtime_quote.get('source', 'Yahoo Finance')} (Current Price) + Yahoo Finance (Historical & Fundamentals)"
        
        return AnalysisResponse(
            success=True,
            ticker=request.ticker.upper(),
            final_recommendation=result.get("final_recommendation", ""),
            analysis_results=result.get("analysis_results", {}),
            realtime_quote=realtime_quote if not realtime_quote.get('error') else None,
            data_source=data_source,
            messages=result.get("messages", []),
            errors=result.get("errors", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Quick Analysis Endpoints (Single Agent) ==============

@app.post("/quick/market-data", response_model=MarketDataResponse, tags=["Quick Analysis"])
async def quick_market_data(request: QuickAnalysisRequest):
    """
    Quick market data analysis (Market Data Agent only)
    
    Returns:
    - Stock price data
    - Technical indicators
    - Fundamental data
    - Return metrics
    """
    try:
        # Use static methods directly (no LLM required)
        stock_data = MarketDataAgent.fetch_stock_data(request.ticker)
        if not stock_data.get("success"):
            raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
        
        technical_indicators = MarketDataAgent.compute_technical_indicators(request.ticker)
        fundamental_data = MarketDataAgent.get_fundamental_data(request.ticker)
        returns_data = MarketDataAgent.calculate_returns(request.ticker)
        
        return MarketDataResponse(
            success=True,
            ticker=request.ticker.upper(),
            stock_data=stock_data,
            technical_indicators=technical_indicators,
            fundamental_data=fundamental_data,
            returns_data=returns_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quick/technical", response_model=TechnicalResponse, tags=["Quick Analysis"])
async def quick_technical(request: QuickAnalysisRequest):
    """
    Quick technical analysis (Technical Agent only)
    
    Returns:
    - Market regime classification
    - Support/Resistance levels
    - Chart patterns
    - Trend structure analysis
    """
    try:
        # Use static methods directly (no LLM required)
        regime_analysis = TechnicalAgent.identify_market_regime(request.ticker)
        support_resistance = TechnicalAgent.find_support_resistance(request.ticker)
        chart_patterns = TechnicalAgent.detect_chart_patterns(request.ticker)
        trend_structure = TechnicalAgent.analyze_trend_structure(request.ticker)
        
        if not regime_analysis.get("success"):
            raise HTTPException(status_code=404, detail=f"Insufficient data for {request.ticker}")
        
        return TechnicalResponse(
            success=True,
            ticker=request.ticker.upper(),
            regime_analysis=regime_analysis,
            support_resistance=support_resistance,
            chart_patterns=chart_patterns,
            trend_structure=trend_structure
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Technical analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quick/risk", response_model=RiskResponse, tags=["Quick Analysis"])
async def quick_risk(request: PositionSizeRequest):
    """
    Quick risk analysis (Risk Agent only)
    
    Returns:
    - Position sizing recommendations
    - Value at Risk (VaR)
    - Drawdown analysis
    - Risk limits and guardrails
    """
    try:
        # Use static methods directly (no LLM required)
        position_sizing = RiskAgent.calculate_position_size(
            request.ticker,
            request.account_size,
            request.risk_per_trade_pct,
            request.stop_loss_pct
        )
        
        if not position_sizing.get("success"):
            raise HTTPException(status_code=404, detail=f"Insufficient data for {request.ticker}")
        
        var_analysis = RiskAgent.compute_var(request.ticker)
        drawdown_analysis = RiskAgent.analyze_drawdown(request.ticker)
        risk_limits = RiskAgent.generate_risk_limits(request.ticker, request.account_size)
        
        return RiskResponse(
            success=True,
            ticker=request.ticker.upper(),
            position_sizing=position_sizing,
            var_analysis=var_analysis,
            drawdown_analysis=drawdown_analysis,
            risk_limits=risk_limits
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Portfolio Analysis Endpoints ==============

@app.post("/portfolio/risk", response_model=PortfolioRiskResponse, tags=["Portfolio"])
async def portfolio_risk(request: PortfolioAnalysisRequest):
    """
    Portfolio-level risk analysis
    
    Returns:
    - Portfolio metrics (return, volatility, Sharpe)
    - Correlation matrix
    - Risk contribution by asset
    - Diversification ratio
    """
    try:
        # Validate weights if provided
        if request.weights:
            if len(request.weights) != len(request.tickers):
                raise HTTPException(
                    status_code=400,
                    detail="Number of weights must match number of tickers"
                )
            if abs(sum(request.weights) - 1.0) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail="Weights must sum to 1.0"
                )
        
        # Use static method directly (no LLM required)
        result = RiskAgent.portfolio_risk_metrics(
            request.tickers,
            request.weights
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Analysis failed"))
        
        return PortfolioRiskResponse(
            success=True,
            tickers=request.tickers,
            portfolio_metrics=result.get("portfolio_metrics"),
            correlation_matrix=result.get("correlation_matrix"),
            risk_contribution=result.get("risk_contribution")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio risk error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Utility Endpoints ==============

@app.get("/indicators/{ticker}", tags=["Utilities"])
async def get_indicators(ticker: str):
    """Get technical indicators for a stock"""
    try:
        # Use static method directly without instantiating agent
        result = MarketDataAgent.compute_technical_indicators(ticker)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indicators error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regime/{ticker}", tags=["Utilities"])
async def get_regime(ticker: str):
    """Get market regime classification for a stock"""
    try:
        # Use static method directly
        result = TechnicalAgent.identify_market_regime(ticker)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Regime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/support-resistance/{ticker}", tags=["Utilities"])
async def get_support_resistance(ticker: str):
    """Get support and resistance levels for a stock"""
    try:
        # Use static method directly
        result = TechnicalAgent.find_support_resistance(ticker)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"S/R error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns/{ticker}", tags=["Utilities"])
async def get_patterns(ticker: str):
    """Get chart patterns for a stock"""
    try:
        # Use static method directly
        result = TechnicalAgent.detect_chart_patterns(ticker)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patterns error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/var/{ticker}", tags=["Utilities"])
async def get_var(ticker: str, position_value: float = 10000.0):
    """Get Value at Risk analysis for a stock"""
    try:
        # Use static method directly
        result = RiskAgent.compute_var(ticker, position_value)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"VaR error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drawdown/{ticker}", tags=["Utilities"])
async def get_drawdown(ticker: str):
    """Get drawdown analysis for a stock"""
    try:
        # Use static method directly
        result = RiskAgent.analyze_drawdown(ticker)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drawdown error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart/{ticker}", tags=["Utilities"])
async def get_chart_data(ticker: str, period: str = "6mo", interval: str = "1d"):
    """
    Get OHLCV chart data for a stock
    
    Args:
        ticker: Stock symbol (e.g., RELIANCE.NS)
        period: Data period (1mo, 3mo, 6mo, 1y, 2y, 5y)
        interval: Data interval (1d, 1wk, 1mo)
    
    Returns:
        OHLCV data formatted for charting
    """
    import yfinance as yf
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        # Format data for lightweight-charts
        candles = []
        volumes = []
        
        for idx, row in df.iterrows():
            timestamp = int(idx.timestamp())
            candles.append({
                "time": timestamp,
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2)
            })
            volumes.append({
                "time": timestamp,
                "value": int(row["Volume"]),
                "color": "rgba(38, 166, 154, 0.5)" if row["Close"] >= row["Open"] else "rgba(239, 83, 80, 0.5)"
            })
        
        # Calculate moving averages
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        
        sma20 = []
        sma50 = []
        
        for idx, row in df.iterrows():
            timestamp = int(idx.timestamp())
            if not pd.isna(row["SMA20"]):
                sma20.append({"time": timestamp, "value": round(row["SMA20"], 2)})
            if not pd.isna(row["SMA50"]):
                sma50.append({"time": timestamp, "value": round(row["SMA50"], 2)})
        
        return {
            "success": True,
            "ticker": ticker.upper(),
            "candles": candles,
            "volumes": volumes,
            "sma20": sma20,
            "sma50": sma50,
            "period": period,
            "interval": interval,
            "data_points": len(candles)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SCREENER ENDPOINTS ====================

@app.get("/screener/presets", tags=["Screener"])
async def get_screener_presets():
    """Get all available preset screening strategies"""
    return {
        "presets": ScreenerAgent.get_preset_screens(),
        "sectors": list(INDIAN_STOCK_UNIVERSE.keys()),
        "total_stocks": sum(len(v) for v in INDIAN_STOCK_UNIVERSE.values())
    }


@app.get("/screener/universe", tags=["Screener"])
async def get_stock_universe(sectors: str = None):
    """
    Get the stock universe for screening
    
    Args:
        sectors: Comma-separated list of sectors (e.g., "banking,it,pharma")
    """
    sector_list = sectors.split(",") if sectors else None
    stocks = ScreenerAgent.get_stock_universe(sector_list)
    return {
        "sectors": sector_list or ["large_cap"],
        "stock_count": len(stocks),
        "stocks": stocks
    }


@app.post("/screener/run", tags=["Screener"])
async def run_screener(
    screen_type: str = "value_picks",
    sectors: str = None,
    top_n: int = 15
):
    """
    Run a stock screener with preset or custom filters
    
    Args:
        screen_type: Preset screen type (value_picks, growth_stars, momentum_leaders, etc.)
        sectors: Comma-separated list of sectors
        top_n: Number of top results to return
    """
    try:
        screener = ScreenerAgent()
        sector_list = sectors.split(",") if sectors else None
        
        result = await screener.run_screen_with_ai(
            screen_type=screen_type,
            sectors=sector_list,
            top_n=top_n
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Screening failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Screener error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screener/custom", tags=["Screener"])
async def run_custom_screener(
    filters: Dict[str, Any],
    sectors: str = None,
    top_n: int = 15
):
    """
    Run a custom stock screener with user-defined filters
    
    Available filters:
    - pe_max, pe_min: P/E ratio bounds
    - pb_max: Price to Book maximum
    - peg_max: PEG ratio maximum
    - roe_min: Return on Equity minimum (%)
    - profit_margin_min: Profit margin minimum (%)
    - revenue_growth_min: Revenue growth minimum (%)
    - earnings_growth_min: Earnings growth minimum (%)
    - rsi_max, rsi_min: RSI bounds
    - above_sma_50: Must be above 50-day SMA (boolean)
    - above_sma_200: Must be above 200-day SMA (boolean)
    - returns_1m_min, returns_3m_min: Minimum returns (%)
    - pct_from_high_max: Maximum % from 52-week high
    - volume_ratio_min: Minimum volume ratio vs average
    - debt_to_equity_max: Maximum debt to equity ratio
    - dividend_yield_min: Minimum dividend yield (%)
    - market_cap_min, market_cap_max: Market cap bounds (in Cr)
    
    Example:
    {
        "pe_max": 20,
        "roe_min": 15,
        "above_sma_50": true,
        "debt_to_equity_max": 100
    }
    """
    try:
        screener = ScreenerAgent()
        sector_list = sectors.split(",") if sectors else None
        
        result = await screener.run_screen_with_ai(
            screen_type="custom",
            sectors=sector_list,
            custom_filters=filters,
            top_n=top_n
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Screening failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Custom screener error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/screener/stock/{ticker}", tags=["Screener"])
async def get_stock_metrics(ticker: str):
    """Get all screening metrics for a single stock"""
    try:
        result = ScreenerAgent.fetch_stock_metrics(ticker)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/screener/compare", tags=["Screener"])
async def compare_stocks(tickers: str):
    """
    Compare multiple stocks side by side
    
    Args:
        tickers: Comma-separated list of tickers (e.g., "RELIANCE.NS,TCS.NS,INFY.NS")
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        
        if len(ticker_list) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 tickers to compare")
        
        if len(ticker_list) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 tickers allowed for comparison")
        
        results = []
        for ticker in ticker_list:
            data = ScreenerAgent.fetch_stock_metrics(ticker)
            if data.get("success"):
                results.append(data)
        
        if not results:
            raise HTTPException(status_code=404, detail="No valid data found for provided tickers")
        
        return {
            "comparison": results,
            "count": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TRADING SIGNALS ENDPOINTS ====================

@app.get("/signals/{ticker}", tags=["Signals"])
async def get_trading_signal(ticker: str):
    """
    Get trading signal for a stock with entry/exit levels
    
    Returns:
    - Signal type (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
    - Signal strength (1-5 stars)
    - Entry zone (price range to buy)
    - Stop loss level with risk %
    - Target prices (T1, T2, T3)
    - Risk/Reward ratio
    - Bullish and bearish reasons
    - All technical indicators used
    """
    try:
        result = SignalsAgent.generate_signal(ticker.upper())
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", f"No data found for {ticker}"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/{ticker}/analyze", tags=["Signals"])
async def get_signal_with_ai_analysis(ticker: str):
    """
    Get trading signal with AI-powered analysis and recommendations
    
    Returns everything from /signals/{ticker} plus:
    - AI analysis with action plan
    - Risk warnings
    - Timing recommendations
    - Position sizing suggestions
    """
    try:
        signals_agent = SignalsAgent()
        signal_data = SignalsAgent.generate_signal(ticker.upper())
        
        if not signal_data.get("success"):
            raise HTTPException(status_code=404, detail=signal_data.get("error", f"No data found for {ticker}"))
        
        # Generate AI analysis
        ai_analysis = await signals_agent.generate_ai_analysis(signal_data)
        signal_data["ai_analysis"] = ai_analysis
        
        return signal_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signal analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/scan", tags=["Signals"])
async def scan_stocks_for_signals(
    signal_filter: str = None,
    sectors: str = None,
    limit: int = 20
):
    """
    Scan multiple stocks for trading signals
    
    Args:
        signal_filter: Filter by signal type ("BUY" or "SELL")
        sectors: Comma-separated sectors to scan (e.g., "banking,it")
        limit: Maximum number of results
    
    Returns list of stocks with their signals, sorted by signal strength
    """
    try:
        # Get stock universe
        sector_list = sectors.split(",") if sectors else None
        tickers = ScreenerAgent.get_stock_universe(sector_list)[:50]  # Limit to 50 for performance
        
        result = SignalsAgent.scan_for_signals(tickers, signal_filter)
        
        if result.get("results"):
            result["results"] = result["results"][:limit]
        
        return result
        
    except Exception as e:
        logger.error(f"Signal scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/indicators/{ticker}", tags=["Signals"])
async def get_signal_indicators(ticker: str):
    """
    Get all technical indicators used for signal generation
    
    Returns detailed indicator values:
    - RSI with signal
    - MACD line, signal, histogram
    - Moving averages (20, 50, 200)
    - Bollinger Bands
    - ATR
    - Volume analysis
    - Support/Resistance levels
    - Trend analysis
    """
    try:
        result = SignalsAgent.calculate_indicators(ticker.upper())
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", f"No data found for {ticker}"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indicators error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REAL-TIME DATA ENDPOINTS ====================

@app.get("/realtime/quote/{symbol}", tags=["Real-Time Data"])
async def get_realtime_quote(symbol: str, force_realtime: bool = True):
    """
    Get real-time stock quote from NSE India
    
    This endpoint provides LIVE market data during trading hours (9:15 AM - 3:30 PM IST).
    Outside market hours, it falls back to Yahoo Finance.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS", or "RELIANCE.NS")
        force_realtime: Force NSE real-time data even outside market hours
    
    Returns:
        Real-time quote with last price, change, OHLC, volume
    """
    try:
        provider = get_data_provider(prefer_realtime=True)
        quote = provider.get_quote(symbol, force_realtime=force_realtime)
        
        if quote.get('error'):
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        return quote
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Real-time quote error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/market-status", tags=["Real-Time Data"])
async def get_market_status():
    """
    Get current Indian market status
    
    Returns whether the market is open/closed and timestamp
    """
    try:
        provider = get_data_provider()
        return provider.get_market_status()
    except Exception as e:
        logger.error(f"Market status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/index/{index_name}", tags=["Real-Time Data"])
async def get_index_quote(index_name: str = "NIFTY 50"):
    """
    Get real-time index quote
    
    Args:
        index_name: Index name (e.g., "NIFTY 50", "NIFTY BANK", "NIFTY IT")
    
    Returns:
        Index value, change, advances/declines
    """
    try:
        provider = get_data_provider()
        quote = provider.get_index(index_name.upper())
        
        if not quote:
            raise HTTPException(status_code=404, detail=f"Index {index_name} not found")
        
        return quote
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Index quote error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/gainers", tags=["Real-Time Data"])
async def get_top_gainers(index: str = "NIFTY 50"):
    """
    Get top 10 gaining stocks in an index
    
    Args:
        index: Index name (default: "NIFTY 50")
    
    Returns:
        List of top gainers with symbol, price, change %
    """
    try:
        provider = get_data_provider()
        gainers = provider.get_top_gainers(index.upper())
        return {"index": index, "gainers": gainers, "count": len(gainers)}
    except Exception as e:
        logger.error(f"Top gainers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/losers", tags=["Real-Time Data"])
async def get_top_losers(index: str = "NIFTY 50"):
    """
    Get top 10 losing stocks in an index
    
    Args:
        index: Index name (default: "NIFTY 50")
    
    Returns:
        List of top losers with symbol, price, change %
    """
    try:
        provider = get_data_provider()
        losers = provider.get_top_losers(index.upper())
        return {"index": index, "losers": losers, "count": len(losers)}
    except Exception as e:
        logger.error(f"Top losers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/bulk-quotes", tags=["Real-Time Data"])
async def get_bulk_quotes(symbols: str):
    """
    Get real-time quotes for multiple stocks
    
    Args:
        symbols: Comma-separated list of symbols (e.g., "RELIANCE,TCS,INFY")
    
    Returns:
        List of quotes for all requested symbols
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        if len(symbol_list) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed")
        
        provider = get_data_provider()
        quotes = []
        
        for symbol in symbol_list:
            quote = provider.get_quote(symbol, force_realtime=True)
            if not quote.get('error'):
                quotes.append(quote)
        
        return {
            "requested": len(symbol_list),
            "found": len(quotes),
            "quotes": quotes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk quotes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

