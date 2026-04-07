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
from app.agents.base import normalize_ticker, _model_name_suggests_groq
from app.data.provider import get_data_provider, DataProvider
from app.json_util import to_json_safe

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Multi-Agent Stock Analysis System...")
    logger.info(f"📊 Version: {__version__}")
    gq = (settings.groq_api_key or "").strip()
    gx = (settings.grok_api_key or "").strip()
    if gq:
        logger.info(f"🤖 LLM: Groq — model {settings.groq_model} @ {settings.groq_api_base}")
    elif gx and _model_name_suggests_groq(settings.grok_model):
        logger.info(
            f"🤖 LLM: Groq (GROK_* + Llama-style model) — {settings.grok_model} @ {settings.groq_api_base}"
        )
    else:
        logger.info(f"🤖 LLM: xAI Grok — model {settings.grok_model} @ {settings.grok_api_base}")

    if not gq and not gx:
        logger.warning("⚠️ Set GROQ_API_KEY (Groq) or GROK_API_KEY (xAI) — LLM features will not work")

    yield

    logger.info("👋 Shutting down Multi-Agent Stock Analysis System...")


app = FastAPI(
    title="Multi-Agent Stock Analysis System",
    description="Multi-agent stock analysis API.",
    version=__version__,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
async def root():
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


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_stock(request: AnalysisRequest):
    try:
        ticker = normalize_ticker(request.ticker)
        logger.info(f"Starting analysis for {ticker}")

        data_provider = get_data_provider(prefer_realtime=True)
        realtime_quote = data_provider.get_quote(ticker, force_realtime=True)
        logger.info(f"Real-time quote for {ticker}: ₹{realtime_quote.get('last_price')} [{realtime_quote.get('source')}]")

        result = await run_analysis(
            ticker=ticker,
            task=request.task,
            account_size=request.account_size,
            risk_tolerance=request.risk_tolerance.value
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {result.get('error', 'Unknown error')}"
            )

        data_source = f"{realtime_quote.get('source', 'Yahoo Finance')} (Current Price) + Yahoo Finance (Historical & Fundamentals)"

        safe_quote = (
            to_json_safe(realtime_quote)
            if realtime_quote and not realtime_quote.get("error")
            else None
        )

        return AnalysisResponse(
            success=True,
            ticker=ticker,
            final_recommendation=str(result.get("final_recommendation") or ""),
            analysis_results=to_json_safe(result.get("analysis_results", {})),
            realtime_quote=safe_quote,
            data_source=data_source,
            messages=to_json_safe(result.get("messages", [])),
            errors=[str(e) for e in (result.get("errors") or [])],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quick/market-data", response_model=MarketDataResponse, tags=["Quick Analysis"])
async def quick_market_data(request: QuickAnalysisRequest):
    try:
        ticker = normalize_ticker(request.ticker)
        stock_data = MarketDataAgent.fetch_stock_data(ticker)
        if not stock_data.get("success"):
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        technical_indicators = MarketDataAgent.compute_technical_indicators(ticker)
        fundamental_data = MarketDataAgent.get_fundamental_data(ticker)
        returns_data = MarketDataAgent.calculate_returns(ticker)

        return MarketDataResponse(
            success=True,
            ticker=ticker,
            stock_data=to_json_safe(stock_data),
            technical_indicators=to_json_safe(technical_indicators),
            fundamental_data=to_json_safe(fundamental_data),
            returns_data=to_json_safe(returns_data),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quick/technical", response_model=TechnicalResponse, tags=["Quick Analysis"])
async def quick_technical(request: QuickAnalysisRequest):
    try:
        ticker = normalize_ticker(request.ticker)
        regime_analysis = TechnicalAgent.identify_market_regime(ticker)
        support_resistance = TechnicalAgent.find_support_resistance(ticker)
        chart_patterns = TechnicalAgent.detect_chart_patterns(ticker)
        trend_structure = TechnicalAgent.analyze_trend_structure(ticker)

        if not regime_analysis.get("success"):
            raise HTTPException(status_code=404, detail=f"Insufficient data for {ticker}")

        return TechnicalResponse(
            success=True,
            ticker=ticker,
            regime_analysis=to_json_safe(regime_analysis),
            support_resistance=to_json_safe(support_resistance),
            chart_patterns=to_json_safe(chart_patterns),
            trend_structure=to_json_safe(trend_structure),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Technical analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quick/risk", response_model=RiskResponse, tags=["Quick Analysis"])
async def quick_risk(request: PositionSizeRequest):
    try:
        ticker = normalize_ticker(request.ticker)
        position_sizing = RiskAgent.calculate_position_size(
            ticker,
            request.account_size,
            request.risk_per_trade_pct,
            request.stop_loss_pct
        )

        if not position_sizing.get("success"):
            raise HTTPException(status_code=404, detail=f"Insufficient data for {ticker}")

        var_analysis = RiskAgent.compute_var(ticker)
        drawdown_analysis = RiskAgent.analyze_drawdown(ticker)
        risk_limits = RiskAgent.generate_risk_limits(ticker, request.account_size)

        return RiskResponse(
            success=True,
            ticker=ticker,
            position_sizing=to_json_safe(position_sizing),
            var_analysis=to_json_safe(var_analysis),
            drawdown_analysis=to_json_safe(drawdown_analysis),
            risk_limits=to_json_safe(risk_limits),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/risk", response_model=PortfolioRiskResponse, tags=["Portfolio"])
async def portfolio_risk(request: PortfolioAnalysisRequest):
    try:
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

        result = RiskAgent.portfolio_risk_metrics(
            request.tickers,
            request.weights
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Analysis failed"))

        return PortfolioRiskResponse(
            success=True,
            tickers=request.tickers,
            portfolio_metrics=to_json_safe(result.get("portfolio_metrics")),
            correlation_matrix=to_json_safe(result.get("correlation_matrix")),
            risk_contribution=to_json_safe(result.get("risk_contribution")),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio risk error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indicators/{ticker}", tags=["Utilities"])
async def get_indicators(ticker: str):
    try:
        ticker = normalize_ticker(ticker)
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
    try:
        ticker = normalize_ticker(ticker)
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
    try:
        ticker = normalize_ticker(ticker)
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
    try:
        ticker = normalize_ticker(ticker)
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
    try:
        ticker = normalize_ticker(ticker)
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
    try:
        ticker = normalize_ticker(ticker)
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
    import yfinance as yf

    try:
        ticker = normalize_ticker(ticker)
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

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


@app.get("/screener/presets", tags=["Screener"])
async def get_screener_presets():
    return {
        "presets": ScreenerAgent.get_preset_screens(),
        "sectors": list(INDIAN_STOCK_UNIVERSE.keys()),
        "total_stocks": sum(len(v) for v in INDIAN_STOCK_UNIVERSE.values())
    }


@app.get("/screener/universe", tags=["Screener"])
async def get_stock_universe(sectors: str = None):
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
    try:
        ticker = normalize_ticker(ticker)
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
    try:
        ticker_list = [normalize_ticker(t.strip()) for t in tickers.split(",")]

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


@app.get("/signals/{ticker}", tags=["Signals"])
async def get_trading_signal(ticker: str):
    try:
        ticker = normalize_ticker(ticker)
        result = SignalsAgent.generate_signal(ticker)

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
    try:
        ticker = normalize_ticker(ticker)
        signals_agent = SignalsAgent()
        signal_data = SignalsAgent.generate_signal(ticker)

        if not signal_data.get("success"):
            raise HTTPException(status_code=404, detail=signal_data.get("error", f"No data found for {ticker}"))

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
    try:
        sector_list = sectors.split(",") if sectors else None
        tickers = ScreenerAgent.get_stock_universe(sector_list)[:50]

        result = SignalsAgent.scan_for_signals(tickers, signal_filter)

        if result.get("results"):
            result["results"] = result["results"][:limit]

        return result

    except Exception as e:
        logger.error(f"Signal scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/indicators/{ticker}", tags=["Signals"])
async def get_signal_indicators(ticker: str):
    try:
        ticker = normalize_ticker(ticker)
        result = SignalsAgent.calculate_indicators(ticker)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", f"No data found for {ticker}"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indicators error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/quote/{symbol}", tags=["Real-Time Data"])
async def get_realtime_quote(symbol: str, force_realtime: bool = True):
    try:
        symbol = normalize_ticker(symbol)
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
    try:
        provider = get_data_provider()
        return provider.get_market_status()
    except Exception as e:
        logger.error(f"Market status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/index/{index_name}", tags=["Real-Time Data"])
async def get_index_quote(index_name: str = "NIFTY 50"):
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
    try:
        provider = get_data_provider()
        gainers = provider.get_top_gainers(index.upper())
        return {"index": index, "gainers": gainers, "count": len(gainers)}
    except Exception as e:
        logger.error(f"Top gainers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/losers", tags=["Real-Time Data"])
async def get_top_losers(index: str = "NIFTY 50"):
    try:
        provider = get_data_provider()
        losers = provider.get_top_losers(index.upper())
        return {"index": index, "losers": losers, "count": len(losers)}
    except Exception as e:
        logger.error(f"Top losers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/bulk-quotes", tags=["Real-Time Data"])
async def get_bulk_quotes(symbols: str):
    try:
        symbol_list = [normalize_ticker(s.strip()) for s in symbols.split(",")]

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


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
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
        port=8001,
        reload=settings.debug
    )
