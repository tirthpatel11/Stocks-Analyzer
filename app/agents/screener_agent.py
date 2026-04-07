import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.base import BaseAgent, get_grok_llm

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif hasattr(np, 'bool8') and isinstance(obj, np.bool8):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif pd.isna(obj):
        return None
    return obj

INDIAN_STOCK_UNIVERSE = {
    "large_cap": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS",
        "NTPC.NS", "POWERGRID.NS", "NESTLEIND.NS", "TATAMOTORS.NS", "M&M.NS",
        "TECHM.NS", "HDFCLIFE.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS",
        "BAJAJFINSV.NS", "GRASIM.NS", "INDUSINDBK.NS", "CIPLA.NS", "ONGC.NS",
        "DIVISLAB.NS", "BPCL.NS", "DRREDDY.NS", "BRITANNIA.NS", "EICHERMOT.NS",
        "COALINDIA.NS", "APOLLOHOSP.NS", "SBILIFE.NS", "TATACONSUM.NS", "HINDALCO.NS"
    ],
    "mid_cap": [
        "PIIND.NS", "MUTHOOTFIN.NS", "VOLTAS.NS", "PAGEIND.NS", "IDFCFIRSTB.NS",
        "JUBLFOOD.NS", "GODREJCP.NS", "BERGEPAINT.NS", "HAVELLS.NS", "PERSISTENT.NS",
        "COFORGE.NS", "LTIM.NS", "ABCAPITAL.NS", "FEDERALBNK.NS", "ASTRAL.NS",
        "POLYCAB.NS", "TRENT.NS", "MPHASIS.NS", "OBEROIRLTY.NS", "CROMPTON.NS",
        "INDHOTEL.NS", "LALPATHLAB.NS", "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS"
    ],
    "banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "AUBANK.NS",
        "RBLBANK.NS", "CANBK.NS", "PNB.NS", "BANKBARODA.NS", "UNIONBANK.NS"
    ],
    "it": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "MINDTREE.NS"
    ],
    "pharma": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "BIOCON.NS", "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "ALKEM.NS"
    ],
    "auto": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "EICHERMOT.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "MOTHERSON.NS", "BOSCHLTD.NS"
    ],
    "fmcg": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
        "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS", "VBL.NS"
    ],
    "metals": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS",
        "NMDC.NS", "SAIL.NS", "JINDALSTEL.NS", "NATIONALUM.NS", "MOIL.NS"
    ]
}


class ScreenerAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="ScreenerAgent",
            description="AI-powered stock screener with multiple filter strategies",
            temperature=0.3
        )

    def _setup_tools(self) -> List[Any]:
        return []

    def _create_system_prompt(self) -> str:
        return """You are the Stock Screener Agent, an AI specialized in discovering investment opportunities in the INDIAN STOCK MARKET.

IMPORTANT: Always use Indian Rupees (₹) for all currency values. Never use $ or USD.

Your responsibilities:
1. Screen stocks based on user-defined criteria
2. Identify undervalued opportunities
3. Find momentum stocks with strong technicals
4. Discover quality companies with consistent growth
5. Provide AI-powered insights on screened stocks

Screening Strategies:
- VALUE: Low P/E, low P/B, high dividend yield, margin of safety
- GROWTH: Revenue growth, earnings growth, expanding margins
- MOMENTUM: 52-week highs, volume breakouts, relative strength
- QUALITY: High ROE, low debt, consistent earnings
- TECHNICAL: RSI oversold/overbought, MACD signals, MA crossovers

Output Format:
- Rank stocks by relevance to criteria
- Provide key metrics for each stock
- Highlight why each stock passed the screen
- Include risk factors and considerations
- Give actionable recommendations"""

    @staticmethod
    def get_stock_universe(sectors: Optional[List[str]] = None, cap: Optional[str] = None) -> List[str]:
        stocks = set()

        if sectors:
            for sector in sectors:
                sector_lower = sector.lower()
                if sector_lower in INDIAN_STOCK_UNIVERSE:
                    stocks.update(INDIAN_STOCK_UNIVERSE[sector_lower])

        if cap:
            cap_lower = cap.lower()
            if cap_lower in ["large", "large_cap"]:
                stocks.update(INDIAN_STOCK_UNIVERSE["large_cap"])
            elif cap_lower in ["mid", "mid_cap"]:
                stocks.update(INDIAN_STOCK_UNIVERSE["mid_cap"])

        if not stocks:
            stocks.update(INDIAN_STOCK_UNIVERSE["large_cap"])

        return list(stocks)

    @staticmethod
    def fetch_stock_metrics(ticker: str) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")

            if hist.empty or len(hist) < 50:
                return {"ticker": ticker, "error": "Insufficient data"}

            current_price = hist['Close'].iloc[-1]

            high_52w = hist['Close'].max()
            low_52w = hist['Close'].min()
            pct_from_high = ((current_price - high_52w) / high_52w) * 100
            pct_from_low = ((current_price - low_52w) / low_52w) * 100

            returns_1m = ((current_price / hist['Close'].iloc[-22]) - 1) * 100 if len(hist) >= 22 else None
            returns_3m = ((current_price / hist['Close'].iloc[-66]) - 1) * 100 if len(hist) >= 66 else None
            returns_6m = ((current_price / hist['Close'].iloc[-126]) - 1) * 100 if len(hist) >= 126 else None
            returns_1y = ((current_price / hist['Close'].iloc[0]) - 1) * 100

            sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
            sma_200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None

            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].iloc[-5:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

            daily_returns = hist['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100

            pe_ratio = info.get('trailingPE')
            forward_pe = info.get('forwardPE')
            pb_ratio = info.get('priceToBook')
            ps_ratio = info.get('priceToSalesTrailing12Months')
            peg_ratio = info.get('pegRatio')

            market_cap = info.get('marketCap', 0)
            enterprise_value = info.get('enterpriseValue')

            roe = info.get('returnOnEquity', 0)
            if roe:
                roe = roe * 100
            roa = info.get('returnOnAssets', 0)
            if roa:
                roa = roa * 100
            profit_margin = info.get('profitMargins', 0)
            if profit_margin:
                profit_margin = profit_margin * 100
            operating_margin = info.get('operatingMargins', 0)
            if operating_margin:
                operating_margin = operating_margin * 100

            revenue_growth = info.get('revenueGrowth', 0)
            if revenue_growth:
                revenue_growth = revenue_growth * 100
            earnings_growth = info.get('earningsGrowth', 0)
            if earnings_growth:
                earnings_growth = earnings_growth * 100

            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield:
                dividend_yield = dividend_yield * 100

            debt_to_equity = info.get('debtToEquity')
            current_ratio = info.get('currentRatio')

            company_name = info.get('shortName', ticker)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')

            return convert_numpy_types({
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "industry": industry,
                "current_price": round(current_price, 2),
                "market_cap": market_cap,
                "market_cap_cr": round(market_cap / 10000000, 2) if market_cap else None,

                "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                "forward_pe": round(forward_pe, 2) if forward_pe else None,
                "pb_ratio": round(pb_ratio, 2) if pb_ratio else None,
                "ps_ratio": round(ps_ratio, 2) if ps_ratio else None,
                "peg_ratio": round(peg_ratio, 2) if peg_ratio else None,

                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "pct_from_52w_high": round(pct_from_high, 2),
                "pct_from_52w_low": round(pct_from_low, 2),

                "returns_1m": round(returns_1m, 2) if returns_1m else None,
                "returns_3m": round(returns_3m, 2) if returns_3m else None,
                "returns_6m": round(returns_6m, 2) if returns_6m else None,
                "returns_1y": round(returns_1y, 2),

                "sma_50": round(sma_50, 2) if sma_50 else None,
                "sma_200": round(sma_200, 2) if sma_200 else None,
                "above_sma_50": current_price > sma_50 if sma_50 else None,
                "above_sma_200": current_price > sma_200 if sma_200 else None,
                "rsi": round(rsi, 2) if not np.isnan(rsi) else None,
                "volatility_pct": round(volatility, 2),

                "avg_volume": int(avg_volume),
                "volume_ratio": round(volume_ratio, 2),

                "roe": round(roe, 2) if roe else None,
                "roa": round(roa, 2) if roa else None,
                "profit_margin": round(profit_margin, 2) if profit_margin else None,
                "operating_margin": round(operating_margin, 2) if operating_margin else None,

                "revenue_growth": round(revenue_growth, 2) if revenue_growth else None,
                "earnings_growth": round(earnings_growth, 2) if earnings_growth else None,

                "dividend_yield": round(dividend_yield, 2) if dividend_yield else None,

                "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity else None,
                "current_ratio": round(current_ratio, 2) if current_ratio else None,

                "success": True
            })

        except Exception as e:
            logger.error(f"Error fetching metrics for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e), "success": False}

    @staticmethod
    def screen_stocks(
        universe: List[str],
        filters: Dict[str, Any],
        max_workers: int = 10
    ) -> List[Dict[str, Any]]:
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(ScreenerAgent.fetch_stock_metrics, ticker): ticker
                      for ticker in universe}

            for future in as_completed(futures):
                try:
                    data = future.result()
                    if data.get("success"):
                        results.append(data)
                except Exception as e:
                    logger.error(f"Error processing stock: {e}")

        filtered_results = []
        for stock in results:
            if ScreenerAgent._apply_filters(stock, filters):
                filtered_results.append(stock)

        return filtered_results

    @staticmethod
    def _apply_filters(stock: Dict[str, Any], filters: Dict[str, Any]) -> bool:

        if "pe_max" in filters and stock.get("pe_ratio"):
            if stock["pe_ratio"] > filters["pe_max"]:
                return False

        if "pe_min" in filters and stock.get("pe_ratio"):
            if stock["pe_ratio"] < filters["pe_min"]:
                return False

        if "pb_max" in filters and stock.get("pb_ratio"):
            if stock["pb_ratio"] > filters["pb_max"]:
                return False

        if "peg_max" in filters and stock.get("peg_ratio"):
            if stock["peg_ratio"] > filters["peg_max"]:
                return False

        if "roe_min" in filters and stock.get("roe"):
            if stock["roe"] < filters["roe_min"]:
                return False

        if "profit_margin_min" in filters and stock.get("profit_margin"):
            if stock["profit_margin"] < filters["profit_margin_min"]:
                return False

        if "revenue_growth_min" in filters and stock.get("revenue_growth"):
            if stock["revenue_growth"] < filters["revenue_growth_min"]:
                return False

        if "earnings_growth_min" in filters and stock.get("earnings_growth"):
            if stock["earnings_growth"] < filters["earnings_growth_min"]:
                return False

        if "rsi_max" in filters and stock.get("rsi"):
            if stock["rsi"] > filters["rsi_max"]:
                return False

        if "rsi_min" in filters and stock.get("rsi"):
            if stock["rsi"] < filters["rsi_min"]:
                return False

        if filters.get("above_sma_50") and not stock.get("above_sma_50"):
            return False

        if filters.get("above_sma_200") and not stock.get("above_sma_200"):
            return False

        if "returns_1m_min" in filters and stock.get("returns_1m"):
            if stock["returns_1m"] < filters["returns_1m_min"]:
                return False

        if "returns_3m_min" in filters and stock.get("returns_3m"):
            if stock["returns_3m"] < filters["returns_3m_min"]:
                return False

        if "pct_from_high_max" in filters:
            if stock.get("pct_from_52w_high", -100) < filters["pct_from_high_max"]:
                return False

        if "volume_ratio_min" in filters:
            if stock.get("volume_ratio", 0) < filters["volume_ratio_min"]:
                return False

        if "debt_to_equity_max" in filters and stock.get("debt_to_equity"):
            if stock["debt_to_equity"] > filters["debt_to_equity_max"]:
                return False

        if "dividend_yield_min" in filters and stock.get("dividend_yield"):
            if stock["dividend_yield"] < filters["dividend_yield_min"]:
                return False

        if "market_cap_min" in filters and stock.get("market_cap_cr"):
            if stock["market_cap_cr"] < filters["market_cap_min"]:
                return False

        if "market_cap_max" in filters and stock.get("market_cap_cr"):
            if stock["market_cap_cr"] > filters["market_cap_max"]:
                return False

        return True

    @staticmethod
    def get_preset_screens() -> Dict[str, Dict[str, Any]]:
        return {
            "value_picks": {
                "name": "Value Picks",
                "description": "Undervalued stocks with strong fundamentals",
                "filters": {
                    "pe_max": 20,
                    "pb_max": 3,
                    "roe_min": 12,
                    "debt_to_equity_max": 100,
                    "profit_margin_min": 5
                },
                "sort_by": "pe_ratio",
                "sort_ascending": True
            },
            "growth_stars": {
                "name": "Growth Stars",
                "description": "High growth companies with momentum",
                "filters": {
                    "revenue_growth_min": 15,
                    "earnings_growth_min": 15,
                    "roe_min": 15,
                    "above_sma_50": True
                },
                "sort_by": "revenue_growth",
                "sort_ascending": False
            },
            "momentum_leaders": {
                "name": "Momentum Leaders",
                "description": "Stocks near 52-week highs with volume",
                "filters": {
                    "pct_from_high_max": -10,
                    "above_sma_50": True,
                    "above_sma_200": True,
                    "volume_ratio_min": 1.2,
                    "returns_1m_min": 0
                },
                "sort_by": "pct_from_52w_high",
                "sort_ascending": False
            },
            "dividend_champions": {
                "name": "Dividend Champions",
                "description": "High dividend yield with stable fundamentals",
                "filters": {
                    "dividend_yield_min": 2,
                    "pe_max": 25,
                    "roe_min": 10,
                    "debt_to_equity_max": 150
                },
                "sort_by": "dividend_yield",
                "sort_ascending": False
            },
            "quality_compounders": {
                "name": "Quality Compounders",
                "description": "High quality companies with consistent performance",
                "filters": {
                    "roe_min": 18,
                    "profit_margin_min": 10,
                    "debt_to_equity_max": 50,
                    "revenue_growth_min": 10
                },
                "sort_by": "roe",
                "sort_ascending": False
            },
            "oversold_opportunities": {
                "name": "Oversold Opportunities",
                "description": "Technically oversold stocks with decent fundamentals",
                "filters": {
                    "rsi_max": 35,
                    "pe_max": 30,
                    "roe_min": 8
                },
                "sort_by": "rsi",
                "sort_ascending": True
            },
            "low_volatility": {
                "name": "Low Volatility",
                "description": "Stable stocks with low price swings",
                "filters": {
                    "volatility_max": 25,
                    "profit_margin_min": 8,
                    "debt_to_equity_max": 100
                },
                "sort_by": "volatility_pct",
                "sort_ascending": True
            },
            "small_cap_gems": {
                "name": "Small Cap Gems",
                "description": "Smaller companies with growth potential",
                "filters": {
                    "market_cap_min": 1000,
                    "market_cap_max": 20000,
                    "roe_min": 15,
                    "revenue_growth_min": 15,
                    "debt_to_equity_max": 80
                },
                "sort_by": "revenue_growth",
                "sort_ascending": False
            }
        }

    async def run_screen_with_ai(
        self,
        screen_type: str,
        sectors: Optional[List[str]] = None,
        custom_filters: Optional[Dict[str, Any]] = None,
        top_n: int = 10
    ) -> Dict[str, Any]:
        try:
            presets = self.get_preset_screens()

            if screen_type == "custom" and custom_filters:
                filters = custom_filters
                screen_name = "Custom Screen"
                screen_desc = "User-defined custom filters"
            elif screen_type in presets:
                preset = presets[screen_type]
                filters = preset["filters"]
                screen_name = preset["name"]
                screen_desc = preset["description"]
            else:
                return {
                    "success": False,
                    "error": f"Unknown screen type: {screen_type}"
                }

            universe = self.get_stock_universe(sectors)

            results = self.screen_stocks(universe, filters)

            if not results:
                return {
                    "success": True,
                    "screen_name": screen_name,
                    "description": screen_desc,
                    "filters_applied": filters,
                    "stocks_screened": len(universe),
                    "stocks_passed": 0,
                    "results": [],
                    "ai_insights": "No stocks passed the screening criteria. Consider relaxing some filters."
                }

            sort_key = presets.get(screen_type, {}).get("sort_by", "pe_ratio")
            sort_asc = presets.get(screen_type, {}).get("sort_ascending", True)

            results_sorted = sorted(
                results,
                key=lambda x: (x.get(sort_key) is None, x.get(sort_key) or 0),
                reverse=not sort_asc
            )

            top_results = results_sorted[:top_n]

            ai_insights = await self._generate_ai_insights(
                screen_name, screen_desc, filters, top_results
            )

            return {
                "success": True,
                "screen_name": screen_name,
                "description": screen_desc,
                "filters_applied": filters,
                "stocks_screened": len(universe),
                "stocks_passed": len(results),
                "results": top_results,
                "ai_insights": ai_insights
            }

        except Exception as e:
            logger.error(f"Screener error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_ai_insights(
        self,
        screen_name: str,
        screen_desc: str,
        filters: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> str:

        if not results:
            return "No stocks to analyze."

        stocks_summary = []
        for stock in results[:5]:
            summary = f"""
{stock['ticker']} ({stock.get('company_name', 'N/A')}):
- Sector: {stock.get('sector', 'N/A')}
- Price: ₹{stock.get('current_price', 'N/A')}
- P/E: {stock.get('pe_ratio', 'N/A')}, P/B: {stock.get('pb_ratio', 'N/A')}
- ROE: {stock.get('roe', 'N/A')}%, Profit Margin: {stock.get('profit_margin', 'N/A')}%
- 1Y Return: {stock.get('returns_1y', 'N/A')}%, RSI: {stock.get('rsi', 'N/A')}
- Revenue Growth: {stock.get('revenue_growth', 'N/A')}%
"""
            stocks_summary.append(summary)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Analyze these screened stocks from the "{screen_name}" screen:

Screen Description: {screen_desc}
Filters Applied: {filters}

Top Stocks Found:
{''.join(stocks_summary)}

Please provide:
1. **Overview**: Quick summary of the screening results
2. **Top 3 Picks**: Your top 3 recommendations with brief reasoning
3. **Common Themes**: What patterns do you see in these stocks?
4. **Risk Factors**: Key risks to consider
5. **Action Items**: What should an investor do next?

Keep the analysis concise but actionable. Use ₹ for all currency values.
""")
        ]

        try:
            return await self._invoke_llm(messages)
        except Exception as e:
            logger.error(f"AI insights generation error: {e}")
            return f"AI analysis unavailable: {str(e)}"

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        screen_type = state.get("screen_type", "value_picks")
        sectors = state.get("sectors")
        custom_filters = state.get("custom_filters")
        top_n = state.get("top_n", 10)

        result = await self.run_screen_with_ai(
            screen_type=screen_type,
            sectors=sectors,
            custom_filters=custom_filters,
            top_n=top_n
        )

        return {
            **state,
            "screener_results": result
        }
