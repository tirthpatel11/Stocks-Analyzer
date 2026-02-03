"""
Market Data & Feature Engineering Agent

This agent is responsible for:
- Fetching real-time and historical market data
- Computing technical indicators and features
- Data preprocessing and normalization
- Feature engineering for analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import ta
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import logging

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from app.agents.base import BaseAgent, get_grok_llm

logger = logging.getLogger(__name__)


class MarketDataAgent(BaseAgent):
    """
    Market Data & Feature Engineering Agent
    
    Specializes in:
    - Real-time market data retrieval
    - Historical data analysis
    - Technical indicator computation
    - Feature engineering for ML models
    """
    
    def __init__(self):
        super().__init__(
            name="MarketDataAgent",
            description="Fetches market data and engineers features for stock analysis",
            temperature=0.0  # Low temperature for precise data handling
        )
        
    def _setup_tools(self) -> List[Any]:
        """Set up market data tools"""
        return [
            self.fetch_stock_data,
            self.compute_technical_indicators,
            self.get_fundamental_data,
            self.calculate_returns
        ]
    
    def _create_system_prompt(self) -> str:
        return """You are the Market Data & Feature Engineering Agent, a specialized AI for financial data processing focused on the INDIAN STOCK MARKET.

IMPORTANT: Always use Indian Rupees (₹) for all currency values. Never use $ or USD.

Your responsibilities:
1. Fetch accurate and timely market data for requested securities
2. Compute comprehensive technical indicators
3. Engineer features that capture market dynamics
4. Provide clean, normalized data for downstream analysis

When analyzing data:
- Always verify data quality and completeness
- Flag any anomalies or missing data points
- Explain the significance of computed features
- Provide context for unusual market conditions

Output Format:
- Present data summaries in a clear, structured format
- Include key statistics (mean, std, min, max) for numerical features
- Highlight any data quality issues or concerns
- Recommend additional data points if needed for comprehensive analysis

You work as part of a team with Technical Analysis and Risk Management agents.
Provide your analysis in a way that supports their specialized functions."""
    
    @staticmethod
    def fetch_stock_data(
        ticker: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary containing OHLCV data and metadata
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                return {
                    "success": False,
                    "error": f"No data found for ticker {ticker}",
                    "ticker": ticker
                }
            
            # Convert to serializable format
            data = {
                "success": True,
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "data_points": len(df),
                "date_range": {
                    "start": df.index[0].strftime("%Y-%m-%d"),
                    "end": df.index[-1].strftime("%Y-%m-%d")
                },
                "latest": {
                    "date": df.index[-1].strftime("%Y-%m-%d"),
                    "open": round(df['Open'].iloc[-1], 2),
                    "high": round(df['High'].iloc[-1], 2),
                    "low": round(df['Low'].iloc[-1], 2),
                    "close": round(df['Close'].iloc[-1], 2),
                    "volume": int(df['Volume'].iloc[-1])
                },
                "statistics": {
                    "avg_close": round(df['Close'].mean(), 2),
                    "std_close": round(df['Close'].std(), 2),
                    "min_close": round(df['Close'].min(), 2),
                    "max_close": round(df['Close'].max(), 2),
                    "avg_volume": int(df['Volume'].mean()),
                    "total_volume": int(df['Volume'].sum())
                },
                "price_data": df[['Open', 'High', 'Low', 'Close', 'Volume']].round(2).to_dict()
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    @staticmethod
    def compute_technical_indicators(ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Compute comprehensive technical indicators for a stock
        
        Args:
            ticker: Stock ticker symbol
            period: Historical data period
            
        Returns:
            Dictionary containing technical indicators
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 50:
                return {
                    "success": False,
                    "error": "Insufficient data for indicator calculation",
                    "ticker": ticker
                }
            
            # Moving Averages
            df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
            df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
            df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
            df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
            df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
            
            # MACD
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()
            
            # RSI
            df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = bb.bollinger_wband()
            
            # ATR (Average True Range)
            df['ATR'] = AverageTrueRange(
                high=df['High'], low=df['Low'], close=df['Close'], window=14
            ).average_true_range()
            
            # ADX (Average Directional Index)
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['ADX'] = adx.adx()
            df['DI_Plus'] = adx.adx_pos()
            df['DI_Minus'] = adx.adx_neg()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Volume Indicators
            df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Compute signals
            signals = []
            
            # Moving average signals
            if latest['Close'] > latest['SMA_50']:
                signals.append("Price above SMA50 (Bullish)")
            else:
                signals.append("Price below SMA50 (Bearish)")
                
            if latest['SMA_50'] > latest['SMA_200']:
                signals.append("Golden Cross active (Bullish)")
            elif latest['SMA_50'] < latest['SMA_200']:
                signals.append("Death Cross active (Bearish)")
            
            # RSI signals
            if latest['RSI'] > 70:
                signals.append(f"RSI Overbought ({latest['RSI']:.1f})")
            elif latest['RSI'] < 30:
                signals.append(f"RSI Oversold ({latest['RSI']:.1f})")
            else:
                signals.append(f"RSI Neutral ({latest['RSI']:.1f})")
            
            # MACD signals
            if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                signals.append("MACD Bullish Crossover")
            elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                signals.append("MACD Bearish Crossover")
            
            # Bollinger Band signals
            if latest['Close'] > latest['BB_Upper']:
                signals.append("Price above upper Bollinger Band (Overbought)")
            elif latest['Close'] < latest['BB_Lower']:
                signals.append("Price below lower Bollinger Band (Oversold)")
            
            # ADX trend strength
            if latest['ADX'] > 25:
                signals.append(f"Strong trend (ADX: {latest['ADX']:.1f})")
            else:
                signals.append(f"Weak/No trend (ADX: {latest['ADX']:.1f})")
            
            return {
                "success": True,
                "ticker": ticker,
                "date": df.index[-1].strftime("%Y-%m-%d"),
                "indicators": {
                    "moving_averages": {
                        "SMA_20": round(latest['SMA_20'], 2),
                        "SMA_50": round(latest['SMA_50'], 2),
                        "SMA_200": round(latest['SMA_200'], 2) if pd.notna(latest['SMA_200']) else None,
                        "EMA_12": round(latest['EMA_12'], 2),
                        "EMA_26": round(latest['EMA_26'], 2)
                    },
                    "momentum": {
                        "RSI": round(latest['RSI'], 2),
                        "MACD": round(latest['MACD'], 4),
                        "MACD_Signal": round(latest['MACD_Signal'], 4),
                        "MACD_Histogram": round(latest['MACD_Histogram'], 4),
                        "Stoch_K": round(latest['Stoch_K'], 2),
                        "Stoch_D": round(latest['Stoch_D'], 2)
                    },
                    "volatility": {
                        "BB_Upper": round(latest['BB_Upper'], 2),
                        "BB_Middle": round(latest['BB_Middle'], 2),
                        "BB_Lower": round(latest['BB_Lower'], 2),
                        "BB_Width": round(latest['BB_Width'], 4),
                        "ATR": round(latest['ATR'], 2)
                    },
                    "trend": {
                        "ADX": round(latest['ADX'], 2),
                        "DI_Plus": round(latest['DI_Plus'], 2),
                        "DI_Minus": round(latest['DI_Minus'], 2)
                    },
                    "volume": {
                        "OBV": int(latest['OBV']),
                        "Volume": int(latest['Volume']),
                        "Avg_Volume_20": int(df['Volume'].tail(20).mean())
                    }
                },
                "signals": signals,
                "current_price": round(latest['Close'], 2)
            }
            
        except Exception as e:
            logger.error(f"Error computing indicators for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    @staticmethod
    def get_fundamental_data(ticker: str) -> Dict[str, Any]:
        """
        Get fundamental data for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing fundamental metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "success": True,
                "ticker": ticker,
                "company_info": {
                    "name": info.get("longName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "country": info.get("country", "N/A"),
                    "employees": info.get("fullTimeEmployees", "N/A"),
                    "website": info.get("website", "N/A")
                },
                "valuation": {
                    "market_cap": info.get("marketCap", "N/A"),
                    "enterprise_value": info.get("enterpriseValue", "N/A"),
                    "pe_ratio": info.get("trailingPE", "N/A"),
                    "forward_pe": info.get("forwardPE", "N/A"),
                    "peg_ratio": info.get("pegRatio", "N/A"),
                    "price_to_book": info.get("priceToBook", "N/A"),
                    "price_to_sales": info.get("priceToSalesTrailing12Months", "N/A")
                },
                "profitability": {
                    "profit_margin": info.get("profitMargins", "N/A"),
                    "operating_margin": info.get("operatingMargins", "N/A"),
                    "gross_margin": info.get("grossMargins", "N/A"),
                    "roe": info.get("returnOnEquity", "N/A"),
                    "roa": info.get("returnOnAssets", "N/A")
                },
                "financial_health": {
                    "current_ratio": info.get("currentRatio", "N/A"),
                    "debt_to_equity": info.get("debtToEquity", "N/A"),
                    "total_cash": info.get("totalCash", "N/A"),
                    "total_debt": info.get("totalDebt", "N/A"),
                    "free_cash_flow": info.get("freeCashflow", "N/A")
                },
                "dividends": {
                    "dividend_rate": info.get("dividendRate", "N/A"),
                    "dividend_yield": info.get("dividendYield", "N/A"),
                    "payout_ratio": info.get("payoutRatio", "N/A"),
                    "ex_dividend_date": info.get("exDividendDate", "N/A")
                },
                "growth": {
                    "earnings_growth": info.get("earningsGrowth", "N/A"),
                    "revenue_growth": info.get("revenueGrowth", "N/A"),
                    "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth", "N/A")
                },
                "analyst_targets": {
                    "target_high": info.get("targetHighPrice", "N/A"),
                    "target_low": info.get("targetLowPrice", "N/A"),
                    "target_mean": info.get("targetMeanPrice", "N/A"),
                    "recommendation": info.get("recommendationKey", "N/A")
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    @staticmethod
    def calculate_returns(ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Calculate various return metrics for a stock
        
        Args:
            ticker: Stock ticker symbol
            period: Historical data period
            
        Returns:
            Dictionary containing return metrics
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                return {
                    "success": False,
                    "error": f"No data found for ticker {ticker}",
                    "ticker": ticker
                }
            
            # Calculate returns
            df['Daily_Return'] = df['Close'].pct_change()
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Cumulative returns
            total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            
            # Annualized metrics (assuming 252 trading days)
            trading_days = len(df)
            annualization_factor = 252 / trading_days if trading_days > 0 else 1
            
            mean_daily_return = df['Daily_Return'].mean()
            std_daily_return = df['Daily_Return'].std()
            
            annualized_return = mean_daily_return * 252 * 100
            annualized_volatility = std_daily_return * np.sqrt(252) * 100
            
            # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
            sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0
            
            # Maximum Drawdown
            rolling_max = df['Close'].expanding().max()
            drawdown = (df['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Sortino Ratio (downside deviation)
            negative_returns = df['Daily_Return'][df['Daily_Return'] < 0]
            downside_std = negative_returns.std() * np.sqrt(252) * 100
            sortino_ratio = (annualized_return / downside_std) if downside_std != 0 else 0
            
            # Calmar Ratio
            calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0
            
            # Period returns
            if len(df) >= 5:
                week_return = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
            else:
                week_return = None
                
            if len(df) >= 21:
                month_return = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100
            else:
                month_return = None
                
            if len(df) >= 63:
                quarter_return = (df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) * 100
            else:
                quarter_return = None
            
            return {
                "success": True,
                "ticker": ticker,
                "period": period,
                "returns": {
                    "total_return_pct": round(total_return, 2),
                    "annualized_return_pct": round(annualized_return, 2),
                    "week_return_pct": round(week_return, 2) if week_return else None,
                    "month_return_pct": round(month_return, 2) if month_return else None,
                    "quarter_return_pct": round(quarter_return, 2) if quarter_return else None
                },
                "risk_metrics": {
                    "annualized_volatility_pct": round(annualized_volatility, 2),
                    "max_drawdown_pct": round(max_drawdown, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "sortino_ratio": round(sortino_ratio, 2),
                    "calmar_ratio": round(calmar_ratio, 2)
                },
                "statistics": {
                    "mean_daily_return_pct": round(mean_daily_return * 100, 4),
                    "std_daily_return_pct": round(std_daily_return * 100, 4),
                    "trading_days": trading_days,
                    "positive_days": int((df['Daily_Return'] > 0).sum()),
                    "negative_days": int((df['Daily_Return'] < 0).sum()),
                    "win_rate_pct": round((df['Daily_Return'] > 0).sum() / trading_days * 100, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating returns for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Market Data Agent's analysis
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with market data analysis
        """
        ticker = state.get("ticker", "")
        
        if not ticker:
            return {
                **state,
                "errors": state.get("errors", []) + ["No ticker provided to Market Data Agent"]
            }
        
        try:
            # Fetch all market data
            stock_data = self.fetch_stock_data(ticker)
            technical_indicators = self.compute_technical_indicators(ticker)
            fundamental_data = self.get_fundamental_data(ticker)
            returns_data = self.calculate_returns(ticker)
            
            # Prepare summary for LLM analysis
            data_summary = f"""
Market Data Analysis for {ticker}:

PRICE DATA:
- Current Price: ₹{stock_data.get('latest', {}).get('close', 'N/A')}
- 52-Week Range: ₹{stock_data.get('statistics', {}).get('min_close', 'N/A')} - ₹{stock_data.get('statistics', {}).get('max_close', 'N/A')}
- Average Volume: {stock_data.get('statistics', {}).get('avg_volume', 'N/A'):,}

TECHNICAL SIGNALS:
{chr(10).join(['- ' + s for s in technical_indicators.get('signals', [])])}

KEY INDICATORS:
- RSI: {technical_indicators.get('indicators', {}).get('momentum', {}).get('RSI', 'N/A')}
- MACD: {technical_indicators.get('indicators', {}).get('momentum', {}).get('MACD', 'N/A')}
- ADX: {technical_indicators.get('indicators', {}).get('trend', {}).get('ADX', 'N/A')}

RETURNS:
- Total Return ({returns_data.get('period', '1y')}): {returns_data.get('returns', {}).get('total_return_pct', 'N/A')}%
- Annualized Return: {returns_data.get('returns', {}).get('annualized_return_pct', 'N/A')}%
- Sharpe Ratio: {returns_data.get('risk_metrics', {}).get('sharpe_ratio', 'N/A')}

FUNDAMENTALS:
- P/E Ratio: {fundamental_data.get('valuation', {}).get('pe_ratio', 'N/A')}
- Market Cap: ₹{fundamental_data.get('valuation', {}).get('market_cap', 'N/A'):,}
- Sector: {fundamental_data.get('company_info', {}).get('sector', 'N/A')}
"""
            
            # Get LLM analysis
            messages = self._format_messages(state)
            messages.append(HumanMessage(content=f"""
Analyze the following market data and provide insights:

{data_summary}

Provide a comprehensive summary of:
1. Data quality assessment
2. Key observations from the data
3. Notable patterns or anomalies
4. Recommendations for further analysis by Technical and Risk agents
"""))
            
            llm_analysis = await self._invoke_llm(messages)
            
            # Update state with results
            analysis_results = state.get("analysis_results", {})
            analysis_results["market_data"] = {
                "stock_data": stock_data,
                "technical_indicators": technical_indicators,
                "fundamental_data": fundamental_data,
                "returns_data": returns_data,
                "llm_analysis": llm_analysis
            }
            
            return {
                **state,
                "analysis_results": analysis_results,
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "agent": self.name,
                    "content": llm_analysis
                }]
            }
            
        except Exception as e:
            logger.error(f"Market Data Agent execution error: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Market Data Agent error: {str(e)}"]
            }

