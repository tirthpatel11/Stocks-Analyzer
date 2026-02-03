"""
Technical & Regime Agent — Chart + Market Structure

This agent is responsible for:
- Chart pattern recognition
- Market structure analysis
- Trend and regime detection
- Support/Resistance identification
- Price action analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from scipy.signal import argrelextrema
from scipy.stats import linregress
import logging

from langchain_core.messages import HumanMessage
from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    """
    Technical & Regime Agent - Chart + Market Structure Analysis
    
    Specializes in:
    - Chart pattern identification
    - Market regime classification
    - Support/Resistance level detection
    - Trend analysis and structure
    - Price action signals
    """
    
    def __init__(self):
        super().__init__(
            name="TechnicalAgent",
            description="Analyzes chart patterns, market structure, and regime detection",
            temperature=0.2
        )
        
    def _setup_tools(self) -> List[Any]:
        """Set up technical analysis tools"""
        return [
            self.identify_market_regime,
            self.find_support_resistance,
            self.detect_chart_patterns,
            self.analyze_trend_structure
        ]
    
    def _create_system_prompt(self) -> str:
        return """You are the Technical & Regime Agent, a specialized AI for chart analysis and market structure focused on the INDIAN STOCK MARKET.

IMPORTANT: Always use Indian Rupees (₹) for all currency values. Never use $ or USD.

Your responsibilities:
1. Identify current market regime (trending, ranging, volatile)
2. Detect chart patterns and their implications
3. Find key support and resistance levels
4. Analyze market structure (higher highs, lower lows, etc.)
5. Provide actionable technical insights

Analysis Framework:
- TREND ANALYSIS: Determine primary and secondary trends
- STRUCTURE: Identify market structure shifts and pivots
- PATTERNS: Recognize classical and candlestick patterns
- LEVELS: Map significant price levels
- REGIME: Classify current market phase

Output Format:
- Clear regime classification with confidence level
- Specific price levels for support/resistance
- Pattern identification with success probabilities
- Trend direction and strength assessment
- Actionable recommendations for trading/investing

You work with Market Data and Risk Management agents. Your analysis should:
- Build upon the data provided by the Market Data Agent
- Provide technical context for the Risk Agent's position sizing
- Highlight key levels and patterns that affect risk/reward"""
    
    @staticmethod
    def identify_market_regime(ticker: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Identify the current market regime
        
        Regimes:
        - TRENDING_UP: Strong uptrend with higher highs and higher lows
        - TRENDING_DOWN: Strong downtrend with lower highs and lower lows
        - RANGING: Sideways movement within a range
        - VOLATILE: High volatility with no clear direction
        - BREAKOUT: Transitioning from range to trend
        
        Args:
            ticker: Stock ticker symbol
            period: Analysis period
            
        Returns:
            Dictionary with regime classification and metrics
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 50:
                return {
                    "success": False,
                    "error": "Insufficient data for regime analysis",
                    "ticker": ticker
                }
            
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            
            # Calculate trend metrics
            # Linear regression slope for trend direction
            x = np.arange(len(close))
            slope, intercept, r_value, p_value, std_err = linregress(x, close)
            trend_strength = abs(r_value)  # R-squared indicates trend strength
            
            # Normalize slope as percentage of price
            normalized_slope = (slope / close.mean()) * 100
            
            # Calculate volatility metrics
            returns = pd.Series(close).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            recent_volatility = returns.tail(20).std() * np.sqrt(252) * 100
            
            # Calculate ADX-like trend strength
            # Using price momentum
            momentum_20 = (close[-1] / close[-20] - 1) * 100 if len(close) >= 20 else 0
            momentum_50 = (close[-1] / close[-50] - 1) * 100 if len(close) >= 50 else 0
            
            # Identify swing highs and lows for structure
            order = 5  # Look for local extrema within 5 bars
            local_max_idx = argrelextrema(high, np.greater, order=order)[0]
            local_min_idx = argrelextrema(low, np.less, order=order)[0]
            
            # Analyze structure
            recent_highs = high[local_max_idx[-4:]] if len(local_max_idx) >= 4 else high[local_max_idx]
            recent_lows = low[local_min_idx[-4:]] if len(local_min_idx) >= 4 else low[local_min_idx]
            
            # Determine higher highs/lower lows pattern
            if len(recent_highs) >= 2:
                higher_highs = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
                lower_highs = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
            else:
                higher_highs = False
                lower_highs = False
                
            if len(recent_lows) >= 2:
                higher_lows = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))
                lower_lows = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))
            else:
                higher_lows = False
                lower_lows = False
            
            # Calculate range metrics
            range_high = high[-20:].max()
            range_low = low[-20:].min()
            range_pct = ((range_high - range_low) / close[-1]) * 100
            
            # Determine regime
            regime = "UNKNOWN"
            confidence = 0.5
            
            if trend_strength > 0.7 and normalized_slope > 0.1 and (higher_highs or higher_lows):
                regime = "TRENDING_UP"
                confidence = min(0.9, trend_strength)
            elif trend_strength > 0.7 and normalized_slope < -0.1 and (lower_highs or lower_lows):
                regime = "TRENDING_DOWN"
                confidence = min(0.9, trend_strength)
            elif recent_volatility > volatility * 1.5:
                regime = "VOLATILE"
                confidence = 0.7
            elif range_pct < 15 and trend_strength < 0.5:
                regime = "RANGING"
                confidence = 0.8
            elif abs(close[-1] - range_high) < (range_high - range_low) * 0.1 or \
                 abs(close[-1] - range_low) < (range_high - range_low) * 0.1:
                regime = "BREAKOUT"
                confidence = 0.6
            else:
                if normalized_slope > 0:
                    regime = "TRENDING_UP"
                else:
                    regime = "TRENDING_DOWN"
                confidence = max(0.4, trend_strength)
            
            return {
                "success": True,
                "ticker": ticker,
                "regime": regime,
                "confidence": round(confidence, 2),
                "metrics": {
                    "trend_strength_r2": round(float(trend_strength), 3),
                    "trend_slope_pct": round(float(normalized_slope), 4),
                    "annualized_volatility_pct": round(float(volatility), 2),
                    "recent_volatility_pct": round(float(recent_volatility), 2),
                    "momentum_20d_pct": round(float(momentum_20), 2),
                    "momentum_50d_pct": round(float(momentum_50), 2),
                    "range_pct": round(float(range_pct), 2)
                },
                "structure": {
                    "higher_highs": bool(higher_highs),
                    "higher_lows": bool(higher_lows),
                    "lower_highs": bool(lower_highs),
                    "lower_lows": bool(lower_lows),
                    "swing_highs_count": int(len(local_max_idx)),
                    "swing_lows_count": int(len(local_min_idx))
                },
                "current_price": round(float(close[-1]), 2),
                "range": {
                    "high": round(float(range_high), 2),
                    "low": round(float(range_low), 2),
                    "mid": round(float((range_high + range_low) / 2), 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error identifying regime for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    @staticmethod
    def find_support_resistance(ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Find key support and resistance levels
        
        Uses multiple methods:
        - Pivot points
        - Price clustering
        - Volume profile approximation
        - Recent swing highs/lows
        
        Args:
            ticker: Stock ticker symbol
            period: Analysis period
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 50:
                return {
                    "success": False,
                    "error": "Insufficient data for S/R analysis",
                    "ticker": ticker
                }
            
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values
            current_price = close[-1]
            
            # Method 1: Swing highs and lows
            order = 10
            swing_high_idx = argrelextrema(high, np.greater, order=order)[0]
            swing_low_idx = argrelextrema(low, np.less, order=order)[0]
            
            swing_highs = high[swing_high_idx]
            swing_lows = low[swing_low_idx]
            
            # Method 2: Pivot points (standard)
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            r1 = 2 * pivot - low[-1]
            r2 = pivot + (high[-1] - low[-1])
            r3 = high[-1] + 2 * (pivot - low[-1])
            s1 = 2 * pivot - high[-1]
            s2 = pivot - (high[-1] - low[-1])
            s3 = low[-1] - 2 * (high[-1] - pivot)
            
            # Method 3: Price clustering using histogram
            price_range = np.linspace(low.min(), high.max(), 50)
            price_counts = np.zeros(len(price_range) - 1)
            
            for i in range(len(df)):
                for j in range(len(price_range) - 1):
                    if price_range[j] <= close[i] <= price_range[j + 1]:
                        price_counts[j] += volume[i]
                        break
            
            # Find high-volume price levels
            cluster_threshold = np.percentile(price_counts, 80)
            cluster_levels = []
            for i in range(len(price_counts)):
                if price_counts[i] > cluster_threshold:
                    cluster_levels.append((price_range[i] + price_range[i+1]) / 2)
            
            # Combine and sort all levels
            all_levels = list(swing_highs) + list(swing_lows) + cluster_levels
            all_levels = sorted(set([round(l, 2) for l in all_levels]))
            
            # Separate into support and resistance
            resistance_levels = sorted([l for l in all_levels if l > current_price])[:5]
            support_levels = sorted([l for l in all_levels if l < current_price], reverse=True)[:5]
            
            # Calculate distance to nearest levels
            nearest_resistance = resistance_levels[0] if resistance_levels else None
            nearest_support = support_levels[0] if support_levels else None
            
            return {
                "success": True,
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "pivot_points": {
                    "pivot": round(pivot, 2),
                    "r1": round(r1, 2),
                    "r2": round(r2, 2),
                    "r3": round(r3, 2),
                    "s1": round(s1, 2),
                    "s2": round(s2, 2),
                    "s3": round(s3, 2)
                },
                "key_levels": {
                    "resistance": [round(r, 2) for r in resistance_levels],
                    "support": [round(s, 2) for s in support_levels]
                },
                "nearest_levels": {
                    "resistance": round(nearest_resistance, 2) if nearest_resistance else None,
                    "support": round(nearest_support, 2) if nearest_support else None,
                    "distance_to_resistance_pct": round((nearest_resistance / current_price - 1) * 100, 2) if nearest_resistance else None,
                    "distance_to_support_pct": round((nearest_support / current_price - 1) * 100, 2) if nearest_support else None
                },
                "price_range": {
                    "52w_high": round(high.max(), 2),
                    "52w_low": round(low.min(), 2),
                    "position_in_range_pct": round((current_price - low.min()) / (high.max() - low.min()) * 100, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error finding S/R for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    @staticmethod
    def detect_chart_patterns(ticker: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Detect common chart patterns
        
        Patterns detected:
        - Double Top/Bottom
        - Head and Shoulders
        - Triangles (ascending, descending, symmetric)
        - Wedges
        - Channels
        
        Args:
            ticker: Stock ticker symbol
            period: Analysis period
            
        Returns:
            Dictionary with detected patterns
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 50:
                return {
                    "success": False,
                    "error": "Insufficient data for pattern detection",
                    "ticker": ticker
                }
            
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            
            detected_patterns = []
            
            # Find swing points
            order = 5
            swing_high_idx = argrelextrema(high, np.greater, order=order)[0]
            swing_low_idx = argrelextrema(low, np.less, order=order)[0]
            
            swing_highs = [(idx, high[idx]) for idx in swing_high_idx]
            swing_lows = [(idx, low[idx]) for idx in swing_low_idx]
            
            # Double Top Detection
            if len(swing_highs) >= 2:
                recent_highs = swing_highs[-4:]
                for i in range(len(recent_highs) - 1):
                    for j in range(i + 1, len(recent_highs)):
                        price_diff = abs(recent_highs[i][1] - recent_highs[j][1]) / recent_highs[i][1]
                        if price_diff < 0.03:  # Within 3%
                            detected_patterns.append({
                                "pattern": "DOUBLE_TOP",
                                "confidence": round(1 - price_diff * 10, 2),
                                "price_level": round((recent_highs[i][1] + recent_highs[j][1]) / 2, 2),
                                "implication": "BEARISH",
                                "description": "Two peaks at similar levels suggesting resistance"
                            })
                            break
            
            # Double Bottom Detection
            if len(swing_lows) >= 2:
                recent_lows = swing_lows[-4:]
                for i in range(len(recent_lows) - 1):
                    for j in range(i + 1, len(recent_lows)):
                        price_diff = abs(recent_lows[i][1] - recent_lows[j][1]) / recent_lows[i][1]
                        if price_diff < 0.03:  # Within 3%
                            detected_patterns.append({
                                "pattern": "DOUBLE_BOTTOM",
                                "confidence": round(1 - price_diff * 10, 2),
                                "price_level": round((recent_lows[i][1] + recent_lows[j][1]) / 2, 2),
                                "implication": "BULLISH",
                                "description": "Two troughs at similar levels suggesting support"
                            })
                            break
            
            # Trend Channel Detection
            if len(close) >= 20:
                # Upper trendline (resistance)
                x_high = np.array([h[0] for h in swing_highs[-5:]]) if len(swing_highs) >= 5 else np.arange(5)
                y_high = np.array([h[1] for h in swing_highs[-5:]]) if len(swing_highs) >= 5 else high[-5:]
                
                # Lower trendline (support)
                x_low = np.array([l[0] for l in swing_lows[-5:]]) if len(swing_lows) >= 5 else np.arange(5)
                y_low = np.array([l[1] for l in swing_lows[-5:]]) if len(swing_lows) >= 5 else low[-5:]
                
                if len(x_high) >= 2 and len(x_low) >= 2:
                    slope_high, _, r_high, _, _ = linregress(x_high, y_high)
                    slope_low, _, r_low, _, _ = linregress(x_low, y_low)
                    
                    # Channel detection
                    if abs(r_high) > 0.7 and abs(r_low) > 0.7:
                        if slope_high > 0 and slope_low > 0:
                            detected_patterns.append({
                                "pattern": "ASCENDING_CHANNEL",
                                "confidence": round((abs(r_high) + abs(r_low)) / 2, 2),
                                "implication": "BULLISH_CONTINUATION",
                                "description": "Price moving in upward parallel channel"
                            })
                        elif slope_high < 0 and slope_low < 0:
                            detected_patterns.append({
                                "pattern": "DESCENDING_CHANNEL",
                                "confidence": round((abs(r_high) + abs(r_low)) / 2, 2),
                                "implication": "BEARISH_CONTINUATION",
                                "description": "Price moving in downward parallel channel"
                            })
                    
                    # Triangle Detection
                    if slope_high < 0 and slope_low > 0:
                        detected_patterns.append({
                            "pattern": "SYMMETRIC_TRIANGLE",
                            "confidence": 0.7,
                            "implication": "NEUTRAL_BREAKOUT_PENDING",
                            "description": "Converging trendlines - breakout imminent"
                        })
                    elif abs(slope_high) < 0.001 and slope_low > 0:
                        detected_patterns.append({
                            "pattern": "ASCENDING_TRIANGLE",
                            "confidence": 0.75,
                            "implication": "BULLISH_BREAKOUT",
                            "description": "Flat resistance with rising support"
                        })
                    elif slope_high < 0 and abs(slope_low) < 0.001:
                        detected_patterns.append({
                            "pattern": "DESCENDING_TRIANGLE",
                            "confidence": 0.75,
                            "implication": "BEARISH_BREAKOUT",
                            "description": "Descending resistance with flat support"
                        })
            
            # Candlestick patterns (last 5 candles)
            candle_patterns = []
            
            # Check for Doji
            for i in range(-5, 0):
                body = abs(close[i] - df['Open'].values[i])
                range_size = high[i] - low[i]
                if range_size > 0 and body / range_size < 0.1:
                    candle_patterns.append({
                        "pattern": "DOJI",
                        "position": i,
                        "implication": "INDECISION"
                    })
            
            # Check for Hammer/Hanging Man
            for i in range(-5, 0):
                body = abs(close[i] - df['Open'].values[i])
                lower_shadow = min(close[i], df['Open'].values[i]) - low[i]
                upper_shadow = high[i] - max(close[i], df['Open'].values[i])
                
                if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                    candle_patterns.append({
                        "pattern": "HAMMER" if close[i] > df['Open'].values[i] else "HANGING_MAN",
                        "position": i,
                        "implication": "POTENTIAL_REVERSAL"
                    })
            
            # Check for Engulfing
            for i in range(-4, 0):
                prev_body = close[i-1] - df['Open'].values[i-1]
                curr_body = close[i] - df['Open'].values[i]
                
                if prev_body < 0 and curr_body > 0:  # Bullish engulfing
                    if (df['Open'].values[i] < close[i-1] and 
                        close[i] > df['Open'].values[i-1]):
                        candle_patterns.append({
                            "pattern": "BULLISH_ENGULFING",
                            "position": i,
                            "implication": "BULLISH_REVERSAL"
                        })
                elif prev_body > 0 and curr_body < 0:  # Bearish engulfing
                    if (df['Open'].values[i] > close[i-1] and 
                        close[i] < df['Open'].values[i-1]):
                        candle_patterns.append({
                            "pattern": "BEARISH_ENGULFING",
                            "position": i,
                            "implication": "BEARISH_REVERSAL"
                        })
            
            return {
                "success": True,
                "ticker": ticker,
                "chart_patterns": detected_patterns if detected_patterns else [{
                    "pattern": "NO_CLEAR_PATTERN",
                    "confidence": 0,
                    "implication": "NEUTRAL",
                    "description": "No significant chart patterns detected"
                }],
                "candlestick_patterns": candle_patterns,
                "analysis_period": period,
                "current_price": round(close[-1], 2)
            }
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    @staticmethod
    def analyze_trend_structure(ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Analyze market structure and trend quality
        
        Args:
            ticker: Stock ticker symbol
            period: Analysis period
            
        Returns:
            Dictionary with trend structure analysis
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 50:
                return {
                    "success": False,
                    "error": "Insufficient data for structure analysis",
                    "ticker": ticker
                }
            
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            
            # Calculate multiple timeframe trends
            def get_trend(data, window):
                if len(data) < window:
                    return "INSUFFICIENT_DATA"
                x = np.arange(window)
                slope, _, r_value, _, _ = linregress(x, data[-window:])
                if r_value > 0.5:
                    return "UPTREND" if slope > 0 else "DOWNTREND"
                return "SIDEWAYS"
            
            short_trend = get_trend(close, 20)
            medium_trend = get_trend(close, 50)
            long_trend = get_trend(close, 200) if len(close) >= 200 else "N/A"
            
            # Moving average positions
            sma_20 = close[-20:].mean() if len(close) >= 20 else None
            sma_50 = close[-50:].mean() if len(close) >= 50 else None
            sma_200 = close[-200:].mean() if len(close) >= 200 else None
            
            # MA alignment
            ma_aligned_bullish = False
            ma_aligned_bearish = False
            
            if sma_20 and sma_50 and sma_200:
                ma_aligned_bullish = close[-1] > sma_20 > sma_50 > sma_200
                ma_aligned_bearish = close[-1] < sma_20 < sma_50 < sma_200
            
            # Calculate trend changes
            order = 10
            swing_high_idx = argrelextrema(high, np.greater, order=order)[0]
            swing_low_idx = argrelextrema(low, np.less, order=order)[0]
            
            # Count higher highs/lows and lower highs/lows
            hh_count = 0
            lh_count = 0
            hl_count = 0
            ll_count = 0
            
            if len(swing_high_idx) >= 2:
                for i in range(1, len(swing_high_idx)):
                    if high[swing_high_idx[i]] > high[swing_high_idx[i-1]]:
                        hh_count += 1
                    else:
                        lh_count += 1
                        
            if len(swing_low_idx) >= 2:
                for i in range(1, len(swing_low_idx)):
                    if low[swing_low_idx[i]] > low[swing_low_idx[i-1]]:
                        hl_count += 1
                    else:
                        ll_count += 1
            
            # Determine structure
            if hh_count > lh_count and hl_count > ll_count:
                structure = "BULLISH"
                structure_description = "Higher highs and higher lows - healthy uptrend structure"
            elif lh_count > hh_count and ll_count > hl_count:
                structure = "BEARISH"
                structure_description = "Lower highs and lower lows - healthy downtrend structure"
            elif hh_count == lh_count or hl_count == ll_count:
                structure = "TRANSITIONAL"
                structure_description = "Mixed signals - potential trend change underway"
            else:
                structure = "NEUTRAL"
                structure_description = "No clear structure - ranging or choppy market"
            
            # Momentum analysis
            recent_momentum = (close[-1] / close[-10] - 1) * 100 if len(close) >= 10 else 0
            monthly_momentum = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0
            
            return {
                "success": True,
                "ticker": ticker,
                "current_price": round(float(close[-1]), 2),
                "trend_analysis": {
                    "short_term_20d": short_trend,
                    "medium_term_50d": medium_trend,
                    "long_term_200d": long_trend,
                    "primary_trend": long_trend if long_trend != "N/A" else medium_trend
                },
                "moving_averages": {
                    "sma_20": round(float(sma_20), 2) if sma_20 else None,
                    "sma_50": round(float(sma_50), 2) if sma_50 else None,
                    "sma_200": round(float(sma_200), 2) if sma_200 else None,
                    "price_vs_sma20_pct": round(float((close[-1] / sma_20 - 1) * 100), 2) if sma_20 else None,
                    "price_vs_sma50_pct": round(float((close[-1] / sma_50 - 1) * 100), 2) if sma_50 else None,
                    "price_vs_sma200_pct": round(float((close[-1] / sma_200 - 1) * 100), 2) if sma_200 else None
                },
                "ma_alignment": {
                    "bullish_aligned": bool(ma_aligned_bullish),
                    "bearish_aligned": bool(ma_aligned_bearish)
                },
                "structure": {
                    "type": structure,
                    "description": structure_description,
                    "higher_highs": int(hh_count),
                    "lower_highs": int(lh_count),
                    "higher_lows": int(hl_count),
                    "lower_lows": int(ll_count)
                },
                "momentum": {
                    "10d_momentum_pct": round(float(recent_momentum), 2),
                    "21d_momentum_pct": round(float(monthly_momentum), 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing structure for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Technical Agent's analysis
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with technical analysis
        """
        ticker = state.get("ticker", "")
        
        if not ticker:
            return {
                **state,
                "errors": state.get("errors", []) + ["No ticker provided to Technical Agent"]
            }
        
        try:
            # Perform all technical analyses
            regime_analysis = self.identify_market_regime(ticker)
            support_resistance = self.find_support_resistance(ticker)
            chart_patterns = self.detect_chart_patterns(ticker)
            trend_structure = self.analyze_trend_structure(ticker)
            
            # Prepare summary for LLM
            technical_summary = f"""
Technical Analysis for {ticker}:

MARKET REGIME:
- Current Regime: {regime_analysis.get('regime', 'N/A')}
- Confidence: {regime_analysis.get('confidence', 0) * 100:.0f}%
- Trend Strength (R²): {regime_analysis.get('metrics', {}).get('trend_strength_r2', 'N/A')}
- Volatility: {regime_analysis.get('metrics', {}).get('annualized_volatility_pct', 'N/A')}%

SUPPORT & RESISTANCE:
- Current Price: ₹{support_resistance.get('current_price', 'N/A')}
- Nearest Resistance: ₹{support_resistance.get('nearest_levels', {}).get('resistance', 'N/A')}
- Nearest Support: ₹{support_resistance.get('nearest_levels', {}).get('support', 'N/A')}
- Position in 52W Range: {support_resistance.get('price_range', {}).get('position_in_range_pct', 'N/A')}%

CHART PATTERNS:
{chr(10).join([f"- {p.get('pattern')}: {p.get('implication')} (Confidence: {p.get('confidence', 0) * 100:.0f}%)" for p in chart_patterns.get('chart_patterns', [])])}

TREND STRUCTURE:
- Structure Type: {trend_structure.get('structure', {}).get('type', 'N/A')}
- Short-term Trend (20d): {trend_structure.get('trend_analysis', {}).get('short_term_20d', 'N/A')}
- Medium-term Trend (50d): {trend_structure.get('trend_analysis', {}).get('medium_term_50d', 'N/A')}
- Long-term Trend (200d): {trend_structure.get('trend_analysis', {}).get('long_term_200d', 'N/A')}
- MA Alignment Bullish: {trend_structure.get('ma_alignment', {}).get('bullish_aligned', False)}
"""
            
            # Get LLM analysis
            messages = self._format_messages(state)
            messages.append(HumanMessage(content=f"""
Based on the following technical analysis data, provide a comprehensive market structure assessment:

{technical_summary}

Please analyze:
1. Overall market regime and what it means for trading
2. Key support and resistance levels to watch
3. Any actionable chart patterns and their implications
4. Trend health and potential direction changes
5. Specific recommendations for entry/exit points based on technical levels
6. Risk factors from a technical perspective
"""))
            
            llm_analysis = await self._invoke_llm(messages)
            
            # Update state with results
            analysis_results = state.get("analysis_results", {})
            analysis_results["technical"] = {
                "regime_analysis": regime_analysis,
                "support_resistance": support_resistance,
                "chart_patterns": chart_patterns,
                "trend_structure": trend_structure,
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
            logger.error(f"Technical Agent execution error: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Technical Agent error: {str(e)}"]
            }

