"""
Trading Signals Agent — Buy/Sell Signal Generation

This agent is responsible for:
- Generating clear BUY/SELL/HOLD signals
- Calculating entry, stop loss, and target prices
- Scoring signal strength based on multiple indicators
- Providing reasoning for each signal
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
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


class SignalType(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalsAgent(BaseAgent):
    """
    Trading Signals Agent - Generates actionable buy/sell signals
    
    Uses multiple indicators:
    - RSI (momentum)
    - MACD (trend)
    - Moving Averages (trend)
    - Support/Resistance (levels)
    - Volume (confirmation)
    - Bollinger Bands (volatility)
    """
    
    def __init__(self):
        super().__init__(
            name="SignalsAgent",
            description="Generates trading signals with entry/exit levels and reasoning",
            temperature=0.2
        )
    
    def _setup_tools(self) -> List[Any]:
        return []
    
    def _create_system_prompt(self) -> str:
        return """You are the Trading Signals Agent, specialized in generating actionable buy/sell signals for the INDIAN STOCK MARKET.

IMPORTANT: Always use Indian Rupees (₹) for all currency values. Never use $ or USD.

Your responsibilities:
1. Analyze multiple technical indicators to generate signals
2. Provide clear BUY/SELL/HOLD recommendations
3. Calculate optimal entry, stop loss, and target prices
4. Explain the reasoning behind each signal
5. Assess signal strength and confidence level

Signal Generation Rules:
- STRONG_BUY: 4+ bullish indicators align, oversold with reversal
- BUY: 3+ bullish indicators, trend support
- HOLD: Mixed signals, wait for clarity
- SELL: 3+ bearish indicators, trend resistance
- STRONG_SELL: 4+ bearish indicators, overbought with reversal

Always provide:
- Clear signal with strength rating (1-5 stars)
- Entry zone (price range to buy)
- Stop loss level with % risk
- Target prices (T1, T2, T3)
- Risk/Reward ratio
- Key reasons for the signal"""

    @staticmethod
    def calculate_indicators(ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Calculate all technical indicators needed for signal generation"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 50:
                return {"success": False, "error": "Insufficient data"}
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            current_price = close.iloc[-1]
            
            # === MOVING AVERAGES ===
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean() if len(close) >= 200 else pd.Series([np.nan] * len(close))
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            
            # === RSI ===
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # === MACD ===
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - signal_line
            
            # === BOLLINGER BANDS ===
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # === ATR (for stop loss calculation) ===
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            
            # === VOLUME ANALYSIS ===
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Recent volume trend (last 5 days vs previous 5 days)
            recent_vol = volume.iloc[-5:].mean()
            prev_vol = volume.iloc[-10:-5].mean()
            volume_trend = "increasing" if recent_vol > prev_vol * 1.1 else "decreasing" if recent_vol < prev_vol * 0.9 else "stable"
            
            # === SUPPORT & RESISTANCE ===
            # Find local minima and maxima
            window = 20
            local_max = high.rolling(window, center=True).max()
            local_min = low.rolling(window, center=True).min()
            
            # Recent highs and lows for S/R
            high_52w = high.max()
            low_52w = low.min()
            recent_high = high.iloc[-20:].max()
            recent_low = low.iloc[-20:].min()
            
            # Calculate support and resistance levels
            resistances = []
            supports = []
            
            # Use price clusters
            price_range = high_52w - low_52w
            levels = np.linspace(low_52w, high_52w, 20)
            
            for level in levels:
                if level > current_price * 1.01:  # Resistance
                    resistances.append(level)
                elif level < current_price * 0.99:  # Support
                    supports.append(level)
            
            nearest_resistance = min(resistances) if resistances else recent_high
            nearest_support = max(supports) if supports else recent_low
            
            # === TREND ANALYSIS ===
            trend_short = "bullish" if current_price > sma_20.iloc[-1] else "bearish"
            trend_medium = "bullish" if current_price > sma_50.iloc[-1] else "bearish"
            trend_long = "bullish" if not np.isnan(sma_200.iloc[-1]) and current_price > sma_200.iloc[-1] else "bearish" if not np.isnan(sma_200.iloc[-1]) else "unknown"
            
            # MA alignment
            ma_aligned_bullish = sma_20.iloc[-1] > sma_50.iloc[-1] > (sma_200.iloc[-1] if not np.isnan(sma_200.iloc[-1]) else 0)
            ma_aligned_bearish = sma_20.iloc[-1] < sma_50.iloc[-1] < (sma_200.iloc[-1] if not np.isnan(sma_200.iloc[-1]) else float('inf'))
            
            # === MOMENTUM ===
            price_change_5d = ((current_price / close.iloc[-5]) - 1) * 100
            price_change_20d = ((current_price / close.iloc[-20]) - 1) * 100
            
            # MACD signals
            macd_bullish = macd_histogram.iloc[-1] > 0 and macd_histogram.iloc[-1] > macd_histogram.iloc[-2]
            macd_bearish = macd_histogram.iloc[-1] < 0 and macd_histogram.iloc[-1] < macd_histogram.iloc[-2]
            macd_crossover = macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
            macd_crossunder = macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]
            
            # === 52-WEEK POSITION ===
            pct_from_high = ((current_price - high_52w) / high_52w) * 100
            pct_from_low = ((current_price - low_52w) / low_52w) * 100
            
            return convert_numpy_types({
                "success": True,
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "indicators": {
                    "rsi": round(current_rsi, 2),
                    "rsi_signal": "oversold" if current_rsi < 30 else "overbought" if current_rsi > 70 else "neutral",
                    "macd": {
                        "line": round(macd_line.iloc[-1], 4),
                        "signal": round(signal_line.iloc[-1], 4),
                        "histogram": round(macd_histogram.iloc[-1], 4),
                        "bullish": macd_bullish,
                        "bearish": macd_bearish,
                        "crossover": macd_crossover,
                        "crossunder": macd_crossunder
                    },
                    "moving_averages": {
                        "sma_20": round(sma_20.iloc[-1], 2),
                        "sma_50": round(sma_50.iloc[-1], 2),
                        "sma_200": round(sma_200.iloc[-1], 2) if not np.isnan(sma_200.iloc[-1]) else None,
                        "above_sma_20": current_price > sma_20.iloc[-1],
                        "above_sma_50": current_price > sma_50.iloc[-1],
                        "above_sma_200": current_price > sma_200.iloc[-1] if not np.isnan(sma_200.iloc[-1]) else None,
                        "ma_aligned_bullish": ma_aligned_bullish,
                        "ma_aligned_bearish": ma_aligned_bearish
                    },
                    "bollinger": {
                        "upper": round(bb_upper.iloc[-1], 2),
                        "middle": round(bb_middle.iloc[-1], 2),
                        "lower": round(bb_lower.iloc[-1], 2),
                        "position": round(bb_position, 2),  # 0 = at lower, 1 = at upper
                        "squeeze": bb_std.iloc[-1] < bb_std.iloc[-20:].mean() * 0.8
                    },
                    "atr": round(atr, 2),
                    "atr_pct": round((atr / current_price) * 100, 2)
                },
                "trend": {
                    "short_term": trend_short,
                    "medium_term": trend_medium,
                    "long_term": trend_long,
                    "price_change_5d": round(price_change_5d, 2),
                    "price_change_20d": round(price_change_20d, 2)
                },
                "volume": {
                    "current": int(current_volume),
                    "average": int(avg_volume),
                    "ratio": round(volume_ratio, 2),
                    "trend": volume_trend
                },
                "levels": {
                    "resistance": round(nearest_resistance, 2),
                    "support": round(nearest_support, 2),
                    "high_52w": round(high_52w, 2),
                    "low_52w": round(low_52w, 2),
                    "pct_from_high": round(pct_from_high, 2),
                    "pct_from_low": round(pct_from_low, 2)
                }
            })
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def generate_signal(ticker: str) -> Dict[str, Any]:
        """Generate trading signal with entry/exit levels"""
        try:
            # Get all indicators
            data = SignalsAgent.calculate_indicators(ticker)
            
            if not data.get("success"):
                return {"success": False, "error": data.get("error", "Failed to calculate indicators")}
            
            current_price = data["current_price"]
            indicators = data["indicators"]
            trend = data["trend"]
            volume = data["volume"]
            levels = data["levels"]
            
            # === SCORING SYSTEM ===
            bullish_score = 0
            bearish_score = 0
            reasons_bullish = []
            reasons_bearish = []
            
            # RSI Analysis
            rsi = indicators["rsi"]
            if rsi < 30:
                bullish_score += 2
                reasons_bullish.append(f"RSI oversold at {rsi:.0f} - potential bounce")
            elif rsi < 40:
                bullish_score += 1
                reasons_bullish.append(f"RSI approaching oversold ({rsi:.0f})")
            elif rsi > 70:
                bearish_score += 2
                reasons_bearish.append(f"RSI overbought at {rsi:.0f} - potential pullback")
            elif rsi > 60:
                bearish_score += 1
                reasons_bearish.append(f"RSI approaching overbought ({rsi:.0f})")
            
            # MACD Analysis
            macd = indicators["macd"]
            if macd["crossover"]:
                bullish_score += 2
                reasons_bullish.append("MACD bullish crossover - momentum shifting up")
            elif macd["bullish"]:
                bullish_score += 1
                reasons_bullish.append("MACD histogram increasing - bullish momentum")
            elif macd["crossunder"]:
                bearish_score += 2
                reasons_bearish.append("MACD bearish crossover - momentum shifting down")
            elif macd["bearish"]:
                bearish_score += 1
                reasons_bearish.append("MACD histogram decreasing - bearish momentum")
            
            # Moving Average Analysis
            ma = indicators["moving_averages"]
            if ma["ma_aligned_bullish"]:
                bullish_score += 2
                reasons_bullish.append("Moving averages aligned bullishly (20 > 50 > 200)")
            elif ma["ma_aligned_bearish"]:
                bearish_score += 2
                reasons_bearish.append("Moving averages aligned bearishly (20 < 50 < 200)")
            
            if ma["above_sma_50"]:
                bullish_score += 1
                reasons_bullish.append("Price above 50-day SMA - medium-term uptrend")
            else:
                bearish_score += 1
                reasons_bearish.append("Price below 50-day SMA - medium-term downtrend")
            
            if ma["above_sma_200"]:
                bullish_score += 1
                reasons_bullish.append("Price above 200-day SMA - long-term uptrend")
            elif ma["above_sma_200"] is False:
                bearish_score += 1
                reasons_bearish.append("Price below 200-day SMA - long-term downtrend")
            
            # Bollinger Bands Analysis
            bb = indicators["bollinger"]
            if bb["position"] < 0.2:
                bullish_score += 1
                reasons_bullish.append("Price near lower Bollinger Band - potential bounce")
            elif bb["position"] > 0.8:
                bearish_score += 1
                reasons_bearish.append("Price near upper Bollinger Band - potential pullback")
            
            if bb["squeeze"]:
                reasons_bullish.append("Bollinger squeeze detected - big move coming")
            
            # Support/Resistance Analysis
            support_distance = ((current_price - levels["support"]) / current_price) * 100
            resistance_distance = ((levels["resistance"] - current_price) / current_price) * 100
            
            if support_distance < 3:
                bullish_score += 1
                reasons_bullish.append(f"Price near support at ₹{levels['support']:.0f}")
            
            if resistance_distance < 3:
                bearish_score += 1
                reasons_bearish.append(f"Price near resistance at ₹{levels['resistance']:.0f}")
            
            # Volume Confirmation
            if volume["ratio"] > 1.5 and trend["price_change_5d"] > 0:
                bullish_score += 1
                reasons_bullish.append("High volume on up move - strong buying interest")
            elif volume["ratio"] > 1.5 and trend["price_change_5d"] < 0:
                bearish_score += 1
                reasons_bearish.append("High volume on down move - strong selling pressure")
            
            # 52-Week Position
            if levels["pct_from_high"] > -10:
                bullish_score += 1
                reasons_bullish.append(f"Near 52-week high ({levels['pct_from_high']:.1f}%) - strength")
            elif levels["pct_from_low"] < 20:
                bullish_score += 1
                reasons_bullish.append(f"Near 52-week low - potential value")
            
            # === DETERMINE SIGNAL ===
            net_score = bullish_score - bearish_score
            
            if net_score >= 5:
                signal = SignalType.STRONG_BUY
                signal_strength = 5
            elif net_score >= 3:
                signal = SignalType.BUY
                signal_strength = 4
            elif net_score >= 1:
                signal = SignalType.BUY
                signal_strength = 3
            elif net_score <= -5:
                signal = SignalType.STRONG_SELL
                signal_strength = 5
            elif net_score <= -3:
                signal = SignalType.SELL
                signal_strength = 4
            elif net_score <= -1:
                signal = SignalType.SELL
                signal_strength = 3
            else:
                signal = SignalType.HOLD
                signal_strength = 2
            
            # === CALCULATE ENTRY/EXIT LEVELS ===
            atr = indicators["atr"]
            
            if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                # Entry zone: current price to slight pullback
                entry_low = round(current_price * 0.98, 2)
                entry_high = round(current_price * 1.01, 2)
                
                # Stop loss: below support or 2x ATR
                stop_loss = round(max(levels["support"] * 0.98, current_price - (atr * 2)), 2)
                stop_loss_pct = round(((current_price - stop_loss) / current_price) * 100, 2)
                
                # Targets based on risk/reward
                risk = current_price - stop_loss
                target_1 = round(current_price + (risk * 1.5), 2)  # 1.5:1 R/R
                target_2 = round(current_price + (risk * 2.5), 2)  # 2.5:1 R/R
                target_3 = round(min(current_price + (risk * 4), levels["resistance"] * 1.05), 2)  # 4:1 R/R or resistance
                
                target_1_pct = round(((target_1 - current_price) / current_price) * 100, 2)
                target_2_pct = round(((target_2 - current_price) / current_price) * 100, 2)
                target_3_pct = round(((target_3 - current_price) / current_price) * 100, 2)
                
            elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
                # For sell signals, show where NOT to buy and exit levels
                entry_low = None
                entry_high = None
                
                stop_loss = round(levels["resistance"] * 1.02, 2)  # Above resistance
                stop_loss_pct = round(((stop_loss - current_price) / current_price) * 100, 2)
                
                # Downside targets
                target_1 = round(current_price * 0.95, 2)
                target_2 = round(current_price * 0.90, 2)
                target_3 = round(max(levels["support"], current_price * 0.85), 2)
                
                target_1_pct = round(((current_price - target_1) / current_price) * 100, 2)
                target_2_pct = round(((current_price - target_2) / current_price) * 100, 2)
                target_3_pct = round(((current_price - target_3) / current_price) * 100, 2)
                
            else:  # HOLD
                entry_low = round(levels["support"], 2)
                entry_high = round(current_price * 0.98, 2)
                stop_loss = round(levels["support"] * 0.95, 2)
                stop_loss_pct = round(((current_price - stop_loss) / current_price) * 100, 2)
                target_1 = round(levels["resistance"], 2)
                target_2 = round(levels["resistance"] * 1.05, 2)
                target_3 = round(levels["high_52w"], 2)
                target_1_pct = round(((target_1 - current_price) / current_price) * 100, 2)
                target_2_pct = round(((target_2 - current_price) / current_price) * 100, 2)
                target_3_pct = round(((target_3 - current_price) / current_price) * 100, 2)
            
            # Risk/Reward calculation
            if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                risk_amount = current_price - stop_loss
                reward_amount = target_2 - current_price
                risk_reward = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0
            else:
                risk_reward = 0
            
            return convert_numpy_types({
                "success": True,
                "ticker": ticker,
                "current_price": current_price,
                "signal": signal.value,
                "signal_strength": signal_strength,
                "signal_description": {
                    SignalType.STRONG_BUY.value: "Strong buying opportunity - multiple indicators aligned",
                    SignalType.BUY.value: "Good buying opportunity - favorable conditions",
                    SignalType.HOLD.value: "Wait for better entry - mixed signals",
                    SignalType.SELL.value: "Consider selling/avoiding - unfavorable conditions",
                    SignalType.STRONG_SELL.value: "Strong sell signal - exit positions"
                }[signal.value],
                "entry": {
                    "zone_low": entry_low,
                    "zone_high": entry_high,
                    "ideal_entry": round(current_price * 0.99, 2) if signal in [SignalType.STRONG_BUY, SignalType.BUY] else None
                },
                "stop_loss": {
                    "price": stop_loss,
                    "percentage": stop_loss_pct
                },
                "targets": {
                    "target_1": {"price": target_1, "percentage": target_1_pct, "risk_reward": "1.5:1"},
                    "target_2": {"price": target_2, "percentage": target_2_pct, "risk_reward": "2.5:1"},
                    "target_3": {"price": target_3, "percentage": target_3_pct, "risk_reward": "4:1"}
                },
                "risk_reward_ratio": risk_reward,
                "scores": {
                    "bullish": bullish_score,
                    "bearish": bearish_score,
                    "net": net_score
                },
                "reasons": {
                    "bullish": reasons_bullish,
                    "bearish": reasons_bearish
                },
                "indicators": data["indicators"],
                "trend": data["trend"],
                "volume": data["volume"],
                "levels": data["levels"],
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def scan_for_signals(tickers: List[str], signal_filter: Optional[str] = None) -> Dict[str, Any]:
        """Scan multiple stocks for signals"""
        results = []
        
        for ticker in tickers:
            try:
                signal_data = SignalsAgent.generate_signal(ticker)
                if signal_data.get("success"):
                    # Apply filter if specified
                    if signal_filter:
                        if signal_filter.upper() == "BUY" and signal_data["signal"] not in ["STRONG_BUY", "BUY"]:
                            continue
                        elif signal_filter.upper() == "SELL" and signal_data["signal"] not in ["STRONG_SELL", "SELL"]:
                            continue
                    
                    results.append({
                        "ticker": ticker,
                        "signal": signal_data["signal"],
                        "signal_strength": signal_data["signal_strength"],
                        "current_price": signal_data["current_price"],
                        "entry_zone": signal_data["entry"],
                        "stop_loss": signal_data["stop_loss"],
                        "target_1": signal_data["targets"]["target_1"],
                        "risk_reward": signal_data["risk_reward_ratio"],
                        "reasons": signal_data["reasons"]["bullish"][:2] if signal_data["signal"] in ["STRONG_BUY", "BUY"] else signal_data["reasons"]["bearish"][:2]
                    })
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        # Sort by signal strength
        results.sort(key=lambda x: (
            {"STRONG_BUY": 5, "BUY": 4, "HOLD": 3, "SELL": 2, "STRONG_SELL": 1}[x["signal"]],
            x["signal_strength"]
        ), reverse=True)
        
        return {
            "success": True,
            "scanned": len(tickers),
            "signals_found": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_ai_analysis(self, signal_data: Dict[str, Any]) -> str:
        """Generate AI analysis for the signal"""
        if not signal_data.get("success"):
            return "Unable to generate analysis due to data error."
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Analyze this trading signal and provide actionable advice:

Ticker: {signal_data['ticker']}
Current Price: ₹{signal_data['current_price']}
Signal: {signal_data['signal']} (Strength: {signal_data['signal_strength']}/5)

Entry Zone: ₹{signal_data['entry'].get('zone_low', 'N/A')} - ₹{signal_data['entry'].get('zone_high', 'N/A')}
Stop Loss: ₹{signal_data['stop_loss']['price']} ({signal_data['stop_loss']['percentage']}% risk)
Target 1: ₹{signal_data['targets']['target_1']['price']} (+{signal_data['targets']['target_1']['percentage']}%)
Target 2: ₹{signal_data['targets']['target_2']['price']} (+{signal_data['targets']['target_2']['percentage']}%)
Risk/Reward: {signal_data['risk_reward_ratio']}:1

Technical Indicators:
- RSI: {signal_data['indicators']['rsi']} ({signal_data['indicators']['rsi_signal']})
- MACD: {'Bullish' if signal_data['indicators']['macd']['bullish'] else 'Bearish' if signal_data['indicators']['macd']['bearish'] else 'Neutral'}
- Trend: {signal_data['trend']['short_term']} (short), {signal_data['trend']['medium_term']} (medium)
- Volume: {signal_data['volume']['ratio']}x average ({signal_data['volume']['trend']})

Bullish Factors:
{chr(10).join(['- ' + r for r in signal_data['reasons']['bullish']])}

Bearish Factors:
{chr(10).join(['- ' + r for r in signal_data['reasons']['bearish']])}

Please provide:
1. **Signal Summary**: Quick interpretation of the signal
2. **Action Plan**: Specific steps to take
3. **Risk Warning**: Key risks to watch
4. **Timing**: When to enter/exit
5. **Position Sizing**: Suggested allocation based on signal strength

Keep it concise and actionable. Use ₹ for all prices.
""")
        ]
        
        try:
            return await self._invoke_llm(messages)
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return f"AI analysis unavailable: {str(e)}"

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute signal generation"""
        ticker = state.get("ticker", "")
        
        if not ticker:
            return {**state, "errors": ["No ticker provided"]}
        
        signal_data = self.generate_signal(ticker)
        
        if signal_data.get("success"):
            ai_analysis = await self.generate_ai_analysis(signal_data)
            signal_data["ai_analysis"] = ai_analysis
        
        return {
            **state,
            "signal_data": signal_data
        }

