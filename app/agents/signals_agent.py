import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging

from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator

from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.base import BaseAgent

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


class SignalType(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalsAgent(BaseAgent):

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

            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean() if len(close) >= 200 else pd.Series([np.nan] * len(close))
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()

            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - signal_line

            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]

            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            recent_vol = volume.iloc[-5:].mean()
            prev_vol = volume.iloc[-10:-5].mean()
            volume_trend = "increasing" if recent_vol > prev_vol * 1.1 else "decreasing" if recent_vol < prev_vol * 0.9 else "stable"

            window = 20
            local_max = high.rolling(window, center=True).max()
            local_min = low.rolling(window, center=True).min()

            high_52w = high.max()
            low_52w = low.min()
            recent_high = high.iloc[-20:].max()
            recent_low = low.iloc[-20:].min()

            resistances = []
            supports = []

            price_range = high_52w - low_52w
            levels = np.linspace(low_52w, high_52w, 20)

            for level in levels:
                if level > current_price * 1.01:
                    resistances.append(level)
                elif level < current_price * 0.99:
                    supports.append(level)

            nearest_resistance = min(resistances) if resistances else recent_high
            nearest_support = max(supports) if supports else recent_low

            trend_short = "bullish" if current_price > sma_20.iloc[-1] else "bearish"
            trend_medium = "bullish" if current_price > sma_50.iloc[-1] else "bearish"
            trend_long = "bullish" if not np.isnan(sma_200.iloc[-1]) and current_price > sma_200.iloc[-1] else "bearish" if not np.isnan(sma_200.iloc[-1]) else "unknown"

            ma_aligned_bullish = sma_20.iloc[-1] > sma_50.iloc[-1] > (sma_200.iloc[-1] if not np.isnan(sma_200.iloc[-1]) else 0)
            ma_aligned_bearish = sma_20.iloc[-1] < sma_50.iloc[-1] < (sma_200.iloc[-1] if not np.isnan(sma_200.iloc[-1]) else float('inf'))

            price_change_5d = ((current_price / close.iloc[-5]) - 1) * 100
            price_change_20d = ((current_price / close.iloc[-20]) - 1) * 100

            macd_bullish = macd_histogram.iloc[-1] > 0 and macd_histogram.iloc[-1] > macd_histogram.iloc[-2]
            macd_bearish = macd_histogram.iloc[-1] < 0 and macd_histogram.iloc[-1] < macd_histogram.iloc[-2]
            macd_crossover = macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
            macd_crossunder = macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]

            pct_from_high = ((current_price - high_52w) / high_52w) * 100
            pct_from_low = ((current_price - low_52w) / low_52w) * 100

            try:
                adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
                current_adx = adx_ind.adx().iloc[-1]
                current_plus_di = adx_ind.adx_pos().iloc[-1]
                current_minus_di = adx_ind.adx_neg().iloc[-1]
                if pd.isna(current_adx):
                    current_adx = 0
                if pd.isna(current_plus_di):
                    current_plus_di = 0
                if pd.isna(current_minus_di):
                    current_minus_di = 0
            except Exception:
                current_adx, current_plus_di, current_minus_di = 0, 0, 0

            try:
                stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
                stoch_k_val = stoch.stoch().iloc[-1]
                stoch_d_val = stoch.stoch_signal().iloc[-1]
                if pd.isna(stoch_k_val):
                    stoch_k_val = 50
                if pd.isna(stoch_d_val):
                    stoch_d_val = 50
            except Exception:
                stoch_k_val, stoch_d_val = 50, 50

            try:
                obv_ind = OnBalanceVolumeIndicator(close=close, volume=volume)
                obv_series = obv_ind.on_balance_volume()
                obv_sma_20 = obv_series.rolling(20).mean()
                obv_trending_up = bool(obv_series.iloc[-1] > obv_sma_20.iloc[-1])
                obv_5d_ago = obv_series.iloc[-5] if len(obv_series) >= 5 else obv_series.iloc[0]
                obv_rising = bool(obv_series.iloc[-1] > obv_5d_ago)
            except Exception:
                obv_trending_up = True
                obv_rising = True

            recent_20 = close.iloc[-20:]
            mid_pt = len(recent_20) // 2
            first_half_hi = recent_20.iloc[:mid_pt].max()
            second_half_hi = recent_20.iloc[mid_pt:].max()
            first_half_lo = recent_20.iloc[:mid_pt].min()
            second_half_lo = recent_20.iloc[mid_pt:].min()
            making_higher_highs = bool(second_half_hi > first_half_hi)
            making_higher_lows = bool(second_half_lo > first_half_lo)

            lookback = min(20, len(close) - 1)
            half_lb = lookback // 2
            bullish_divergence = False
            bearish_divergence = False
            if half_lb >= 3:
                price_min_old = close.iloc[-lookback:-half_lb].min()
                price_min_new = close.iloc[-half_lb:].min()
                rsi_min_old = rsi.iloc[-lookback:-half_lb].min()
                rsi_min_new = rsi.iloc[-half_lb:].min()

                price_max_old = close.iloc[-lookback:-half_lb].max()
                price_max_new = close.iloc[-half_lb:].max()
                rsi_max_old = rsi.iloc[-lookback:-half_lb].max()
                rsi_max_new = rsi.iloc[-half_lb:].max()

                bullish_divergence = bool(
                    price_min_new < price_min_old and rsi_min_new > rsi_min_old
                )
                bearish_divergence = bool(
                    price_max_new > price_max_old and rsi_max_new < rsi_max_old
                )

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
                        "position": round(bb_position, 2),
                        "squeeze": bb_std.iloc[-1] < bb_std.iloc[-20:].mean() * 0.8
                    },
                    "atr": round(atr, 2),
                    "atr_pct": round((atr / current_price) * 100, 2),
                    "adx": {
                        "value": round(current_adx, 2),
                        "plus_di": round(current_plus_di, 2),
                        "minus_di": round(current_minus_di, 2),
                        "trend_strength": "strong" if current_adx > 25 else "weak" if current_adx < 20 else "moderate"
                    },
                    "stochastic": {
                        "k": round(stoch_k_val, 2),
                        "d": round(stoch_d_val, 2),
                        "signal": "oversold" if stoch_k_val < 20 else "overbought" if stoch_k_val > 80 else "neutral"
                    }
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
                    "trend": volume_trend,
                    "obv_trending_up": obv_trending_up,
                    "obv_rising": obv_rising
                },
                "levels": {
                    "resistance": round(nearest_resistance, 2),
                    "support": round(nearest_support, 2),
                    "high_52w": round(high_52w, 2),
                    "low_52w": round(low_52w, 2),
                    "pct_from_high": round(pct_from_high, 2),
                    "pct_from_low": round(pct_from_low, 2)
                },
                "price_structure": {
                    "making_higher_highs": making_higher_highs,
                    "making_higher_lows": making_higher_lows,
                    "structure": "bullish" if making_higher_highs and making_higher_lows else "bearish" if not making_higher_highs and not making_higher_lows else "mixed"
                },
                "divergence": {
                    "bullish": bullish_divergence,
                    "bearish": bearish_divergence
                }
            })

        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def generate_signal(ticker: str) -> Dict[str, Any]:
        try:
            data = SignalsAgent.calculate_indicators(ticker)

            if not data.get("success"):
                return {"success": False, "error": data.get("error", "Failed to calculate indicators")}

            current_price = data["current_price"]
            indicators = data["indicators"]
            trend = data["trend"]
            volume = data["volume"]
            levels = data["levels"]
            price_structure = data.get("price_structure", {})
            divergence = data.get("divergence", {})

            bullish_score = 0
            bearish_score = 0
            reasons_bullish = []
            reasons_bearish = []

            rsi = indicators["rsi"]
            medium_trend = trend["medium_term"]

            if rsi < 30:
                if medium_trend == "bullish":
                    bullish_score += 3
                    reasons_bullish.append(f"RSI oversold ({rsi:.0f}) in uptrend — high-probability dip buy")
                else:
                    bullish_score += 1
                    reasons_bullish.append(f"RSI oversold ({rsi:.0f}) but downtrend active — risky bounce")
            elif rsi < 40:
                if medium_trend == "bullish":
                    bullish_score += 1
                    reasons_bullish.append(f"RSI pulling back ({rsi:.0f}) in uptrend — potential entry zone")
            elif rsi > 70:
                if medium_trend == "bearish":
                    bearish_score += 3
                    reasons_bearish.append(f"RSI overbought ({rsi:.0f}) in downtrend — strong sell signal")
                else:
                    bearish_score += 1
                    reasons_bearish.append(f"RSI elevated ({rsi:.0f}) — trend strong but watch for pullback")
            elif rsi > 60:
                if medium_trend == "bearish":
                    bearish_score += 1
                    reasons_bearish.append(f"RSI rising ({rsi:.0f}) against downtrend — likely to fade")

            macd = indicators["macd"]
            if macd["crossover"]:
                bullish_score += 2
                reasons_bullish.append("MACD bullish crossover — fresh momentum shift up")
            elif macd["bullish"]:
                bullish_score += 1
                reasons_bullish.append("MACD histogram expanding — bullish momentum building")
            elif macd["crossunder"]:
                bearish_score += 2
                reasons_bearish.append("MACD bearish crossover — momentum shifting down")
            elif macd["bearish"]:
                bearish_score += 1
                reasons_bearish.append("MACD histogram declining — bearish momentum building")

            ma = indicators["moving_averages"]
            if ma["ma_aligned_bullish"]:
                bullish_score += 2
                reasons_bullish.append("All MAs aligned bullishly (20 > 50 > 200) — strong trend")
            elif ma["ma_aligned_bearish"]:
                bearish_score += 2
                reasons_bearish.append("All MAs aligned bearishly (20 < 50 < 200) — strong downtrend")

            if ma["above_sma_50"]:
                bullish_score += 1
                reasons_bullish.append("Price above 50-day SMA — medium-term uptrend intact")
            else:
                bearish_score += 1
                reasons_bearish.append("Price below 50-day SMA — medium-term trend broken")

            if ma["above_sma_200"] is True:
                bullish_score += 1
                reasons_bullish.append("Price above 200-day SMA — long-term bullish")
            elif ma["above_sma_200"] is False:
                bearish_score += 1
                reasons_bearish.append("Price below 200-day SMA — long-term bearish")

            adx_data = indicators.get("adx", {})
            adx_val = adx_data.get("value", 0) or 0
            plus_di = adx_data.get("plus_di", 0) or 0
            minus_di = adx_data.get("minus_di", 0) or 0

            if adx_val > 30:
                if plus_di > minus_di:
                    bullish_score += 2
                    reasons_bullish.append(f"Strong uptrend confirmed by ADX ({adx_val:.0f}) with DI+ > DI−")
                else:
                    bearish_score += 2
                    reasons_bearish.append(f"Strong downtrend confirmed by ADX ({adx_val:.0f}) with DI− > DI+")
            elif adx_val > 25:
                if plus_di > minus_di:
                    bullish_score += 1
                    reasons_bullish.append(f"Trending up (ADX: {adx_val:.0f}, DI+: {plus_di:.0f} > DI−: {minus_di:.0f})")
                else:
                    bearish_score += 1
                    reasons_bearish.append(f"Trending down (ADX: {adx_val:.0f}, DI−: {minus_di:.0f} > DI+: {plus_di:.0f})")
            elif adx_val < 20:
                reasons_bearish.append(f"Weak trend (ADX: {adx_val:.0f}) — range-bound, directional trades risky")

            stoch_data = indicators.get("stochastic", {})
            stoch_k = stoch_data.get("k", 50) or 50
            stoch_d = stoch_data.get("d", 50) or 50

            if stoch_k < 20 and stoch_d < 20:
                bullish_score += 1
                reasons_bullish.append(f"Stochastic oversold ({stoch_k:.0f}/{stoch_d:.0f}) — confirms buying opportunity")
            elif stoch_k > 80 and stoch_d > 80:
                bearish_score += 1
                reasons_bearish.append(f"Stochastic overbought ({stoch_k:.0f}/{stoch_d:.0f}) — confirms selling pressure")
            if stoch_k < 30 and stoch_k > stoch_d:
                bullish_score += 1
                reasons_bullish.append("Stochastic bullish crossover in oversold zone")
            elif stoch_k > 70 and stoch_k < stoch_d:
                bearish_score += 1
                reasons_bearish.append("Stochastic bearish crossover in overbought zone")

            short_t = trend["short_term"]
            medium_t = trend["medium_term"]
            long_t = trend["long_term"]

            if short_t == medium_t == long_t and short_t != "unknown":
                if short_t == "bullish":
                    bullish_score += 2
                    reasons_bullish.append("All timeframes aligned BULLISH (20d/50d/200d) — highest conviction")
                else:
                    bearish_score += 2
                    reasons_bearish.append("All timeframes aligned BEARISH (20d/50d/200d) — highest conviction")
            elif short_t != medium_t:
                reasons_bearish.append(f"Timeframe conflict: short-term {short_t} vs medium-term {medium_t} — wait for alignment")

            obv_up = volume.get("obv_trending_up")
            obv_rising = volume.get("obv_rising", True)

            if volume["ratio"] > 1.5 and trend["price_change_5d"] > 0:
                bullish_score += 1
                reasons_bullish.append("High volume on up move — institutional buying interest")
            elif volume["ratio"] > 1.5 and trend["price_change_5d"] < 0:
                bearish_score += 1
                reasons_bearish.append("High volume on down move — institutional selling pressure")

            if obv_up is not None:
                if trend["price_change_20d"] > 0 and not obv_up:
                    bearish_score += 1
                    reasons_bearish.append("Price rising but OBV declining — rally not confirmed by volume (distribution)")
                elif trend["price_change_20d"] < 0 and obv_up:
                    bullish_score += 1
                    reasons_bullish.append("Price falling but OBV rising — smart money accumulating")

            if divergence.get("bullish"):
                bullish_score += 2
                reasons_bullish.append("Bullish RSI divergence — price making lower lows but momentum improving")
            if divergence.get("bearish"):
                bearish_score += 2
                reasons_bearish.append("Bearish RSI divergence — price making higher highs but momentum weakening")

            structure = price_structure.get("structure", "mixed")
            if structure == "bullish":
                bullish_score += 1
                reasons_bullish.append("Healthy price structure — making higher highs and higher lows")
            elif structure == "bearish":
                bearish_score += 1
                reasons_bearish.append("Deteriorating price structure — making lower highs and lower lows")

            bb = indicators["bollinger"]
            if bb["position"] < 0.15:
                bullish_score += 1
                reasons_bullish.append("Price at lower Bollinger Band — oversold by volatility measure")
            elif bb["position"] > 0.85:
                bearish_score += 1
                reasons_bearish.append("Price at upper Bollinger Band — overbought by volatility measure")

            if bb["squeeze"]:
                reasons_bullish.append("Bollinger Band squeeze — big directional move imminent")

            support_distance = ((current_price - levels["support"]) / current_price) * 100
            resistance_distance = ((levels["resistance"] - current_price) / current_price) * 100

            if support_distance < 2:
                bullish_score += 1
                reasons_bullish.append(f"Price near key support at ₹{levels['support']:.0f} — potential bounce zone")

            if resistance_distance < 2:
                bearish_score += 1
                reasons_bearish.append(f"Price near key resistance at ₹{levels['resistance']:.0f} — potential rejection")

            if levels["pct_from_high"] > -5:
                bearish_score += 1
                reasons_bearish.append(f"Near 52-week high ({levels['pct_from_high']:.1f}%) — extended, risk of pullback")
            elif levels["pct_from_high"] < -30:
                bullish_score += 1
                reasons_bullish.append(f"Down {abs(levels['pct_from_high']):.1f}% from 52-week high — deep correction, potential value")

            if levels["pct_from_low"] < 10:
                bearish_score += 1
                reasons_bearish.append(f"Near 52-week low ({levels['pct_from_low']:.1f}% above) — weak momentum, falling knife risk")

            net_score = bullish_score - bearish_score

            if net_score >= 8:
                signal = SignalType.STRONG_BUY
                signal_strength = 5
            elif net_score >= 5:
                signal = SignalType.BUY
                signal_strength = 4
            elif net_score >= 4:
                signal = SignalType.BUY
                signal_strength = 3
            elif net_score <= -8:
                signal = SignalType.STRONG_SELL
                signal_strength = 5
            elif net_score <= -5:
                signal = SignalType.SELL
                signal_strength = 4
            elif net_score <= -4:
                signal = SignalType.SELL
                signal_strength = 3
            else:
                signal = SignalType.HOLD
                signal_strength = 2

            atr = indicators["atr"]
            atr_pct = indicators["atr_pct"]

            if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                entry_low = round(current_price - (atr * 0.5), 2)
                entry_high = round(current_price, 2)

                atr_stop = current_price - (atr * 2)
                support_stop = levels["support"] * 0.99
                stop_loss = round(max(atr_stop, support_stop), 2)

                max_stop = current_price * 0.92
                stop_loss = max(stop_loss, max_stop)

                stop_loss_pct = round(((current_price - stop_loss) / current_price) * 100, 2)

                risk = current_price - stop_loss
                target_1 = round(current_price + (risk * 2), 2)
                target_2 = round(current_price + (risk * 3), 2)
                target_3 = round(min(current_price + (risk * 5), levels["resistance"] * 1.02), 2)

                target_1_pct = round(((target_1 - current_price) / current_price) * 100, 2)
                target_2_pct = round(((target_2 - current_price) / current_price) * 100, 2)
                target_3_pct = round(((target_3 - current_price) / current_price) * 100, 2)

            elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
                entry_low = None
                entry_high = None

                stop_loss = round(min(levels["resistance"] * 1.01, current_price + (atr * 2)), 2)
                stop_loss_pct = round(((stop_loss - current_price) / current_price) * 100, 2)

                target_1 = round(current_price - (atr * 2), 2)
                target_2 = round(current_price - (atr * 4), 2)
                target_3 = round(max(levels["support"], current_price - (atr * 6)), 2)

                target_1_pct = round(((current_price - target_1) / current_price) * 100, 2)
                target_2_pct = round(((current_price - target_2) / current_price) * 100, 2)
                target_3_pct = round(((current_price - target_3) / current_price) * 100, 2)

            else:
                entry_low = round(levels["support"], 2)
                entry_high = round(current_price - (atr * 0.5), 2)
                stop_loss = round(levels["support"] - (atr * 1.5), 2)
                stop_loss_pct = round(((current_price - stop_loss) / current_price) * 100, 2)
                target_1 = round(levels["resistance"], 2)
                target_2 = round(levels["resistance"] + (atr * 2), 2)
                target_3 = round(levels["high_52w"], 2)
                target_1_pct = round(((target_1 - current_price) / current_price) * 100, 2)
                target_2_pct = round(((target_2 - current_price) / current_price) * 100, 2)
                target_3_pct = round(((target_3 - current_price) / current_price) * 100, 2)

            if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                risk_amount = current_price - stop_loss
                reward_amount = target_2 - current_price
                risk_reward = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0

                if risk_reward < 1.5:
                    signal = SignalType.HOLD
                    signal_strength = 2
                    reasons_bearish.append(f"Signal downgraded: Risk/Reward ({risk_reward}:1) below 1.5:1 minimum")
            elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
                risk_amount = stop_loss - current_price
                reward_amount = current_price - target_2
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
                    SignalType.STRONG_BUY.value: "Strong buying opportunity — multiple indicators and timeframes aligned",
                    SignalType.BUY.value: "Good buying opportunity — favorable conditions with confirmed trend",
                    SignalType.HOLD.value: "Wait for better entry — mixed or conflicting signals",
                    SignalType.SELL.value: "Consider selling/avoiding — bearish conditions confirmed",
                    SignalType.STRONG_SELL.value: "Strong sell signal — exit positions, multiple bearish confirmations"
                }[signal.value],
                "entry": {
                    "zone_low": entry_low,
                    "zone_high": entry_high,
                    "ideal_entry": round(current_price - (atr * 0.3), 2) if signal in [SignalType.STRONG_BUY, SignalType.BUY] else None
                },
                "stop_loss": {
                    "price": stop_loss,
                    "percentage": stop_loss_pct,
                    "method": "2x ATR from entry, capped at support level"
                },
                "targets": {
                    "target_1": {"price": target_1, "percentage": target_1_pct, "risk_reward": "2:1"},
                    "target_2": {"price": target_2, "percentage": target_2_pct, "risk_reward": "3:1"},
                    "target_3": {"price": target_3, "percentage": target_3_pct, "risk_reward": "5:1"}
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
                "price_structure": price_structure,
                "divergence": divergence,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def scan_for_signals(tickers: List[str], signal_filter: Optional[str] = None) -> Dict[str, Any]:
        results = []

        for ticker in tickers:
            try:
                signal_data = SignalsAgent.generate_signal(ticker)
                if signal_data.get("success"):
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
- ADX: {signal_data['indicators'].get('adx', {}).get('value', 'N/A')} ({signal_data['indicators'].get('adx', {}).get('trend_strength', 'N/A')} trend)
- Stochastic: {signal_data['indicators'].get('stochastic', {}).get('k', 'N/A')}/{signal_data['indicators'].get('stochastic', {}).get('d', 'N/A')} ({signal_data['indicators'].get('stochastic', {}).get('signal', 'N/A')})
- Trend: {signal_data['trend']['short_term']} (short), {signal_data['trend']['medium_term']} (medium), {signal_data['trend']['long_term']} (long)
- Volume: {signal_data['volume']['ratio']}x average ({signal_data['volume']['trend']})
- Price Structure: {signal_data.get('price_structure', {}).get('structure', 'N/A')}
- RSI Divergence: {'Bullish' if signal_data.get('divergence', {}).get('bullish') else 'Bearish' if signal_data.get('divergence', {}).get('bearish') else 'None'}

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
