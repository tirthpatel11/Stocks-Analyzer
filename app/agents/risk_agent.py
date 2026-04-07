import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize
import logging

from langchain_core.messages import HumanMessage
from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="RiskAgent",
            description="Manages risk, position sizing, and portfolio construction guardrails",
            temperature=0.0
        )

    def _setup_tools(self) -> List[Any]:
        return [
            self.calculate_position_size,
            self.compute_var,
            self.analyze_drawdown,
            self.portfolio_risk_metrics,
            self.generate_risk_limits
        ]

    def _create_system_prompt(self) -> str:
        return """You are the Risk & Portfolio Construction Agent, specialized in protecting capital and optimizing risk-adjusted returns for the INDIAN STOCK MARKET.

IMPORTANT: Always use Indian Rupees (₹) for all currency values. Never use $ or USD.

Your responsibilities:
1. Calculate appropriate position sizes based on risk tolerance
2. Compute Value at Risk (VaR) and other risk metrics
3. Analyze historical drawdowns and stress scenarios
4. Recommend portfolio allocations and diversification
5. Enforce risk limits and guardrails

Risk Management Principles:
- Capital preservation is the primary objective
- Never risk more than the acceptable loss per trade
- Consider correlation and portfolio-level risk
- Account for liquidity and market conditions
- Use multiple risk metrics for comprehensive assessment

Output Format:
- Clear position sizing recommendations with justification
- Risk metrics presented with confidence intervals
- Specific guardrails and limits with reasoning
- Warning signals and risk factors
- Actionable risk management recommendations

You work with Market Data and Technical agents. Your analysis should:
- Use volatility data from Market Data Agent for position sizing
- Consider technical levels from Technical Agent for stop placement
- Provide final risk-adjusted recommendations to the Supervisor"""

    @staticmethod
    def calculate_position_size(
        ticker: str,
        account_size: float = 100000.0,
        risk_per_trade_pct: float = 2.0,
        stop_loss_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo")

            if df.empty or len(df) < 20:
                return {
                    "success": False,
                    "error": "Insufficient data for position sizing",
                    "ticker": ticker
                }

            current_price = df['Close'].iloc[-1]

            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values

            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_14 = np.mean(true_range[-14:])
            atr_pct = (atr_14 / current_price) * 100

            returns = pd.Series(close).pct_change().dropna()
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252) * 100

            dollar_risk = account_size * (risk_per_trade_pct / 100)

            if stop_loss_pct:
                stop_distance = current_price * (stop_loss_pct / 100)
            else:
                stop_distance = atr_14 * 2
                stop_loss_pct = (stop_distance / current_price) * 100

            shares_fixed_risk = int(dollar_risk / stop_distance)
            position_value_fixed = shares_fixed_risk * current_price

            target_position_volatility = 0.01
            volatility_adjusted_allocation = target_position_volatility / daily_volatility
            position_value_vol_adjusted = account_size * min(volatility_adjusted_allocation, 0.25)
            shares_vol_adjusted = int(position_value_vol_adjusted / current_price)

            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            if len(positive_returns) > 0 and len(negative_returns) > 0:
                win_rate = len(positive_returns) / len(returns)
                avg_win = positive_returns.mean()
                avg_loss = abs(negative_returns.mean())
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1

                kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio
                kelly_pct = max(0, min(kelly_pct, 0.25))

                half_kelly_pct = kelly_pct / 2
                position_value_kelly = account_size * half_kelly_pct
                shares_kelly = int(position_value_kelly / current_price)
            else:
                kelly_pct = 0
                half_kelly_pct = 0
                shares_kelly = 0
                win_rate = 0.5
                win_loss_ratio = 1

            recommended_shares = min(shares_fixed_risk, shares_vol_adjusted, max(1, shares_kelly))
            recommended_value = recommended_shares * current_price

            return {
                "success": True,
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "account_size": account_size,
                "risk_parameters": {
                    "risk_per_trade_pct": risk_per_trade_pct,
                    "dollar_risk": round(dollar_risk, 2),
                    "stop_loss_pct": round(stop_loss_pct, 2),
                    "stop_price": round(current_price * (1 - stop_loss_pct / 100), 2)
                },
                "volatility_metrics": {
                    "atr_14": round(atr_14, 2),
                    "atr_pct": round(atr_pct, 2),
                    "daily_volatility_pct": round(daily_volatility * 100, 2),
                    "annualized_volatility_pct": round(annualized_volatility, 2)
                },
                "position_sizing": {
                    "fixed_risk": {
                        "shares": shares_fixed_risk,
                        "value": round(position_value_fixed, 2),
                        "pct_of_account": round(position_value_fixed / account_size * 100, 2)
                    },
                    "volatility_adjusted": {
                        "shares": shares_vol_adjusted,
                        "value": round(position_value_vol_adjusted, 2),
                        "pct_of_account": round(position_value_vol_adjusted / account_size * 100, 2)
                    },
                    "kelly_criterion": {
                        "full_kelly_pct": round(kelly_pct * 100, 2),
                        "half_kelly_pct": round(half_kelly_pct * 100, 2),
                        "shares": shares_kelly,
                        "win_rate": round(win_rate * 100, 2),
                        "win_loss_ratio": round(win_loss_ratio, 2)
                    },
                    "recommended": {
                        "shares": recommended_shares,
                        "value": round(recommended_value, 2),
                        "pct_of_account": round(recommended_value / account_size * 100, 2),
                        "method": "Conservative (minimum of all methods)"
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error calculating position size for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }

    @staticmethod
    def compute_var(
        ticker: str,
        position_value: float = 10000.0,
        confidence_levels: List[float] = [0.95, 0.99],
        time_horizon_days: int = 1
    ) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")

            if df.empty or len(df) < 252:
                return {
                    "success": False,
                    "error": "Insufficient data for VaR calculation (need at least 1 year)",
                    "ticker": ticker
                }

            returns = df['Close'].pct_change().dropna()

            mean_return = returns.mean()
            std_return = returns.std()
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)

            var_results = {}

            for conf in confidence_levels:
                conf_str = f"{int(conf * 100)}%"
                alpha = 1 - conf

                historical_var = np.percentile(returns, alpha * 100)
                historical_var_dollar = abs(historical_var) * position_value

                z_score = stats.norm.ppf(alpha)
                parametric_var = mean_return + z_score * std_return
                parametric_var_dollar = abs(parametric_var) * position_value

                cf_adjustment = (
                    z_score +
                    (z_score**2 - 1) * skewness / 6 +
                    (z_score**3 - 3 * z_score) * (kurtosis - 3) / 24 -
                    (2 * z_score**3 - 5 * z_score) * skewness**2 / 36
                )
                modified_var = mean_return + cf_adjustment * std_return
                modified_var_dollar = abs(modified_var) * position_value

                simulations = 10000
                simulated_returns = np.random.normal(mean_return, std_return, simulations)
                monte_carlo_var = np.percentile(simulated_returns, alpha * 100)
                monte_carlo_var_dollar = abs(monte_carlo_var) * position_value

                time_scaling = np.sqrt(time_horizon_days)

                var_results[conf_str] = {
                    "historical": {
                        "var_pct": round(abs(historical_var) * 100 * time_scaling, 2),
                        "var_dollar": round(historical_var_dollar * time_scaling, 2)
                    },
                    "parametric": {
                        "var_pct": round(abs(parametric_var) * 100 * time_scaling, 2),
                        "var_dollar": round(parametric_var_dollar * time_scaling, 2)
                    },
                    "modified": {
                        "var_pct": round(abs(modified_var) * 100 * time_scaling, 2),
                        "var_dollar": round(modified_var_dollar * time_scaling, 2)
                    },
                    "monte_carlo": {
                        "var_pct": round(abs(monte_carlo_var) * 100 * time_scaling, 2),
                        "var_dollar": round(monte_carlo_var_dollar * time_scaling, 2)
                    }
                }

            cvar_results = {}
            for conf in confidence_levels:
                conf_str = f"{int(conf * 100)}%"
                alpha = 1 - conf
                var_threshold = np.percentile(returns, alpha * 100)
                cvar = returns[returns <= var_threshold].mean()
                cvar_dollar = abs(cvar) * position_value * np.sqrt(time_horizon_days)

                cvar_results[conf_str] = {
                    "cvar_pct": round(abs(cvar) * 100 * np.sqrt(time_horizon_days), 2),
                    "cvar_dollar": round(cvar_dollar, 2)
                }

            return {
                "success": True,
                "ticker": ticker,
                "position_value": position_value,
                "time_horizon_days": time_horizon_days,
                "return_statistics": {
                    "mean_daily_return_pct": round(mean_return * 100, 4),
                    "std_daily_return_pct": round(std_return * 100, 4),
                    "skewness": round(skewness, 4),
                    "kurtosis": round(kurtosis, 4),
                    "distribution": "Fat-tailed" if kurtosis > 3 else "Normal-tailed"
                },
                "value_at_risk": var_results,
                "conditional_var": cvar_results,
                "interpretation": {
                    "95%": f"95% confident that losses will not exceed the 95% VaR in {time_horizon_days} day(s)",
                    "99%": f"99% confident that losses will not exceed the 99% VaR in {time_horizon_days} day(s)"
                }
            }

        except Exception as e:
            logger.error(f"Error computing VaR for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }

    @staticmethod
    def analyze_drawdown(ticker: str, period: str = "5y") -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty or len(df) < 50:
                return {
                    "success": False,
                    "error": "Insufficient data for drawdown analysis",
                    "ticker": ticker
                }

            close = df['Close']

            running_max = close.expanding().max()
            drawdown = (close - running_max) / running_max

            max_drawdown = drawdown.min()
            max_dd_idx = drawdown.idxmin()

            peak_idx = close[:max_dd_idx].idxmax()
            peak_value = close[peak_idx]
            trough_value = close[max_dd_idx]

            post_trough = close[max_dd_idx:]
            recovery_idx = post_trough[post_trough >= peak_value].first_valid_index()

            if recovery_idx:
                recovery_days = (recovery_idx - max_dd_idx).days
                recovered = True
            else:
                recovery_days = None
                recovered = False

            drawdown_days = (max_dd_idx - peak_idx).days

            significant_dds = []
            in_drawdown = False
            dd_start = None

            for date, dd in drawdown.items():
                if dd < -0.10 and not in_drawdown:
                    in_drawdown = True
                    dd_start = date
                elif dd >= -0.02 and in_drawdown:
                    in_drawdown = False
                    dd_end = date
                    period_dd = drawdown[dd_start:dd_end].min()
                    significant_dds.append({
                        "start": dd_start.strftime("%Y-%m-%d"),
                        "end": dd_end.strftime("%Y-%m-%d"),
                        "max_drawdown_pct": round(period_dd * 100, 2),
                        "duration_days": (dd_end - dd_start).days
                    })

            avg_drawdown = drawdown[drawdown < 0].mean()

            underwater_pct = (drawdown < 0).sum() / len(drawdown) * 100

            ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100

            return {
                "success": True,
                "ticker": ticker,
                "period": period,
                "maximum_drawdown": {
                    "value_pct": round(max_drawdown * 100, 2),
                    "peak_date": peak_idx.strftime("%Y-%m-%d"),
                    "peak_value": round(peak_value, 2),
                    "trough_date": max_dd_idx.strftime("%Y-%m-%d"),
                    "trough_value": round(trough_value, 2),
                    "drawdown_duration_days": drawdown_days,
                    "recovered": recovered,
                    "recovery_days": recovery_days
                },
                "drawdown_statistics": {
                    "average_drawdown_pct": round(avg_drawdown * 100, 2),
                    "time_underwater_pct": round(underwater_pct, 2),
                    "ulcer_index": round(ulcer_index, 2),
                    "significant_drawdowns_count": len(significant_dds)
                },
                "significant_drawdowns": significant_dds[-5:],
                "current_status": {
                    "current_drawdown_pct": round(drawdown.iloc[-1] * 100, 2),
                    "current_price": round(close.iloc[-1], 2),
                    "all_time_high": round(running_max.iloc[-1], 2),
                    "distance_from_ath_pct": round((close.iloc[-1] / running_max.iloc[-1] - 1) * 100, 2)
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing drawdown for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }

    @staticmethod
    def portfolio_risk_metrics(
        tickers: List[str],
        weights: Optional[List[float]] = None,
        period: str = "1y"
    ) -> Dict[str, Any]:
        try:
            if weights is None:
                weights = [1.0 / len(tickers)] * len(tickers)

            if len(weights) != len(tickers):
                return {
                    "success": False,
                    "error": "Number of weights must match number of tickers"
                }

            returns_data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                df = stock.history(period=period)
                if not df.empty:
                    returns_data[ticker] = df['Close'].pct_change().dropna()

            if len(returns_data) < len(tickers):
                missing = set(tickers) - set(returns_data.keys())
                return {
                    "success": False,
                    "error": f"Could not fetch data for: {missing}"
                }

            returns_df = pd.DataFrame(returns_data)

            returns_df = returns_df.dropna()

            if len(returns_df) < 50:
                return {
                    "success": False,
                    "error": "Insufficient overlapping data for portfolio analysis"
                }

            weights = np.array(weights)

            individual_metrics = {}
            for ticker in tickers:
                r = returns_df[ticker]
                individual_metrics[ticker] = {
                    "annualized_return_pct": round(r.mean() * 252 * 100, 2),
                    "annualized_volatility_pct": round(r.std() * np.sqrt(252) * 100, 2),
                    "sharpe_ratio": round((r.mean() * 252) / (r.std() * np.sqrt(252)), 2)
                }

            corr_matrix = returns_df.corr()

            portfolio_return = (returns_df * weights).sum(axis=1)

            port_mean = portfolio_return.mean() * 252
            port_std = portfolio_return.std() * np.sqrt(252)
            port_sharpe = port_mean / port_std if port_std > 0 else 0

            cov_matrix = returns_df.cov() * 252

            port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            port_volatility = np.sqrt(port_variance)

            weighted_vols = np.sum(weights * np.array([returns_df[t].std() * np.sqrt(252) for t in tickers]))
            diversification_ratio = weighted_vols / port_volatility if port_volatility > 0 else 1

            try:
                spy = yf.Ticker("SPY").history(period=period)['Close'].pct_change().dropna()
                spy = spy.reindex(portfolio_return.index).dropna()
                port_aligned = portfolio_return.reindex(spy.index).dropna()

                if len(port_aligned) > 20:
                    covariance = np.cov(port_aligned, spy)[0, 1]
                    market_variance = np.var(spy)
                    beta = covariance / market_variance if market_variance > 0 else 1
                else:
                    beta = 1.0
            except Exception:
                beta = 1.0

            cum_return = (1 + portfolio_return).cumprod()
            running_max = cum_return.expanding().max()
            drawdown = (cum_return - running_max) / running_max
            max_drawdown = drawdown.min()

            return {
                "success": True,
                "tickers": tickers,
                "weights": {t: round(w, 4) for t, w in zip(tickers, weights)},
                "individual_metrics": individual_metrics,
                "correlation_matrix": {
                    t1: {t2: round(corr_matrix.loc[t1, t2], 3) for t2 in tickers}
                    for t1 in tickers
                },
                "portfolio_metrics": {
                    "annualized_return_pct": round(port_mean * 100, 2),
                    "annualized_volatility_pct": round(port_volatility * 100, 2),
                    "sharpe_ratio": round(port_sharpe, 2),
                    "beta": round(beta, 2),
                    "max_drawdown_pct": round(max_drawdown * 100, 2),
                    "diversification_ratio": round(diversification_ratio, 2)
                },
                "risk_contribution": {
                    t: round((weights[i] * np.dot(cov_matrix.iloc[i], weights)) / port_variance * 100, 2)
                    for i, t in enumerate(tickers)
                }
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def generate_risk_limits(
        ticker: str,
        account_size: float = 100000.0,
        risk_tolerance: str = "moderate"
    ) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")

            if df.empty:
                return {
                    "success": False,
                    "error": f"No data found for {ticker}",
                    "ticker": ticker
                }

            current_price = df['Close'].iloc[-1]
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            tolerance_params = {
                "conservative": {
                    "max_position_pct": 5,
                    "max_sector_pct": 15,
                    "max_daily_loss_pct": 1,
                    "max_total_risk_pct": 10,
                    "stop_loss_atr_mult": 1.5,
                    "profit_target_atr_mult": 2.0
                },
                "moderate": {
                    "max_position_pct": 10,
                    "max_sector_pct": 25,
                    "max_daily_loss_pct": 2,
                    "max_total_risk_pct": 20,
                    "stop_loss_atr_mult": 2.0,
                    "profit_target_atr_mult": 3.0
                },
                "aggressive": {
                    "max_position_pct": 20,
                    "max_sector_pct": 40,
                    "max_daily_loss_pct": 3,
                    "max_total_risk_pct": 30,
                    "stop_loss_atr_mult": 2.5,
                    "profit_target_atr_mult": 4.0
                }
            }

            params = tolerance_params.get(risk_tolerance, tolerance_params["moderate"])

            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values

            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(true_range[-14:])

            max_position_value = account_size * (params["max_position_pct"] / 100)
            max_shares = int(max_position_value / current_price)

            stop_loss_distance = atr * params["stop_loss_atr_mult"]
            stop_loss_price = current_price - stop_loss_distance
            stop_loss_pct = (stop_loss_distance / current_price) * 100

            profit_target_distance = atr * params["profit_target_atr_mult"]
            profit_target_price = current_price + profit_target_distance
            risk_reward_ratio = profit_target_distance / stop_loss_distance

            max_daily_loss = account_size * (params["max_daily_loss_pct"] / 100)
            max_total_at_risk = account_size * (params["max_total_risk_pct"] / 100)

            target_position_vol = 0.02
            vol_adjusted_max_pct = (target_position_vol / (volatility / np.sqrt(252))) * 100
            vol_adjusted_max_pct = min(vol_adjusted_max_pct, params["max_position_pct"])

            return {
                "success": True,
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "risk_tolerance": risk_tolerance,
                "account_size": account_size,
                "position_limits": {
                    "max_position_pct": params["max_position_pct"],
                    "max_position_value": round(max_position_value, 2),
                    "max_shares": max_shares,
                    "vol_adjusted_max_pct": round(vol_adjusted_max_pct, 2),
                    "vol_adjusted_max_value": round(account_size * vol_adjusted_max_pct / 100, 2)
                },
                "stop_loss": {
                    "method": f"{params['stop_loss_atr_mult']}x ATR",
                    "atr_14": round(atr, 2),
                    "stop_distance": round(stop_loss_distance, 2),
                    "stop_price": round(stop_loss_price, 2),
                    "stop_loss_pct": round(stop_loss_pct, 2)
                },
                "profit_target": {
                    "method": f"{params['profit_target_atr_mult']}x ATR",
                    "target_distance": round(profit_target_distance, 2),
                    "target_price": round(profit_target_price, 2),
                    "target_pct": round((profit_target_distance / current_price) * 100, 2)
                },
                "risk_reward": {
                    "ratio": round(risk_reward_ratio, 2),
                    "assessment": "Favorable" if risk_reward_ratio >= 2 else "Marginal" if risk_reward_ratio >= 1.5 else "Unfavorable"
                },
                "account_limits": {
                    "max_daily_loss": round(max_daily_loss, 2),
                    "max_daily_loss_pct": params["max_daily_loss_pct"],
                    "max_total_at_risk": round(max_total_at_risk, 2),
                    "max_total_risk_pct": params["max_total_risk_pct"],
                    "max_sector_pct": params["max_sector_pct"]
                },
                "guardrails": [
                    f"Never exceed {params['max_position_pct']}% of account in a single position",
                    f"Set stop loss at ₹{round(stop_loss_price, 2)} ({round(stop_loss_pct, 2)}% below entry)",
                    f"Consider taking profits at ₹{round(profit_target_price, 2)}",
                    f"Maximum daily loss limit: ₹{round(max_daily_loss, 2)}",
                    f"If volatility increases >50%, reduce position size proportionally",
                    f"Review position if loss exceeds {round(stop_loss_pct * 1.5, 1)}%"
                ]
            }

        except Exception as e:
            logger.error(f"Error generating risk limits for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state.get("ticker", "")
        account_size = state.get("account_size", 100000.0)
        risk_tolerance = state.get("risk_tolerance", "moderate")

        if not ticker:
            return {
                **state,
                "errors": state.get("errors", []) + ["No ticker provided to Risk Agent"]
            }

        try:
            position_sizing = self.calculate_position_size(ticker, account_size)
            var_analysis = self.compute_var(ticker)
            drawdown_analysis = self.analyze_drawdown(ticker)
            risk_limits = self.generate_risk_limits(ticker, account_size, risk_tolerance)

            def _fmt(val, spec=",.2f"):
                if val is None or val == 'N/A' or isinstance(val, str):
                    return 'N/A'
                try:
                    return format(val, spec)
                except (ValueError, TypeError):
                    return str(val)

            pos_value = position_sizing.get('position_sizing', {}).get('recommended', {}).get('value', 'N/A')
            hist_var = var_analysis.get('value_at_risk', {}).get('95%', {}).get('historical', {}).get('var_dollar', 'N/A')
            mod_var = var_analysis.get('value_at_risk', {}).get('95%', {}).get('modified', {}).get('var_dollar', 'N/A')

            risk_summary = f"""
Risk Analysis for {ticker}:

POSITION SIZING (Account: ₹{account_size:,.0f}):
- Recommended Position: {position_sizing.get('position_sizing', {}).get('recommended', {}).get('shares', 'N/A')} shares
- Position Value: ₹{_fmt(pos_value)}
- % of Account: {position_sizing.get('position_sizing', {}).get('recommended', {}).get('pct_of_account', 'N/A')}%

VALUE AT RISK (95% confidence, 1-day):
- Historical VaR: ₹{_fmt(hist_var)}
- Modified VaR: ₹{_fmt(mod_var)}

DRAWDOWN ANALYSIS:
- Maximum Drawdown: {drawdown_analysis.get('maximum_drawdown', {}).get('value_pct', 'N/A')}%
- Current Drawdown: {drawdown_analysis.get('current_status', {}).get('current_drawdown_pct', 'N/A')}%
- Time Underwater: {drawdown_analysis.get('drawdown_statistics', {}).get('time_underwater_pct', 'N/A')}%

RISK LIMITS ({risk_tolerance.upper()}):
- Stop Loss: ₹{risk_limits.get('stop_loss', {}).get('stop_price', 'N/A')} ({risk_limits.get('stop_loss', {}).get('stop_loss_pct', 'N/A')}%)
- Profit Target: ₹{risk_limits.get('profit_target', {}).get('target_price', 'N/A')}
- Risk/Reward Ratio: {risk_limits.get('risk_reward', {}).get('ratio', 'N/A')}

GUARDRAILS:
{chr(10).join(['- ' + g for g in risk_limits.get('guardrails', [])])}
"""

            messages = self._format_messages(state)

            market_data = state.get("analysis_results", {}).get("market_data", {})
            technical = state.get("analysis_results", {}).get("technical", {})

            context = ""
            if market_data:
                context += f"\nMarket Data Context: RSI={market_data.get('technical_indicators', {}).get('indicators', {}).get('momentum', {}).get('RSI', 'N/A')}"
            if technical:
                context += f"\nTechnical Context: Regime={technical.get('regime_analysis', {}).get('regime', 'N/A')}"

            messages.append(HumanMessage(content=f"""
Based on the following risk analysis, provide comprehensive risk management recommendations:

{risk_summary}
{context}

Please provide:
1. Overall risk assessment (Low/Medium/High) with justification
2. Specific position sizing recommendation
3. Stop loss and profit target strategy
4. Key risk factors to monitor
5. Portfolio-level considerations
6. Final trading/investing recommendation with risk-adjusted view
"""))

            analysis_results = state.get("analysis_results", {})
            analysis_results["risk"] = {
                "position_sizing": position_sizing,
                "var_analysis": var_analysis,
                "drawdown_analysis": drawdown_analysis,
                "risk_limits": risk_limits,
            }

            llm_analysis = ""
            try:
                llm_analysis = await self._invoke_llm(messages)
                analysis_results["risk"]["llm_analysis"] = llm_analysis
            except Exception as llm_err:
                logger.error(f"Risk Agent LLM error: {llm_err}")
                analysis_results["risk"]["llm_analysis"] = ""
                analysis_results["risk"]["llm_error"] = str(llm_err)
                state = {
                    **state,
                    "errors": state.get("errors", [])
                    + [f"Risk Agent LLM error: {str(llm_err)}"],
                }

            out_messages = state.get("messages", [])
            if llm_analysis:
                out_messages = out_messages + [
                    {"role": "assistant", "agent": self.name, "content": llm_analysis}
                ]

            return {**state, "analysis_results": analysis_results, "messages": out_messages}

        except Exception as e:
            logger.error(f"Risk Agent execution error: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Risk Agent error: {str(e)}"]
            }
