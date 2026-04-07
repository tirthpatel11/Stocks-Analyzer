import json
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
from operator import add
import logging

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.agents.base import BaseAgent, get_grok_llm
from app.agents.market_data_agent import MarketDataAgent
from app.agents.technical_agent import TechnicalAgent
from app.agents.risk_agent import RiskAgent

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add]
    ticker: str
    task: str
    account_size: float
    risk_tolerance: str
    current_agent: str
    analysis_results: Dict[str, Any]
    errors: List[str]
    final_recommendation: str
    completed: bool


class SupervisorAgent(BaseAgent):
    
    def __init__(self):
        super().__init__(
            name="SupervisorAgent",
            description="Coordinates and synthesizes analysis from all specialized agents",
            temperature=0.3
        )
        
        self.market_data_agent = MarketDataAgent()
        self.technical_agent = TechnicalAgent()
        self.risk_agent = RiskAgent()
        
    def _setup_tools(self) -> List[Any]:
        return []
    
    def _create_system_prompt(self) -> str:
        return """You are the Supervisor Agent for a sophisticated multi-agent stock analysis system focused on the INDIAN STOCK MARKET.

IMPORTANT: Always use Indian Rupees (₹) for all currency values. Never use $ or USD.

Your team consists of:
1. **Market Data Agent**: Fetches and processes market data, computes technical indicators, 
   handles feature engineering. Provides fundamental data, price statistics, and return metrics.

2. **Technical Agent**: Analyzes chart patterns, market structure, and regime detection.
   Identifies support/resistance levels, trend analysis, and price action signals.

3. **Risk Agent**: Manages position sizing, calculates VaR, analyzes drawdowns,
   and provides portfolio construction guardrails. Ensures proper risk management.

Your responsibilities:
1. Understand the user's analysis request
2. Coordinate the workflow between agents
3. Synthesize insights from all agents into a cohesive analysis
4. Provide clear, actionable recommendations
5. Highlight key risks and opportunities

When synthesizing the final analysis:
- Combine quantitative data with qualitative insights
- Identify confluence between different analysis dimensions
- Clearly state the confidence level of recommendations
- Present a balanced view of opportunities and risks
- Provide specific, actionable next steps

Output Format:
Present your final synthesis in a structured format:
1. Executive Summary
2. Key Findings by Category
3. Risk Assessment
4. Recommendation with Confidence Level
5. Action Items and Monitoring Points"""

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state.get("ticker", "")
        analysis_results = state.get("analysis_results", {})
        
        if not ticker:
            return {
                **state,
                "errors": state.get("errors", []) + ["No ticker provided"],
                "completed": True
            }
        
        try:
            market_data = analysis_results.get("market_data", {})
            technical = analysis_results.get("technical", {})
            risk = analysis_results.get("risk", {})
            
            synthesis_data = self._prepare_synthesis_data(ticker, market_data, technical, risk)
            
            messages = [SystemMessage(content=self.system_prompt)]
            messages.append(HumanMessage(content=f"""
Please synthesize the following multi-agent analysis for {ticker} and provide a comprehensive recommendation:

{synthesis_data}

User's original request: {state.get('task', 'Comprehensive stock analysis')}
Account Size: ₹{state.get('account_size', 100000):,.0f}
Risk Tolerance: {state.get('risk_tolerance', 'moderate').upper()}

Please provide:
1. EXECUTIVE SUMMARY (2-3 sentences)
2. KEY FINDINGS:
   - Market Data Insights
   - Technical Analysis Insights  
   - Risk Assessment Insights
3. CONFLUENCE ANALYSIS (where do multiple signals agree?)
4. RISK FACTORS TO MONITOR
5. FINAL RECOMMENDATION:
   - Action: [BUY/HOLD/SELL/AVOID]
   - Confidence: [HIGH/MEDIUM/LOW]
   - Position Size Suggestion
   - Entry Strategy
   - Exit Strategy (Stop Loss & Take Profit)
6. KEY MONITORING POINTS
"""))
            
            try:
                final_analysis = await self._invoke_llm(messages)
            except Exception as llm_err:
                logger.error(f"Supervisor LLM error: {llm_err}")
                final_analysis = (
                    "AI synthesis was unavailable. Review the Market Data, Technical Analysis, "
                    f"and Risk sections for quantitative results. ({llm_err})"
                )
                state = {
                    **state,
                    "errors": state.get("errors", [])
                    + [f"Supervisor LLM error: {str(llm_err)}"],
                }

            return {
                **state,
                "final_recommendation": final_analysis,
                "current_agent": "SupervisorAgent",
                "completed": True,
                "messages": state.get("messages", [])
                + [
                    {
                        "role": "assistant",
                        "agent": self.name,
                        "content": final_analysis,
                    }
                ],
            }

        except Exception as e:
            logger.error(f"Supervisor execution error: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Supervisor error: {str(e)}"],
                "completed": True
            }
    
    @staticmethod
    def _fmt(val, spec=",.2f"):
        if val is None or val == 'N/A' or isinstance(val, str):
            return 'N/A'
        try:
            return format(val, spec)
        except (ValueError, TypeError):
            return str(val)
    
    def _prepare_synthesis_data(
        self, 
        ticker: str, 
        market_data: Dict, 
        technical: Dict, 
        risk: Dict
    ) -> str:
        
        synthesis = f"""
=== MULTI-AGENT ANALYSIS SYNTHESIS FOR {ticker} ===

📊 MARKET DATA AGENT FINDINGS:
"""
        if market_data:
            stock_data = market_data.get("stock_data", {})
            tech_indicators = market_data.get("technical_indicators", {})
            fundamentals = market_data.get("fundamental_data", {})
            returns = market_data.get("returns_data", {})
            
            avg_volume = stock_data.get('statistics', {}).get('avg_volume', 'N/A')
            avg_volume_str = f"{avg_volume:,}" if isinstance(avg_volume, (int, float)) else str(avg_volume)
            
            synthesis += f"""
Price & Volume:
- Current Price: ₹{stock_data.get('latest', {}).get('close', 'N/A')}
- Average Volume: {avg_volume_str}

Key Indicators:
- RSI: {tech_indicators.get('indicators', {}).get('momentum', {}).get('RSI', 'N/A')}
- MACD: {tech_indicators.get('indicators', {}).get('momentum', {}).get('MACD', 'N/A')}
- ADX: {tech_indicators.get('indicators', {}).get('trend', {}).get('ADX', 'N/A')}

Technical Signals:
{chr(10).join(['  • ' + s for s in tech_indicators.get('signals', ['No signals'])])}

Fundamentals:
- P/E Ratio: {fundamentals.get('valuation', {}).get('pe_ratio', 'N/A')}
- Sector: {fundamentals.get('company_info', {}).get('sector', 'N/A')}
- Analyst Target: ₹{fundamentals.get('analyst_targets', {}).get('target_mean', 'N/A')}

Returns:
- Total Return (1Y): {returns.get('returns', {}).get('total_return_pct', 'N/A')}%
- Sharpe Ratio: {returns.get('risk_metrics', {}).get('sharpe_ratio', 'N/A')}

Market Data Agent Analysis:
{market_data.get('llm_analysis', 'No analysis available')}
"""
        
        synthesis += """
📈 TECHNICAL AGENT FINDINGS:
"""
        if technical:
            regime = technical.get("regime_analysis", {})
            sr = technical.get("support_resistance", {})
            patterns = technical.get("chart_patterns", {})
            structure = technical.get("trend_structure", {})
            
            synthesis += f"""
Market Regime:
- Current Regime: {regime.get('regime', 'N/A')}
- Confidence: {regime.get('confidence', 0) * 100:.0f}%
- ADX (Trend Strength): {regime.get('metrics', {}).get('adx', 'N/A')}
- Volatility Regime: {regime.get('metrics', {}).get('volatility_regime', 'N/A')}

Support & Resistance:
- Nearest Resistance: ₹{sr.get('nearest_levels', {}).get('resistance', 'N/A')}
- Nearest Support: ₹{sr.get('nearest_levels', {}).get('support', 'N/A')}
- Position in Range: {sr.get('price_range', {}).get('position_in_range_pct', 'N/A')}%

Chart Patterns:
{chr(10).join(['  • ' + p.get('pattern', 'Unknown') + ': ' + p.get('implication', 'N/A') for p in patterns.get('chart_patterns', [{'pattern': 'None detected'}])])}

Trend Structure:
- Structure Type: {structure.get('structure', {}).get('type', 'N/A')}
- Short-term Trend: {structure.get('trend_analysis', {}).get('short_term_20d', 'N/A')}
- Medium-term Trend: {structure.get('trend_analysis', {}).get('medium_term_50d', 'N/A')}
- Long-term Trend: {structure.get('trend_analysis', {}).get('long_term_200d', 'N/A')}

Technical Agent Analysis:
{technical.get('llm_analysis', 'No analysis available')}
"""
        
        synthesis += """
⚠️ RISK AGENT FINDINGS:
"""
        if risk:
            position = risk.get("position_sizing", {})
            var = risk.get("var_analysis", {})
            drawdown = risk.get("drawdown_analysis", {})
            limits = risk.get("risk_limits", {})
            
            pos_value = position.get('position_sizing', {}).get('recommended', {}).get('value', 'N/A')
            hist_var = var.get('value_at_risk', {}).get('95%', {}).get('historical', {}).get('var_dollar', 'N/A')
            
            synthesis += f"""
Position Sizing:
- Recommended Shares: {position.get('position_sizing', {}).get('recommended', {}).get('shares', 'N/A')}
- Position Value: ₹{self._fmt(pos_value)}
- % of Account: {position.get('position_sizing', {}).get('recommended', {}).get('pct_of_account', 'N/A')}%

Value at Risk (95%, 1-day):
- Historical VaR: ₹{self._fmt(hist_var)}
- VaR %: {var.get('value_at_risk', {}).get('95%', {}).get('historical', {}).get('var_pct', 'N/A')}%

Drawdown Analysis:
- Max Historical Drawdown: {drawdown.get('maximum_drawdown', {}).get('value_pct', 'N/A')}%
- Current Drawdown: {drawdown.get('current_status', {}).get('current_drawdown_pct', 'N/A')}%
- Time Underwater: {drawdown.get('drawdown_statistics', {}).get('time_underwater_pct', 'N/A')}%

Risk Limits:
- Stop Loss: ₹{limits.get('stop_loss', {}).get('stop_price', 'N/A')} ({limits.get('stop_loss', {}).get('stop_loss_pct', 'N/A')}%)
- Profit Target: ₹{limits.get('profit_target', {}).get('target_price', 'N/A')}
- Risk/Reward: {limits.get('risk_reward', {}).get('ratio', 'N/A')}

Guardrails:
{chr(10).join(['  • ' + g for g in limits.get('guardrails', ['No guardrails defined'])])}

Risk Agent Analysis:
{risk.get('llm_analysis', 'No analysis available')}
"""
        
        return synthesis


def create_stock_analysis_graph() -> StateGraph:
    
    market_data_agent = MarketDataAgent()
    technical_agent = TechnicalAgent()
    risk_agent = RiskAgent()
    supervisor_agent = SupervisorAgent()
    
    async def market_data_node(state: AgentState) -> AgentState:
        logger.info(f"Executing Market Data Agent for {state.get('ticker')}")
        result = await market_data_agent.execute(dict(state))
        return {**state, **result, "current_agent": "MarketDataAgent"}
    
    async def technical_node(state: AgentState) -> AgentState:
        logger.info(f"Executing Technical Agent for {state.get('ticker')}")
        result = await technical_agent.execute(dict(state))
        return {**state, **result, "current_agent": "TechnicalAgent"}
    
    async def risk_node(state: AgentState) -> AgentState:
        logger.info(f"Executing Risk Agent for {state.get('ticker')}")
        result = await risk_agent.execute(dict(state))
        return {**state, **result, "current_agent": "RiskAgent"}
    
    async def supervisor_node(state: AgentState) -> AgentState:
        logger.info(f"Executing Supervisor Agent for {state.get('ticker')}")
        result = await supervisor_agent.execute(dict(state))
        return {**state, **result, "current_agent": "SupervisorAgent"}
    
    def should_continue(state: AgentState) -> Literal["continue", "end"]:
        if state.get("completed", False):
            return "end"
        if len(state.get("errors", [])) > 3:
            return "end"
        return "continue"
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("market_data", market_data_node)
    workflow.add_node("technical", technical_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("supervisor", supervisor_node)
    
    workflow.set_entry_point("market_data")
    workflow.add_edge("market_data", "technical")
    workflow.add_edge("technical", "risk")
    workflow.add_edge("risk", "supervisor")
    workflow.add_edge("supervisor", END)
    
    return workflow.compile()


async def run_analysis(
    ticker: str,
    task: str = "Comprehensive stock analysis",
    account_size: float = 100000.0,
    risk_tolerance: str = "moderate"
) -> Dict[str, Any]:
    
    initial_state: AgentState = {
        "messages": [],
        "ticker": ticker.upper(),
        "task": task,
        "account_size": account_size,
        "risk_tolerance": risk_tolerance,
        "current_agent": "",
        "analysis_results": {},
        "errors": [],
        "final_recommendation": "",
        "completed": False
    }
    
    graph = create_stock_analysis_graph()
    
    try:
        final_state = await graph.ainvoke(initial_state)
        
        return {
            "success": True,
            "ticker": ticker,
            "analysis_results": final_state.get("analysis_results", {}),
            "final_recommendation": final_state.get("final_recommendation", ""),
            "messages": final_state.get("messages", []),
            "errors": final_state.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"Analysis workflow error: {e}")
        return {
            "success": False,
            "ticker": ticker,
            "error": str(e),
            "errors": [str(e)]
        }
