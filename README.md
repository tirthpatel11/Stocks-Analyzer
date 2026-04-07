# Multi-Agent Stock Analysis System

A sophisticated multi-agent system for comprehensive stock analysis built with FastAPI, LangChain, LangGraph, and Grok API.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-purple.svg)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              SUPERVISOR AGENT                        │   │
│  │         (Orchestration & Synthesis)                  │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│          ┌───────────────┼───────────────┐                 │
│          │               │               │                 │
│          ▼               ▼               ▼                 │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐    │
│  │ MARKET DATA   │ │  TECHNICAL    │ │    RISK       │    │
│  │    AGENT      │ │    AGENT      │ │   AGENT       │    │
│  ├───────────────┤ ├───────────────┤ ├───────────────┤    │
│  │• Price Data   │ │• Regime       │ │• Position     │    │
│  │• Indicators   │ │• Patterns     │ │  Sizing       │    │
│  │• Fundamentals │ │• S/R Levels   │ │• VaR          │    │
│  │• Returns      │ │• Trend        │ │• Drawdown     │    │
│  └───────────────┘ └───────────────┘ └───────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   LANGGRAPH                          │   │
│  │              (Workflow Orchestration)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              GROK API (LLM)                          │   │
│  │         (Analysis & Recommendations)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Agents

### 1. Market Data & Feature Engineering Agent
- **Purpose**: Fetch and process market data
- **Capabilities**:
  - Real-time and historical price data (via Yahoo Finance)
  - Technical indicator computation (RSI, MACD, Bollinger Bands, etc.)
  - Fundamental data retrieval
  - Return and performance metrics calculation

### 2. Technical & Regime Agent
- **Purpose**: Chart analysis and market structure
- **Capabilities**:
  - Market regime classification (Trending, Ranging, Volatile, Breakout)
  - Chart pattern detection (Double Top/Bottom, Triangles, Channels)
  - Support and resistance level identification
  - Trend structure analysis (Higher Highs/Lows)

### 3. Risk & Portfolio Construction Agent
- **Purpose**: Risk management guardrails
- **Capabilities**:
  - Position sizing (Fixed Risk, Volatility-adjusted, Kelly Criterion)
  - Value at Risk (VaR) calculation (Historical, Parametric, Monte Carlo)
  - Drawdown analysis
  - Portfolio-level risk metrics
  - Risk limits and guardrails generation

### 4. Supervisor Agent
- **Purpose**: Orchestrate and synthesize
- **Capabilities**:
  - Coordinates workflow between agents
  - Synthesizes insights into comprehensive analysis
  - Generates final actionable recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Grok API Key from [x.ai](https://x.ai)

### Installation

1. **Clone and navigate to the project**:
```bash
cd "Multi Agent System"
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
# Copy the example config
cp config.example.env .env

# Edit .env with your API key
# GROK_API_KEY=your_grok_api_key_here
```

5. **Run the server** (default **port 8001** so it does not clash with other apps that often use 8000):
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

6. **Access the API**:
- API Documentation: http://localhost:8001/docs
- Alternative Docs: http://localhost:8001/redoc

## 📡 API Endpoints

### Full Analysis
```http
POST /analyze
```
Run comprehensive multi-agent analysis.

**Request Body**:
```json
{
  "ticker": "AAPL",
  "task": "Should I buy this stock?",
  "account_size": 50000,
  "risk_tolerance": "moderate"
}
```

### Quick Analysis (Single Agent)

| Endpoint | Agent | Description |
|----------|-------|-------------|
| `POST /quick/market-data` | Market Data | Price, indicators, fundamentals |
| `POST /quick/technical` | Technical | Regime, patterns, S/R levels |
| `POST /quick/risk` | Risk | Position sizing, VaR, drawdowns |

### Portfolio Analysis
```http
POST /portfolio/risk
```
Analyze portfolio-level risk metrics.

**Request Body**:
```json
{
  "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN"],
  "weights": [0.25, 0.25, 0.25, 0.25]
}
```

### Utility Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /indicators/{ticker}` | Technical indicators |
| `GET /regime/{ticker}` | Market regime classification |
| `GET /support-resistance/{ticker}` | S/R levels |
| `GET /patterns/{ticker}` | Chart patterns |
| `GET /var/{ticker}` | Value at Risk |
| `GET /drawdown/{ticker}` | Drawdown analysis |

## 📊 Example Usage

### Python Client
```python
import requests

# Full analysis
response = requests.post(
    "http://localhost:8001/analyze",
    json={
        "ticker": "NVDA",
        "task": "Analyze risk/reward for a swing trade",
        "account_size": 25000,
        "risk_tolerance": "moderate"
    }
)
result = response.json()
print(result["final_recommendation"])
```

### cURL
```bash
# Quick technical analysis
curl -X POST "http://localhost:8001/quick/technical" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA"}'

# Get support/resistance levels
curl "http://localhost:8001/support-resistance/AAPL"

# Portfolio risk analysis
curl -X POST "http://localhost:8001/portfolio/risk" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "GOOGL", "MSFT"], "weights": [0.4, 0.3, 0.3]}'
```

## 🔧 Configuration

Environment variables (in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | **Groq** API key (use with `GROQ_*`; takes priority if set) |
| `GROQ_API_BASE` | `https://api.groq.com/openai/v1` | Groq OpenAI-compatible base URL |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `GROK_API_KEY` | - | **xAI Grok** API key (if you use Grok, not Groq) |
| `GROK_API_BASE` | `https://api.x.ai/v1` | xAI API base URL |
| `GROK_MODEL` | `grok-3-mini` | xAI model id. If the name looks like Groq (e.g. `llama-…`), the app uses **Groq’s** base URL with this key. |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |

## 📁 Project Structure

```
Multi Agent System/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── main.py             # FastAPI application
│   ├── models.py           # Pydantic models
│   └── agents/
│       ├── __init__.py
│       ├── base.py         # Base agent class
│       ├── market_data_agent.py
│       ├── technical_agent.py
│       ├── risk_agent.py
│       └── supervisor.py   # LangGraph workflow
├── requirements.txt
├── config.example.env
└── README.md
```

## 🔬 Technical Details

### LangGraph Workflow
The system uses LangGraph for orchestrating the multi-agent workflow:

```
START → Market Data Agent → Technical Agent → Risk Agent → Supervisor → END
```

Each agent:
1. Performs its specialized analysis
2. Updates the shared state
3. Passes results to the next agent

The Supervisor synthesizes all insights into a final recommendation.

### Risk Metrics Calculated

- **Value at Risk (VaR)**: Historical, Parametric, Modified (Cornish-Fisher), Monte Carlo
- **Position Sizing**: Fixed Risk, Volatility-adjusted, Kelly Criterion
- **Performance Ratios**: Sharpe, Sortino, Calmar
- **Drawdown Metrics**: Max Drawdown, Time Underwater, Ulcer Index
- **Portfolio Metrics**: Correlation, Beta, Diversification Ratio

### Technical Indicators

- Moving Averages (SMA, EMA)
- MACD & Signal Line
- RSI (Relative Strength Index)
- Bollinger Bands
- ATR (Average True Range)
- ADX (Average Directional Index)
- Stochastic Oscillator
- On-Balance Volume

## ⚠️ Disclaimer

This system is for educational and informational purposes only. It should not be considered financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

