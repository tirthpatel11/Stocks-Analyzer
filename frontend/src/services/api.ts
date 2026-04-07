import axios from 'axios';
import type { AnalysisRequest, AnalysisResponse, QuickTechnicalResponse } from '../types';
import { getAxiosBaseURL, apiPath } from '../config/apiBase';

const api = axios.create({
  baseURL: getAxiosBaseURL(),
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000,
});

export async function verifyStockAnalyzerBackend(): Promise<void> {
  let data: { agents?: unknown };
  try {
    const response = await axios.get(apiPath('/health'), {
      baseURL: getAxiosBaseURL(),
      timeout: 8000,
    });
    data = response.data;
  } catch (e) {
    if (axios.isAxiosError(e) && !e.response) {
      throw new Error(
        'Cannot reach the Stock Analyzer API. Start it with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001'
      );
    }
    if (axios.isAxiosError(e) && e.response?.status === 404) {
      throw new Error(
        'Stock Analyzer API not found on this URL. Start it with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001'
      );
    }
    throw e;
  }
  const agents = data?.agents;
  if (!Array.isArray(agents) || !agents.includes('MarketDataAgent')) {
    throw new Error(
      'Another app is responding on this API address (not the Stock Analyzer). This project uses port 8001 by default. Start the analyzer with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001'
    );
  }
}

const encTicker = (symbol: string) => encodeURIComponent(symbol);

export const stockApi = {
  analyze: async (request: AnalysisRequest): Promise<AnalysisResponse> => {
    const response = await api.post<AnalysisResponse>(apiPath('/analyze'), request);
    return response.data;
  },

  quickMarketData: async (ticker: string) => {
    const response = await api.post(apiPath('/quick/market-data'), { ticker });
    return response.data;
  },

  quickTechnical: async (ticker: string): Promise<QuickTechnicalResponse> => {
    const response = await api.post<QuickTechnicalResponse>(apiPath('/quick/technical'), { ticker });
    return response.data;
  },

  quickRisk: async (ticker: string, accountSize: number = 100000) => {
    const response = await api.post(apiPath('/quick/risk'), {
      ticker,
      account_size: accountSize,
      risk_per_trade_pct: 2.0,
    });
    return response.data;
  },

  getIndicators: async (ticker: string) => {
    const response = await api.get(apiPath(`/indicators/${encTicker(ticker)}`));
    return response.data;
  },

  getRegime: async (ticker: string) => {
    const response = await api.get(apiPath(`/regime/${encTicker(ticker)}`));
    return response.data;
  },

  getSupportResistance: async (ticker: string) => {
    const response = await api.get(apiPath(`/support-resistance/${encTicker(ticker)}`));
    return response.data;
  },

  getPatterns: async (ticker: string) => {
    const response = await api.get(apiPath(`/patterns/${encTicker(ticker)}`));
    return response.data;
  },

  getVaR: async (ticker: string, positionValue: number = 10000) => {
    const response = await api.get(apiPath(`/var/${encTicker(ticker)}`), {
      params: { position_value: positionValue },
    });
    return response.data;
  },

  getDrawdown: async (ticker: string) => {
    const response = await api.get(apiPath(`/drawdown/${encTicker(ticker)}`));
    return response.data;
  },

  getChartData: async (ticker: string, period: string = '6mo', interval: string = '1d') => {
    const response = await api.get(apiPath(`/chart/${encTicker(ticker)}`), {
      params: { period, interval },
    });
    return response.data;
  },

  portfolioRisk: async (tickers: string[], weights?: number[]) => {
    const response = await api.post(apiPath('/portfolio/risk'), { tickers, weights });
    return response.data;
  },

  health: async () => {
    const response = await api.get(apiPath('/health'));
    return response.data;
  },

  screener: {
    getPresets: async () => {
      const response = await api.get(apiPath('/screener/presets'));
      return response.data;
    },

    getUniverse: async (sectors?: string) => {
      const response = await api.get(apiPath('/screener/universe'), {
        params: sectors ? { sectors } : {},
      });
      return response.data;
    },

    runScreen: async (screenType: string, sectors?: string, topN: number = 15) => {
      const response = await api.post(apiPath('/screener/run'), null, {
        params: { screen_type: screenType, sectors, top_n: topN },
      });
      return response.data;
    },

    runCustomScreen: async (filters: Record<string, any>, sectors?: string, topN: number = 15) => {
      const response = await api.post(apiPath('/screener/custom'), filters, {
        params: { sectors, top_n: topN },
      });
      return response.data;
    },

    getStockMetrics: async (ticker: string) => {
      const response = await api.get(apiPath(`/screener/stock/${encTicker(ticker)}`));
      return response.data;
    },

    compareStocks: async (tickers: string[]) => {
      const response = await api.get(apiPath('/screener/compare'), {
        params: { tickers: tickers.join(',') },
      });
      return response.data;
    },
  },
};

export default stockApi;
