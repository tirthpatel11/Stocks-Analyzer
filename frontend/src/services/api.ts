import axios from 'axios';
import type { AnalysisRequest, AnalysisResponse, QuickTechnicalResponse } from '../types';

const API_BASE_URL = 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const stockApi = {
  // Full multi-agent analysis
  analyze: async (request: AnalysisRequest): Promise<AnalysisResponse> => {
    const response = await api.post<AnalysisResponse>('/analyze', request);
    return response.data;
  },

  // Quick analysis endpoints
  quickMarketData: async (ticker: string) => {
    const response = await api.post('/quick/market-data', { ticker });
    return response.data;
  },

  quickTechnical: async (ticker: string): Promise<QuickTechnicalResponse> => {
    const response = await api.post<QuickTechnicalResponse>('/quick/technical', { ticker });
    return response.data;
  },

  quickRisk: async (ticker: string, accountSize: number = 100000) => {
    const response = await api.post('/quick/risk', { 
      ticker, 
      account_size: accountSize,
      risk_per_trade_pct: 2.0 
    });
    return response.data;
  },

  // Utility endpoints
  getIndicators: async (ticker: string) => {
    const response = await api.get(`/indicators/${ticker}`);
    return response.data;
  },

  getRegime: async (ticker: string) => {
    const response = await api.get(`/regime/${ticker}`);
    return response.data;
  },

  getSupportResistance: async (ticker: string) => {
    const response = await api.get(`/support-resistance/${ticker}`);
    return response.data;
  },

  getPatterns: async (ticker: string) => {
    const response = await api.get(`/patterns/${ticker}`);
    return response.data;
  },

  getVaR: async (ticker: string, positionValue: number = 10000) => {
    const response = await api.get(`/var/${ticker}`, {
      params: { position_value: positionValue }
    });
    return response.data;
  },

  getDrawdown: async (ticker: string) => {
    const response = await api.get(`/drawdown/${ticker}`);
    return response.data;
  },

  getChartData: async (ticker: string, period: string = '6mo', interval: string = '1d') => {
    const response = await api.get(`/chart/${ticker}`, {
      params: { period, interval }
    });
    return response.data;
  },

  // Portfolio
  portfolioRisk: async (tickers: string[], weights?: number[]) => {
    const response = await api.post('/portfolio/risk', { tickers, weights });
    return response.data;
  },

  // Health check
  health: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Screener endpoints
  screener: {
    getPresets: async () => {
      const response = await api.get('/screener/presets');
      return response.data;
    },

    getUniverse: async (sectors?: string) => {
      const response = await api.get('/screener/universe', {
        params: sectors ? { sectors } : {}
      });
      return response.data;
    },

    runScreen: async (screenType: string, sectors?: string, topN: number = 15) => {
      const response = await api.post('/screener/run', null, {
        params: { screen_type: screenType, sectors, top_n: topN }
      });
      return response.data;
    },

    runCustomScreen: async (filters: Record<string, any>, sectors?: string, topN: number = 15) => {
      const response = await api.post('/screener/custom', filters, {
        params: { sectors, top_n: topN }
      });
      return response.data;
    },

    getStockMetrics: async (ticker: string) => {
      const response = await api.get(`/screener/stock/${ticker}`);
      return response.data;
    },

    compareStocks: async (tickers: string[]) => {
      const response = await api.get('/screener/compare', {
        params: { tickers: tickers.join(',') }
      });
      return response.data;
    }
  }
};

export default stockApi;

