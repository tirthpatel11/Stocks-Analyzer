export interface AnalysisRequest {
  ticker: string;
  task: string;
  account_size: number;
  risk_tolerance: 'conservative' | 'moderate' | 'aggressive';
}

export interface RealtimeQuote {
  symbol: string;
  name?: string;
  last_price: number;
  change?: number;
  change_pct?: number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
  '52w_high'?: number;
  '52w_low'?: number;
  timestamp: string;
  source: string;
}

export interface AnalysisResponse {
  success: boolean;
  ticker: string;
  final_recommendation: string;
  analysis_results: {
    market_data?: MarketDataResult;
    technical?: TechnicalResult;
    risk?: RiskResult;
  };
  realtime_quote?: RealtimeQuote;
  data_source?: string;
  messages: AgentMessage[];
  errors: string[];
}

export interface AgentMessage {
  role: string;
  agent: string;
  content: string;
}

export interface MarketDataResult {
  stock_data: StockData;
  technical_indicators: TechnicalIndicators;
  fundamental_data: FundamentalData;
  returns_data: ReturnsData;
  llm_analysis?: string;
}

export interface StockData {
  success: boolean;
  ticker: string;
  latest: {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  };
  statistics: {
    avg_close: number;
    std_close: number;
    min_close: number;
    max_close: number;
    avg_volume: number;
  };
}

export interface TechnicalIndicators {
  success: boolean;
  ticker: string;
  date: string;
  indicators: {
    moving_averages: {
      SMA_20: number;
      SMA_50: number;
      SMA_200: number | null;
      EMA_12: number;
      EMA_26: number;
    };
    momentum: {
      RSI: number;
      MACD: number;
      MACD_Signal: number;
      MACD_Histogram: number;
      Stoch_K: number;
      Stoch_D: number;
    };
    volatility: {
      BB_Upper: number;
      BB_Middle: number;
      BB_Lower: number;
      BB_Width: number;
      ATR: number;
    };
    trend: {
      ADX: number;
      DI_Plus: number;
      DI_Minus: number;
    };
    volume: {
      OBV: number;
      Volume: number;
      Avg_Volume_20: number;
    };
  };
  signals: string[];
  current_price: number;
}

export interface FundamentalData {
  success: boolean;
  ticker: string;
  company_info: {
    name: string;
    sector: string;
    industry: string;
  };
  valuation: {
    market_cap: number;
    pe_ratio: number | string;
    forward_pe: number | string;
    price_to_book: number | string;
  };
  analyst_targets: {
    target_high: number;
    target_low: number;
    target_mean: number;
    recommendation: string;
  };
}

export interface ReturnsData {
  success: boolean;
  ticker: string;
  returns: {
    total_return_pct: number;
    annualized_return_pct: number;
    week_return_pct: number | null;
    month_return_pct: number | null;
  };
  risk_metrics: {
    annualized_volatility_pct: number;
    max_drawdown_pct: number;
    sharpe_ratio: number;
    sortino_ratio: number;
  };
}

export interface TechnicalResult {
  regime_analysis: RegimeAnalysis;
  support_resistance: SupportResistance;
  chart_patterns: ChartPatterns;
  trend_structure: TrendStructure;
  llm_analysis?: string;
}

export interface RegimeAnalysis {
  success: boolean;
  ticker: string;
  regime: string;
  confidence: number;
  metrics: {
    trend_strength_r2: number;
    annualized_volatility_pct: number;
    momentum_20d_pct: number;
  };
  current_price: number;
}

export interface SupportResistance {
  success: boolean;
  ticker: string;
  current_price: number;
  key_levels: {
    resistance: number[];
    support: number[];
  };
  nearest_levels: {
    resistance: number | null;
    support: number | null;
    distance_to_resistance_pct: number | null;
    distance_to_support_pct: number | null;
  };
  price_range: {
    '52w_high': number;
    '52w_low': number;
    position_in_range_pct: number;
  };
}

export interface ChartPatterns {
  success: boolean;
  ticker: string;
  chart_patterns: {
    pattern: string;
    confidence: number;
    implication: string;
    description?: string;
  }[];
}

export interface TrendStructure {
  success: boolean;
  ticker: string;
  trend_analysis: {
    short_term_20d: string;
    medium_term_50d: string;
    long_term_200d: string;
  };
  ma_alignment: {
    bullish_aligned: boolean;
    bearish_aligned: boolean;
  };
  structure: {
    type: string;
    description: string;
  };
}

export interface RiskResult {
  position_sizing: PositionSizing;
  var_analysis: VaRAnalysis;
  drawdown_analysis: DrawdownAnalysis;
  risk_limits: RiskLimits;
  llm_analysis?: string;
}

export interface PositionSizing {
  success: boolean;
  ticker: string;
  current_price: number;
  position_sizing: {
    recommended: {
      shares: number;
      value: number;
      pct_of_account: number;
    };
  };
  volatility_metrics: {
    atr_14: number;
    annualized_volatility_pct: number;
  };
}

export interface VaRAnalysis {
  success: boolean;
  ticker: string;
  value_at_risk: {
    '95%': {
      historical: { var_pct: number; var_dollar: number };
      modified: { var_pct: number; var_dollar: number };
    };
    '99%': {
      historical: { var_pct: number; var_dollar: number };
      modified: { var_pct: number; var_dollar: number };
    };
  };
  return_statistics: {
    skewness: number;
    kurtosis: number;
    distribution: string;
  };
}

export interface DrawdownAnalysis {
  success: boolean;
  ticker: string;
  maximum_drawdown: {
    value_pct: number;
    peak_date: string;
    trough_date: string;
    recovered: boolean;
  };
  current_status: {
    current_drawdown_pct: number;
    distance_from_ath_pct: number;
  };
}

export interface RiskLimits {
  success: boolean;
  ticker: string;
  stop_loss: {
    stop_price: number;
    stop_loss_pct: number;
  };
  profit_target: {
    target_price: number;
    target_pct: number;
  };
  risk_reward: {
    ratio: number;
    assessment: string;
  };
  guardrails: string[];
}

export interface QuickTechnicalResponse {
  success: boolean;
  ticker: string;
  regime_analysis: RegimeAnalysis;
  support_resistance: SupportResistance;
  chart_patterns: ChartPatterns;
  trend_structure: TrendStructure;
}

