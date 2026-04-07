import { useState } from 'react';
import { 
  TrendingUp, TrendingDown, Minus, Search, Loader2, 
  Target, Shield, AlertTriangle, CheckCircle, XCircle,
  ChevronDown, ChevronUp, Zap, Star, Activity,
  ArrowUpRight, BarChart3
} from 'lucide-react';
import { buildApiUrl } from '../config/apiBase';

interface SignalData {
  success: boolean;
  ticker: string;
  current_price: number;
  signal: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  signal_strength: number;
  signal_description: string;
  entry: {
    zone_low: number | null;
    zone_high: number | null;
    ideal_entry: number | null;
  };
  stop_loss: {
    price: number;
    percentage: number;
  };
  targets: {
    target_1: { price: number; percentage: number; risk_reward: string };
    target_2: { price: number; percentage: number; risk_reward: string };
    target_3: { price: number; percentage: number; risk_reward: string };
  };
  risk_reward_ratio: number;
  scores: {
    bullish: number;
    bearish: number;
    net: number;
  };
  reasons: {
    bullish: string[];
    bearish: string[];
  };
  indicators: {
    rsi: number;
    rsi_signal: string;
    macd: {
      bullish: boolean;
      bearish: boolean;
      crossover: boolean;
      crossunder: boolean;
    };
    moving_averages: {
      above_sma_50: boolean;
      above_sma_200: boolean | null;
    };
    bollinger: {
      position: number;
      squeeze: boolean;
    };
    atr: number;
    atr_pct: number;
  };
  trend: {
    short_term: string;
    medium_term: string;
    long_term: string;
    price_change_5d: number;
    price_change_20d: number;
  };
  volume: {
    ratio: number;
    trend: string;
  };
  levels: {
    support: number;
    resistance: number;
    high_52w: number;
    low_52w: number;
    pct_from_high: number;
  };
  ai_analysis?: string;
}

const SIGNAL_CONFIG = {
  STRONG_BUY: {
    color: 'emerald',
    icon: TrendingUp,
    bgGradient: 'from-emerald-500/20 to-emerald-600/10',
    borderColor: 'border-emerald-500/50',
    textColor: 'text-emerald-400',
    label: 'STRONG BUY'
  },
  BUY: {
    color: 'green',
    icon: TrendingUp,
    bgGradient: 'from-green-500/20 to-green-600/10',
    borderColor: 'border-green-500/50',
    textColor: 'text-green-400',
    label: 'BUY'
  },
  HOLD: {
    color: 'amber',
    icon: Minus,
    bgGradient: 'from-amber-500/20 to-amber-600/10',
    borderColor: 'border-amber-500/50',
    textColor: 'text-amber-400',
    label: 'HOLD'
  },
  SELL: {
    color: 'orange',
    icon: TrendingDown,
    bgGradient: 'from-orange-500/20 to-orange-600/10',
    borderColor: 'border-orange-500/50',
    textColor: 'text-orange-400',
    label: 'SELL'
  },
  STRONG_SELL: {
    color: 'rose',
    icon: TrendingDown,
    bgGradient: 'from-rose-500/20 to-rose-600/10',
    borderColor: 'border-rose-500/50',
    textColor: 'text-rose-400',
    label: 'STRONG SELL'
  }
};

export function Signals() {
  const [ticker, setTicker] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [signal, setSignal] = useState<SignalData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showIndicators, setShowIndicators] = useState(false);
  const [showAIAnalysis, setShowAIAnalysis] = useState(true);

  const normalizeTicker = (t: string): string => {
    const upper = t.trim().toUpperCase();
    if (!upper) return upper;
    if (!upper.endsWith('.NS') && !upper.endsWith('.BO')) {
      return `${upper}.NS`;
    }
    return upper;
  };

  const fetchSignal = async (withAI: boolean = true, tickerOverride?: string) => {
    const raw = (tickerOverride ?? ticker).trim();
    if (!raw) return;

    const normalizedTicker = normalizeTicker(raw);
    setTicker(normalizedTicker);

    setIsLoading(true);
    setError(null);

    try {
      const enc = encodeURIComponent(normalizedTicker);
      const path = withAI ? `/signals/${enc}/analyze` : `/signals/${enc}`;
      const response = await fetch(buildApiUrl(path), {
        method: withAI ? 'POST' : 'GET',
      });

      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
          const body = await response.json();
          if (body && typeof body.detail === 'string') {
            detail = body.detail;
          }
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }

      const data = await response.json();
      setSignal(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch signal');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchSignal(true);
  };

  const popularTickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'SBIN.NS'];

  const config = signal ? SIGNAL_CONFIG[signal.signal] : null;
  const SignalIcon = config?.icon || Activity;

  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 rounded-xl bg-gradient-to-br from-amber-500/20 to-orange-500/20">
            <Zap className="w-6 h-6 text-amber-400" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">Trade Alerts</h2>
            <p className="text-slate-400">Get AI-powered buy/sell signals with entry & exit levels</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="Enter stock ticker (e.g., RELIANCE.NS)"
                className="input-field pl-12 text-lg font-mono uppercase"
                disabled={isLoading}
              />
            </div>
            
            <button
              type="submit"
              disabled={isLoading || !ticker.trim()}
              className="btn-primary flex items-center justify-center gap-2 min-w-[180px]"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>Get Signal</span>
                </>
              )}
            </button>
          </div>

          <div className="flex flex-wrap gap-2">
            <span className="text-sm text-slate-400">Popular:</span>
            {popularTickers.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => fetchSignal(true, t)}
                className="px-3 py-1 text-sm font-mono bg-slate-800/50 hover:bg-slate-700/50 
                           border border-slate-600/50 rounded-lg transition-colors text-slate-300"
              >
                {t.replace('.NS', '')}
              </button>
            ))}
          </div>
        </form>

        {error && (
          <div className="mt-4 p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl flex items-center gap-3">
            <XCircle className="w-5 h-5 text-rose-400" />
            <span className="text-rose-300">{error}</span>
          </div>
        )}
      </div>

      {signal && config && (
        <div className="space-y-6">
          <div className={`glass-card overflow-hidden ${config.borderColor} border-2`}>
            <div className={`p-6 bg-gradient-to-r ${config.bgGradient} to-transparent`}>
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-4">
                  <div className={`p-4 rounded-2xl bg-${config.color}-500/20`}>
                    <SignalIcon className={`w-10 h-10 ${config.textColor}`} />
                  </div>
                  <div>
                    <div className="text-sm text-slate-400 mb-1">Signal for</div>
                    <h2 className="text-3xl font-bold text-white font-mono">{signal.ticker}</h2>
                    <div className="text-slate-300">₹{signal.current_price.toLocaleString()}</div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className={`text-4xl font-bold ${config.textColor}`}>
                    {config.label}
                  </div>
                  <div className="flex items-center justify-end gap-1 mt-2">
                    {[...Array(5)].map((_, i) => (
                      <Star 
                        key={i} 
                        className={`w-5 h-5 ${i < signal.signal_strength ? config.textColor : 'text-slate-700'}`}
                        fill={i < signal.signal_strength ? 'currentColor' : 'none'}
                      />
                    ))}
                  </div>
                  <div className="text-sm text-slate-400 mt-1">{signal.signal_description}</div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-6 border-t border-slate-700/50">
              {signal.entry.zone_low && (
                <div className="bg-slate-800/30 rounded-xl p-4">
                  <div className="flex items-center gap-2 text-sm text-slate-400 mb-2">
                    <Target className="w-4 h-4 text-sky-400" />
                    Entry Zone
                  </div>
                  <div className="text-xl font-bold text-white">
                    ₹{signal.entry.zone_low?.toLocaleString()}
                  </div>
                  <div className="text-sm text-slate-400">
                    to ₹{signal.entry.zone_high?.toLocaleString()}
                  </div>
                </div>
              )}

              <div className="bg-slate-800/30 rounded-xl p-4">
                <div className="flex items-center gap-2 text-sm text-slate-400 mb-2">
                  <Shield className="w-4 h-4 text-rose-400" />
                  Stop Loss
                </div>
                <div className="text-xl font-bold text-rose-400">
                  ₹{signal.stop_loss.price.toLocaleString()}
                </div>
                <div className="text-sm text-rose-400/70">
                  -{signal.stop_loss.percentage}% risk
                </div>
              </div>

              <div className="bg-slate-800/30 rounded-xl p-4">
                <div className="flex items-center gap-2 text-sm text-slate-400 mb-2">
                  <ArrowUpRight className="w-4 h-4 text-emerald-400" />
                  Target 1
                </div>
                <div className="text-xl font-bold text-emerald-400">
                  ₹{signal.targets.target_1.price.toLocaleString()}
                </div>
                <div className="text-sm text-emerald-400/70">
                  +{signal.targets.target_1.percentage}%
                </div>
              </div>

              <div className="bg-slate-800/30 rounded-xl p-4">
                <div className="flex items-center gap-2 text-sm text-slate-400 mb-2">
                  <ArrowUpRight className="w-4 h-4 text-emerald-400" />
                  Target 2
                </div>
                <div className="text-xl font-bold text-emerald-400">
                  ₹{signal.targets.target_2.price.toLocaleString()}
                </div>
                <div className="text-sm text-emerald-400/70">
                  +{signal.targets.target_2.percentage}%
                </div>
              </div>
            </div>

            {signal.risk_reward_ratio > 0 && (
              <div className="px-6 pb-6">
                <div className="bg-gradient-to-r from-emerald-500/10 to-sky-500/10 border border-emerald-500/20 rounded-xl p-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <BarChart3 className="w-6 h-6 text-emerald-400" />
                    <div>
                      <div className="text-sm text-slate-400">Risk/Reward Ratio</div>
                      <div className="text-2xl font-bold text-white">{signal.risk_reward_ratio}:1</div>
                    </div>
                  </div>
                  <div className={`px-4 py-2 rounded-lg ${
                    signal.risk_reward_ratio >= 2 ? 'bg-emerald-500/20 text-emerald-400' :
                    signal.risk_reward_ratio >= 1.5 ? 'bg-amber-500/20 text-amber-400' :
                    'bg-rose-500/20 text-rose-400'
                  }`}>
                    {signal.risk_reward_ratio >= 2 ? 'Favorable' :
                     signal.risk_reward_ratio >= 1.5 ? 'Acceptable' : 'Risky'}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
                <CheckCircle className="w-5 h-5 text-emerald-400" />
                Bullish Factors ({signal.scores.bullish})
              </h3>
              <ul className="space-y-2">
                {signal.reasons.bullish.map((reason, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <TrendingUp className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <span className="text-slate-300">{reason}</span>
                  </li>
                ))}
                {signal.reasons.bullish.length === 0 && (
                  <li className="text-slate-500 text-sm">No bullish signals detected</li>
                )}
              </ul>
            </div>

            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
                <AlertTriangle className="w-5 h-5 text-rose-400" />
                Bearish Factors ({signal.scores.bearish})
              </h3>
              <ul className="space-y-2">
                {signal.reasons.bearish.map((reason, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <TrendingDown className="w-4 h-4 text-rose-400 mt-0.5 flex-shrink-0" />
                    <span className="text-slate-300">{reason}</span>
                  </li>
                ))}
                {signal.reasons.bearish.length === 0 && (
                  <li className="text-slate-500 text-sm">No bearish signals detected</li>
                )}
              </ul>
            </div>
          </div>

          {signal.ai_analysis && (
            <div className="glass-card overflow-hidden">
              <button
                onClick={() => setShowAIAnalysis(!showAIAnalysis)}
                className="w-full p-4 flex items-center justify-between hover:bg-slate-800/30 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Zap className="w-5 h-5 text-amber-400" />
                  <h3 className="text-lg font-semibold text-white">AI Analysis & Action Plan</h3>
                </div>
                {showAIAnalysis ? (
                  <ChevronUp className="w-5 h-5 text-slate-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-slate-400" />
                )}
              </button>
              {showAIAnalysis && (
                <div className="p-6 pt-0 border-t border-slate-700/50">
                  <div className="whitespace-pre-wrap text-slate-200 leading-relaxed text-sm">
                    {signal.ai_analysis}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="glass-card overflow-hidden">
            <button
              onClick={() => setShowIndicators(!showIndicators)}
              className="w-full p-4 flex items-center justify-between hover:bg-slate-800/30 transition-colors"
            >
              <div className="flex items-center gap-3">
                <Activity className="w-5 h-5 text-sky-400" />
                <h3 className="text-lg font-semibold text-white">Technical Indicators</h3>
              </div>
              {showIndicators ? (
                <ChevronUp className="w-5 h-5 text-slate-400" />
              ) : (
                <ChevronDown className="w-5 h-5 text-slate-400" />
              )}
            </button>
            {showIndicators && (
              <div className="p-6 pt-0 border-t border-slate-700/50">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">RSI (14)</div>
                    <div className={`text-2xl font-bold ${
                      signal.indicators.rsi < 30 ? 'text-emerald-400' :
                      signal.indicators.rsi > 70 ? 'text-rose-400' :
                      'text-white'
                    }`}>
                      {signal.indicators.rsi}
                    </div>
                    <div className={`text-xs mt-1 ${
                      signal.indicators.rsi_signal === 'oversold' ? 'text-emerald-400' :
                      signal.indicators.rsi_signal === 'overbought' ? 'text-rose-400' :
                      'text-slate-500'
                    }`}>
                      {signal.indicators.rsi_signal.toUpperCase()}
                    </div>
                  </div>

                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">MACD</div>
                    <div className={`text-2xl font-bold ${
                      signal.indicators.macd.bullish ? 'text-emerald-400' :
                      signal.indicators.macd.bearish ? 'text-rose-400' :
                      'text-white'
                    }`}>
                      {signal.indicators.macd.bullish ? '↑' : signal.indicators.macd.bearish ? '↓' : '—'}
                    </div>
                    <div className={`text-xs mt-1 ${
                      signal.indicators.macd.crossover ? 'text-emerald-400' :
                      signal.indicators.macd.crossunder ? 'text-rose-400' :
                      'text-slate-500'
                    }`}>
                      {signal.indicators.macd.crossover ? 'BULLISH CROSS' :
                       signal.indicators.macd.crossunder ? 'BEARISH CROSS' :
                       signal.indicators.macd.bullish ? 'BULLISH' :
                       signal.indicators.macd.bearish ? 'BEARISH' : 'NEUTRAL'}
                    </div>
                  </div>

                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Trend</div>
                    <div className={`text-lg font-bold ${
                      signal.trend.short_term === 'bullish' ? 'text-emerald-400' : 'text-rose-400'
                    }`}>
                      {signal.trend.short_term.toUpperCase()}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      5D: {signal.trend.price_change_5d > 0 ? '+' : ''}{signal.trend.price_change_5d}%
                    </div>
                  </div>

                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Volume</div>
                    <div className={`text-2xl font-bold ${
                      signal.volume.ratio > 1.5 ? 'text-amber-400' : 'text-white'
                    }`}>
                      {signal.volume.ratio}x
                    </div>
                    <div className={`text-xs mt-1 ${
                      signal.volume.trend === 'increasing' ? 'text-emerald-400' :
                      signal.volume.trend === 'decreasing' ? 'text-rose-400' :
                      'text-slate-500'
                    }`}>
                      {signal.volume.trend.toUpperCase()}
                    </div>
                  </div>

                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Support</div>
                    <div className="text-xl font-bold text-emerald-400">
                      ₹{signal.levels.support.toLocaleString()}
                    </div>
                  </div>

                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Resistance</div>
                    <div className="text-xl font-bold text-rose-400">
                      ₹{signal.levels.resistance.toLocaleString()}
                    </div>
                  </div>

                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">52W High</div>
                    <div className="text-xl font-bold text-white">
                      ₹{signal.levels.high_52w.toLocaleString()}
                    </div>
                    <div className={`text-xs mt-1 ${
                      signal.levels.pct_from_high > -10 ? 'text-emerald-400' : 'text-slate-500'
                    }`}>
                      {signal.levels.pct_from_high}%
                    </div>
                  </div>

                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">ATR (Volatility)</div>
                    <div className="text-xl font-bold text-white">
                      ₹{signal.indicators.atr}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      {signal.indicators.atr_pct}% daily
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {!signal && !isLoading && (
        <div className="text-center py-16">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-slate-800/50 mb-6">
            <Zap className="w-10 h-10 text-amber-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Get Trade Alerts</h3>
          <p className="text-slate-400 max-w-md mx-auto">
            Enter a stock ticker above to get AI-powered buy/sell signals with precise entry, 
            stop loss, and target prices.
          </p>
        </div>
      )}
    </div>
  );
}

export default Signals;

