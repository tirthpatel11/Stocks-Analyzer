import { useState } from 'react';
import { 
  TrendingUp, TrendingDown, Activity, Shield, Target, 
  AlertTriangle, CheckCircle, BarChart3,
  Layers, Gauge, IndianRupee, ChevronDown, ChevronUp
} from 'lucide-react';
import type { AnalysisResponse } from '../types';
import { MetricCard, MetricRow } from './MetricCard';

interface AnalysisResultsProps {
  data: AnalysisResponse;
}

export function AnalysisResults({ data }: AnalysisResultsProps) {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    recommendation: true,
    market: false,
    technical: false,
    risk: false,
  });

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const { analysis_results, final_recommendation, ticker, realtime_quote, data_source, errors } = data;
  const { market_data, technical, risk } = analysis_results;
  
  // Check if we have rate limit errors
  const hasRateLimitError = errors && errors.some(e => e.includes('rate_limit') || e.includes('Rate limit'));
  const hasErrors = errors && errors.length > 0;
  
  // Use real-time price if available, otherwise fall back to historical
  const currentPrice = realtime_quote?.last_price || market_data?.stock_data?.latest?.close;
  const priceChange = realtime_quote?.change_pct;
  const isRealtime = realtime_quote?.source === 'NSE_REALTIME';

  // Parse recommendation for action
  const getRecommendationAction = () => {
    const lower = final_recommendation.toLowerCase();
    if (lower.includes('action: buy') || lower.includes('**action**: buy')) {
      return { action: 'BUY', color: 'emerald' as const, icon: TrendingUp };
    } else if (lower.includes('action: sell') || lower.includes('**action**: sell')) {
      return { action: 'SELL', color: 'rose' as const, icon: TrendingDown };
    } else if (lower.includes('action: hold') || lower.includes('**action**: hold')) {
      return { action: 'HOLD', color: 'amber' as const, icon: Activity };
    }
    return { action: 'ANALYZE', color: 'sky' as const, icon: BarChart3 };
  };

  const recommendation = getRecommendationAction();
  const RecommendationIcon = recommendation.icon;

  const getColorClasses = (color: 'emerald' | 'rose' | 'amber' | 'sky') => {
    const colors = {
      emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', gradient: 'from-emerald-500/10' },
      rose: { bg: 'bg-rose-500/10', border: 'border-rose-500/30', text: 'text-rose-400', gradient: 'from-rose-500/10' },
      amber: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', gradient: 'from-amber-500/10' },
      sky: { bg: 'bg-sky-500/10', border: 'border-sky-500/30', text: 'text-sky-400', gradient: 'from-sky-500/10' },
    };
    return colors[color];
  };

  const colorClasses = getColorClasses(recommendation.color);

  return (
    <div className="space-y-6">
      {/* Error Banner */}
      {hasErrors && (
        <div className={`glass-card overflow-hidden border ${hasRateLimitError ? 'border-amber-500/30 bg-amber-500/5' : 'border-rose-500/30 bg-rose-500/5'}`}>
          <div className="p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${hasRateLimitError ? 'text-amber-400' : 'text-rose-400'}`} />
              <div className="flex-1">
                <h3 className={`font-semibold ${hasRateLimitError ? 'text-amber-400' : 'text-rose-400'}`}>
                  {hasRateLimitError ? 'API Rate Limit Reached' : 'Analysis Errors'}
                </h3>
                <p className="text-sm text-slate-400 mt-1">
                  {hasRateLimitError 
                    ? 'Your Groq API has hit its daily token limit. The analysis below shows available data without AI insights. Rate limits reset daily.'
                    : 'Some agents encountered errors during analysis. Partial results are shown below.'}
                </p>
                <details className="mt-2">
                  <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-400">
                    Show technical details ({errors?.length} error{errors?.length !== 1 ? 's' : ''})
                  </summary>
                  <ul className="mt-2 text-xs text-slate-500 space-y-1 max-h-32 overflow-y-auto">
                    {errors?.map((err, i) => (
                      <li key={i} className="bg-slate-800/50 p-2 rounded text-slate-400 break-all">
                        {err.split(':').slice(0, 2).join(':')}...
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Hero Card - Recommendation */}
      <div className={`glass-card overflow-hidden ${colorClasses.border}`}>
        {/* Header */}
        <div className={`p-6 border-b border-slate-700/50 bg-gradient-to-r ${colorClasses.gradient} to-transparent`}>
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-xl ${colorClasses.bg}`}>
                <RecommendationIcon className={`w-8 h-8 ${colorClasses.text}`} />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white font-mono">{ticker}</h2>
                <p className="text-slate-400">
                  {market_data?.fundamental_data?.company_info?.name || 'Stock Analysis'}
                </p>
              </div>
            </div>
            
            <div className="text-right">
              <div className={`text-4xl font-bold ${colorClasses.text}`}>
                {recommendation.action}
              </div>
              <p className="text-sm text-slate-400">AI Recommendation</p>
            </div>
          </div>
        </div>

        {/* Data Source Badge */}
        {data_source && (
          <div className="px-6 pt-4 -mb-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/50 text-xs text-slate-400">
              <span className={`w-2 h-2 rounded-full ${isRealtime ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
              {isRealtime ? 'Live Data' : 'Delayed Data'} • {data_source}
            </div>
          </div>
        )}

        {/* Quick Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-6">
          <MetricCard
            label={isRealtime ? "Live Price" : "Current Price"}
            value={`₹${currentPrice?.toFixed(2) || 'N/A'}`}
            trend={priceChange !== undefined ? (priceChange >= 0 ? 'up' : 'down') : undefined}
            icon={IndianRupee}
          />
          <MetricCard
            label="RSI"
            value={market_data?.technical_indicators?.indicators?.momentum?.RSI?.toFixed(1) || 'N/A'}
            trend={
              (market_data?.technical_indicators?.indicators?.momentum?.RSI || 50) > 70 ? 'down' :
              (market_data?.technical_indicators?.indicators?.momentum?.RSI || 50) < 30 ? 'up' : 'neutral'
            }
            icon={Gauge}
          />
          <MetricCard
            label="Market Regime"
            value={technical?.regime_analysis?.regime || 'N/A'}
            icon={Layers}
          />
          <MetricCard
            label="Risk/Reward"
            value={risk?.risk_limits?.risk_reward?.ratio?.toFixed(2) || 'N/A'}
            trend={
              (risk?.risk_limits?.risk_reward?.ratio || 0) >= 2 ? 'up' :
              (risk?.risk_limits?.risk_reward?.ratio || 0) >= 1.5 ? 'neutral' : 'down'
            }
            icon={Target}
          />
        </div>
      </div>

      {/* Recommendation Section */}
      <CollapsibleSection
        title="AI Recommendation"
        icon={CheckCircle}
        isExpanded={expandedSections.recommendation}
        onToggle={() => toggleSection('recommendation')}
        iconColor="text-sky-400"
      >
        <div className="whitespace-pre-wrap text-slate-200 leading-relaxed">
          {formatMarkdown(final_recommendation)}
        </div>
      </CollapsibleSection>

      {/* Market Data Section */}
      {market_data && (
        <CollapsibleSection
          title="Market Data Analysis"
          icon={BarChart3}
          isExpanded={expandedSections.market}
          onToggle={() => toggleSection('market')}
          iconColor="text-blue-400"
        >
          <div className="grid md:grid-cols-2 gap-6">
            {/* Price Info */}
            <div className="space-y-3">
              <h4 className="font-semibold text-white flex items-center gap-2">
                <IndianRupee className="w-4 h-4 text-sky-400" />
                Price Data
              </h4>
              <div className="bg-slate-800/30 rounded-xl p-4">
                <MetricRow label="Current Price" value={`₹${market_data.stock_data?.latest?.close?.toFixed(2)}`} />
                <MetricRow label="Day High" value={`₹${market_data.stock_data?.latest?.high?.toFixed(2)}`} />
                <MetricRow label="Day Low" value={`₹${market_data.stock_data?.latest?.low?.toFixed(2)}`} />
                <MetricRow label="52W High" value={`₹${market_data.stock_data?.statistics?.max_close?.toFixed(2)}`} />
                <MetricRow label="52W Low" value={`₹${market_data.stock_data?.statistics?.min_close?.toFixed(2)}`} />
                <MetricRow label="Avg Volume" value={market_data.stock_data?.statistics?.avg_volume?.toLocaleString()} />
              </div>
            </div>

            {/* Technical Indicators */}
            <div className="space-y-3">
              <h4 className="font-semibold text-white flex items-center gap-2">
                <Activity className="w-4 h-4 text-sky-400" />
                Technical Indicators
              </h4>
              <div className="bg-slate-800/30 rounded-xl p-4">
                <MetricRow 
                  label="RSI (14)" 
                  value={market_data.technical_indicators?.indicators?.momentum?.RSI?.toFixed(2)}
                  trend={
                    (market_data.technical_indicators?.indicators?.momentum?.RSI || 50) > 70 ? 'down' :
                    (market_data.technical_indicators?.indicators?.momentum?.RSI || 50) < 30 ? 'up' : 'neutral'
                  }
                />
                <MetricRow label="MACD" value={market_data.technical_indicators?.indicators?.momentum?.MACD?.toFixed(4)} />
                <MetricRow label="ADX" value={market_data.technical_indicators?.indicators?.trend?.ADX?.toFixed(2)} />
                <MetricRow label="ATR" value={`₹${market_data.technical_indicators?.indicators?.volatility?.ATR?.toFixed(2)}`} />
                <MetricRow label="SMA 50" value={`₹${market_data.technical_indicators?.indicators?.moving_averages?.SMA_50?.toFixed(2)}`} />
                <MetricRow label="SMA 200" value={`₹${market_data.technical_indicators?.indicators?.moving_averages?.SMA_200?.toFixed(2)}`} />
              </div>
            </div>

            {/* Signals */}
            {market_data.technical_indicators?.signals && (
              <div className="md:col-span-2 space-y-3">
                <h4 className="font-semibold text-white">Trade Alerts</h4>
                <div className="flex flex-wrap gap-2">
                  {market_data.technical_indicators.signals.map((signal, i) => (
                    <SignalBadge key={i} signal={signal} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </CollapsibleSection>
      )}

      {/* Technical Analysis Section */}
      {technical && (
        <CollapsibleSection
          title="Technical Analysis"
          icon={TrendingUp}
          isExpanded={expandedSections.technical}
          onToggle={() => toggleSection('technical')}
          iconColor="text-purple-400"
        >
          <div className="grid md:grid-cols-2 gap-6">
            {/* Regime */}
            <div className="space-y-3">
              <h4 className="font-semibold text-white">Market Regime</h4>
              <div className="bg-slate-800/30 rounded-xl p-4">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-2xl font-bold text-white">
                    {technical.regime_analysis?.regime}
                  </span>
                  <span className="text-sm text-slate-400">
                    {((technical.regime_analysis?.confidence || 0) * 100).toFixed(0)}% confidence
                  </span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div 
                    className="bg-sky-500 h-2 rounded-full transition-all"
                    style={{ width: `${(technical.regime_analysis?.confidence || 0) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Support/Resistance */}
            <div className="space-y-3">
              <h4 className="font-semibold text-white">Support & Resistance</h4>
              <div className="bg-slate-800/30 rounded-xl p-4">
                <MetricRow 
                  label="Nearest Resistance" 
                  value={`₹${technical.support_resistance?.nearest_levels?.resistance?.toFixed(2)}`}
                  trend="down"
                />
                <MetricRow 
                  label="Nearest Support" 
                  value={`₹${technical.support_resistance?.nearest_levels?.support?.toFixed(2)}`}
                  trend="up"
                />
                <MetricRow 
                  label="Position in Range" 
                  value={`${technical.support_resistance?.price_range?.position_in_range_pct?.toFixed(1)}%`}
                />
              </div>
            </div>

            {/* Trend Structure */}
            <div className="space-y-3">
              <h4 className="font-semibold text-white">Trend Analysis</h4>
              <div className="bg-slate-800/30 rounded-xl p-4">
                <MetricRow label="Short-term (20d)" value={technical.trend_structure?.trend_analysis?.short_term_20d} />
                <MetricRow label="Medium-term (50d)" value={technical.trend_structure?.trend_analysis?.medium_term_50d} />
                <MetricRow label="Long-term (200d)" value={technical.trend_structure?.trend_analysis?.long_term_200d} />
                <MetricRow label="Structure" value={technical.trend_structure?.structure?.type} />
              </div>
            </div>

            {/* Chart Patterns */}
            <div className="space-y-3">
              <h4 className="font-semibold text-white">Chart Patterns</h4>
              <div className="bg-slate-800/30 rounded-xl p-4 space-y-2">
                {technical.chart_patterns?.chart_patterns?.map((pattern, i) => (
                  <div key={i} className="flex justify-between items-center py-2 border-b border-slate-700/30 last:border-0">
                    <span className="font-medium text-white">{pattern.pattern}</span>
                    <span className={`text-sm px-2 py-0.5 rounded ${
                      pattern.implication?.includes('BULLISH') ? 'bg-emerald-500/20 text-emerald-400' :
                      pattern.implication?.includes('BEARISH') ? 'bg-rose-500/20 text-rose-400' :
                      'bg-amber-500/20 text-amber-400'
                    }`}>
                      {pattern.implication}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CollapsibleSection>
      )}

      {/* Risk Analysis Section */}
      {risk && (
        <CollapsibleSection
          title="Risk Analysis"
          icon={Shield}
          isExpanded={expandedSections.risk}
          onToggle={() => toggleSection('risk')}
          iconColor="text-rose-400"
        >
          <div className="grid md:grid-cols-2 gap-6">
            {/* Position Sizing */}
            <div className="space-y-3">
              <h4 className="font-semibold text-white">Position Sizing</h4>
              <div className="bg-slate-800/30 rounded-xl p-4">
                <MetricRow label="Recommended Shares" value={risk.position_sizing?.position_sizing?.recommended?.shares} />
                <MetricRow label="Position Value" value={`₹${risk.position_sizing?.position_sizing?.recommended?.value?.toFixed(2)}`} />
                <MetricRow label="% of Account" value={`${risk.position_sizing?.position_sizing?.recommended?.pct_of_account?.toFixed(2)}%`} />
                <MetricRow label="Volatility" value={`${risk.position_sizing?.volatility_metrics?.annualized_volatility_pct?.toFixed(2)}%`} />
              </div>
            </div>

            {/* Risk Limits */}
            <div className="space-y-3">
              <h4 className="font-semibold text-white">Trade Setup</h4>
              <div className="bg-slate-800/30 rounded-xl p-4">
                <MetricRow label="Stop Loss" value={`₹${risk.risk_limits?.stop_loss?.stop_price?.toFixed(2)}`} trend="down" />
                <MetricRow label="Stop Loss %" value={`${risk.risk_limits?.stop_loss?.stop_loss_pct?.toFixed(2)}%`} />
                <MetricRow label="Take Profit" value={`₹${risk.risk_limits?.profit_target?.target_price?.toFixed(2)}`} trend="up" />
                <MetricRow label="Risk/Reward" value={risk.risk_limits?.risk_reward?.ratio?.toFixed(2)} />
              </div>
            </div>

            {/* Guardrails */}
            {risk.risk_limits?.guardrails && (
              <div className="md:col-span-2 space-y-3">
                <h4 className="font-semibold text-white flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  Risk Guardrails
                </h4>
                <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4">
                  <ul className="space-y-2">
                    {risk.risk_limits.guardrails.map((guardrail, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-amber-200">
                        <Shield className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                        {guardrail}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </CollapsibleSection>
      )}
    </div>
  );
}

// Collapsible Section Component
interface CollapsibleSectionProps {
  title: string;
  icon: React.ElementType;
  isExpanded: boolean;
  onToggle: () => void;
  iconColor: string;
  children: React.ReactNode;
}

function CollapsibleSection({ title, icon: Icon, isExpanded, onToggle, iconColor, children }: CollapsibleSectionProps) {
  return (
    <div className="glass-card overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full p-4 flex items-center justify-between hover:bg-slate-800/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Icon className={`w-5 h-5 ${iconColor}`} />
          <h3 className="text-lg font-semibold text-white">{title}</h3>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-slate-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-slate-400" />
        )}
      </button>
      {isExpanded && (
        <div className="p-6 pt-0 border-t border-slate-700/50">
          {children}
        </div>
      )}
    </div>
  );
}

// Signal Badge Component
function SignalBadge({ signal }: { signal: string }) {
  const isBullish = signal.toLowerCase().includes('bullish') || signal.toLowerCase().includes('above');
  const isBearish = signal.toLowerCase().includes('bearish') || signal.toLowerCase().includes('below');
  
  return (
    <span className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
      isBullish ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
      isBearish ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' :
      'bg-slate-700/50 text-slate-300 border border-slate-600/30'
    }`}>
      {signal}
    </span>
  );
}

// Simple markdown formatter
function formatMarkdown(text: string): React.ReactNode {
  return text.split('\n').map((line, i) => {
    if (line.startsWith('### ')) {
      return <h3 key={i} className="text-lg font-bold text-white mt-4 mb-2">{line.replace('### ', '')}</h3>;
    }
    if (line.startsWith('## ')) {
      return <h2 key={i} className="text-xl font-bold text-white mt-4 mb-2">{line.replace('## ', '')}</h2>;
    }
    if (line.startsWith('# ')) {
      return <h1 key={i} className="text-2xl font-bold text-white mt-4 mb-2">{line.replace('# ', '')}</h1>;
    }
    if (line.startsWith('- ')) {
      return <li key={i} className="ml-4 text-slate-200">{formatInline(line.replace('- ', ''))}</li>;
    }
    if (!line.trim()) {
      return <br key={i} />;
    }
    return <p key={i} className="text-slate-200 mb-2">{formatInline(line)}</p>;
  });
}

function formatInline(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} className="text-white font-semibold">{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}
