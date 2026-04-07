import { useState, useEffect } from 'react';
import { 
  Search, Filter, TrendingUp, TrendingDown, BarChart3, 
  Target, Shield, Zap, Gift, Star,
  ChevronDown, ChevronUp, Loader2, Sparkles
} from 'lucide-react';
import stockApi from '../services/api';

interface StockResult {
  ticker: string;
  company_name: string;
  sector: string;
  current_price: number;
  market_cap_cr: number;
  pe_ratio: number | null;
  pb_ratio: number | null;
  roe: number | null;
  profit_margin: number | null;
  revenue_growth: number | null;
  returns_1y: number;
  returns_1m: number | null;
  rsi: number | null;
  above_sma_50: boolean;
  above_sma_200: boolean;
  dividend_yield: number | null;
  debt_to_equity: number | null;
  pct_from_52w_high: number;
}

interface ScreenResult {
  success: boolean;
  screen_name: string;
  description: string;
  filters_applied: Record<string, any>;
  stocks_screened: number;
  stocks_passed: number;
  results: StockResult[];
  ai_insights: string;
}

interface PresetScreen {
  name: string;
  description: string;
  filters: Record<string, any>;
  sort_by: string;
  sort_ascending: boolean;
}

const PRESET_ICONS: Record<string, React.ElementType> = {
  value_picks: Target,
  growth_stars: TrendingUp,
  momentum_leaders: Zap,
  dividend_champions: Gift,
  quality_compounders: Star,
  oversold_opportunities: TrendingDown,
  low_volatility: Shield,
  small_cap_gems: Sparkles,
};

const PRESET_COLORS: Record<string, string> = {
  value_picks: 'from-blue-500/20 to-blue-600/10 border-blue-500/30',
  growth_stars: 'from-emerald-500/20 to-emerald-600/10 border-emerald-500/30',
  momentum_leaders: 'from-amber-500/20 to-amber-600/10 border-amber-500/30',
  dividend_champions: 'from-purple-500/20 to-purple-600/10 border-purple-500/30',
  quality_compounders: 'from-yellow-500/20 to-yellow-600/10 border-yellow-500/30',
  oversold_opportunities: 'from-rose-500/20 to-rose-600/10 border-rose-500/30',
  low_volatility: 'from-cyan-500/20 to-cyan-600/10 border-cyan-500/30',
  small_cap_gems: 'from-pink-500/20 to-pink-600/10 border-pink-500/30',
};

export function Screener() {
  const [presets, setPresets] = useState<Record<string, PresetScreen>>({});
  const [sectors, setSectors] = useState<string[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('value_picks');
  const [selectedSectors, setSelectedSectors] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ScreenResult | null>(null);
  const [showCustomFilters, setShowCustomFilters] = useState(false);
  const [showAIInsights, setShowAIInsights] = useState(true);
  const [customFilters, setCustomFilters] = useState<Record<string, any>>({});
  const [sortConfig, setSortConfig] = useState<{ key: string; ascending: boolean }>({ 
    key: 'returns_1y', 
    ascending: false 
  });

  useEffect(() => {
    const fetchPresets = async () => {
      try {
        const data = await stockApi.screener.getPresets();
        setPresets(data.presets);
        setSectors(data.sectors);
      } catch (error) {
        console.error('Failed to fetch presets:', error);
      }
    };
    fetchPresets();
  }, []);

  const runScreen = async () => {
    setIsLoading(true);
    try {
      const sectorParam = selectedSectors.length > 0 ? selectedSectors.join(',') : undefined;
      
      let data;
      if (showCustomFilters && Object.keys(customFilters).length > 0) {
        data = await stockApi.screener.runCustomScreen(customFilters, sectorParam, 20);
      } else {
        data = await stockApi.screener.runScreen(selectedPreset, sectorParam, 20);
      }
      
      setResults(data);
    } catch (error) {
      console.error('Screening failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleSector = (sector: string) => {
    setSelectedSectors(prev => 
      prev.includes(sector) 
        ? prev.filter(s => s !== sector)
        : [...prev, sector]
    );
  };

  const handleSort = (key: string) => {
    setSortConfig(prev => ({
      key,
      ascending: prev.key === key ? !prev.ascending : false
    }));
  };

  const sortedResults = results?.results ? [...results.results].sort((a, b) => {
    const aVal = a[sortConfig.key as keyof StockResult];
    const bVal = b[sortConfig.key as keyof StockResult];
    
    if (aVal === null || aVal === undefined) return 1;
    if (bVal === null || bVal === undefined) return -1;
    
    const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
    return sortConfig.ascending ? comparison : -comparison;
  }) : [];

  const updateCustomFilter = (key: string, value: any) => {
    if (value === '' || value === null || value === undefined) {
      const { [key]: _, ...rest } = customFilters;
      setCustomFilters(rest);
    } else {
      setCustomFilters(prev => ({ ...prev, [key]: value }));
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 rounded-xl bg-gradient-to-br from-sky-500/20 to-purple-500/20">
            <Filter className="w-6 h-6 text-sky-400" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">AI Stock Finder</h2>
            <p className="text-slate-400">Discover stocks using AI-powered filters</p>
          </div>
        </div>

        <div className="mb-6">
          <h3 className="text-sm font-medium text-slate-400 mb-3">Select Strategy</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {Object.entries(presets).map(([key, preset]) => {
              const Icon = PRESET_ICONS[key] || BarChart3;
              const colors = PRESET_COLORS[key] || 'from-slate-500/20 to-slate-600/10 border-slate-500/30';
              const isSelected = selectedPreset === key && !showCustomFilters;
              
              return (
                <button
                  key={key}
                  onClick={() => {
                    setSelectedPreset(key);
                    setShowCustomFilters(false);
                  }}
                  className={`p-4 rounded-xl border transition-all text-left ${
                    isSelected 
                      ? `bg-gradient-to-br ${colors} border-opacity-100` 
                      : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600/50'
                  }`}
                >
                  <Icon className={`w-5 h-5 mb-2 ${isSelected ? 'text-white' : 'text-slate-400'}`} />
                  <div className={`font-medium ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                    {preset.name}
                  </div>
                  <div className="text-xs text-slate-500 mt-1 line-clamp-2">
                    {preset.description}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <button
          onClick={() => setShowCustomFilters(!showCustomFilters)}
          className="text-sm text-sky-400 hover:text-sky-300 transition-colors flex items-center gap-1 mb-4"
        >
          {showCustomFilters ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          {showCustomFilters ? 'Hide Custom Filters' : 'Show Custom Filters'}
        </button>

        {showCustomFilters && (
          <div className="bg-slate-800/30 rounded-xl p-4 mb-6 space-y-4">
            <h4 className="font-medium text-white flex items-center gap-2">
              <Filter className="w-4 h-4 text-sky-400" />
              Custom Filters
            </h4>
            
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Max P/E Ratio</label>
                <input
                  type="number"
                  placeholder="e.g., 25"
                  className="input-field text-sm"
                  onChange={(e) => updateCustomFilter('pe_max', e.target.value ? Number(e.target.value) : null)}
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Max P/B Ratio</label>
                <input
                  type="number"
                  placeholder="e.g., 3"
                  className="input-field text-sm"
                  onChange={(e) => updateCustomFilter('pb_max', e.target.value ? Number(e.target.value) : null)}
                />
              </div>
              
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Min ROE (%)</label>
                <input
                  type="number"
                  placeholder="e.g., 15"
                  className="input-field text-sm"
                  onChange={(e) => updateCustomFilter('roe_min', e.target.value ? Number(e.target.value) : null)}
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Min Profit Margin (%)</label>
                <input
                  type="number"
                  placeholder="e.g., 10"
                  className="input-field text-sm"
                  onChange={(e) => updateCustomFilter('profit_margin_min', e.target.value ? Number(e.target.value) : null)}
                />
              </div>
              
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Min Revenue Growth (%)</label>
                <input
                  type="number"
                  placeholder="e.g., 10"
                  className="input-field text-sm"
                  onChange={(e) => updateCustomFilter('revenue_growth_min', e.target.value ? Number(e.target.value) : null)}
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Max Debt/Equity</label>
                <input
                  type="number"
                  placeholder="e.g., 100"
                  className="input-field text-sm"
                  onChange={(e) => updateCustomFilter('debt_to_equity_max', e.target.value ? Number(e.target.value) : null)}
                />
              </div>
              
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Max RSI</label>
                <input
                  type="number"
                  placeholder="e.g., 70"
                  className="input-field text-sm"
                  onChange={(e) => updateCustomFilter('rsi_max', e.target.value ? Number(e.target.value) : null)}
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Min Dividend Yield (%)</label>
                <input
                  type="number"
                  placeholder="e.g., 2"
                  className="input-field text-sm"
                  onChange={(e) => updateCustomFilter('dividend_yield_min', e.target.value ? Number(e.target.value) : null)}
                />
              </div>
            </div>

            <div className="flex flex-wrap gap-4 pt-2">
              <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  className="rounded bg-slate-700 border-slate-600"
                  onChange={(e) => updateCustomFilter('above_sma_50', e.target.checked || null)}
                />
                Above 50-day SMA
              </label>
              <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  className="rounded bg-slate-700 border-slate-600"
                  onChange={(e) => updateCustomFilter('above_sma_200', e.target.checked || null)}
                />
                Above 200-day SMA
              </label>
            </div>
          </div>
        )}

        <div className="mb-6">
          <h3 className="text-sm font-medium text-slate-400 mb-3">Filter by Sector (Optional)</h3>
          <div className="flex flex-wrap gap-2">
            {sectors.map(sector => (
              <button
                key={sector}
                onClick={() => toggleSector(sector)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  selectedSectors.includes(sector)
                    ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30'
                    : 'bg-slate-800/50 text-slate-400 border border-slate-700/50 hover:border-slate-600/50'
                }`}
              >
                {sector.replace('_', ' ').toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={runScreen}
          disabled={isLoading}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Screening Stocks...</span>
            </>
          ) : (
            <>
              <Search className="w-5 h-5" />
              <span>Find Stocks</span>
            </>
          )}
        </button>
      </div>

      {results && (
        <div className="space-y-6">
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold text-white">{results.screen_name}</h3>
                <p className="text-slate-400 text-sm">{results.description}</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-sky-400">{results.stocks_passed}</div>
                <div className="text-xs text-slate-500">of {results.stocks_screened} stocks</div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              {Object.entries(results.filters_applied).map(([key, value]) => (
                <span key={key} className="px-2 py-1 rounded-md bg-slate-800/50 text-xs text-slate-300">
                  {key.replace(/_/g, ' ')}: {String(value)}
                </span>
              ))}
            </div>
          </div>

          {results.ai_insights && (
            <div className="glass-card overflow-hidden">
              <button
                onClick={() => setShowAIInsights(!showAIInsights)}
                className="w-full p-4 flex items-center justify-between hover:bg-slate-800/30 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Sparkles className="w-5 h-5 text-purple-400" />
                  <h3 className="text-lg font-semibold text-white">AI Insights</h3>
                </div>
                {showAIInsights ? (
                  <ChevronUp className="w-5 h-5 text-slate-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-slate-400" />
                )}
              </button>
              {showAIInsights && (
                <div className="p-6 pt-0 border-t border-slate-700/50">
                  <div className="whitespace-pre-wrap text-slate-200 leading-relaxed text-sm">
                    {results.ai_insights}
                  </div>
                </div>
              )}
            </div>
          )}

          {sortedResults.length > 0 && (
            <div className="glass-card overflow-hidden">
              <div className="p-4 border-b border-slate-700/50">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-sky-400" />
                  Screened Stocks ({sortedResults.length})
                </h3>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <SortableHeader label="Stock" sortKey="ticker" sortConfig={sortConfig} onSort={handleSort} />
                      <SortableHeader label="Price" sortKey="current_price" sortConfig={sortConfig} onSort={handleSort} />
                      <SortableHeader label="Mkt Cap (Cr)" sortKey="market_cap_cr" sortConfig={sortConfig} onSort={handleSort} />
                      <SortableHeader label="P/E" sortKey="pe_ratio" sortConfig={sortConfig} onSort={handleSort} />
                      <SortableHeader label="ROE %" sortKey="roe" sortConfig={sortConfig} onSort={handleSort} />
                      <SortableHeader label="1Y Return" sortKey="returns_1y" sortConfig={sortConfig} onSort={handleSort} />
                      <SortableHeader label="RSI" sortKey="rsi" sortConfig={sortConfig} onSort={handleSort} />
                      <SortableHeader label="From 52W High" sortKey="pct_from_52w_high" sortConfig={sortConfig} onSort={handleSort} />
                    </tr>
                  </thead>
                  <tbody>
                    {sortedResults.map((stock, idx) => (
                      <tr 
                        key={stock.ticker} 
                        className={`border-b border-slate-700/30 hover:bg-slate-800/30 transition-colors ${
                          idx % 2 === 0 ? 'bg-slate-900/20' : ''
                        }`}
                      >
                        <td className="px-4 py-3">
                          <div>
                            <div className="font-mono font-medium text-white">{stock.ticker.replace('.NS', '')}</div>
                            <div className="text-xs text-slate-500 truncate max-w-[150px]">{stock.company_name}</div>
                          </div>
                        </td>
                        <td className="px-4 py-3 font-mono text-slate-200">
                          ₹{stock.current_price?.toLocaleString()}
                        </td>
                        <td className="px-4 py-3 text-slate-300">
                          {stock.market_cap_cr?.toLocaleString()}
                        </td>
                        <td className="px-4 py-3 text-slate-300">
                          {stock.pe_ratio ?? '-'}
                        </td>
                        <td className="px-4 py-3">
                          <span className={stock.roe && stock.roe >= 15 ? 'text-emerald-400' : 'text-slate-300'}>
                            {stock.roe ?? '-'}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`font-medium ${
                            stock.returns_1y >= 0 ? 'text-emerald-400' : 'text-rose-400'
                          }`}>
                            {stock.returns_1y >= 0 ? '+' : ''}{stock.returns_1y?.toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`${
                            stock.rsi && stock.rsi > 70 ? 'text-rose-400' :
                            stock.rsi && stock.rsi < 30 ? 'text-emerald-400' :
                            'text-slate-300'
                          }`}>
                            {stock.rsi?.toFixed(0) ?? '-'}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`${
                            stock.pct_from_52w_high > -5 ? 'text-emerald-400' :
                            stock.pct_from_52w_high < -20 ? 'text-rose-400' :
                            'text-amber-400'
                          }`}>
                            {stock.pct_from_52w_high?.toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SortableHeader({ 
  label, 
  sortKey, 
  sortConfig, 
  onSort 
}: { 
  label: string; 
  sortKey: string; 
  sortConfig: { key: string; ascending: boolean }; 
  onSort: (key: string) => void;
}) {
  const isActive = sortConfig.key === sortKey;
  
  return (
    <th 
      className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider cursor-pointer hover:text-slate-200 transition-colors"
      onClick={() => onSort(sortKey)}
    >
      <div className="flex items-center gap-1">
        {label}
        {isActive && (
          sortConfig.ascending 
            ? <ChevronUp className="w-3 h-3" /> 
            : <ChevronDown className="w-3 h-3" />
        )}
      </div>
    </th>
  );
}

export default Screener;

