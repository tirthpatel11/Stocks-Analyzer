import { useState } from 'react';
import { Search, Loader2, IndianRupee, Shield } from 'lucide-react';

interface SearchInputProps {
  onAnalyze: (ticker: string, task: string, accountSize: number, riskTolerance: string) => void;
  isLoading: boolean;
}

export function SearchInput({ onAnalyze, isLoading }: SearchInputProps) {
  const [ticker, setTicker] = useState('');
  const [task, setTask] = useState('Comprehensive analysis - should I invest?');
  const [accountSize, setAccountSize] = useState(50000);
  const [riskTolerance, setRiskTolerance] = useState<'conservative' | 'moderate' | 'aggressive'>('moderate');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const normalizeTicker = (t: string): string => {
    const upper = t.trim().toUpperCase();
    if (!upper) return upper;
    if (!upper.endsWith('.NS') && !upper.endsWith('.BO')) {
      return `${upper}.NS`;
    }
    return upper;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (ticker.trim()) {
      const normalized = normalizeTicker(ticker);
      setTicker(normalized);
      onAnalyze(normalized, task, accountSize, riskTolerance);
    }
  };

  const runQuickAnalyze = (symbol: string) => {
    if (isLoading) return;
    const normalized = normalizeTicker(symbol);
    setTicker(normalized);
    onAnalyze(normalized, task, accountSize, riskTolerance);
  };

  const popularTickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS'];

  return (
    <div className="glass-card p-6 space-y-6">
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
            className="btn-primary flex items-center justify-center gap-2 min-w-[160px]"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Analyzing<span className="loading-dots"></span></span>
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                <span>Analyze</span>
              </>
            )}
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          <span className="text-sm text-slate-400">Popular:</span>
          {popularTickers.map((sym) => (
            <button
              key={sym}
              type="button"
              disabled={isLoading}
              onClick={() => runQuickAnalyze(sym)}
              className="px-3 py-1 text-sm font-mono bg-slate-800/50 hover:bg-slate-700/50 
                         border border-slate-600/50 rounded-lg transition-colors text-slate-300
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {sym}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-sky-400 hover:text-sky-300 transition-colors"
        >
          {showAdvanced ? '− Hide' : '+ Show'} Advanced Options
        </button>

        {showAdvanced && (
          <div className="grid sm:grid-cols-3 gap-4 pt-2">
            <div>
              <label className="block text-sm text-slate-400 mb-2">
                <IndianRupee className="inline w-4 h-4 mr-1" />
                Account Size
              </label>
              <input
                type="number"
                value={accountSize}
                onChange={(e) => setAccountSize(Number(e.target.value))}
                className="input-field"
                min={1000}
              />
            </div>
            
            <div>
              <label className="block text-sm text-slate-400 mb-2">
                <Shield className="inline w-4 h-4 mr-1" />
                Risk Tolerance
              </label>
              <select
                value={riskTolerance}
                onChange={(e) => setRiskTolerance(e.target.value as typeof riskTolerance)}
                className="input-field"
              >
                <option value="conservative">Conservative</option>
                <option value="moderate">Moderate</option>
                <option value="aggressive">Aggressive</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm text-slate-400 mb-2">Analysis Task</label>
              <input
                type="text"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                className="input-field"
                placeholder="What would you like to know?"
              />
            </div>
          </div>
        )}
      </form>
    </div>
  );
}
