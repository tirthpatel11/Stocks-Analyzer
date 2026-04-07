import { useState, useEffect } from 'react';
import axios from 'axios';
import { Header } from './components/Header';
import { SearchInput } from './components/SearchInput';
import { AnalysisResults } from './components/AnalysisResults';
import { LoadingState } from './components/LoadingState';
import { Screener } from './components/Screener';
import { Signals } from './components/Signals';
import { AlertCircle, Sparkles, BarChart3, Filter, Zap } from 'lucide-react';
import stockApi, { verifyStockAnalyzerBackend } from './services/api';
import type { AnalysisResponse } from './types';

type TabType = 'analyzer' | 'screener' | 'signals';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('analyzer');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingAgent, setLoadingAgent] = useState(0);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isLoading) {
      const interval = setInterval(() => {
        setLoadingAgent(prev => (prev < 3 ? prev + 1 : prev));
      }, 5000);
      return () => clearInterval(interval);
    } else {
      setLoadingAgent(0);
    }
  }, [isLoading]);

  const handleAnalyze = async (
    ticker: string, 
    task: string, 
    accountSize: number, 
    riskTolerance: string
  ) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setLoadingAgent(0);

    try {
      await verifyStockAnalyzerBackend();
      const data = await stockApi.analyze({
        ticker,
        task,
        account_size: accountSize,
        risk_tolerance: riskTolerance as 'conservative' | 'moderate' | 'aggressive',
      });
      
      setResult(data);
    } catch (err: unknown) {
      console.error('Analysis error:', err);
      let errorMessage = 'Failed to analyze stock. Please try again.';
      if (axios.isAxiosError(err)) {
        const d = err.response?.data;
        if (d && typeof d === 'object' && 'detail' in d) {
          const detail = (d as { detail: unknown }).detail;
          errorMessage = typeof detail === 'string' ? detail : JSON.stringify(detail);
        } else {
          errorMessage = err.message;
        }
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-sky-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-r from-sky-500/5 to-purple-500/5 rounded-full blur-3xl"></div>
      </div>

      <div className="relative">
        <Header />
        
        <div className="max-w-7xl mx-auto px-6 pt-6">
          <div className="flex gap-2 p-1 bg-slate-900/50 rounded-xl w-fit border border-slate-800/50">
            <button
              onClick={() => setActiveTab('analyzer')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === 'analyzer'
                  ? 'bg-gradient-to-r from-sky-500 to-blue-600 text-white shadow-lg shadow-sky-500/25'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <BarChart3 className="w-5 h-5" />
              <span>Stock Analyzer</span>
            </button>
            <button
              onClick={() => setActiveTab('screener')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === 'screener'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg shadow-purple-500/25'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <Filter className="w-5 h-5" />
              <span>AI Stock Finder</span>
            </button>
            <button
              onClick={() => setActiveTab('signals')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === 'signals'
                  ? 'bg-gradient-to-r from-amber-500 to-orange-600 text-white shadow-lg shadow-amber-500/25'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <Zap className="w-5 h-5" />
              <span>Trade Alerts</span>
            </button>
          </div>
        </div>
        
        <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
          {activeTab === 'analyzer' && (
            <>
              {!result && !isLoading && (
                <div className="text-center py-12">
                  <div className="inline-flex items-center gap-2 px-4 py-2 bg-sky-500/10 border border-sky-500/20 rounded-full mb-6">
                    <Sparkles className="w-4 h-4 text-sky-400" />
                    <span className="text-sm text-sky-300">Powered by Multi-Agent AI</span>
                  </div>
                  <h1 className="text-4xl sm:text-5xl font-bold text-white mb-4">
                    Intelligent Stock Analysis
                  </h1>
                  <p className="text-lg text-slate-400 max-w-2xl mx-auto">
                    Get comprehensive analysis of NSE/BSE stocks powered by 4 specialized AI agents: Market Data, 
                    Technical Analysis, Risk Management, and a Supervisor that synthesizes actionable insights.
                  </p>
                </div>
              )}

              <SearchInput onAnalyze={handleAnalyze} isLoading={isLoading} />

              {error && (
                <div className="glass-card border-rose-500/30 p-6">
                  <div className="flex items-start gap-4">
                    <div className="p-2 bg-rose-500/20 rounded-lg">
                      <AlertCircle className="w-6 h-6 text-rose-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-rose-300">Analysis Failed</h3>
                      <p className="text-slate-300 mt-1">{error}</p>
                      <button
                        onClick={() => setError(null)}
                        className="mt-3 text-sm text-sky-400 hover:text-sky-300"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {isLoading && <LoadingState currentAgent={loadingAgent} />}

              {result && !isLoading && <AnalysisResults data={result} />}

              {!result && !isLoading && (
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-8">
                  {[
                    { title: 'Market Data', desc: 'Real-time prices, indicators, fundamentals', color: 'blue' },
                    { title: 'Technical Analysis', desc: 'Patterns, regimes, support/resistance', color: 'purple' },
                    { title: 'Risk Management', desc: 'Position sizing, VaR, drawdowns', color: 'rose' },
                    { title: 'AI Synthesis', desc: 'Actionable recommendations', color: 'emerald' },
                  ].map((feature, i) => (
                    <div key={i} className="glass-card p-5">
                      <div className={`w-10 h-10 rounded-lg mb-3 flex items-center justify-center ${
                        feature.color === 'blue' ? 'bg-blue-500/20' :
                        feature.color === 'purple' ? 'bg-purple-500/20' :
                        feature.color === 'rose' ? 'bg-rose-500/20' :
                        'bg-emerald-500/20'
                      }`}>
                        <div className={`w-3 h-3 rounded-full ${
                          feature.color === 'blue' ? 'bg-blue-400' :
                          feature.color === 'purple' ? 'bg-purple-400' :
                          feature.color === 'rose' ? 'bg-rose-400' :
                          'bg-emerald-400'
                        }`}></div>
                      </div>
                      <h3 className="font-semibold text-white">{feature.title}</h3>
                      <p className="text-sm text-slate-400 mt-1">{feature.desc}</p>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {activeTab === 'screener' && <Screener />}

          {activeTab === 'signals' && <Signals />}
        </main>

        
      </div>
    </div>
  );
}

export default App;
