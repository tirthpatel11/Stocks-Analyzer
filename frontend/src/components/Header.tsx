import { TrendingUp, Bot } from 'lucide-react';

export function Header() {
  return (
    <header className="sticky top-0 z-50 glass-card border-b border-slate-700/50 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-sky-500 blur-lg opacity-50"></div>
              <div className="relative p-2 bg-gradient-to-br from-sky-500 to-sky-600 rounded-xl">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
                Indian StockAI
              </h1>
              <p className="text-xs text-slate-400">NSE/BSE Analysis • Multi-Agent AI</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-xs font-medium text-emerald-400">Live</span>
            </div>
            
            <div className="hidden sm:flex items-center gap-2 text-slate-400 text-sm">
              <Bot className="w-4 h-4" />
              <span>4 Agents Active</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
