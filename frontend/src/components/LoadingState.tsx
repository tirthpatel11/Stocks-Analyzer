import { Bot, Database, TrendingUp, Shield, Brain } from 'lucide-react';

const agents = [
  { name: 'Market Data Agent', icon: Database, description: 'Fetching market data & indicators...' },
  { name: 'Technical Agent', icon: TrendingUp, description: 'Analyzing charts & patterns...' },
  { name: 'Risk Agent', icon: Shield, description: 'Calculating risk metrics...' },
  { name: 'Supervisor Agent', icon: Brain, description: 'Synthesizing recommendations...' },
];

export function LoadingState({ currentAgent = 0 }: { currentAgent?: number }) {
  return (
    <div className="glass-card p-8">
      <div className="text-center mb-8">
        <div className="relative inline-block">
          <div className="absolute inset-0 bg-sky-500 blur-2xl opacity-30 animate-pulse"></div>
          <Bot className="relative w-16 h-16 text-sky-400 mx-auto animate-bounce" />
        </div>
        <h3 className="text-xl font-bold text-white mt-4">AI Agents Working</h3>
        <p className="text-slate-400 mt-2">Analyzing your stock with multi-agent intelligence...</p>
      </div>

      <div className="space-y-4 max-w-md mx-auto">
        {agents.map((agent, index) => {
          const Icon = agent.icon;
          const isActive = index === currentAgent;
          const isComplete = index < currentAgent;
          
          return (
            <div
              key={agent.name}
              className={`flex items-center gap-4 p-4 rounded-xl transition-all duration-500 ${
                isActive ? 'bg-sky-500/10 border border-sky-500/30' :
                isComplete ? 'bg-emerald-500/10 border border-emerald-500/30' :
                'bg-slate-800/30 border border-slate-700/30'
              }`}
            >
              <div className={`p-2 rounded-lg ${
                isActive ? 'bg-sky-500/20' :
                isComplete ? 'bg-emerald-500/20' :
                'bg-slate-700/50'
              }`}>
                <Icon className={`w-5 h-5 ${
                  isActive ? 'text-sky-400 animate-pulse' :
                  isComplete ? 'text-emerald-400' :
                  'text-slate-500'
                }`} />
              </div>
              
              <div className="flex-1">
                <p className={`font-medium ${
                  isActive ? 'text-sky-300' :
                  isComplete ? 'text-emerald-300' :
                  'text-slate-400'
                }`}>
                  {agent.name}
                </p>
                {isActive && (
                  <p className="text-sm text-slate-400 mt-0.5">{agent.description}</p>
                )}
              </div>
              
              {isComplete && (
                <span className="text-emerald-400 text-sm font-medium">✓ Done</span>
              )}
              {isActive && (
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-sky-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-2 h-2 bg-sky-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-2 h-2 bg-sky-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <p className="text-center text-slate-500 text-sm mt-8">
        This usually takes 15-30 seconds
      </p>
    </div>
  );
}
