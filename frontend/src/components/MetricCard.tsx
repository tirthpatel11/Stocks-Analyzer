import type { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export function MetricCard({ label, value, subValue, icon: Icon, trend, className = '' }: MetricCardProps) {
  const getTrendColor = () => {
    if (trend === 'up') return 'text-emerald-400';
    if (trend === 'down') return 'text-rose-400';
    if (trend === 'neutral') return 'text-amber-400';
    return 'text-white';
  };

  return (
    <div className={`metric-card ${className}`}>
      <div className="flex items-start justify-between mb-2">
        <span className="text-sm text-slate-400">{label}</span>
        {Icon && <Icon className="w-4 h-4 text-slate-500" />}
      </div>
      <div className="flex items-end gap-2">
        <span className={`text-2xl font-bold ${getTrendColor()}`}>
          {typeof value === 'number' ? value.toLocaleString() : value}
        </span>
        {subValue && (
          <span className="text-sm text-slate-400 mb-1">{subValue}</span>
        )}
      </div>
    </div>
  );
}

interface MetricRowProps {
  label: string;
  value: string | number | null | undefined;
  trend?: 'up' | 'down' | 'neutral';
}

export function MetricRow({ label, value, trend }: MetricRowProps) {
  if (value === null || value === undefined || value === 'N/A') {
    return null;
  }

  const getTrendColor = () => {
    if (trend === 'up') return 'text-emerald-400';
    if (trend === 'down') return 'text-rose-400';
    if (trend === 'neutral') return 'text-amber-400';
    return 'text-white';
  };
  
  return (
    <div className="flex justify-between items-center py-2 border-b border-slate-700/30 last:border-0">
      <span className="text-sm text-slate-400">{label}</span>
      <span className={`font-mono font-medium ${getTrendColor()}`}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </span>
    </div>
  );
}
