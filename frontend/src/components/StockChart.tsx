import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CandlestickSeries, HistogramSeries, LineSeries } from 'lightweight-charts';
import { TrendingUp, TrendingDown, BarChart3, RefreshCw } from 'lucide-react';
import stockApi from '../services/api';

interface StockChartProps {
  ticker: string;
}

export function StockChart({ ticker }: StockChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const candleSeriesRef = useRef<ReturnType<typeof CandlestickSeries.prototype.create> | null>(null);
  const volumeSeriesRef = useRef<ReturnType<typeof HistogramSeries.prototype.create> | null>(null);
  const sma20SeriesRef = useRef<ReturnType<typeof LineSeries.prototype.create> | null>(null);
  const sma50SeriesRef = useRef<ReturnType<typeof LineSeries.prototype.create> | null>(null);
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [period, setPeriod] = useState('6mo');
  const [showSMA20, setShowSMA20] = useState(true);
  const [showSMA50, setShowSMA50] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [lastPrice, setLastPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number | null>(null);

  const periods = [
    { label: '1M', value: '1mo' },
    { label: '3M', value: '3mo' },
    { label: '6M', value: '6mo' },
    { label: '1Y', value: '1y' },
    { label: '2Y', value: '2y' },
  ];

  const fetchChartData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await stockApi.getChartData(ticker, period);
      
      if (data.success && chartRef.current) {
        // Update candle series
        if (candleSeriesRef.current) {
          candleSeriesRef.current.setData(data.candles);
        }
        
        // Update volume series
        if (volumeSeriesRef.current) {
          volumeSeriesRef.current.setData(data.volumes);
        }
        
        // Update SMA lines
        if (sma20SeriesRef.current) {
          sma20SeriesRef.current.setData(data.sma20);
        }
        if (sma50SeriesRef.current) {
          sma50SeriesRef.current.setData(data.sma50);
        }
        
        // Calculate price change
        if (data.candles.length > 1) {
          const latest = data.candles[data.candles.length - 1];
          const previous = data.candles[data.candles.length - 2];
          const change = ((latest.close - previous.close) / previous.close) * 100;
          setLastPrice(latest.close);
          setPriceChange(change);
        }
        
        // Fit content
        chartRef.current.timeScale().fitContent();
      }
    } catch (err) {
      console.error('Chart data error:', err);
      setError('Failed to load chart data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!chartContainerRef.current) return;
    
    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: '#6366f1',
          style: 2,
        },
        horzLine: {
          width: 1,
          color: '#6366f1',
          style: 2,
        },
      },
      rightPriceScale: {
        borderColor: 'rgba(42, 46, 57, 0.8)',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      timeScale: {
        borderColor: 'rgba(42, 46, 57, 0.8)',
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: {
        vertTouchDrag: false,
      },
    });
    
    chartRef.current = chart;
    
    // Add candlestick series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    });
    candleSeriesRef.current = candleSeries;
    
    // Add volume series
    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: '#6366f1',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });
    volumeSeriesRef.current = volumeSeries;
    
    // Add SMA 20 line
    const sma20Series = chart.addSeries(LineSeries, {
      color: '#f59e0b',
      lineWidth: 1,
      title: 'SMA 20',
    });
    sma20SeriesRef.current = sma20Series;
    
    // Add SMA 50 line
    const sma50Series = chart.addSeries(LineSeries, {
      color: '#8b5cf6',
      lineWidth: 1,
      title: 'SMA 50',
    });
    sma50SeriesRef.current = sma50Series;
    
    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: 400,
        });
      }
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    fetchChartData();
  }, [ticker, period]);

  useEffect(() => {
    if (sma20SeriesRef.current) {
      sma20SeriesRef.current.applyOptions({ visible: showSMA20 });
    }
  }, [showSMA20]);

  useEffect(() => {
    if (sma50SeriesRef.current) {
      sma50SeriesRef.current.applyOptions({ visible: showSMA50 });
    }
  }, [showVolume]);

  useEffect(() => {
    if (volumeSeriesRef.current) {
      volumeSeriesRef.current.applyOptions({ visible: showVolume });
    }
  }, [showVolume]);

  return (
    <div className="glass-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-5 h-5 text-sky-400" />
          <h3 className="font-semibold text-white">Price Chart</h3>
          {lastPrice !== null && (
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold text-white">₹{lastPrice.toFixed(2)}</span>
              {priceChange !== null && (
                <span className={`flex items-center text-sm ${priceChange >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {priceChange >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                </span>
              )}
            </div>
          )}
        </div>
        
        <button
          onClick={fetchChartData}
          disabled={isLoading}
          className="p-2 rounded-lg bg-slate-700/50 hover:bg-slate-600/50 text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>
      
      {/* Period Selector */}
      <div className="flex items-center gap-2 mb-4 flex-wrap">
        <div className="flex bg-slate-800/50 rounded-lg p-1">
          {periods.map((p) => (
            <button
              key={p.value}
              onClick={() => setPeriod(p.value)}
              className={`px-3 py-1 text-sm rounded-md transition-all ${
                period === p.value
                  ? 'bg-sky-500 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
        
        <div className="flex gap-2 ml-auto">
          <button
            onClick={() => setShowSMA20(!showSMA20)}
            className={`px-3 py-1 text-xs rounded-full border transition-all ${
              showSMA20
                ? 'border-amber-500 bg-amber-500/20 text-amber-400'
                : 'border-slate-600 text-slate-500'
            }`}
          >
            SMA 20
          </button>
          <button
            onClick={() => setShowSMA50(!showSMA50)}
            className={`px-3 py-1 text-xs rounded-full border transition-all ${
              showSMA50
                ? 'border-purple-500 bg-purple-500/20 text-purple-400'
                : 'border-slate-600 text-slate-500'
            }`}
          >
            SMA 50
          </button>
          <button
            onClick={() => setShowVolume(!showVolume)}
            className={`px-3 py-1 text-xs rounded-full border transition-all ${
              showVolume
                ? 'border-indigo-500 bg-indigo-500/20 text-indigo-400'
                : 'border-slate-600 text-slate-500'
            }`}
          >
            Volume
          </button>
        </div>
      </div>
      
      {/* Chart Container */}
      <div className="relative">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50 z-10 rounded-lg">
            <div className="flex items-center gap-2 text-slate-400">
              <RefreshCw className="w-5 h-5 animate-spin" />
              <span>Loading chart...</span>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50 z-10 rounded-lg">
            <div className="text-rose-400">{error}</div>
          </div>
        )}
        
        <div
          ref={chartContainerRef}
          className="w-full rounded-lg overflow-hidden"
          style={{ height: '400px' }}
        />
      </div>
      
      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-xs text-slate-500">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-emerald-500"></div>
          <span>Bullish</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-rose-500"></div>
          <span>Bearish</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-1 bg-amber-500"></div>
          <span>SMA 20</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-1 bg-purple-500"></div>
          <span>SMA 50</span>
        </div>
      </div>
    </div>
  );
}

export default StockChart;
