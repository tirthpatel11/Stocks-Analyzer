"""
Data Provider Abstraction Layer

Supports multiple data sources:
- NSE India (Real-time, Free, No account needed)
- Yahoo Finance (Delayed 15-20 min, Free)
- Angel One SmartAPI (Real-time, Free with account)

Default: NSE India for real-time quotes, Yahoo Finance for historical data
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import yfinance as yf
import logging
import json
import time
from functools import lru_cache

logger = logging.getLogger(__name__)


class NSEDataProvider:
    """
    NSE India Direct API - Real-time data without any account
    
    Features:
    - Real-time quotes (no delay)
    - Live market status
    - Index data (Nifty 50, Bank Nifty, etc.)
    - No API key required
    
    Note: NSE has anti-scraping protection. Falls back to Yahoo Finance if blocked.
    """
    
    BASE_URL = "https://www.nseindia.com"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Pragma': 'no-cache',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
        })
        self._cookies_initialized = False
        self._init_session()
    
    def _init_session(self):
        """Initialize session with NSE cookies"""
        try:
            # First request to get cookies
            response = self.session.get(self.BASE_URL, timeout=15)
            if response.status_code == 200:
                self._cookies_initialized = True
                # Update headers for API calls
                self.session.headers.update({
                    'Accept': 'application/json, text/plain, */*',
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'same-origin',
                    'Referer': 'https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE',
                })
                time.sleep(1)  # Small delay after getting cookies
            else:
                logger.warning(f"NSE returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to initialize NSE session: {e}")
    
    def _make_request(self, endpoint: str, retries: int = 2) -> Optional[Dict]:
        """Make request to NSE API with retry logic"""
        if not self._cookies_initialized:
            self._init_session()
            if not self._cookies_initialized:
                return None
        
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    try:
                        return response.json()
                    except:
                        logger.warning(f"NSE returned non-JSON response")
                        return None
                elif response.status_code in [401, 403]:
                    # Re-initialize session if unauthorized
                    self._cookies_initialized = False
                    self._init_session()
                else:
                    logger.warning(f"NSE returned status {response.status_code}")
            except requests.exceptions.Timeout:
                logger.warning(f"NSE request timed out (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"NSE request failed (attempt {attempt + 1}): {e}")
            
            if attempt < retries - 1:
                time.sleep(0.5)
        
        return None
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a stock
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            
        Returns:
            Dictionary with real-time quote data
        """
        # Remove .NS suffix if present
        symbol = symbol.replace('.NS', '').replace('.BO', '').upper()
        
        data = self._make_request(f"/api/quote-equity?symbol={symbol}")
        
        if not data:
            return None
        
        try:
            price_info = data.get('priceInfo', {})
            
            return {
                'symbol': symbol,
                'name': data.get('info', {}).get('companyName', symbol),
                'last_price': price_info.get('lastPrice'),
                'change': price_info.get('change'),
                'change_pct': price_info.get('pChange'),
                'open': price_info.get('open'),
                'high': price_info.get('intraDayHighLow', {}).get('max'),
                'low': price_info.get('intraDayHighLow', {}).get('min'),
                'close': price_info.get('previousClose'),
                'volume': data.get('preOpenMarket', {}).get('totalTradedVolume'),
                'bid': price_info.get('upperCP'),
                'ask': price_info.get('lowerCP'),
                '52w_high': price_info.get('weekHighLow', {}).get('max'),
                '52w_low': price_info.get('weekHighLow', {}).get('min'),
                'timestamp': datetime.now().isoformat(),
                'source': 'NSE_REALTIME'
            }
        except Exception as e:
            logger.error(f"Error parsing NSE quote for {symbol}: {e}")
            return None
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        data = self._make_request("/api/marketStatus")
        
        if data:
            return {
                'status': data.get('marketState', [{}])[0].get('marketStatus', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            }
        return {'status': 'Unknown', 'timestamp': datetime.now().isoformat()}
    
    def get_index_quote(self, index: str = "NIFTY 50") -> Optional[Dict[str, Any]]:
        """Get real-time index quote"""
        index_map = {
            'NIFTY 50': 'NIFTY%2050',
            'NIFTY BANK': 'NIFTY%20BANK',
            'NIFTY IT': 'NIFTY%20IT',
            'NIFTY PHARMA': 'NIFTY%20PHARMA',
        }
        
        encoded_index = index_map.get(index.upper(), index.replace(' ', '%20'))
        data = self._make_request(f"/api/equity-stockIndices?index={encoded_index}")
        
        if data:
            advance = data.get('advance', {})
            return {
                'index': index,
                'last_price': data.get('data', [{}])[0].get('last'),
                'change': data.get('data', [{}])[0].get('change'),
                'change_pct': data.get('data', [{}])[0].get('pChange'),
                'advances': advance.get('advances'),
                'declines': advance.get('declines'),
                'unchanged': advance.get('unchanged'),
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def get_top_gainers(self, index: str = "NIFTY 50") -> List[Dict[str, Any]]:
        """Get top gaining stocks"""
        encoded_index = index.replace(' ', '%20')
        data = self._make_request(f"/api/equity-stockIndices?index={encoded_index}")
        
        if data and 'data' in data:
            stocks = data['data'][1:]  # Skip first (index itself)
            gainers = sorted(stocks, key=lambda x: x.get('pChange', 0), reverse=True)[:10]
            return [
                {
                    'symbol': s.get('symbol'),
                    'last_price': s.get('lastPrice'),
                    'change_pct': s.get('pChange'),
                    'volume': s.get('totalTradedVolume')
                }
                for s in gainers
            ]
        return []
    
    def get_top_losers(self, index: str = "NIFTY 50") -> List[Dict[str, Any]]:
        """Get top losing stocks"""
        encoded_index = index.replace(' ', '%20')
        data = self._make_request(f"/api/equity-stockIndices?index={encoded_index}")
        
        if data and 'data' in data:
            stocks = data['data'][1:]
            losers = sorted(stocks, key=lambda x: x.get('pChange', 0))[:10]
            return [
                {
                    'symbol': s.get('symbol'),
                    'last_price': s.get('lastPrice'),
                    'change_pct': s.get('pChange'),
                    'volume': s.get('totalTradedVolume')
                }
                for s in losers
            ]
        return []


class YahooFinanceProvider:
    """
    Yahoo Finance Provider - For historical data and fundamentals
    
    Features:
    - Historical OHLCV data
    - Fundamental data (P/E, ROE, etc.)
    - Company information
    - Delayed quotes (15-20 min)
    """
    
    @staticmethod
    def get_quote(symbol: str) -> Optional[Dict[str, Any]]:
        """Get quote from Yahoo Finance"""
        try:
            # Ensure .NS suffix for NSE stocks
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period='1d')
            
            if hist.empty:
                return None
            
            return {
                'symbol': symbol,
                'name': info.get('shortName', symbol),
                'last_price': hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice'),
                'change': info.get('regularMarketChange'),
                'change_pct': info.get('regularMarketChangePercent'),
                'open': hist['Open'].iloc[-1] if not hist.empty else info.get('open'),
                'high': hist['High'].iloc[-1] if not hist.empty else info.get('dayHigh'),
                'low': hist['Low'].iloc[-1] if not hist.empty else info.get('dayLow'),
                'close': info.get('previousClose'),
                'volume': int(hist['Volume'].iloc[-1]) if not hist.empty else info.get('volume'),
                '52w_high': info.get('fiftyTwoWeekHigh'),
                '52w_low': info.get('fiftyTwoWeekLow'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'timestamp': datetime.now().isoformat(),
                'source': 'YAHOO_FINANCE'
            }
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    @staticmethod
    def get_historical(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        try:
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df if not df.empty else None
        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {e}")
            return None
    
    @staticmethod
    def get_fundamentals(symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data"""
        try:
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                'symbol': symbol,
                'name': info.get('shortName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'beta': info.get('beta'),
                'analyst_target': info.get('targetMeanPrice'),
                'recommendation': info.get('recommendationKey')
            }
        except Exception as e:
            logger.error(f"Fundamentals error for {symbol}: {e}")
            return None


class DataProvider:
    """
    Unified Data Provider - Combines multiple sources
    
    Strategy:
    - Real-time quotes: NSE India (during market hours)
    - Historical data: Yahoo Finance
    - Fundamentals: Yahoo Finance
    - Fallback: Yahoo Finance for everything
    """
    
    def __init__(self, prefer_realtime: bool = True):
        """
        Initialize data provider
        
        Args:
            prefer_realtime: If True, use NSE for real-time during market hours
        """
        self.prefer_realtime = prefer_realtime
        self.nse = NSEDataProvider() if prefer_realtime else None
        self.yahoo = YahooFinanceProvider()
        self._cache = {}
        self._cache_ttl = 60  # Cache TTL in seconds
    
    def _is_market_hours(self) -> bool:
        """Check if Indian market is open (9:15 AM - 3:30 PM IST)"""
        from datetime import timezone
        
        # Get current IST time
        ist_offset = timedelta(hours=5, minutes=30)
        utc_now = datetime.now(timezone.utc)
        ist_now = utc_now + ist_offset
        
        # Market hours: 9:15 AM to 3:30 PM, Monday to Friday
        market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_weekday = ist_now.weekday() < 5
        is_market_time = market_open <= ist_now <= market_close
        
        return is_weekday and is_market_time
    
    def get_quote(self, symbol: str, force_realtime: bool = False) -> Dict[str, Any]:
        """
        Get stock quote - automatically chooses best source
        
        Args:
            symbol: Stock symbol (with or without .NS suffix)
            force_realtime: Force NSE real-time data
            
        Returns:
            Quote dictionary with price data
        """
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '').upper()
        
        # Try NSE real-time first during market hours
        if self.nse and (force_realtime or (self.prefer_realtime and self._is_market_hours())):
            quote = self.nse.get_quote(clean_symbol)
            if quote:
                return quote
        
        # Fallback to Yahoo Finance
        yahoo_quote = self.yahoo.get_quote(symbol)
        if yahoo_quote:
            return yahoo_quote
        
        return {
            'symbol': symbol,
            'error': 'Failed to fetch quote',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_historical(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data (always from Yahoo Finance)"""
        return self.yahoo.get_historical(symbol, period)
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data (from Yahoo Finance)"""
        return self.yahoo.get_fundamentals(symbol)
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get market status"""
        if self.nse:
            return self.nse.get_market_status()
        return {'status': 'Unknown', 'timestamp': datetime.now().isoformat()}
    
    def get_index(self, index: str = "NIFTY 50") -> Optional[Dict[str, Any]]:
        """Get index quote"""
        if self.nse:
            return self.nse.get_index_quote(index)
        return None
    
    def get_top_gainers(self, index: str = "NIFTY 50") -> List[Dict[str, Any]]:
        """Get top gainers"""
        if self.nse:
            return self.nse.get_top_gainers(index)
        return []
    
    def get_top_losers(self, index: str = "NIFTY 50") -> List[Dict[str, Any]]:
        """Get top losers"""
        if self.nse:
            return self.nse.get_top_losers(index)
        return []


# Global instance
_data_provider = None

def get_data_provider(prefer_realtime: bool = True) -> DataProvider:
    """Get or create global data provider instance"""
    global _data_provider
    if _data_provider is None:
        _data_provider = DataProvider(prefer_realtime=prefer_realtime)
    return _data_provider

