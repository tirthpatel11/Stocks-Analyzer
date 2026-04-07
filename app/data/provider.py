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
        try:
            response = self.session.get(self.BASE_URL, timeout=15)
            if response.status_code == 200:
                self._cookies_initialized = True
                self.session.headers.update({
                    'Accept': 'application/json, text/plain, */*',
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'same-origin',
                    'Referer': 'https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE',
                })
                time.sleep(1)
            else:
                logger.warning(f"NSE returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to initialize NSE session: {e}")

    def _make_request(self, endpoint: str, retries: int = 2) -> Optional[Dict]:
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
        data = self._make_request("/api/marketStatus")

        if data:
            return {
                'status': data.get('marketState', [{}])[0].get('marketStatus', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            }
        return {'status': 'Unknown', 'timestamp': datetime.now().isoformat()}

    def get_index_quote(self, index: str = "NIFTY 50") -> Optional[Dict[str, Any]]:
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
        encoded_index = index.replace(' ', '%20')
        data = self._make_request(f"/api/equity-stockIndices?index={encoded_index}")

        if data and 'data' in data:
            stocks = data['data'][1:]
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

    @staticmethod
    def get_quote(symbol: str) -> Optional[Dict[str, Any]]:
        try:
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

    def __init__(self, prefer_realtime: bool = True):
        self.prefer_realtime = prefer_realtime
        self.nse = NSEDataProvider() if prefer_realtime else None
        self.yahoo = YahooFinanceProvider()
        self._cache = {}
        self._cache_ttl = 60

    def _is_market_hours(self) -> bool:
        from datetime import timezone

        ist_offset = timedelta(hours=5, minutes=30)
        utc_now = datetime.now(timezone.utc)
        ist_now = utc_now + ist_offset

        market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)

        is_weekday = ist_now.weekday() < 5
        is_market_time = market_open <= ist_now <= market_close

        return is_weekday and is_market_time

    def get_quote(self, symbol: str, force_realtime: bool = False) -> Dict[str, Any]:
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '').upper()

        if self.nse and (force_realtime or (self.prefer_realtime and self._is_market_hours())):
            quote = self.nse.get_quote(clean_symbol)
            if quote:
                return quote

        yahoo_quote = self.yahoo.get_quote(symbol)
        if yahoo_quote:
            return yahoo_quote

        return {
            'symbol': symbol,
            'error': 'Failed to fetch quote',
            'timestamp': datetime.now().isoformat()
        }

    def get_historical(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        return self.yahoo.get_historical(symbol, period)

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.yahoo.get_fundamentals(symbol)

    def get_market_status(self) -> Dict[str, Any]:
        if self.nse:
            return self.nse.get_market_status()
        return {'status': 'Unknown', 'timestamp': datetime.now().isoformat()}

    def get_index(self, index: str = "NIFTY 50") -> Optional[Dict[str, Any]]:
        if self.nse:
            return self.nse.get_index_quote(index)
        return None

    def get_top_gainers(self, index: str = "NIFTY 50") -> List[Dict[str, Any]]:
        if self.nse:
            return self.nse.get_top_gainers(index)
        return []

    def get_top_losers(self, index: str = "NIFTY 50") -> List[Dict[str, Any]]:
        if self.nse:
            return self.nse.get_top_losers(index)
        return []


_data_provider = None

def get_data_provider(prefer_realtime: bool = True) -> DataProvider:
    global _data_provider
    if _data_provider is None:
        _data_provider = DataProvider(prefer_realtime=prefer_realtime)
    return _data_provider
