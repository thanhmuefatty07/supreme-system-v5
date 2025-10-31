"""
Alpha Vantage Connector - Stock and macroeconomic data
Free tier: 5 calls/min, 500 calls/day - excellent for traditional markets
"""

import asyncio
import time
from typing import Dict, Optional, List, Any

import aiohttp
from loguru import logger

class AlphaVantageConnector:
    """
    Alpha Vantage API connector - Stock and macroeconomic data
    Free tier: 5 calls/min, 500 calls/day, 100+ global markets
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Alpha Vantage connector"""
        self.base_url = "https://www.alphavantage.co/query"
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting (5 calls/min free = 12s between calls)
        self.rate_limit_delay = 12.0
        self.last_call = 0.0

        # Symbol mapping for different markets
        self.symbol_map = {
            # US Stocks
            'AAPL-USDT': 'AAPL',
            'TSLA-USDT': 'TSLA',
            'MSFT-USDT': 'MSFT',
            'NVDA-USDT': 'NVDA',
            'AMZN-USDT': 'AMZN',

            # ETFs
            'SPY-USDT': 'SPY',
            'QQQ-USDT': 'QQQ',
            'VOO-USDT': 'VOO',

            # Forex
            'EUR-USD': 'EUR/USD',
            'USD-JPY': 'USD/JPY',
            'GBP-USD': 'GBP/USD',

            # Crypto (Alpha Vantage uses different format)
            'BTC-USD': 'BTC',
            'ETH-USD': 'ETH',
            'BNB-USD': 'BNB'
        }

    async def connect(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Supreme-System-V5/1.0',
                    'Accept': 'application/json'
                }
            )

            if not self.api_key:
                logger.warning("⚠️ Alpha Vantage running without API key - using demo mode")
            else:
                logger.info("✅ Alpha Vantage connector initialized with API key")

        logger.info("✅ Alpha Vantage connector ready")

    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("✅ Alpha Vantage connector disconnected")

    async def _enforce_rate_limit(self):
        """Enforce API rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_call

        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)

        self.last_call = time.time()

    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time price data for symbol
        Supports stocks, ETFs, forex, and crypto
        """
        await self._enforce_rate_limit()

        if symbol not in self.symbol_map:
            logger.warning(f"⚠️ Symbol {symbol} not mapped for Alpha Vantage")
            return None

        alpha_symbol = self.symbol_map[symbol]

        try:
            # Determine endpoint based on symbol type
            if '-' in alpha_symbol:  # Forex
                endpoint = "CURRENCY_EXCHANGE_RATE"
                params = {
                    'function': endpoint,
                    'from_currency': alpha_symbol.split('/')[0],
                    'to_currency': alpha_symbol.split('/')[1],
                    'apikey': self.api_key or 'demo'
                }
            elif len(alpha_symbol) <= 5:  # Stocks/ETFs
                endpoint = "GLOBAL_QUOTE"
                params = {
                    'function': endpoint,
                    'symbol': alpha_symbol,
                    'apikey': self.api_key or 'demo'
                }
            else:  # Crypto
                endpoint = "DIGITAL_CURRENCY_DAILY"
                params = {
                    'function': endpoint,
                    'symbol': alpha_symbol,
                    'market': 'USD',
                    'apikey': self.api_key or 'demo'
                }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_response(endpoint, data, symbol)
                else:
                    logger.error(f"❌ Alpha Vantage API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"❌ Alpha Vantage error for {symbol}: {e}")
            return None

    def _parse_response(self, endpoint: str, data: Dict, symbol: str) -> Dict[str, Any]:
        """Parse Alpha Vantage API response"""
        result = {
            'symbol': symbol,
            'source': 'alpha_vantage',
            'timestamp': time.time()
        }

        try:
            if endpoint == "GLOBAL_QUOTE":  # Stocks/ETFs
                quote = data.get('Global Quote', {})
                result.update({
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': float(quote.get('10. change percent', '0').rstrip('%')),
                    'volume': int(quote.get('06. volume', 0)),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'open': float(quote.get('02. open', 0))
                })

            elif endpoint == "CURRENCY_EXCHANGE_RATE":  # Forex
                rate = data.get('Realtime Currency Exchange Rate', {})
                result.update({
                    'price': float(rate.get('5. Exchange Rate', 0)),
                    'bid': float(rate.get('8. Bid Price', 0)),
                    'ask': float(rate.get('9. Ask Price', 0)),
                    'last_refreshed': rate.get('6. Last Refreshed', '')
                })

            elif endpoint == "DIGITAL_CURRENCY_DAILY":  # Crypto
                # Get latest day's data
                time_series = data.get('Time Series (Digital Currency Daily)', {})
                if time_series:
                    latest_date = sorted(time_series.keys())[-1]
                    daily_data = time_series[latest_date]
                    result.update({
                        'price': float(daily_data.get('4a. close (USD)', 0)),
                        'high': float(daily_data.get('2a. high (USD)', 0)),
                        'low': float(daily_data.get('3a. low (USD)', 0)),
                        'open': float(daily_data.get('1a. open (USD)', 0)),
                        'volume': float(daily_data.get('5. volume', 0)),
                        'market_cap': float(daily_data.get('6. market cap (USD)', 0))
                    })

            # Add metadata
            result['endpoint'] = endpoint
            result['success'] = True

            logger.debug(f"✅ Alpha Vantage: {symbol} = ${result.get('price', 0):.4f}")
            return result

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ Alpha Vantage parsing error: {e}")
            result['success'] = False
            result['error'] = str(e)
            return result

    async def get_macro_data(self, indicator: str) -> Optional[Dict[str, Any]]:
        """
        Get macroeconomic data (GDP, inflation, unemployment, etc.)
        """
        await self._enforce_rate_limit()

        macro_functions = {
            'GDP': 'REAL_GDP',
            'inflation': 'CPI',
            'unemployment': 'UNEMPLOYMENT',
            'interest_rate': 'FEDERAL_FUNDS_RATE',
            'retail_sales': 'RETAIL_SALES'
        }

        if indicator not in macro_functions:
            logger.warning(f"⚠️ Macro indicator {indicator} not supported")
            return None

        try:
            params = {
                'function': macro_functions[indicator],
                'interval': 'quarterly',  # Most conservative rate usage
                'apikey': self.api_key or 'demo'
            }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_macro_response(indicator, data)
                else:
                    logger.error(f"❌ Alpha Vantage macro API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"❌ Alpha Vantage macro error for {indicator}: {e}")
            return None

    def _parse_macro_response(self, indicator: str, data: Dict) -> Dict[str, Any]:
        """Parse macroeconomic data response"""
        result = {
            'indicator': indicator,
            'source': 'alpha_vantage',
            'timestamp': time.time()
        }

        try:
            # Get data series (different structure for each indicator)
            data_key = None
            for key in data.keys():
                if 'data' in key.lower() or 'series' in key.lower():
                    data_key = key
                    break

            if data_key and data_key in data:
                series = data[data_key]
                if isinstance(series, list) and len(series) > 0:
                    latest = series[0]  # Most recent data
                    result.update({
                        'value': float(latest.get('value', 0)),
                        'date': latest.get('date', ''),
                        'units': latest.get('units', ''),
                        'success': True
                    })
                else:
                    result.update({
                        'value': 0.0,
                        'date': '',
                        'units': '',
                        'success': False,
                        'error': 'No data available'
                    })
            else:
                result.update({
                    'value': 0.0,
                    'date': '',
                    'units': '',
                    'success': False,
                    'error': 'Invalid response format'
                })

            logger.debug(f"✅ Alpha Vantage macro: {indicator} = {result.get('value', 0)}")
            return result

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ Alpha Vantage macro parsing error: {e}")
            result['success'] = False
            result['error'] = str(e)
            return result

    async def get_technical_indicators(self, symbol: str, indicator: str, interval: str = 'daily') -> Optional[Dict[str, Any]]:
        """
        Get technical indicators (SMA, EMA, RSI, etc.)
        """
        await self._enforce_rate_limit()

        if symbol not in self.symbol_map:
            logger.warning(f"⚠️ Symbol {symbol} not mapped for Alpha Vantage")
            return None

        alpha_symbol = self.symbol_map[symbol]

        # Map indicator names
        indicator_functions = {
            'SMA': 'SMA',
            'EMA': 'EMA',
            'RSI': 'RSI',
            'MACD': 'MACD',
            'BBANDS': 'BBANDS',
            'STOCH': 'STOCH'
        }

        if indicator not in indicator_functions:
            logger.warning(f"⚠️ Technical indicator {indicator} not supported")
            return None

        try:
            params = {
                'function': indicator_functions[indicator],
                'symbol': alpha_symbol,
                'interval': interval,
                'time_period': 14,  # Default period
                'series_type': 'close',
                'apikey': self.api_key or 'demo'
            }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_technical_response(indicator, data, symbol)
                else:
                    logger.error(f"❌ Alpha Vantage technical API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"❌ Alpha Vantage technical error for {symbol} {indicator}: {e}")
            return None

    def _parse_technical_response(self, indicator: str, data: Dict, symbol: str) -> Dict[str, Any]:
        """Parse technical indicator response"""
        result = {
            'symbol': symbol,
            'indicator': indicator,
            'source': 'alpha_vantage',
            'timestamp': time.time()
        }

        try:
            # Technical indicators have different response formats
            tech_data_key = f"Technical Analysis: {indicator}"
            if tech_data_key in data:
                series = data[tech_data_key]
                if isinstance(series, dict) and len(series) > 0:
                    latest_date = sorted(series.keys())[-1]
                    latest_data = series[latest_date]

                    # Extract indicator values (different for each indicator)
                    values = {}
                    for key, value in latest_data.items():
                        try:
                            values[key.lower()] = float(value)
                        except (ValueError, TypeError):
                            values[key.lower()] = value

                    result.update({
                        'date': latest_date,
                        'values': values,
                        'success': True
                    })
                else:
                    result.update({
                        'values': {},
                        'success': False,
                        'error': 'No technical data available'
                    })
            else:
                result.update({
                    'values': {},
                    'success': False,
                    'error': 'Invalid technical response format'
                })

            logger.debug(f"✅ Alpha Vantage technical: {symbol} {indicator} = {result.get('values', {})}")
            return result

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ Alpha Vantage technical parsing error: {e}")
            result['success'] = False
            result['error'] = str(e)
            return result
