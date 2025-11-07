import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class LiveDataFeed:
    """
    Real-time market data integration với Yahoo Finance
    Foundation cho paper trading
    """
    def __init__(self, symbols, data_type="historical"):
        self.symbols = symbols
        self.data_type = data_type
        self.connected = False
    
    def connect(self):
        """Kết nối đến data source"""
        try:
            # Test connection với AAPL
            test_data = yf.download("AAPL", period="1d", progress=False)
            self.connected = True
            return True
        except Exception as e:
            print(f"Data connection failed: {e}")
            return False
    
    def get_historical_data(self, symbol, period="6mo"):
        """Lấy historical data cho backtesting"""
        try:
            data = yf.download(symbol, period=period, progress=False)
            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns
                data.columns = data.columns.droplevel(1) if data.columns.nlevels > 1 else data.columns
            return data
        except Exception as e:
            print(f"Historical data error for {symbol}: {e}")
            return None
