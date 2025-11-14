"""
Trend Following Strategy for Supreme System V5.

A comprehensive trend following strategy that uses multiple technical indicators
to identify and follow market trends with risk management.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class TrendFollowingAgent(BaseStrategy):
    """
    Trend Following Strategy Agent.

    Uses moving averages, ADX, RSI, and volume analysis to identify
    and follow strong market trends.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize Trend Following Agent.

        Args:
            agent_id: Unique identifier for the agent
            config: Configuration parameters
        """
        # BaseStrategy.__init__() only takes 'name' parameter
        super().__init__(name=agent_id)

        # Strategy parameters with defaults
        self.short_window = config.get('short_window', 20)
        self.long_window = config.get('long_window', 50)
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 25)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.volume_ma_period = config.get('volume_ma_period', 20)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.05)

        # Initialize parameters dict for parent class
        self.parameters = {
            'short_window': self.short_window,
            'long_window': self.long_window,
            'adx_period': self.adx_period,
            'adx_threshold': self.adx_threshold,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'volume_ma_period': self.volume_ma_period,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }

        logger.info(f"Initialized TrendFollowingAgent {agent_id} with parameters: {self.parameters}")

    def generate_signal(self, data: pd.DataFrame, portfolio_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate trading signal based on trend analysis.

        Args:
            market_data: OHLCV market data
            portfolio_value: Current portfolio value

        Returns:
            Dict containing trading signal
        """
        try:
            # Validate data
            if data is None or data.empty or len(data) < self.long_window:
                return {
                    'action': 'HOLD',
                    'symbol': 'AAPL',
                    'quantity': 0,
                    'price': 0.0,
                    'strength': 0.0,
                    'confidence': 0.0,
                    'reason': 'Insufficient or invalid data'
                }

            # Calculate indicators
            indicators = self._calculate_indicators(data)

            # Determine trend direction
            trend_direction = self._determine_trend_direction(indicators)

            # Check entry conditions
            current_price = indicators['close'].iloc[-1]

            if trend_direction == 'UPTREND' and self._check_buy_conditions(indicators):
                position_size = self._calculate_position_size(portfolio_value or 10000, current_price)
                return {
                    'action': 'BUY',
                    'symbol': 'AAPL',
                    'quantity': position_size,
                    'price': current_price,
                    'strength': 0.8,
                    'confidence': 0.9,
                    'reason': 'Strong uptrend with confirmation'
                }
            elif trend_direction == 'DOWNTREND' and self._check_sell_conditions(indicators):
                position_size = self._calculate_position_size(portfolio_value or 10000, current_price)
                return {
                    'action': 'SELL',
                    'symbol': 'AAPL',
                    'quantity': position_size,
                    'price': current_price,
                    'strength': 0.8,
                    'confidence': 0.9,
                    'reason': 'Strong downtrend with confirmation'
                }

            return {
                'action': 'HOLD',
                'symbol': 'AAPL',
                'quantity': 0,
                'price': current_price,
                'strength': 0.0,
                'confidence': 0.5,
                'reason': 'No clear trend signal'
            }

        except Exception as e:
            logger.error(f"Error generating trade signal: {e}")
            return {
                'action': 'HOLD',
                'symbol': 'AAPL',
                'quantity': 0,
                'price': 0.0,
                'strength': 0.0,
                'confidence': 0.0,
                'reason': f'Error: {str(e)}'
            }

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for trend analysis."""
        df = data.copy()

        # Moving averages
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()

        # ADX calculation (simplified)
        df['adx'] = self._calculate_adx(df, self.adx_period)

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)

        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()

        # MACD (simplified)
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX indicator."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), high - high.shift(1), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), low.shift(1) - low, 0)

        # Smoothed values
        atr = tr.rolling(window=period).mean()
        di_plus = 100 * (pd.Series(dm_plus).rolling(window=period).mean() / atr)
        di_minus = 100 * (pd.Series(dm_minus).rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _determine_trend_direction(self, indicators: pd.DataFrame) -> str:
        """Determine overall trend direction."""
        latest = indicators.iloc[-1]

        # Check moving average alignment
        ma_trend = 'UPTREND' if latest['sma_short'] > latest['sma_long'] else 'DOWNTREND'

        # Check ADX for trend strength
        adx_strong = latest['adx'] > self.adx_threshold

        # Check MACD
        macd_trend = 'UPTREND' if latest['macd'] > latest['macd_signal'] else 'DOWNTREND'

        # Consensus
        if ma_trend == macd_trend and adx_strong:
            return ma_trend

        return 'SIDEWAYS'

    def _check_buy_conditions(self, indicators: pd.DataFrame) -> bool:
        """Check conditions for buy signal."""
        latest = indicators.iloc[-1]

        # Short MA above long MA
        ma_condition = latest['sma_short'] > latest['sma_long']

        # ADX shows strong trend
        adx_condition = latest['adx'] > self.adx_threshold

        # RSI not overbought
        rsi_condition = latest['rsi'] < self.rsi_overbought

        # MACD positive
        macd_condition = latest['macd'] > latest['macd_signal']

        # Volume confirmation
        volume_condition = latest['volume'] > latest['volume_ma']

        return all([ma_condition, adx_condition, rsi_condition, macd_condition, volume_condition])

    def _check_sell_conditions(self, indicators: pd.DataFrame) -> bool:
        """Check conditions for sell signal."""
        latest = indicators.iloc[-1]

        # Short MA below long MA
        ma_condition = latest['sma_short'] < latest['sma_long']

        # ADX shows strong trend
        adx_condition = latest['adx'] > self.adx_threshold

        # RSI not oversold
        rsi_condition = latest['rsi'] > self.rsi_oversold

        # MACD negative
        macd_condition = latest['macd'] < latest['macd_signal']

        # Volume confirmation
        volume_condition = latest['volume'] > latest['volume_ma']

        return all([ma_condition, adx_condition, rsi_condition, macd_condition, volume_condition])

    def _calculate_position_size(self, portfolio_value: float, price: float) -> int:
        """Calculate position size based on risk management."""
        # Use 1% of portfolio per trade as default
        risk_amount = portfolio_value * 0.01
        position_size = risk_amount / (price * self.stop_loss_pct)
        return max(1, int(position_size))
