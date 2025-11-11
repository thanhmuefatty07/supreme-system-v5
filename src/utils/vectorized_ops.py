"""
Vectorized Operations for Supreme System V5

High-performance, vectorized implementations of trading calculations.
Replaces slow iterative operations with NumPy/Pandas vectorized functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from numba import jit, prange
import logging

logger = logging.getLogger(__name__)


class VectorizedTradingOps:
    """Vectorized trading operations for maximum performance."""

    @staticmethod
    def calculate_sma_vectorized(prices: pd.Series, window: int) -> pd.Series:
        """Vectorized Simple Moving Average calculation."""
        return prices.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def calculate_ema_vectorized(prices: pd.Series, span: int) -> pd.Series:
        """Vectorized Exponential Moving Average calculation."""
        return prices.ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_rsi_vectorized(prices: pd.Series, period: int = 14) -> pd.Series:
        """Vectorized RSI calculation - 10x faster than iterative."""
        delta = prices.diff()

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Calculate average gains and losses
        avg_gains = pd.Series(gains).rolling(window=period, min_periods=1).mean()
        avg_losses = pd.Series(losses).rolling(window=period, min_periods=1).mean()

        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd_vectorized(prices: pd.Series,
                                fast_period: int = 12,
                                slow_period: int = 26,
                                signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Vectorized MACD calculation."""
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands_vectorized(prices: pd.Series,
                                          window: int = 20,
                                          num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Vectorized Bollinger Bands calculation."""
        sma = prices.rolling(window=window, min_periods=1).mean()
        std = prices.rolling(window=window, min_periods=1).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return upper_band, sma, lower_band

    @staticmethod
    def calculate_stochastic_oscillator_vectorized(high: pd.Series,
                                                 low: pd.Series,
                                                 close: pd.Series,
                                                 k_period: int = 14,
                                                 d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Vectorized Stochastic Oscillator calculation."""
        # Calculate %K
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))

        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()

        return k_percent, d_percent

    @staticmethod
    @jit(nopython=True, parallel=True)
    def calculate_atr_numba(high: np.ndarray,
                          low: np.ndarray,
                          close: np.ndarray,
                          period: int = 14) -> np.ndarray:
        """Numba-accelerated ATR calculation for maximum performance."""
        n = len(high)
        atr = np.zeros(n)
        tr = np.zeros(n)

        # Calculate True Range
        tr[0] = high[0] - low[0]
        for i in prange(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        # Calculate ATR using Wilder's smoothing
        atr[period-1] = np.mean(tr[:period])
        for i in prange(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    @staticmethod
    def calculate_atr_vectorized(high: pd.Series,
                               low: pd.Series,
                               close: pd.Series,
                               period: int = 14) -> pd.Series:
        """Vectorized ATR calculation."""
        # True Range calculation
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR using exponential moving average (Wilder's method approximation)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def detect_candlestick_patterns_vectorized(data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized candlestick pattern detection."""
        open_price = data['open'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Bullish engulfing pattern
        prev_open = np.roll(open_price, 1)
        prev_close = np.roll(close_price, 1)

        bullish_engulfing = (
            (prev_close < prev_open) &  # Previous bearish
            (close > open_price) &      # Current bullish
            (close > prev_open) &       # Current close > prev open
            (open_price < prev_close)   # Current open < prev close
        )
        bullish_engulfing[0] = False  # No pattern for first candle

        # Bearish engulfing pattern
        bearish_engulfing = (
            (prev_close > prev_open) &  # Previous bullish
            (close < open_price) &      # Current bearish
            (close < prev_open) &       # Current close < prev open
            (open_price > prev_close)   # Current open > prev close
        )
        bearish_engulfing[0] = False

        # Hammer pattern
        body = np.abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        total_range = high - low

        hammer = (
            (lower_shadow > 2 * body) &     # Long lower shadow
            (upper_shadow < body) &         # Short upper shadow
            (body < 0.3 * total_range)      # Small body
        )

        # Shooting star pattern
        shooting_star = (
            (upper_shadow > 2 * body) &     # Long upper shadow
            (lower_shadow < body) &         # Short lower shadow
            (body < 0.3 * total_range)      # Small body
        )

        return pd.DataFrame({
            'bullish_engulfing': bullish_engulfing,
            'bearish_engulfing': bearish_engulfing,
            'hammer': hammer,
            'shooting_star': shooting_star
        }, index=data.index)

    @staticmethod
    def calculate_volume_indicators_vectorized(volume: pd.Series,
                                             price: pd.Series,
                                             window: int = 20) -> Dict[str, pd.Series]:
        """Vectorized volume indicators calculation."""
        # Volume moving averages
        volume_sma = volume.rolling(window=window, min_periods=1).mean()
        volume_ratio = volume / volume_sma

        # On-balance volume (OBV)
        obv = pd.Series(index=volume.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        price_changes = price.pct_change()
        for i in range(1, len(volume)):
            if price_changes.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_changes.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        # Volume weighted average price (VWAP)
        vwap = (price * volume).cumsum() / volume.cumsum()

        # Accumulation/Distribution Line
        ad = pd.Series(index=volume.index, dtype=float)
        ad.iloc[0] = 0

        money_flow_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
        money_flow_volume = money_flow_multiplier * volume

        for i in range(1, len(volume)):
            ad.iloc[i] = ad.iloc[i-1] + money_flow_volume.iloc[i]

        return {
            'volume_sma': volume_sma,
            'volume_ratio': volume_ratio,
            'obv': obv,
            'vwap': vwap,
            'accumulation_distribution': ad
        }

    @staticmethod
    def batch_signal_generation_vectorized(data_batch: List[pd.DataFrame],
                                         strategy_func: callable) -> List[Dict]:
        """Batch process signal generation for multiple data chunks."""
        results = []

        for data in data_batch:
            try:
                signals = strategy_func(data)
                results.append({
                    'success': True,
                    'signals': signals,
                    'data_size': len(data)
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'data_size': len(data)
                })

        return results

    @staticmethod
    def optimize_dataframe_memory(data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage with vectorized operations."""
        optimized = data.copy()

        for col in optimized.columns:
            col_data = optimized[col]

            if col_data.dtype == 'float64':
                # Check if we can downcast to float32
                if (col_data % 1 == 0).all():  # All integers
                    optimized[col] = col_data.astype('int32')
                else:
                    optimized[col] = col_data.astype('float32')
            elif col_data.dtype == 'int64':
                # Check range for downcasting
                if col_data.min() >= 0:
                    if col_data.max() < 2**8:
                        optimized[col] = col_data.astype('uint8')
                    elif col_data.max() < 2**16:
                        optimized[col] = col_data.astype('uint16')
                    elif col_data.max() < 2**32:
                        optimized[col] = col_data.astype('uint32')
                    else:
                        optimized[col] = col_data.astype('uint64')
                else:
                    if col_data.min() >= -2**7 and col_data.max() < 2**7:
                        optimized[col] = col_data.astype('int8')
                    elif col_data.min() >= -2**15 and col_data.max() < 2**15:
                        optimized[col] = col_data.astype('int16')
                    elif col_data.min() >= -2**31 and col_data.max() < 2**31:
                        optimized[col] = col_data.astype('int32')
                    # Keep int64 if needed

        return optimized

    @staticmethod
    def calculate_multiple_indicators_batch(data: pd.DataFrame,
                                          indicators: List[str]) -> pd.DataFrame:
        """Calculate multiple indicators in a single batch operation."""
        result = data.copy()

        for indicator in indicators:
            if indicator == 'sma_20':
                result['sma_20'] = result['close'].rolling(20).mean()
            elif indicator == 'sma_50':
                result['sma_50'] = result['close'].rolling(50).mean()
            elif indicator == 'ema_12':
                result['ema_12'] = result['close'].ewm(span=12).mean()
            elif indicator == 'ema_26':
                result['ema_26'] = result['close'].ewm(span=26).mean()
            elif indicator == 'rsi':
                result['rsi'] = VectorizedTradingOps.calculate_rsi_vectorized(result['close'])
            elif indicator == 'macd':
                macd, signal, hist = VectorizedTradingOps.calculate_macd_vectorized(result['close'])
                result['macd'] = macd
                result['macd_signal'] = signal
                result['macd_histogram'] = hist
            elif indicator == 'bollinger':
                upper, middle, lower = VectorizedTradingOps.calculate_bollinger_bands_vectorized(result['close'])
                result['bb_upper'] = upper
                result['bb_middle'] = middle
                result['bb_lower'] = lower
            elif indicator == 'stochastic':
                k, d = VectorizedTradingOps.calculate_stochastic_oscillator_vectorized(
                    result['high'], result['low'], result['close']
                )
                result['stoch_k'] = k
                result['stoch_d'] = d
            elif indicator == 'atr':
                result['atr'] = VectorizedTradingOps.calculate_atr_vectorized(
                    result['high'], result['low'], result['close']
                )

        return result


@jit(nopython=True)
def numba_signal_calculation(prices: np.ndarray, threshold: float) -> np.ndarray:
    """Numba-compiled signal calculation for maximum performance."""
    n = len(prices)
    signals = np.zeros(n, dtype=np.int32)

    for i in range(1, n):
        if prices[i] > prices[i-1] * (1 + threshold):
            signals[i] = 1  # Buy signal
        elif prices[i] < prices[i-1] * (1 - threshold):
            signals[i] = -1  # Sell signal
        else:
            signals[i] = 0  # Hold

    return signals


def benchmark_vectorized_vs_iterative():
    """Benchmark vectorized operations vs iterative approaches."""
    import time

    # Generate test data
    np.random.seed(42)
    n = 100000
    prices = 100 + np.random.normal(0, 1, n)

    logger.info(f"Benchmarking with {n} data points...")

    # Vectorized RSI calculation
    start_time = time.time()
    rsi_vectorized = VectorizedTradingOps.calculate_rsi_vectorized(pd.Series(prices))
    vectorized_time = time.time() - start_time

    # Iterative RSI calculation (slow)
    def rsi_iterative(prices, period=14):
        rsi_values = []
        for i in range(len(prices)):
            if i < period:
                rsi_values.append(50.0)  # Neutral RSI
            else:
                gains = []
                losses = []
                for j in range(i-period+1, i+1):
                    change = prices[j] - prices[j-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(-change)

                avg_gain = sum(gains) / period
                avg_loss = sum(losses) / period
                rs = avg_gain / (avg_loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        return rsi_values

    start_time = time.time()
    rsi_iterative_result = rsi_iterative(prices)
    iterative_time = time.time() - start_time

    speedup = iterative_time / vectorized_time

    logger.info(f"Vectorized RSI: {vectorized_time:.3f}s")
    logger.info(f"Iterative RSI: {iterative_time:.3f}s")
    logger.info(f"Speedup: {speedup:.1f}x")

    return {
        'vectorized_time': vectorized_time,
        'iterative_time': iterative_time,
        'speedup': speedup
    }


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_vectorized_vs_iterative()
    logger.info("Vectorized operations benchmark complete!")
