"""
Supreme System V5 - Core Trading Engine
ULTRA SFL implementation with hybrid Python+Rust architecture
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

# Third-party imports
import numpy as np
import polars as pl
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Try to import Rust engine (graceful fallback)
try:
    import supreme_engine_rs
    RUST_ENGINE_AVAILABLE = True
    logger.info("✅ Rust engine loaded successfully")
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    logger.warning("⚠️ Rust engine not available, using Python fallback")

# Metrics
TRADE_COUNTER = Counter('supreme_trades_total', 'Total number of trades executed')
PNL_GAUGE = Gauge('supreme_pnl_current', 'Current P&L in USD')
LATENCY_HISTOGRAM = Histogram('supreme_latency_seconds', 'Processing latency')

@dataclass
class SystemConfig:
    """
    ULTRA SFL system configuration with validation
    Centralized configuration management
    """
    # Trading parameters
    max_position_size: float = 0.01
    stop_loss_percent: float = 0.005  # 0.5%
    take_profit_percent: float = 0.002  # 0.2%
    
    # System parameters
    max_memory_mb: int = 3500  # i3-4GB limit
    max_cpu_percent: float = 80.0
    update_interval_ms: int = 100  # 100ms for scalping
    
    # Exchange configuration
    primary_exchange: str = "OKX"
    backup_exchange: str = "BINANCE"
    trading_symbols: List[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT"])
    
    # Monitoring
    metrics_port: int = 8000
    log_level: str = "INFO"
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters
        Returns list of validation errors
        """
        errors = []
        
        if self.max_position_size <= 0 or self.max_position_size > 1.0:
            errors.append("max_position_size must be between 0 and 1.0")
            
        if self.max_memory_mb > 4000:
            errors.append("max_memory_mb exceeds i3-4GB hardware limit")
            
        if self.update_interval_ms < 10:
            errors.append("update_interval_ms too aggressive (min 10ms)")
            
        if not self.trading_symbols:
            errors.append("trading_symbols cannot be empty")
            
        return errors

@dataclass 
class MarketData:
    """
    Real-time market data structure
    Optimized for scalping with minimal latency
    """
    symbol: str
    timestamp: float
    price: float
    volume: float
    bid: float
    ask: float
    spread: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.spread = abs(self.ask - self.bid) if self.ask and self.bid else 0.0

@dataclass
class TradingSignal:
    """
    Trading signal with confidence and risk parameters
    """
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)

class SupremeCore:
    """
    ULTRA SFL Core Trading Engine
    Production-ready hybrid Python+Rust implementation
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize Supreme trading core"""
        self.config = config or SystemConfig()
        self.running = False
        self.market_data: Dict[str, MarketData] = {}
        self.active_positions: Dict[str, Dict] = {}
        self.performance_metrics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            raise ValueError(f"Configuration errors: {config_errors}")
            
        logger.info("✅ Supreme Core initialized successfully")
        logger.info(f"🔧 Rust engine: {'Available' if RUST_ENGINE_AVAILABLE else 'Fallback mode'}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information
        Used for health checks and monitoring
        """
        import psutil
        
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            'version': '5.0.0',
            'rust_engine': RUST_ENGINE_AVAILABLE,
            'python_version': sys.version_info[:2],
            'memory_usage_mb': round(memory.used / 1024 / 1024, 1),
            'memory_percent': memory.percent,
            'cpu_percent': cpu_percent,
            'running': self.running,
            'active_symbols': list(self.market_data.keys()),
            'active_positions': len(self.active_positions),
            'total_trades': self.performance_metrics['total_trades'],
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_metrics_server(self):
        """
        Start Prometheus metrics server
        """
        try:
            start_http_server(self.config.metrics_port)
            logger.info(f"📊 Metrics server started on port {self.config.metrics_port}")
        except Exception as e:
            logger.error(f"❌ Metrics server failed: {e}")
    
    def calculate_technical_indicators(self, symbol: str, lookback: int = 100) -> Dict[str, float]:
        """
        Calculate technical indicators for trading decision
        Uses Rust engine if available, Python fallback otherwise
        """
        if not self.market_data.get(symbol):
            return {}

        # Get price history for calculations
        prices = self._get_price_history(symbol, lookback)
        if len(prices) < 14:  # Minimum for RSI
            return {}

        current_price = self.market_data[symbol].price

        if RUST_ENGINE_AVAILABLE:
            # Use Rust engine for ultra-fast calculation
            try:
                import numpy as np

                price_array = np.array(prices, dtype=np.float64)

                # Calculate EMAs (5, 20, 50)
                ema_5 = supreme_engine_rs.fast_ema(price_array, 5)[-1] if len(prices) >= 5 else current_price
                ema_20 = supreme_engine_rs.fast_ema(price_array, 20)[-1] if len(prices) >= 20 else current_price
                ema_50 = supreme_engine_rs.fast_ema(price_array, 50)[-1] if len(prices) >= 50 else current_price

                # Calculate RSI
                rsi_values = supreme_engine_rs.fast_rsi(price_array, 14)
                rsi_14 = rsi_values[-1] if rsi_values else 50.0

                # Calculate MACD
                macd_result = supreme_engine_rs.fast_macd(price_array, 12, 26, 9)
                macd_line, signal_line, histogram = macd_result
                macd = macd_line[-1] - signal_line[-1] if macd_line and signal_line else 0.001

                # Calculate Bollinger Bands
                bb_result = supreme_engine_rs.bollinger_bands(price_array, 20, 2.0)
                upper_bb, sma_20, lower_bb = bb_result

                indicators = {
                    'ema_5': ema_5,
                    'ema_20': ema_20,
                    'ema_50': ema_50,
                    'sma_20': sma_20[-1] if sma_20 else current_price,
                    'rsi_14': rsi_14,
                    'macd': macd,
                    'bb_upper': upper_bb[-1] if upper_bb else current_price * 1.002,
                    'bb_lower': lower_bb[-1] if lower_bb else current_price * 0.998,
                    'current_price': current_price,
                    'price_history_count': len(prices)
                }

                logger.debug(f"⚡ Rust indicators calculated for {symbol}: EMA5={ema_5:.4f}, RSI={rsi_14:.1f}")
                return indicators

            except Exception as e:
                logger.warning(f"Rust indicator calculation failed: {e}, using Python fallback")

        # Python fallback implementation (using technical analysis library)
        try:
            from ta.trend import EMAIndicator, SMAIndicator
            from ta.momentum import RSIIndicator
            from ta.trend import MACD
            from ta.volatility import BollingerBands
            import pandas as pd

            # Create DataFrame for ta library
            df = pd.DataFrame({'close': prices})

            # Calculate EMAs
            ema_5 = EMAIndicator(df['close'], window=5).ema_indicator().iloc[-1] if len(df) >= 5 else current_price
            ema_20 = EMAIndicator(df['close'], window=20).ema_indicator().iloc[-1] if len(df) >= 20 else current_price
            ema_50 = EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1] if len(df) >= 50 else current_price

            # Calculate SMA for comparison
            sma_20 = SMAIndicator(df['close'], window=20).sma_indicator().iloc[-1] if len(df) >= 20 else current_price

            # Calculate RSI
            rsi_14 = RSIIndicator(df['close'], window=14).rsi().iloc[-1] if len(df) >= 14 else 50.0

            # Calculate MACD
            macd_indicator = MACD(df['close'])
            macd = macd_indicator.macd_diff().iloc[-1] if len(df) >= 26 else 0.001

            # Calculate Bollinger Bands
            bb_indicator = BollingerBands(df['close'], window=20, window_dev=2)
            bb_upper = bb_indicator.bollinger_hband().iloc[-1] if len(df) >= 20 else current_price * 1.002
            bb_lower = bb_indicator.bollinger_lband().iloc[-1] if len(df) >= 20 else current_price * 0.998

            indicators = {
                'ema_5': ema_5,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'sma_20': sma_20,
                'rsi_14': rsi_14,
                'macd': macd,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'current_price': current_price,
                'price_history_count': len(prices)
            }

            logger.debug(f"🐍 Python TA indicators calculated for {symbol}: EMA5={ema_5:.4f}, RSI={rsi_14:.1f}")
            return indicators

        except ImportError:
            # Ultimate fallback - simple calculations
            logger.warning("TA library not available, using simple calculations")

            # Simple EMA approximation
            def simple_ema(prices, period):
                if len(prices) < period:
                    return prices[-1] if prices else current_price
                alpha = 2.0 / (period + 1)
                ema = prices[0]
                for price in prices[1:]:
                    ema = alpha * price + (1 - alpha) * ema
                return ema

            ema_5 = simple_ema(prices[-5:] if len(prices) >= 5 else prices, 5)
            ema_20 = simple_ema(prices[-20:] if len(prices) >= 20 else prices, 20)
            ema_50 = simple_ema(prices[-50:] if len(prices) >= 50 else prices, 50)

            # Simple RSI approximation
            def simple_rsi(prices, period=14):
                if len(prices) < period + 1:
                    return 50.0

                gains = []
                losses = []
                for i in range(1, len(prices)):
                    change = prices[i] - prices[i-1]
                    gains.append(max(change, 0))
                    losses.append(max(-change, 0))

                avg_gain = sum(gains[-period:]) / period
                avg_loss = sum(losses[-period:]) / period

                if avg_loss == 0:
                    return 100.0
                rs = avg_gain / avg_loss
                return 100 - (100 / (1 + rs))

            rsi_14 = simple_rsi(prices, 14)

            indicators = {
                'ema_5': ema_5,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'sma_20': sum(prices[-20:]) / min(20, len(prices)) if prices else current_price,
                'rsi_14': rsi_14,
                'macd': 0.001,  # Simplified
                'bb_upper': current_price * 1.02,
                'bb_lower': current_price * 0.98,
                'current_price': current_price,
                'price_history_count': len(prices)
            }

            logger.debug(f"🔧 Simple indicators calculated for {symbol}: EMA5={ema_5:.4f}, RSI={rsi_14:.1f}")
            return indicators

    def _get_price_history(self, symbol: str, max_points: int = 100) -> List[float]:
        """
        Get price history for indicator calculations
        This is a simplified implementation - in production this would
        connect to the data fabric cache
        """
        # Mock price history - in production this would come from cache
        if not hasattr(self, '_price_cache'):
            self._price_cache = {}

        if symbol not in self._price_cache:
            self._price_cache[symbol] = []

        # Add current price to history
        current_price = self.market_data[symbol].price
        self._price_cache[symbol].append(current_price)

        # Keep only recent history
        history = self._price_cache[symbol][-max_points:]

        return history
    
    def generate_trading_signal(self, symbol: str) -> TradingSignal:
        """
        Generate trading signal based on technical analysis
        Core scalping logic implementation
        """
        if symbol not in self.market_data:
            return TradingSignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                reasoning="No market data available"
            )
        
        market_data = self.market_data[symbol]
        indicators = self.calculate_technical_indicators(symbol)
        
        # Scalping strategy logic
        current_price = market_data.price
        sma_5 = indicators.get('sma_5', current_price)
        sma_20 = indicators.get('sma_20', current_price)
        rsi = indicators.get('rsi_14', 50.0)
        
        # Generate signal based on SMA crossover + RSI
        if sma_5 > sma_20 and 30 < rsi < 70:
            # Bullish signal
            stop_loss = current_price * (1 - self.config.stop_loss_percent)
            take_profit = current_price * (1 + self.config.take_profit_percent)
            
            return TradingSignal(
                symbol=symbol,
                action="BUY",
                confidence=0.75,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"SMA5({sma_5:.4f}) > SMA20({sma_20:.4f}), RSI={rsi:.1f}"
            )
        
        elif sma_5 < sma_20 and 30 < rsi < 70:
            # Bearish signal
            stop_loss = current_price * (1 + self.config.stop_loss_percent)
            take_profit = current_price * (1 - self.config.take_profit_percent)
            
            return TradingSignal(
                symbol=symbol,
                action="SELL",
                confidence=0.75,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"SMA5({sma_5:.4f}) < SMA20({sma_20:.4f}), RSI={rsi:.1f}"
            )
        
        else:
            # No clear signal
            return TradingSignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.3,
                entry_price=current_price,
                stop_loss=current_price,
                take_profit=current_price,
                reasoning=f"No clear trend, RSI={rsi:.1f}"
            )
    
    def update_market_data(self, symbol: str, price: float, volume: float, bid: float = 0.0, ask: float = 0.0):
        """
        Update real-time market data
        Optimized for high-frequency updates
        """
        self.market_data[symbol] = MarketData(
            symbol=symbol,
            timestamp=time.time(),
            price=price,
            volume=volume,
            bid=bid or price * 0.9999,  # Mock bid
            ask=ask or price * 1.0001,  # Mock ask
        )
        
        # Update metrics
        LATENCY_HISTOGRAM.observe(time.time() - self.market_data[symbol].timestamp)
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """
        Execute trading signal
        In production: integrate with exchange APIs
        """
        if signal.action == "HOLD":
            return False
            
        # Risk validation
        if signal.confidence < 0.5:
            logger.warning(f"Low confidence signal rejected: {signal.symbol} {signal.confidence:.2f}")
            return False
            
        # Position size calculation
        position_size = self.config.max_position_size
        
        # Mock trade execution (in production: call exchange API)
        trade_id = f"{signal.symbol}_{int(time.time())}"
        
        self.active_positions[trade_id] = {
            'symbol': signal.symbol,
            'action': signal.action,
            'size': position_size,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'timestamp': signal.timestamp,
            'reasoning': signal.reasoning
        }
        
        # Update metrics
        TRADE_COUNTER.inc()
        self.performance_metrics['total_trades'] += 1
        
        logger.info(f"✅ Trade executed: {trade_id} - {signal.action} {signal.symbol} @ {signal.entry_price:.4f}")
        return True
    
    async def trading_loop(self):
        """
        Main trading loop for scalping
        Optimized for minimal latency
        """
        logger.info("🏁 Starting trading loop")
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Process each symbol
                for symbol in self.config.trading_symbols:
                    if symbol in self.market_data:
                        # Generate signal
                        signal = self.generate_trading_signal(symbol)
                        
                        # Execute if valid
                        if signal.action != "HOLD":
                            self.execute_trade(signal)
                
                # Calculate loop latency
                loop_latency = time.time() - loop_start
                LATENCY_HISTOGRAM.observe(loop_latency)
                
                # Maintain target interval
                sleep_time = max(0, self.config.update_interval_ms / 1000 - loop_latency)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(1)  # Error recovery
    
    async def start(self):
        """
        Start the trading system
        """
        logger.info("🚀 Starting Supreme System V5")
        
        # Start metrics server
        await self.start_metrics_server()
        
        # Initialize system
        self.running = True
        
        # Mock market data (in production: connect to real exchanges)
        self.update_market_data("BTC-USDT", 35000.0, 1000000.0)
        self.update_market_data("ETH-USDT", 1800.0, 500000.0)
        
        # Start trading loop
        await self.trading_loop()
    
    async def stop(self):
        """
        Gracefully stop the trading system
        """
        logger.info("🛑 Stopping Supreme System V5")
        self.running = False
        
        # Close all positions (mock)
        if self.active_positions:
            logger.info(f"Closing {len(self.active_positions)} active positions")
            self.active_positions.clear()
        
        logger.info("✅ Supreme System stopped gracefully")

class SupremeSystem:
    """
    Main system orchestrator
    High-level interface for the trading system
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize Supreme System with optional config file"""
        self.config = SystemConfig()
        self.core = SupremeCore(self.config)
        self.start_time = datetime.now()
        
        logger.info("🎆 Supreme System V5 initialized")
        logger.info(f"📅 Start time: {self.start_time}")
        logger.info(f"💻 Hardware target: i3-4GB optimized")
        logger.info(f"🔗 Architecture: Hybrid Python+Rust")
    
    async def run(self):
        """
        Run the complete trading system
        """
        try:
            logger.info("🏁 Supreme System starting...")
            await self.core.start()
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown signal received")
            await self.core.stop()
        except Exception as e:
            logger.error(f"💥 System error: {e}")
            await self.core.stop()
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status for monitoring
        """
        uptime = datetime.now() - self.start_time
        system_info = self.core.get_system_info()
        
        return {
            'system': 'Supreme System V5',
            'version': '5.0.0',
            'status': 'running' if self.core.running else 'stopped',
            'uptime_seconds': uptime.total_seconds(),
            'core_info': system_info,
            'performance': self.core.performance_metrics
        }

# Factory function for easy instantiation
def create_supreme_system(config: Optional[SystemConfig] = None) -> SupremeSystem:
    """
    Factory function to create Supreme System instance
    """
    return SupremeSystem()

# Export main classes
__all__ = [
    'SupremeCore',
    'SupremeSystem', 
    'SystemConfig',
    'MarketData',
    'TradingSignal',
    'create_supreme_system'
]
