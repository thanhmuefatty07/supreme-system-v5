#!/usr/bin/env python3
"""
Live Trading Engine 2.0 - The Trifecta Integration

Integrates Strategy Framework, Risk Management, and Execution Engine
into a unified production-ready live trading system.

Features:
- Real-time market data processing
- Enterprise risk management with Kelly Criterion and Circuit Breaker
- Smart order routing with slippage protection
- Performance tracking and analytics
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..execution.router import SmartRouter
from ..risk.adaptive_kelly import AdaptiveKellyRiskManager, RiskConfig
from ..strategies.base_strategy import BaseStrategy, Signal
from ..utils.validators import validate_market_data, ValidationError
from ..monitoring.metrics_collector import MetricsCollector
from ..data.live_data_manager import LiveDataManager


logger = logging.getLogger(__name__)


class LiveTradingEngineV2:
    """
    Live Trading Engine 2.0 - Enterprise Integration

    Orchestrates the complete trading flow:
    1. Market Data → Strategy → Signal
    2. Signal → Risk Manager → Position Size
    3. Position Size → Smart Router → Execution
    4. Execution → Performance Tracking → Risk Updates
    """

    def __init__(self, exchange_client, strategy: BaseStrategy, config: Dict[str, Any]):
        """
        Initialize the live trading engine.

        Args:
            exchange_client: Exchange API client (ccxt or similar)
            strategy: Trading strategy instance
            config: Engine configuration
        """
        self.logger = logging.getLogger("LiveEngineV2")
        self.strategy = strategy
        self.exchange = exchange_client

        # INTEGRATION 1: Adaptive Kelly Risk Manager with EWMA tracking
        risk_config_dict = config.get('risk_config', {})
        adaptive_risk_config = RiskConfig(
            initial_win_rate=risk_config_dict.get('initial_win_rate', 0.5),
            initial_reward_risk=risk_config_dict.get('initial_reward_risk', 1.5),
            ewma_alpha=risk_config_dict.get('ewma_alpha', 0.05),
            max_daily_loss_pct=risk_config_dict.get('max_daily_loss_pct', 0.05),
            max_consecutive_losses=risk_config_dict.get('max_consecutive_losses', 3),
            max_risk_per_trade=risk_config_dict.get('max_risk_per_trade', 0.02),
            max_position_pct=risk_config_dict.get('max_position_pct', 0.10)
        )

        self.risk_manager = AdaptiveKellyRiskManager(
            config=adaptive_risk_config,
            current_capital=config.get('initial_capital', 10000.0)
        )

        # INTEGRATION 2: Smart Execution Router with Flush-to-Disk logging
        log_path = config.get('trade_log_path', 'trade_history.jsonl')
        self.router = SmartRouter(exchange_client, log_file=log_path)

        # INTEGRATION 4: Live Data Manager for real-time market data
        data_config = config.get('data_config', {})
        self.data_manager = LiveDataManager(data_config)

        # Configure data streams based on strategy requirements
        self._configure_data_streams(strategy, config)

        # Set up data callbacks to receive market data
        self.data_manager.add_data_callback(self._on_market_data_received)

        # Engine state
        self.is_running = False
        self.current_positions = {}  # symbol -> position_info
        self.trade_history = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # Configuration
        self.config = config
        self.update_interval = config.get('update_interval', 60)  # seconds

        # CRITICAL FIX: Add concurrency protection
        self._state_lock = asyncio.Lock()
        self._event_queue = asyncio.Queue()

        # PERFORMANCE MONITORING: Initialize metrics collector
        self.metrics = MetricsCollector()
        self.metrics.initialize(config.get('initial_capital', 10000.0))

        self.logger.info("LiveTradingEngine V2 initialized with Trifecta Integration, Live Data Manager, and Performance Monitoring")

    async def on_market_update(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        The Heartbeat of the System - Process market update and execute trades.

        Flow:
        1. Generate Signal from Strategy
        2. Validate Signal
        3. Risk Check (Kelly + Circuit Breaker)
        4. Execute via Smart Router
        5. Update Performance Metrics

        Args:
            market_data: Current market data (OHLCV + metadata)

        Returns:
            Execution result or None if no action taken
        """
        # PERFORMANCE MONITORING: Start latency measurement
        start_time = time.perf_counter()

        # CRITICAL FIX: Validate input first
        try:
            validated_data = validate_market_data(market_data)
        except ValidationError as e:
            self.logger.warning(f"Invalid market data: {e}")
            return None

        symbol = validated_data.symbol

        # CRITICAL FIX: Use lock to protect shared state
        async with self._state_lock:
            try:
                # STEP 1: Generate Signal from Strategy
                signal = self.strategy.generate_signal(validated_data.model_dump())

                if not signal:
                    return None  # No action needed

                # Validate signal
                if not self.strategy.validate_signal(signal):
                    self.logger.warning(f"Invalid signal for {symbol}")
                    return None

                self.logger.info(f"Signal generated: {signal.side} {symbol} @ {signal.price} (strength: {signal.strength:.2f})")

                # STEP 2: Risk Check - Circuit Breaker First
                if not self.risk_manager.can_trade():
                    self.logger.warning(f"Trading Halted by Circuit Breaker: {self.risk_manager.halt_reason}")
                    return {'status': 'REJECTED', 'reason': 'circuit_breaker_active'}

                # Get Position Size using Adaptive Kelly
                kelly_mode = config.get('kelly_mode', 'half')  # 'full', 'half', or 'quarter'
                target_size = self.risk_manager.get_target_size(mode=kelly_mode)

                if target_size <= 0:
                    self.logger.warning(f"Adaptive Kelly returned zero size for {symbol}")
                    return {'status': 'REJECTED', 'reason': 'kelly_zero_size'}

                # Convert size from $ amount to quantity
                quantity = target_size / signal.price
                self.logger.info(f"Risk Manager approved: ${target_size:.2f} (~{quantity:.4f} {symbol})")

                # STEP 3: Execute via Smart Router
                self.logger.info(f"Executing {signal.side} {quantity:.4f} {symbol} via SmartRouter")

                result = await self.router.execute_order(
                    symbol=symbol,
                    side=signal.side,
                    quantity=quantity
                )

                # STEP 4: Post-Trade Processing
                # Convert ExecutionResult to dict for consistency
                result_dict = result.to_dict() if hasattr(result, 'to_dict') else result

                if result_dict.get('status') == 'FILLED':
                    await self._on_trade_executed(signal, result_dict, target_size)

                return result_dict

            except Exception as e:
                self.logger.error(f"Error in market update processing: {e}", exc_info=True)
                return {'status': 'ERROR', 'error': str(e)}

        # PERFORMANCE MONITORING: Record latency
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        self.metrics.record_latency(latency_ms)

    async def _on_trade_executed(self, signal: Signal, execution_result: Dict[str, Any], position_size: float):
        """
        Handle post-trade updates and tracking.

        Args:
            signal: Original signal that triggered the trade
            execution_result: Result from Smart Router
            position_size: Position size in dollars
        """
        symbol = signal.symbol

        # Update position tracking
        if signal.side == 'buy':
            self.current_positions[symbol] = {
                'side': 'LONG',
                'quantity': execution_result.get('filled_quantity', 0),
                'entry_price': execution_result.get('avg_fill_price', signal.price),
                'timestamp': datetime.now(),
                'signal_metadata': signal.metadata
            }
        elif signal.side == 'sell' and symbol in self.current_positions:
            # Close position
            position = self.current_positions.pop(symbol)
            pnl = self._calculate_pnl(position, execution_result)

            # Update Adaptive Kelly with trade result (EWMA feedback loop)
            was_win = pnl > 0
            self.risk_manager.update_performance(was_win, pnl)

            # Update capital tracking for Adaptive Kelly
            self.risk_manager.current_capital += pnl

            # Update performance stats
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            self.total_pnl += pnl

            # PERFORMANCE MONITORING: Record trade in metrics collector
            self.metrics.record_trade(pnl)

            # Notify strategy
            self.strategy.on_order_filled({
                'symbol': symbol,
                'side': signal.side,
                'quantity': execution_result.get('filled_quantity', 0),
                'price': execution_result.get('avg_fill_price', signal.price),
                'pnl': pnl
            })

            self.logger.info(f"Position closed: {symbol}, PnL: ${pnl:.2f}, Total PnL: ${self.total_pnl:.2f}")

        # Record trade history
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal.to_dict(),
            'execution': execution_result,
            'position_size': position_size
        })

    def _calculate_pnl(self, position: Dict[str, Any], exit_result: Dict[str, Any]) -> float:
        """Calculate realized PnL for a closed position."""
        entry_price = position['entry_price']
        exit_price = exit_result.get('avg_fill_price', 0)
        quantity = position['quantity']

        if position['side'] == 'LONG':
            pnl = (exit_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - exit_price) * quantity

        return pnl

    def _get_strategy_win_rate(self) -> float:
        """Get strategy's historical win rate or use default."""
        if self.total_trades > 0:
            return self.winning_trades / self.total_trades
        return 0.5  # Default 50% win rate

    def _get_strategy_rr_ratio(self) -> float:
        """Get strategy's risk/reward ratio or use default."""
        # TODO: Calculate from actual trade history
        return 2.0  # Default 2:1 reward:risk ratio

    async def start(self):
        """Start the live trading engine with real-time data."""
        await self.start_live_trading()

    async def stop(self):
        """Stop the live trading engine."""
        self.is_running = False
        self.logger.info("Live Trading Engine V2 stopped")

        # Close any open positions (optional, can be configurable)
        if self.current_positions:
            self.logger.warning(f"Stopping with {len(self.current_positions)} open positions")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        return {
            'is_running': self.is_running,
            'strategy': self.strategy.name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self._get_strategy_win_rate(),
            'total_pnl': self.total_pnl,
            'current_positions': len(self.current_positions),
            'circuit_breaker_active': self.risk_manager.is_halted,
            'halt_reason': self.risk_manager.halt_reason,
            'portfolio_value': self.risk_manager.current_capital,
            'adaptive_kelly_metrics': {
                'ewma_win_rate': self.risk_manager.ewma_win_rate,
                'ewma_reward_risk': self.risk_manager.ewma_reward_risk,
                'daily_loss_pct': self.risk_manager.daily_loss_pct,
                'consecutive_losses': self.risk_manager.consecutive_losses,
                'can_trade': self.risk_manager.can_trade()
            }
        }

    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get trade history."""
        if limit:
            return self.trade_history[-limit:]
        return self.trade_history

    def _configure_data_streams(self, strategy: BaseStrategy, config: Dict[str, Any]):
        """
        Configure data streams based on strategy requirements.

        Args:
            strategy: Trading strategy instance
            config: Engine configuration
        """
        # Get symbols from config, default to common pairs if not specified
        symbols = config.get('symbols', ['BTCUSDT', 'ETHUSDT'])

        # Get interval from config, default to 1 minute
        interval = config.get('data_interval', '1m')

        # Add streams for each symbol
        for symbol in symbols:
            self.data_manager.add_stream(
                symbol=symbol,
                interval=interval,
                stream_type="kline"  # OHLCV data
            )

        self.logger.info(f"Configured data streams: {symbols} @ {interval}")

    async def _on_market_data_received(self, market_data: Dict[str, Any]):
        """
        Callback for processing incoming market data from live streams.

        Args:
            market_data: Processed market data from LiveDataManager
        """
        try:
            # Process the market data through our trading engine
            await self.on_market_update(market_data)
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")

    async def start_live_trading(self):
        """
        Start live trading with real-time data streams.

        This replaces the mock loop with actual live data streaming.
        """
        self.is_running = True
        self.logger.info("Starting live trading with real-time data streams...")

        try:
            # Start data streaming and trading loop concurrently
            streaming_task = asyncio.create_task(self.data_manager.start_streaming())

            # Wait for both tasks
            await streaming_task

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Error in live trading loop: {e}")
        finally:
            await self._shutdown()

    async def _shutdown(self):
        """Gracefully shutdown all components."""
        self.logger.info("Shutting down live trading engine...")

        # Stop data manager
        await self.data_manager.disconnect()

        # Close any open positions (in real implementation)
        # This would integrate with position management

        self.is_running = False
        self.logger.info("Live trading engine shutdown complete")

    def get_data_status(self) -> Dict[str, Any]:
        """
        Get status of live data streams.

        Returns:
            Dictionary with data connection and stream status
        """
        status = self.data_manager.get_connection_status()

        return {
            'data_connected': status.connected,
            'last_data_time': status.last_message_time.isoformat() if status.last_message_time else None,
            'reconnect_count': status.reconnect_count,
            'messages_received': status.messages_received,
            'data_errors': status.errors_count,
            'uptime_seconds': status.uptime_seconds,
            'buffered_data_count': len(self.data_manager.data_buffer)
        }
