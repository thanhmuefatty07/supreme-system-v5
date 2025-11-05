#!/usr/bin/env python3
"""
🚀 Supreme System V5 - Paper Trading Validation Pipeline

Advanced paper trading simulation framework for:
- Live market data simulation with realistic conditions
- Order execution simulation with slippage and latency
- Risk management validation under live-like conditions
- Performance comparison against claimed metrics
- Statistical validation of trading signals in real-time

Features:
- Real-time market data simulation (live-like feed)
- Order book simulation with depth and liquidity
- Slippage modeling based on order size and market conditions
- Network latency simulation
- Comprehensive performance analytics
"""

import asyncio
import json
import os
import random
import statistics
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np

# Add python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from supreme_system_v5.strategies import ScalpingStrategy


class MarketDataSimulator:
    """Realistic market data simulator with live-like characteristics"""

    def __init__(self, symbol: str = "ETH-USDT", volatility_regime: str = "normal"):
        self.symbol = symbol
        self.volatility_regime = volatility_regime

        # Market parameters based on regime
        self.regime_params = {
            'low': {
                'base_volatility': 0.001,
                'trend_strength': 0.00002,
                'spread_bps': 2,
                'depth_multiplier': 1.2
            },
            'normal': {
                'base_volatility': 0.002,
                'trend_strength': 0.00005,
                'spread_bps': 5,
                'depth_multiplier': 1.0
            },
            'high': {
                'base_volatility': 0.005,
                'trend_strength': 0.0001,
                'spread_bps': 15,
                'depth_multiplier': 0.7
            },
            'extreme': {
                'base_volatility': 0.012,
                'trend_strength': 0.0003,
                'spread_bps': 50,
                'depth_multiplier': 0.3
            }
        }

        self.params = self.regime_params[volatility_regime]
        self.current_price = 45000.0
        self.last_update_time = time.time()

        # Order book simulation
        self.order_book = {
            'bids': [],  # List of [price, size] tuples
            'asks': []
        }

        # Market statistics
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)

    def generate_tick(self) -> Dict[str, Any]:
        """Generate a single market tick with realistic characteristics"""
        current_time = time.time()

        # Time-based price movement
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time

        # Generate price movement
        trend_component = self.params['trend_strength'] * time_delta * random.choice([-1, 1])
        volatility_component = random.gauss(0, self.params['base_volatility'])
        micro_noise = random.gauss(0, self.params['base_volatility'] * 0.1)

        price_change = trend_component + volatility_component + micro_noise
        self.current_price *= (1 + price_change)

        # Generate spread
        spread = self.current_price * (self.params['spread_bps'] / 10000)
        bid_price = self.current_price - (spread / 2)
        ask_price = self.current_price + (spread / 2)

        # Generate volume (realistic distribution)
        base_volume = random.lognormvariate(6, 1)  # Log-normal distribution for volume
        volume = base_volume * (1 + abs(volatility_component) * 10)  # Higher volume in volatile conditions

        # Update order book
        self._update_order_book(bid_price, ask_price, volume)

        # Create tick data
        tick = {
            'timestamp': current_time,
            'symbol': self.symbol,
            'price': round(self.current_price, 2),
            'bid': round(bid_price, 2),
            'ask': round(ask_price, 2),
            'volume': round(volume, 2),
            'spread_bps': self.params['spread_bps'],
            'volatility_regime': self.volatility_regime,
            'order_book': {
                'bids': self.order_book['bids'][:5],  # Top 5 levels
                'asks': self.order_book['asks'][:5]
            }
        }

        # Update history
        self.price_history.append(tick['price'])
        self.volume_history.append(tick['volume'])

        return tick

    def _update_order_book(self, bid_price: float, ask_price: float, trade_volume: float):
        """Update order book with realistic depth"""
        # Clear old order book
        self.order_book['bids'] = []
        self.order_book['asks'] = []

        # Generate bid side (5 levels)
        for i in range(5):
            level_price = bid_price - (i * self.params['spread_bps'] * bid_price / 10000 * 0.5)
            level_size = random.uniform(0.1, 5.0) * self.params['depth_multiplier']
            level_size *= (1 - i * 0.3)  # Decrease size with distance from best bid
            self.order_book['bids'].append([round(level_price, 2), round(level_size, 3)])

        # Generate ask side (5 levels)
        for i in range(5):
            level_price = ask_price + (i * self.params['spread_bps'] * ask_price / 10000 * 0.5)
            level_size = random.uniform(0.1, 5.0) * self.params['depth_multiplier']
            level_size *= (1 - i * 0.3)  # Decrease size with distance from best ask
            self.order_book['asks'].append([round(level_price, 2), round(level_size, 3)])

    def get_market_stats(self) -> Dict[str, Any]:
        """Get current market statistics"""
        if not self.price_history:
            return {}

        prices = list(self.price_history)
        volumes = list(self.volume_history)

        return {
            'current_price': self.current_price,
            'price_change_1m': ((prices[-1] - prices[0]) / prices[0]) * 100 if len(prices) > 1 else 0,
            'avg_volume': statistics.mean(volumes) if volumes else 0,
            'volatility': statistics.stdev(prices) / statistics.mean(prices) if len(prices) > 1 else 0,
            'spread_bps': self.params['spread_bps'],
            'liquidity_score': self._calculate_liquidity_score()
        }

    def _calculate_liquidity_score(self) -> float:
        """Calculate market liquidity score (0-100)"""
        if not self.order_book['bids'] or not self.order_book['asks']:
            return 0

        # Liquidity based on order book depth
        bid_depth = sum(size for _, size in self.order_book['bids'])
        ask_depth = sum(size for _, size in self.order_book['asks'])

        avg_depth = (bid_depth + ask_depth) / 2
        spread_penalty = self.params['spread_bps'] / 100  # Normalize spread

        # Higher depth and lower spread = higher liquidity
        liquidity_score = (avg_depth * 10) - spread_penalty

        return max(0, min(100, liquidity_score))


class OrderExecutionSimulator:
    """Realistic order execution simulator with slippage and latency"""

    def __init__(self, market_simulator: MarketDataSimulator):
        self.market_simulator = market_simulator

        # Execution parameters
        self.base_latency_ms = 5  # Base network latency
        self.slippage_model = 'realistic'  # 'none', 'fixed', 'realistic'

        # Execution statistics
        self.execution_history = []
        self.slippage_history = []
        self.latency_history = []

    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an order with realistic simulation"""
        start_time = time.time()

        # Simulate network latency
        network_latency = random.gauss(self.base_latency_ms, 2) / 1000  # Convert to seconds
        time.sleep(max(0, network_latency))

        # Get current market state
        market_tick = self.market_simulator.generate_tick()
        market_stats = self.market_simulator.get_market_stats()

        # Calculate execution price with slippage
        execution_price = self._calculate_execution_price(order, market_tick, market_stats)

        # Simulate partial fills (for market orders)
        fill_percentage = self._calculate_fill_percentage(order, market_stats)

        # Calculate total execution time
        execution_time = time.time() - start_time

        # Create execution result
        execution = {
            'order_id': order.get('order_id', f"order_{int(time.time())}"),
            'timestamp': time.time(),
            'symbol': order['symbol'],
            'side': order['side'],
            'order_type': order['type'],
            'requested_price': order.get('price'),
            'requested_size': order['size'],
            'execution_price': execution_price,
            'executed_size': order['size'] * fill_percentage,
            'fill_percentage': fill_percentage,
            'slippage_bps': self._calculate_slippage_bps(order, execution_price),
            'execution_latency_ms': execution_time * 1000,
            'market_conditions': {
                'volatility_regime': market_tick['volatility_regime'],
                'liquidity_score': market_stats.get('liquidity_score', 0),
                'spread_bps': market_tick['spread_bps']
            }
        }

        # Record execution
        self.execution_history.append(execution)
        self.slippage_history.append(execution['slippage_bps'])
        self.latency_history.append(execution['execution_latency_ms'])

        return execution

    def _calculate_execution_price(self, order: Dict[str, Any],
                                 market_tick: Dict[str, Any],
                                 market_stats: Dict[str, Any]) -> float:
        """Calculate execution price with slippage"""
        if order['type'] == 'market':
            # Market order execution
            if order['side'] == 'buy':
                # Buy at ask with slippage
                base_price = market_tick['ask']
            else:
                # Sell at bid with slippage
                base_price = market_tick['bid']

        elif order['type'] == 'limit':
            # Limit order - check if executable
            if order['side'] == 'buy' and order['price'] >= market_tick['ask']:
                base_price = min(order['price'], market_tick['ask'])
            elif order['side'] == 'sell' and order['price'] <= market_tick['bid']:
                base_price = max(order['price'], market_tick['bid'])
            else:
                return 0  # Order not executable

        else:
            return market_tick['price']  # Fallback

        # Apply slippage based on model
        slippage = self._calculate_slippage(order, market_tick, market_stats)
        execution_price = base_price * (1 + slippage)

        return round(execution_price, 2)

    def _calculate_slippage(self, order: Dict[str, Any],
                          market_tick: Dict[str, Any],
                          market_stats: Dict[str, Any]) -> float:
        """Calculate price slippage in basis points"""

        if self.slippage_model == 'none':
            return 0

        # Base slippage components
        size_impact = (order['size'] / 10) * 0.001  # Size-based slippage (0.1% per 10 units)

        # Volatility impact
        volatility_regime = market_tick['volatility_regime']
        volatility_multipliers = {
            'low': 0.5,
            'normal': 1.0,
            'high': 2.0,
            'extreme': 5.0
        }
        volatility_impact = random.gauss(0, 0.001) * volatility_multipliers.get(volatility_regime, 1.0)

        # Liquidity impact (inverse relationship)
        liquidity_score = market_stats.get('liquidity_score', 50)
        liquidity_impact = (100 - liquidity_score) / 100 * 0.002  # Up to 0.2% in illiquid markets

        # Spread impact
        spread_impact = market_tick['spread_bps'] / 10000 * random.uniform(0.5, 1.5)

        # Random market micro-movements
        market_noise = random.gauss(0, 0.0005)

        total_slippage = size_impact + volatility_impact + liquidity_impact + spread_impact + market_noise

        # Direction based on order side (adverse to trader)
        if order['side'] == 'buy':
            return abs(total_slippage)  # Buy orders get worse prices (higher)
        else:
            return -abs(total_slippage)  # Sell orders get worse prices (lower)

    def _calculate_fill_percentage(self, order: Dict[str, Any],
                                 market_stats: Dict[str, Any]) -> float:
        """Calculate order fill percentage (0-1)"""
        if order['type'] == 'limit':
            return 1.0  # Limit orders either fill completely or not at all

        # Market orders may have partial fills
        liquidity_score = market_stats.get('liquidity_score', 50)

        # Higher liquidity = higher fill rate
        base_fill_rate = min(1.0, liquidity_score / 100 + 0.5)

        # Large orders have lower fill rates
        size_penalty = min(0.3, order['size'] / 100)  # Max 30% penalty for large orders

        fill_rate = base_fill_rate - size_penalty

        # Add some randomness
        fill_rate += random.gauss(0, 0.05)
        fill_rate = max(0.1, min(1.0, fill_rate))  # Clamp between 10% and 100%

        return fill_rate

    def _calculate_slippage_bps(self, order: Dict[str, Any], execution_price: float) -> float:
        """Calculate slippage in basis points"""
        if not order.get('price') or execution_price == 0:
            return 0

        slippage = ((execution_price - order['price']) / order['price']) * 10000  # Convert to bps
        return round(slippage, 2)

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {}

        latencies = [e['execution_latency_ms'] for e in self.execution_history]
        slippages = [e['slippage_bps'] for e in self.execution_history]
        fill_rates = [e['fill_percentage'] for e in self.execution_history]

        return {
            'total_executions': len(self.execution_history),
            'avg_latency_ms': statistics.mean(latencies),
            'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)],
            'avg_slippage_bps': statistics.mean(slippages),
            'p95_slippage_bps': sorted(slippages)[int(len(slippages) * 0.95)] if slippages else 0,
            'avg_fill_rate': statistics.mean(fill_rates),
            'successful_executions': len([e for e in self.execution_history if e['executed_size'] > 0])
        }


class PaperTradingValidator:
    """Complete paper trading validation pipeline"""

    def __init__(self, symbol: str = "ETH-USDT", duration_minutes: int = 60):
        self.symbol = symbol
        self.duration_minutes = duration_minutes
        self.output_dir = Path("run_artifacts")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.market_simulator = MarketDataSimulator(symbol)
        self.execution_simulator = OrderExecutionSimulator(self.market_simulator)
        self.strategy = None

        # Trading state
        self.portfolio = {
            'cash': 100000.0,  # Starting capital
            'position': 0.0,   # Current position size
            'entry_price': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0
        }

        # Results tracking
        self.signals = []
        self.orders = []
        self.executions = []
        self.portfolio_history = []

        # Performance metrics
        self.baseline_performance = {
            'target_win_rate': 0.689,
            'target_sharpe': 2.47,
            'max_slippage_bps': 50,  # Maximum acceptable slippage
            'max_latency_ms': 20     # Maximum acceptable latency
        }

    async def run_paper_trading_validation(self) -> Dict[str, Any]:
        """Execute complete paper trading validation"""
        print("🚀 SUPREME SYSTEM V5 - PAPER TRADING VALIDATION")
        print("=" * 60)
        print(f"Symbol: {self.symbol}")
        print(f"Duration: {self.duration_minutes} minutes")
        print(f"Starting Capital: ${self.portfolio['cash']:,.2f}")
        print()

        # Initialize strategy
        await self._initialize_strategy()

        # Run trading simulation
        await self._execute_paper_trading()

        # Generate comprehensive results
        results = self._generate_validation_results()

        # Save artifacts
        artifacts = self._save_artifacts(results)

        print("
✅ Paper trading validation completed"        print(f"📁 Artifacts: {len(artifacts)} files generated")

        return {
            'success': True,
            'duration_minutes': self.duration_minutes,
            'portfolio_final': self.portfolio,
            'trading_stats': results['trading_stats'],
            'execution_stats': results['execution_stats'],
            'validation_results': results['validation_results'],
            'artifacts': artifacts
        }

    async def _initialize_strategy(self):
        """Initialize trading strategy"""
        print("🔧 Initializing trading strategy...")

        config = {
            'symbol': self.symbol,
            'position_size_pct': 0.02,
            'stop_loss_pct': 0.01,
            'take_profit_pct': 0.02,
            'ema_period': 14,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }

        self.strategy = ScalpingStrategy(config)
        print("✅ Strategy initialized")

    async def _execute_paper_trading(self):
        """Execute paper trading simulation"""
        print("🏃 Executing paper trading simulation...")

        start_time = time.time()
        end_time = start_time + (self.duration_minutes * 60)

        tick_count = 0

        while time.time() < end_time:
            # Generate market tick
            market_tick = self.market_simulator.generate_tick()
            tick_count += 1

            # Record portfolio state periodically
            if tick_count % 100 == 0:  # Every 100 ticks
                self.portfolio_history.append({
                    'timestamp': market_tick['timestamp'],
                    'portfolio_value': self._calculate_portfolio_value(market_tick['price']),
                    'cash': self.portfolio['cash'],
                    'position': self.portfolio['position'],
                    'unrealized_pnl': self._calculate_unrealized_pnl(market_tick['price'])
                })

            # Generate trading signal
            signal = self.strategy.add_price_data(
                market_tick['price'],
                market_tick['volume'],
                market_tick['timestamp']
            )

            if signal:
                self.signals.append({
                    'timestamp': market_tick['timestamp'],
                    'signal': signal,
                    'market_data': market_tick
                })

                # Convert signal to order
                order = self._signal_to_order(signal, market_tick)
                if order:
                    self.orders.append(order)

                    # Execute order
                    execution = self.execution_simulator.execute_order(order)
                    self.executions.append(execution)

                    # Update portfolio
                    self._update_portfolio(execution)

                    # Progress reporting
                    if len(self.executions) % 10 == 0:
                        pnl = self.portfolio['total_pnl']
                        win_rate = self.portfolio['winning_trades'] / self.portfolio['total_trades'] if self.portfolio['total_trades'] > 0 else 0
                        print(".2f")

            # Small delay to simulate real-time processing
            await asyncio.sleep(0.01)  # 10ms between ticks

        print("✅ Paper trading simulation completed")

    def _signal_to_order(self, signal: Dict[str, Any], market_tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert trading signal to order"""
        if signal['action'] not in ['BUY', 'SELL']:
            return None

        # Calculate position size
        portfolio_value = self._calculate_portfolio_value(market_tick['price'])
        position_size_pct = 0.02  # 2% of portfolio
        position_size_usd = portfolio_value * position_size_pct

        if signal['action'] == 'BUY':
            # Buy order
            if self.portfolio['position'] >= 0:  # No short position
                price = market_tick['ask']
                size = position_size_usd / price
            else:
                return None  # Close short position first
        else:
            # Sell order
            if self.portfolio['position'] > 0:  # Have long position
                price = market_tick['bid']
                size = abs(self.portfolio['position'])  # Close entire position
            else:
                return None  # No position to sell

        return {
            'order_id': f"order_{int(time.time() * 1000)}",
            'symbol': self.symbol,
            'side': 'buy' if signal['action'] == 'BUY' else 'sell',
            'type': 'market',  # Use market orders for speed
            'size': size,
            'price': price,
            'signal': signal
        }

    def _update_portfolio(self, execution: Dict[str, Any]):
        """Update portfolio based on execution"""
        executed_size = execution['executed_size']
        execution_price = execution['execution_price']

        if execution['side'] == 'buy':
            # Buying
            cost = executed_size * execution_price
            if self.portfolio['cash'] >= cost:
                self.portfolio['cash'] -= cost
                self.portfolio['position'] += executed_size
                self.portfolio['entry_price'] = execution_price
                self.portfolio['total_trades'] += 1

        elif execution['side'] == 'sell':
            # Selling
            proceeds = executed_size * execution_price
            self.portfolio['cash'] += proceeds
            self.portfolio['position'] -= executed_size

            # Calculate P&L
            if self.portfolio['position'] == 0:  # Closed position
                entry_cost = abs(self.portfolio['entry_price'] * executed_size)
                pnl = proceeds - entry_cost
                self.portfolio['total_pnl'] += pnl

                if pnl > 0:
                    self.portfolio['winning_trades'] += 1

                # Reset entry price
                self.portfolio['entry_price'] = 0

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        position_value = self.portfolio['position'] * current_price
        return self.portfolio['cash'] + position_value

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.portfolio['position'] == 0:
            return 0

        position_value = self.portfolio['position'] * current_price
        entry_value = abs(self.portfolio['position']) * self.portfolio['entry_price']

        return position_value - entry_value

    def _generate_validation_results(self) -> Dict[str, Any]:
        """Generate comprehensive validation results"""
        trading_stats = self._calculate_trading_statistics()
        execution_stats = self.execution_simulator.get_execution_stats()

        # Performance validation
        validation_results = {
            'win_rate_validation': self._validate_metric(
                trading_stats.get('win_rate', 0),
                self.baseline_performance['target_win_rate'],
                tolerance=0.10
            ),
            'sharpe_validation': self._validate_metric(
                trading_stats.get('sharpe_ratio', 0),
                self.baseline_performance['target_sharpe'],
                tolerance=0.15
            ),
            'execution_quality_validation': self._validate_execution_quality(execution_stats),
            'portfolio_performance_validation': self._validate_portfolio_performance(trading_stats),
            'overall_validation_passed': False
        }

        # Overall validation
        win_rate_ok = validation_results['win_rate_validation']['passed']
        sharpe_ok = validation_results['sharpe_validation']['passed']
        execution_ok = validation_results['execution_quality_validation']['passed']
        portfolio_ok = validation_results['portfolio_performance_validation']['passed']

        validation_results['overall_validation_passed'] = all([win_rate_ok, sharpe_ok, execution_ok, portfolio_ok])

        return {
            'trading_stats': trading_stats,
            'execution_stats': execution_stats,
            'validation_results': validation_results
        }

    def _calculate_trading_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive trading statistics"""
        if not self.executions:
            return {'error': 'No executions'}

        # Basic metrics
        total_trades = self.portfolio['total_trades']
        winning_trades = self.portfolio['winning_trades']
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L analysis
        total_pnl = self.portfolio['total_pnl']
        final_portfolio_value = self._calculate_portfolio_value(self.market_simulator.current_price)

        # Calculate returns
        returns = []
        if len(self.portfolio_history) > 1:
            for i in range(1, len(self.portfolio_history)):
                prev_value = self.portfolio_history[i-1]['portfolio_value']
                curr_value = self.portfolio_history[i]['portfolio_value']
                if prev_value > 0:
                    ret = (curr_value - prev_value) / prev_value
                    returns.append(ret)

        # Sharpe ratio calculation
        if returns and len(returns) > 1:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            sharpe_ratio = avg_return / std_return * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_portfolio_value': final_portfolio_value,
            'total_return_pct': ((final_portfolio_value - 100000) / 100000) * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown(),
            'avg_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'profit_factor': self._calculate_profit_factor()
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history"""
        if not self.portfolio_history:
            return 0

        values = [h['portfolio_value'] for h in self.portfolio_history]
        peak = values[0]
        max_drawdown = 0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown * 100  # Convert to percentage

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        # This would require tracking individual trade P&L
        # Simplified calculation based on total P&L
        if self.portfolio['total_pnl'] > 0:
            return 1.5  # Placeholder - would need detailed trade analysis
        else:
            return 0.8

    def _validate_metric(self, actual: float, target: float, tolerance: float) -> Dict[str, Any]:
        """Validate a metric against target with tolerance"""
        lower_bound = target * (1 - tolerance)
        upper_bound = target * (1 + tolerance)

        return {
            'actual': actual,
            'target': target,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'within_tolerance': lower_bound <= actual <= upper_bound,
            'deviation_pct': abs(actual - target) / target * 100 if target != 0 else 0,
            'passed': lower_bound <= actual <= upper_bound
        }

    def _validate_execution_quality(self, execution_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Validate execution quality"""
        avg_slippage = execution_stats.get('avg_slippage_bps', 0)
        p95_slippage = execution_stats.get('p95_slippage_bps', 0)
        avg_latency = execution_stats.get('avg_latency_ms', 0)
        p95_latency = execution_stats.get('p95_latency_ms', 0)

        slippage_ok = p95_slippage <= self.baseline_performance['max_slippage_bps']
        latency_ok = p95_latency <= self.baseline_performance['max_latency_ms']
        fill_rate_ok = execution_stats.get('avg_fill_rate', 0) >= 0.95

        return {
            'avg_slippage_bps': avg_slippage,
            'p95_slippage_bps': p95_slippage,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'avg_fill_rate': execution_stats.get('avg_fill_rate', 0),
            'slippage_within_limits': slippage_ok,
            'latency_within_limits': latency_ok,
            'fill_rate_acceptable': fill_rate_ok,
            'passed': slippage_ok and latency_ok and fill_rate_ok
        }

    def _validate_portfolio_performance(self, trading_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Validate portfolio performance"""
        total_return = trading_stats.get('total_return_pct', 0)
        max_drawdown = trading_stats.get('max_drawdown', 0)
        sharpe_ratio = trading_stats.get('sharpe_ratio', 0)

        # Performance criteria
        return_ok = total_return >= -5.0  # Allow small losses
        drawdown_ok = max_drawdown <= 10.0  # Max 10% drawdown
        sharpe_ok = sharpe_ratio >= 1.0  # Minimum Sharpe ratio

        return {
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'return_acceptable': return_ok,
            'drawdown_acceptable': drawdown_ok,
            'sharpe_acceptable': sharpe_ok,
            'passed': return_ok and drawdown_ok and sharpe_ok
        }

    def _save_artifacts(self, results: Dict[str, Any]) -> List[str]:
        """Save all test artifacts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts = []

        # Trading statistics
        trading_file = self.output_dir / f"paper_trading_stats_{timestamp}.json"
        with open(trading_file, 'w') as f:
            json.dump(results['trading_stats'], f, indent=2, default=str)
        artifacts.append(str(trading_file))

        # Execution statistics
        execution_file = self.output_dir / f"paper_trading_execution_{timestamp}.json"
        with open(execution_file, 'w') as f:
            json.dump(results['execution_stats'], f, indent=2, default=str)
        artifacts.append(str(execution_file))

        # Validation results
        validation_file = self.output_dir / f"paper_trading_validation_{timestamp}.json"
        with open(validation_file, 'w') as f:
            json.dump(results['validation_results'], f, indent=2, default=str)
        artifacts.append(str(validation_file))

        # Portfolio history
        portfolio_file = self.output_dir / f"paper_trading_portfolio_{timestamp}.json"
        with open(portfolio_file, 'w') as f:
            json.dump(self.portfolio_history, f, indent=2, default=str)
        artifacts.append(str(portfolio_file))

        # Execution history
        executions_file = self.output_dir / f"paper_trading_executions_{timestamp}.json"
        with open(executions_file, 'w') as f:
            json.dump(self.executions, f, indent=2, default=str)
        artifacts.append(str(executions_file))

        # Signals history
        signals_file = self.output_dir / f"paper_trading_signals_{timestamp}.json"
        with open(signals_file, 'w') as f:
            json.dump(self.signals, f, indent=2, default=str)
        artifacts.append(str(signals_file))

        return artifacts


async def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Supreme System V5 Paper Trading Validator")
    parser.add_argument("--symbol", default="ETH-USDT",
                       help="Trading symbol (default: ETH-USDT)")
    parser.add_argument("--duration", type=int, default=60,
                       help="Simulation duration in minutes (default: 60)")
    parser.add_argument("--volatility", default="normal",
                       choices=['low', 'normal', 'high', 'extreme'],
                       help="Market volatility regime (default: normal)")

    args = parser.parse_args()

    # Create validator
    validator = PaperTradingValidator(
        symbol=args.symbol,
        duration_minutes=args.duration
    )

    # Set market volatility regime
    validator.market_simulator.volatility_regime = args.volatility

    # Run validation
    results = await validator.run_paper_trading_validation()

    # Print summary
    print("\n" + "=" * 80)
    print("🎯 PAPER TRADING VALIDATION RESULTS")
    print("=" * 80)

    if results['success']:
        portfolio = results['portfolio_final']
        trading = results['trading_stats']
        execution = results['execution_stats']
        validation = results['validation_results']

        print("✅ Validation Status: PASSED" if validation['overall_validation_passed'] else "❌ Validation Status: FAILED")
        print("
📊 Portfolio Performance:"        print(".2f"        print(".1%"        print(".2f"        print(f"   Sharpe Ratio: {trading['sharpe_ratio']:.2f}")
        print(f"   Total Trades: {trading['total_trades']}")

        print("
⚡ Execution Quality:"        print(".1f"        print(".1f"        print(".1%"
        print("
🔬 Validation Results:"        print(f"   Win Rate: {'✅ PASSED' if validation['win_rate_validation']['passed'] else '❌ FAILED'}")
        print(f"   Sharpe Ratio: {'✅ PASSED' if validation['sharpe_validation']['passed'] else '❌ FAILED'}")
        print(f"   Execution Quality: {'✅ PASSED' if validation['execution_quality_validation']['passed'] else '❌ FAILED'}")
        print(f"   Portfolio Performance: {'✅ PASSED' if validation['portfolio_performance_validation']['passed'] else '❌ FAILED'}")

        print(f"\n📁 Artifacts saved: {len(results['artifacts'])} files")

    else:
        print(f"❌ Validation Failed: {results.get('error', 'Unknown error')}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
