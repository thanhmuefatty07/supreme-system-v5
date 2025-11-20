#!/usr/bin/env python3
"""
Supreme System V5 - Risk Management System

Real implementation of position sizing, risk management, and backtesting.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..strategies.base_strategy import BaseStrategy


class RiskManager:
    """
    Risk management system with position sizing and backtesting capabilities.

    Handles:
    - Position sizing based on risk percentage
    - Stop loss management
    - Maximum drawdown limits
    - Backtesting with realistic transaction costs
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_size: float = 0.1,  # 10% of capital
        stop_loss_pct: float = 0.02,     # 2% stop loss
        take_profit_pct: float = 0.05,   # 5% take profit
        transaction_fee: float = 0.001   # 0.1% fee
    ):
        """
        Initialize risk manager.

        Args:
            initial_capital: Starting capital amount
            max_position_size: Maximum position size as fraction of capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            transaction_fee: Transaction fee as fraction
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.transaction_fee = transaction_fee

        self.logger = logging.getLogger(__name__)

        # Track positions
        self.positions: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

    def calculate_position_size(self, entry_price: float, capital: Optional[float] = None,
                               risk_pct: float = 0.01) -> float:
        """
        Calculate position size based on risk management rules.

        Args:
            entry_price: Entry price for the position
            capital: Capital to use (default: current capital)
            risk_pct: Risk percentage per trade (default: 1%)

        Returns:
            Position size in base currency
        """
        # Use provided capital or current capital
        capital_to_use = capital if capital is not None else self.current_capital

        # Risk amount is risk_pct of capital
        risk_amount = capital_to_use * risk_pct
        position_size = risk_amount / entry_price

        # Ensure position size doesn't exceed available capital
        max_position_by_capital = self.current_capital / entry_price
        position_size = min(position_size, max_position_by_capital)

        return position_size

    def check_stop_loss(self, entry_price: float, current_price: float, side: Union[str, bool]) -> bool:
        """
        Check if stop loss should be triggered.

        Args:
            entry_price: Position entry price
            current_price: Current market price
            side: Position side ('long', 'short', True for long, False for short)

        Returns:
            True if stop loss triggered, False otherwise
        """
        # Convert side to boolean
        if isinstance(side, str):
            is_long = side.lower() == 'long'
        else:
            is_long = side

        if is_long:
            # Long position: loss when price falls
            loss_pct = (entry_price - current_price) / entry_price
        else:
            # Short position: loss when price rises
            loss_pct = (current_price - entry_price) / entry_price

        return abs(loss_pct) >= self.stop_loss_pct

    def check_take_profit(self, entry_price: float, current_price: float, side: Union[str, bool]) -> bool:
        """
        Check if take profit should be triggered.

        Args:
            entry_price: Position entry price
            current_price: Current market price
            side: Position side ('long', 'short', True for long, False for short)

        Returns:
            True if take profit triggered, False otherwise
        """
        # Convert side to boolean
        if isinstance(side, str):
            is_long = side.lower() == 'long'
        else:
            is_long = side

        if is_long:
            # Long position: profit when price rises
            profit_pct = (current_price - entry_price) / entry_price
        else:
            # Short position: profit when price falls
            profit_pct = (entry_price - current_price) / entry_price

        return abs(profit_pct) >= self.take_profit_pct

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy
    ) -> Dict[str, Any]:
        """
        Run backtest with the given strategy and data.

        Args:
            data: Historical market data
            strategy: Trading strategy to test

        Returns:
            Dictionary with backtest results
        """
        self.logger.info("🚀 Starting backtest...")

        # Reset state
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []

        capital_history = [self.initial_capital]

        for i, row in data.iterrows():
            current_price = row['close']
            timestamp = row['timestamp']

            # Generate trading signal
            signal = strategy.generate_signal(data.iloc[:i+1])

            # Check existing positions for exit conditions
            self._check_position_exits(current_price, timestamp)

            # Enter new positions based on signal
            if signal != 0 and not self.positions:
                self._enter_position(signal, current_price, timestamp)

            # Track capital
            capital_history.append(self.current_capital)

        # Close any remaining positions
        if self.positions:
            self._close_all_positions(data.iloc[-1]['close'], data.iloc[-1]['timestamp'])

        # Calculate performance metrics
        results = self._calculate_performance_metrics(capital_history, data)

        self.logger.info("✅ Backtest completed")
        return results

    def _enter_position(self, signal: int, price: float, timestamp):
        """Enter a new position."""
        position_size = self.calculate_position_size(price)

        if position_size <= 0:
            return

        # Calculate fees
        fee_amount = position_size * price * self.transaction_fee
        total_cost = (position_size * price) + fee_amount

        if total_cost > self.current_capital:
            self.logger.warning("Insufficient capital for position")
            return

        position = {
            'entry_price': price,
            'size': position_size,
            'timestamp': timestamp,
            'is_long': signal > 0,
            'fee': fee_amount
        }

        self.positions.append(position)
        self.current_capital -= total_cost

        self.logger.debug(f"📈 Entered {'LONG' if position['is_long'] else 'SHORT'} position: "
                         f"{position_size:.4f} @ ${price:.2f}")

    def _check_position_exits(self, current_price: float, timestamp):
        """Check and execute position exits."""
        positions_to_remove = []

        for i, position in enumerate(self.positions):
            should_exit = False
            exit_reason = ""

            # Check stop loss
            if self.check_stop_loss(position['entry_price'], current_price, position['is_long']):
                should_exit = True
                exit_reason = "stop_loss"

            # Check take profit
            elif self.check_take_profit(position['entry_price'], current_price, position['is_long']):
                should_exit = True
                exit_reason = "take_profit"

            if should_exit:
                self._exit_position(position, current_price, timestamp, exit_reason)
                positions_to_remove.append(i)

        # Remove closed positions (in reverse order to maintain indices)
        for i in reversed(positions_to_remove):
            self.positions.pop(i)

    def _exit_position(self, position: Dict, exit_price: float, timestamp, reason: str):
        """Exit a position."""
        entry_value = position['entry_price'] * position['size']
        exit_value = exit_price * position['size']

        # Calculate P&L
        if position['is_long']:
            pnl = exit_value - entry_value
        else:
            pnl = entry_value - exit_value

        # Subtract exit fees
        exit_fee = exit_value * self.transaction_fee
        pnl -= exit_fee
        total_fees = position['fee'] + exit_fee

        # Update capital
        self.current_capital += exit_value - exit_fee

        # Record trade
        trade = {
            'entry_time': position['timestamp'],
            'exit_time': timestamp,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'is_long': position['is_long'],
            'pnl': pnl,
            'total_fees': total_fees,
            'exit_reason': reason
        }

        self.trades.append(trade)

        self.logger.debug(f"📉 Exited {'LONG' if position['is_long'] else 'SHORT'} position: "
                         f"${exit_price:.2f}, P&L: ${pnl:.2f}")

    def _close_all_positions(self, final_price: float, timestamp):
        """Close all remaining positions at final price."""
        for position in self.positions:
            self._exit_position(position, final_price, timestamp, "end_of_test")
        self.positions.clear()

    def _calculate_performance_metrics(self, capital_history: List[float], data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        capital_series = pd.Series(capital_history)

        # Basic metrics
        final_capital = capital_series.iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # Calculate returns
        returns = capital_series.pct_change().dropna()

        # Risk metrics
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

        # Maximum drawdown
        peak = capital_series.expanding().max()
        drawdown = (capital_series - peak) / peak
        max_drawdown = drawdown.min()

        # Trade statistics
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        # Profit factor
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades': self.trades,
            'capital_history': capital_history,
            'data_points': len(data)
        }

    def assess_trade_risk(self, symbol: str, quantity: float, entry_price: float, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess risk for a potential trade.

        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            entry_price: Entry price
            current_data: Current market data

        Returns:
            Dict containing risk assessment results
        """
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, self.capital, 0.01)  # 1% risk

        # Check if trade exceeds position size limit
        if quantity > position_size:
            return {
                'approved': False,
                'risk_score': 1.0,
                'recommended_size': position_size,
                'warnings': ['Position size exceeds recommended limit'],
                'reasons': ['Trade size too large']
            }

        # Basic risk assessment
        risk_amount = quantity * entry_price * 0.01  # 1% risk
        if risk_amount > self.capital * 0.02:  # More than 2% of capital
            return {
                'approved': False,
                'risk_score': 0.8,
                'recommended_size': position_size,
                'warnings': ['Risk amount too high'],
                'reasons': ['Exceeds 2% capital risk limit']
            }

        return {
            'approved': True,
            'risk_score': 0.2,
            'recommended_size': quantity,
            'warnings': [],
            'reasons': ['Trade approved']
        }
