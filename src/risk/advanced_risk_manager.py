#!/usr/bin/env python3
"""
Supreme System V5 - Advanced Risk Management

Enterprise-grade risk management for live trading operations.
Includes portfolio optimization, dynamic position sizing, and real-time risk monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .risk_manager import RiskManager


class PortfolioMetrics:
    """Portfolio performance and risk metrics."""

    def __init__(self):
        self.total_value = 0.0
        self.cash = 0.0
        self.positions_value = 0.0
        self.daily_return = 0.0
        self.cumulative_return = 0.0
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.value_at_risk = 0.0
        self.expected_shortfall = 0.0

    def calculate_metrics(self, returns: pd.Series, positions: Dict[str, Any]) -> None:
        """Calculate comprehensive portfolio metrics."""
        if len(returns) < 2:
            return

        # Basic returns
        self.daily_return = returns.iloc[-1]
        self.cumulative_return = (1 + returns).prod() - 1

        # Volatility (annualized)
        self.volatility = returns.std() * np.sqrt(252)  # 252 trading days

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        if excess_returns.std() > 0:
            self.sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        self.max_drawdown = drawdowns.min()

        # Value at Risk (95% confidence, 1-day)
        if len(returns) > 30:
            self.value_at_risk = np.percentile(returns, 5)

            # Expected Shortfall (Conditional VaR)
            var_returns = returns[returns <= self.value_at_risk]
            if len(var_returns) > 0:
                self.expected_shortfall = var_returns.mean()


class DynamicPositionSizer:
    """Dynamic position sizing based on volatility and correlation."""

    def __init__(self, base_risk_pct: float = 0.01):
        self.base_risk_pct = base_risk_pct  # 1% base risk per trade
        self.volatility_lookback = 20
        self.correlation_lookback = 30

    def calculate_optimal_size(
        self,
        capital: float,
        price: float,
        volatility: float,
        portfolio_volatility: float,
        symbol: str,
        current_positions: Dict[str, Any],
        market_regime: str = 'normal'
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion + risk adjustments.

        Args:
            capital: Available capital
            price: Asset price
            volatility: Asset volatility
            portfolio_volatility: Current portfolio volatility
            symbol: Trading symbol
            current_positions: Current portfolio positions
            market_regime: 'normal', 'volatile', 'crisis'

        Returns:
            Optimal position size
        """
        # Base Kelly position size
        kelly_size = self._kelly_criterion(volatility)

        # Adjust for market regime
        regime_multiplier = self._market_regime_adjustment(market_regime)

        # Adjust for portfolio diversification
        diversification_multiplier = self._diversification_adjustment(
            current_positions, symbol
        )

        # Adjust for volatility clustering
        volatility_multiplier = self._volatility_adjustment(volatility, portfolio_volatility)

        # Apply risk limits
        max_risk_pct = min(self.base_risk_pct * regime_multiplier *
                          diversification_multiplier * volatility_multiplier, 0.05)  # Max 5%

        # Calculate position size
        risk_amount = capital * max_risk_pct
        position_size = risk_amount / (price * volatility)

        # Apply bounds
        max_position_pct = 0.10  # Max 10% of capital per position
        max_position_size = capital * max_position_pct / price

        return min(position_size, max_position_size)

    def _kelly_criterion(self, volatility: float) -> float:
        """Calculate Kelly criterion position size."""
        if volatility <= 0:
            return 0.01  # Conservative fallback

        # Simplified Kelly: 1/volatility (inverted volatility)
        return min(1.0 / (volatility * 10), 0.1)  # Cap at 10%

    def _market_regime_adjustment(self, regime: str) -> float:
        """Adjust position size based on market regime."""
        adjustments = {
            'normal': 1.0,
            'volatile': 0.5,    # Reduce size in volatile markets
            'crisis': 0.2       # Very conservative in crisis
        }
        return adjustments.get(regime, 1.0)

    def _diversification_adjustment(self, positions: Dict[str, Any], symbol: str) -> float:
        """Adjust for portfolio diversification."""
        if not positions:
            return 1.0  # No adjustment for first position

        # Count positions in same sector (simplified)
        symbol_prefix = symbol[:3]  # BTC, ETH, etc.
        same_sector_count = sum(1 for pos_symbol in positions.keys()
                               if pos_symbol.startswith(symbol_prefix))

        # Reduce size if already have positions in same sector
        if same_sector_count >= 2:
            return 0.5
        elif same_sector_count >= 1:
            return 0.7

        return 1.0

    def _volatility_adjustment(self, asset_vol: float, portfolio_vol: float) -> float:
        """Adjust for volatility clustering."""
        if portfolio_vol <= 0:
            return 1.0

        vol_ratio = asset_vol / portfolio_vol

        if vol_ratio > 1.5:  # More volatile than portfolio
            return 0.7
        elif vol_ratio < 0.7:  # Less volatile than portfolio
            return 1.2

        return 1.0


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory."""

    def __init__(self):
        self.risk_free_rate = 0.02
        self.max_weight = 0.25  # Max 25% per asset

    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        target_return: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using mean-variance optimization.

        Args:
            returns: Historical returns DataFrame
            current_weights: Current portfolio weights
            target_return: Target portfolio return

        Returns:
            Optimization results
        """
        if len(returns.columns) < 2:
            # Single asset portfolio
            return {
                'weights': {returns.columns[0]: 1.0},
                'expected_return': returns.mean().iloc[0],
                'volatility': returns.std().iloc[0],
                'sharpe_ratio': (returns.mean().iloc[0] - self.risk_free_rate) / returns.std().iloc[0]
            }

        # Calculate expected returns and covariance
        expected_returns = returns.mean()
        cov_matrix = returns.cov()

        # Number of assets
        n_assets = len(returns.columns)

        # Optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]

        # Weight bounds
        bounds = tuple((0, self.max_weight) for _ in range(n_assets))

        # Target return constraint (if specified)
        if target_return:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: expected_returns.dot(x) - target_return
            })

        # Minimize volatility (or maximize Sharpe if no target return)
        if target_return:
            # Minimize volatility for target return
            def objective(weights):
                return np.sqrt(weights.T @ cov_matrix @ weights)
        else:
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = expected_returns.dot(weights)
                portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                return -(portfolio_return - self.risk_free_rate) / portfolio_vol

        # Initial guess (equal weight)
        init_weights = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            optimal_weights = dict(zip(returns.columns, result.x))

            # Calculate portfolio metrics
            portfolio_return = expected_returns.dot(result.x)
            portfolio_vol = np.sqrt(result.x.T @ cov_matrix @ result.x)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol

            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            # Fallback to equal weight
            equal_weights = {col: 1/n_assets for col in returns.columns}
            return {
                'weights': equal_weights,
                'expected_return': expected_returns.mean(),
                'volatility': np.sqrt(cov_matrix.values.mean()),
                'sharpe_ratio': 0.0,
                'optimization_success': False,
                'error': result.message
            }


class AdvancedRiskManager:
    """
    Advanced risk management system for live trading.

    Features:
    - Dynamic position sizing
    - Portfolio optimization
    - Real-time risk monitoring
    - Market regime detection
    - Stress testing capabilities
    """

    def __init__(self, initial_capital: float = 10000, stop_loss_pct: float = 0.02, take_profit_pct: float = 0.05):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.logger = logging.getLogger(__name__)

        # Risk components
        self.position_sizer = DynamicPositionSizer()
        self.portfolio_optimizer = PortfolioOptimizer()

        # Risk limits
        self.max_portfolio_risk = 0.15  # 15% max portfolio risk
        self.max_correlation = 0.8      # Max correlation between assets
        self.max_sector_exposure = 0.4  # Max 40% exposure to one sector

        # Market regime detection
        self.regime_window = 20
        self.volatility_threshold = 0.02  # 2% daily volatility threshold

        # Performance tracking
        self.portfolio_metrics = PortfolioMetrics()
        self.daily_returns: List[float] = []
        self.positions: Dict[str, Any] = {}

        # Risk alerts
        self.alerts: List[Dict[str, Any]] = []

    def assess_trade_risk(
        self,
        symbol: str,
        signal: int,
        price: float,
        confidence: float,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive trade risk assessment.

        Args:
            symbol: Trading symbol
            signal: Trading signal (1, -1, 0)
            price: Current price
            confidence: Signal confidence
            market_data: Historical market data for analysis

        Returns:
            Risk assessment results
        """
        assessment = {
            'approved': False,
            'risk_score': 0.0,
            'recommended_size': 0.0,
            'warnings': [],
            'reasons': []
        }

        # Basic validations
        if signal == 0:
            assessment['reasons'].append('No trading signal')
            return assessment

        if confidence < 0.6:
            assessment['warnings'].append(f'Low confidence: {confidence:.2f}')
            assessment['risk_score'] += 0.3

        # Market regime detection
        market_regime = self._detect_market_regime(market_data)
        if market_regime != 'normal':
            assessment['warnings'].append(f'Market regime: {market_regime}')
            assessment['risk_score'] += 0.2

        # Volatility analysis
        volatility = self._calculate_volatility(symbol, market_data)
        portfolio_vol = self._calculate_portfolio_volatility()

        # Position sizing
        recommended_size = self.position_sizer.calculate_optimal_size(
            capital=self.current_capital,
            price=price,
            volatility=volatility,
            portfolio_volatility=portfolio_vol,
            symbol=symbol,
            current_positions=self.positions,
            market_regime=market_regime
        )

        # Risk checks
        if recommended_size <= 0:
            assessment['reasons'].append('Invalid position size')
            return assessment

        # Portfolio risk limits
        if self._would_exceed_portfolio_limits(symbol, recommended_size, price):
            assessment['warnings'].append('Would exceed portfolio limits')
            assessment['risk_score'] += 0.4

        # Correlation check
        correlation_risk = self._check_correlation_risk(symbol, market_data)
        if correlation_risk > self.max_correlation:
            assessment['warnings'].append(f'High correlation risk: {correlation_risk:.2f}')
            assessment['risk_score'] += 0.2

        # Final decision
        max_risk_score = 0.8
        assessment['approved'] = assessment['risk_score'] <= max_risk_score
        assessment['recommended_size'] = recommended_size if assessment['approved'] else 0.0

        if assessment['approved']:
            assessment['reasons'].append('Trade approved')
        else:
            assessment['reasons'].append('Risk limits exceeded')

        return assessment

    def update_portfolio(self, positions: Dict[str, Any], capital: float) -> None:
        """Update portfolio state and recalculate metrics."""
        self.positions = positions
        self.current_capital = capital

        # Calculate portfolio value
        positions_value = sum(
            pos.get('quantity', 0) * pos.get('current_price', 0)
            for pos in positions.values()
        )

        self.portfolio_metrics.total_value = capital + positions_value
        self.portfolio_metrics.cash = capital
        self.portfolio_metrics.positions_value = positions_value

    def calculate_portfolio_rebalance(
        self,
        target_allocations: Dict[str, float],
        current_positions: Dict[str, Any],
        capital: float
    ) -> List[Dict[str, Any]]:
        """
        Calculate portfolio rebalancing trades.

        Args:
            target_allocations: Target % allocations by symbol
            current_positions: Current positions
            capital: Available capital

        Returns:
            List of rebalancing trades
        """
        trades = []

        # Calculate current allocations
        total_value = capital
        current_allocations = {}

        for symbol, position in current_positions.items():
            position_value = position.get('quantity', 0) * position.get('current_price', 0)
            total_value += position_value
            current_allocations[symbol] = position_value / total_value if total_value > 0 else 0

        # Calculate required trades
        for symbol, target_pct in target_allocations.items():
            current_pct = current_allocations.get(symbol, 0)
            target_value = total_value * target_pct
            current_value = total_value * current_pct

            trade_value = target_value - current_value

            if abs(trade_value) > capital * 0.001:  # Minimum trade threshold
                current_price = current_positions.get(symbol, {}).get('current_price', 0)
                if current_price > 0:
                    quantity = trade_value / current_price

                    trades.append({
                        'symbol': symbol,
                        'action': 'BUY' if quantity > 0 else 'SELL',
                        'quantity': abs(quantity),
                        'reason': 'Portfolio rebalance'
                    })

        return trades

    def stress_test_portfolio(
        self,
        positions: Dict[str, Any],
        scenarios: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Run stress tests on portfolio.

        Args:
            positions: Current positions
            scenarios: List of stress scenarios

        Returns:
            Stress test results
        """
        results = {}

        for scenario in scenarios:
            scenario_name = scenario.get('name', 'Unknown')
            shock_type = scenario.get('type', 'price_shock')
            shock_value = scenario.get('value', 0.0)

            # Apply shock to portfolio
            shocked_value = self._apply_portfolio_shock(positions, shock_type, shock_value)

            original_value = self.portfolio_metrics.total_value
            loss_pct = 0.0
            if original_value != 0:
                loss_pct = (shocked_value - original_value) / original_value

            results[scenario_name] = {
                'original_value': original_value,
                'shocked_value': shocked_value,
                'loss_pct': loss_pct,
                'breach_warnings': self._check_risk_breaches(shocked_value)
            }

        return results

    def _detect_market_regime(self, market_data: Optional[pd.DataFrame]) -> str:
        """
        Detect current market regime based on volatility and trend analysis.
        
        Returns:
            'normal', 'volatile', 'volatile_bullish', 'volatile_bearish', 'crisis', or 'unknown'
        """
        if market_data is None or len(market_data) < self.regime_window:
            return 'unknown'

        # Calculate volatility
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.rolling(self.regime_window).std().iloc[-1]

        # Calculate trend direction and strength
        recent_prices = market_data['close'].tail(self.regime_window).values
        if len(recent_prices) < 2:
            return 'unknown'
        
        # Calculate trend slope using linear regression
        x = np.arange(len(recent_prices))
        trend_slope = np.polyfit(x, recent_prices, 1)[0]
        
        # Calculate trend strength (magnitude) - always non-negative
        mean_price = np.mean(recent_prices)
        if mean_price > 0:
            trend_strength = abs(trend_slope) / mean_price
        else:
            trend_strength = 0.0
        
        # Determine trend direction using original trend_slope (not abs)
        is_bearish = trend_slope < -0.001
        is_bullish = trend_slope > 0.001

        # Classify regime based on volatility and trend
        if volatility > self.volatility_threshold * 2:
            return 'crisis'
        elif volatility > self.volatility_threshold:
            # High volatility: check trend direction
            if is_bearish:
                return 'volatile_bearish'
            elif is_bullish:
                return 'volatile_bullish'
            else:
                return 'volatile'
        else:
            # Normal volatility: still check trend for completeness
            if is_bearish and trend_strength > 0.01:
                return 'normal_bearish'
            elif is_bullish and trend_strength > 0.01:
                return 'normal_bullish'
            else:
                return 'normal'

    def _calculate_volatility(self, symbol: str, market_data: Optional[pd.DataFrame]) -> float:
        """Calculate asset volatility."""
        if market_data is None or len(market_data) < 10:
            return 0.02  # Default 2% volatility

        returns = market_data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualized

    def _calculate_portfolio_volatility(self) -> float:
        """Calculate current portfolio volatility."""
        if len(self.daily_returns) < 10:
            return 0.02  # Default

        return np.std(self.daily_returns) * np.sqrt(252)

    def _would_exceed_portfolio_limits(self, symbol: str, size: float, price: float) -> bool:
        """Check if trade would exceed portfolio limits."""
        position_value = size * price
        new_portfolio_value = self.portfolio_metrics.total_value + position_value

        if new_portfolio_value <= 0:
            return True

        # Check position concentration
        position_pct = position_value / new_portfolio_value
        if position_pct > 0.25:  # Max 25% per position
            return True

        # Check sector exposure (simplified)
        symbol_prefix = symbol[:3]
        sector_positions = [
            pos for pos in self.positions.values()
            if pos.get('symbol', '').startswith(symbol_prefix)
        ]
        sector_value = sum(pos.get('quantity', 0) * pos.get('current_price', 0)
                          for pos in sector_positions) + position_value

        if sector_value / new_portfolio_value > self.max_sector_exposure:
            return True

        return False

    def _check_correlation_risk(self, symbol: str, market_data: Optional[pd.DataFrame]) -> float:
        """Check correlation with existing positions."""
        if market_data is None or not self.positions:
            return 0.0

        # Simplified correlation check
        # In production, would correlate with historical data
        return 0.5  # Placeholder

    def _apply_portfolio_shock(
        self,
        positions: Dict[str, Any],
        shock_type: str,
        shock_value: float
    ) -> float:
        """Apply shock to portfolio value."""
        shocked_value = self.portfolio_metrics.total_value

        if shock_type == 'price_shock':
            # Apply percentage shock to all positions
            for position in positions.values():
                current_value = position.get('quantity', 0) * position.get('current_price', 0)
                shocked_value -= current_value * shock_value

        return shocked_value

    def _check_risk_breaches(self, portfolio_value: float) -> List[str]:
        """Check for risk limit breaches."""
        breaches = []

        # Drawdown check
        drawdown = (portfolio_value - self.initial_capital) / self.initial_capital
        if drawdown < -0.10:  # 10% drawdown
            breaches.append("Portfolio drawdown exceeds 10%")

        # VaR breach (simplified)
        if portfolio_value < self.portfolio_metrics.total_value * 0.95:
            breaches.append("Value at Risk breach")

        return breaches

    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            'portfolio_metrics': {
                'total_value': self.portfolio_metrics.total_value,
                'volatility': self.portfolio_metrics.volatility,
                'sharpe_ratio': self.portfolio_metrics.sharpe_ratio,
                'max_drawdown': self.portfolio_metrics.max_drawdown,
                'value_at_risk': self.portfolio_metrics.value_at_risk
            },
            'risk_limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_correlation': self.max_correlation,
                'max_sector_exposure': self.max_sector_exposure
            },
            'current_exposure': {
                'position_count': len(self.positions),
                'largest_position_pct': self._calculate_largest_position_pct(),
                'sector_diversification': self._calculate_sector_diversification()
            },
            'active_alerts': self.alerts[-10:]  # Last 10 alerts
        }

    def _calculate_largest_position_pct(self) -> float:
        """Calculate percentage of largest position."""
        if not self.positions:
            return 0.0

        position_values = [
            pos.get('quantity', 0) * pos.get('current_price', 0)
            for pos in self.positions.values()
        ]

        if not position_values or self.portfolio_metrics.total_value <= 0:
            return 0.0

        return max(position_values) / self.portfolio_metrics.total_value

    def _calculate_sector_diversification(self) -> Dict[str, float]:
        """Calculate sector diversification."""
        sectors = {}

        for symbol, position in self.positions.items():
            sector = symbol[:3]  # BTC, ETH, etc.
            value = position.get('quantity', 0) * position.get('current_price', 0)

            sectors[sector] = sectors.get(sector, 0) + value

        # Convert to percentages
        total_value = sum(sectors.values())
        if total_value > 0:
            sectors = {k: v/total_value for k, v in sectors.items()}

        return sectors
