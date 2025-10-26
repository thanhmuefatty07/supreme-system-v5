#!/usr/bin/env python3
"""
🛡️ Supreme System V5 - Risk Management for Backtesting
Comprehensive risk controls and position sizing

Features:
- Position sizing algorithms
- Portfolio risk metrics
- Drawdown protection
- Exposure limits
- Risk-adjusted returns
- Dynamic risk adjustment
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskModel(Enum):
    """Risk management models"""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly"
    VOLATILITY_TARGETING = "vol_target"
    MAX_DRAWDOWN = "max_dd"
    VAR_BASED = "var"


class PositionSizeMethod(Enum):
    """Position sizing methods"""
    FIXED_DOLLAR = "fixed_dollar"
    PERCENT_EQUITY = "percent_equity"
    VOLATILITY_ADJUSTED = "vol_adjusted"
    RISK_PARITY = "risk_parity"
    KELLY_OPTIMAL = "kelly_optimal"


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Portfolio limits
    max_portfolio_risk: float = 0.02  # 2% portfolio risk per trade
    max_position_size: float = 0.1    # 10% max position size
    max_sector_exposure: float = 0.3   # 30% max sector exposure
    max_drawdown_limit: float = 0.15  # 15% max drawdown before stop
    
    # Position sizing
    position_size_method: PositionSizeMethod = PositionSizeMethod.VOLATILITY_ADJUSTED
    base_position_size: float = 0.05  # 5% base position size
    
    # Risk model
    risk_model: RiskModel = RiskModel.VOLATILITY_TARGETING
    target_volatility: float = 0.15   # 15% target annual volatility
    
    # Stop loss / Take profit
    default_stop_loss: float = 0.05   # 5% stop loss
    default_take_profit: float = 0.10 # 10% take profit
    trailing_stop: bool = True
    
    # Dynamic adjustments
    adjust_for_volatility: bool = True
    adjust_for_correlation: bool = True
    adjust_for_drawdown: bool = True
    
    # Lookback periods
    volatility_lookback: int = 20     # 20 days for volatility calc
    correlation_lookback: int = 60    # 60 days for correlation
    performance_lookback: int = 252   # 1 year for performance metrics


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    entry_price: float
    position_size: float  # Number of shares/contracts
    entry_time: datetime
    direction: int  # 1 for long, -1 for short
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    
    def get_pnl(self, current_price: float) -> float:
        """Calculate current PnL"""
        return (current_price - self.entry_price) * self.position_size * self.direction
    
    def get_pnl_pct(self, current_price: float) -> float:
        """Calculate PnL percentage"""
        return (current_price - self.entry_price) / self.entry_price * self.direction
    
    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss should trigger"""
        if not self.stop_loss:
            return False
        
        if self.direction == 1:  # Long position
            return current_price <= self.stop_loss
        else:  # Short position
            return current_price >= self.stop_loss
    
    def should_take_profit(self, current_price: float) -> bool:
        """Check if take profit should trigger"""
        if not self.take_profit:
            return False
            
        if self.direction == 1:  # Long position
            return current_price >= self.take_profit
        else:  # Short position
            return current_price <= self.take_profit


class RiskMetrics:
    """Calculate portfolio and strategy risk metrics"""
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)"""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # Annualize daily volatility
        return vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
        volatility = RiskMetrics.calculate_volatility(returns)
        return excess_returns / volatility if volatility > 0 else 0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return excess_returns / downside_deviation if downside_deviation > 0 else 0
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, datetime, datetime]:
        """Calculate maximum drawdown"""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        # Find drawdown period
        max_dd_date = drawdown.idxmin()
        peak_date = running_max.loc[:max_dd_date].idxmax()
        
        return abs(max_dd), peak_date, max_dd_date
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)"""
        return np.percentile(returns, confidence * 100)
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = returns.mean() * 252
        equity_curve = (1 + returns).cumprod()
        max_dd, _, _ = RiskMetrics.calculate_max_drawdown(equity_curve)
        return annual_return / max_dd if max_dd > 0 else 0


class PositionSizer:
    """Advanced position sizing algorithms"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: Optional[float],
        portfolio_value: float,
        volatility: Optional[float] = None,
        signal_strength: float = 1.0
    ) -> float:
        """Calculate optimal position size based on risk parameters"""
        
        if self.config.position_size_method == PositionSizeMethod.FIXED_DOLLAR:
            return self._fixed_dollar_sizing(portfolio_value)
        
        elif self.config.position_size_method == PositionSizeMethod.PERCENT_EQUITY:
            return self._percent_equity_sizing(portfolio_value, entry_price, signal_strength)
        
        elif self.config.position_size_method == PositionSizeMethod.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted_sizing(
                portfolio_value, entry_price, volatility or 0.02, signal_strength
            )
        
        elif self.config.position_size_method == PositionSizeMethod.KELLY_OPTIMAL:
            return self._kelly_position_sizing(
                portfolio_value, entry_price, stop_loss, signal_strength
            )
        
        else:  # Default to percent equity
            return self._percent_equity_sizing(portfolio_value, entry_price, signal_strength)
    
    def _fixed_dollar_sizing(self, portfolio_value: float) -> float:
        """Fixed dollar amount per trade"""
        fixed_amount = portfolio_value * self.config.base_position_size
        return fixed_amount
    
    def _percent_equity_sizing(self, portfolio_value: float, entry_price: float, signal_strength: float) -> float:
        """Percentage of equity sizing"""
        target_value = portfolio_value * self.config.base_position_size * signal_strength
        return target_value / entry_price
    
    def _volatility_adjusted_sizing(
        self, 
        portfolio_value: float, 
        entry_price: float, 
        volatility: float,
        signal_strength: float
    ) -> float:
        """Volatility-adjusted position sizing"""
        # Inverse volatility scaling
        vol_adjustment = self.config.target_volatility / max(volatility, 0.01)
        
        target_value = (
            portfolio_value * 
            self.config.base_position_size * 
            vol_adjustment * 
            signal_strength
        )
        
        # Cap at maximum position size
        max_value = portfolio_value * self.config.max_position_size
        target_value = min(target_value, max_value)
        
        return target_value / entry_price
    
    def _kelly_position_sizing(
        self,
        portfolio_value: float,
        entry_price: float, 
        stop_loss: Optional[float],
        signal_strength: float
    ) -> float:
        """Kelly criterion position sizing"""
        if not stop_loss:
            # Fallback to percent equity if no stop loss
            return self._percent_equity_sizing(portfolio_value, entry_price, signal_strength)
        
        # Simple Kelly approximation
        # Assumes win rate based on signal strength and risk/reward from stop loss
        win_rate = 0.5 + (signal_strength - 1) * 0.1  # Adjust based on signal
        win_rate = np.clip(win_rate, 0.3, 0.8)  # Reasonable bounds
        
        loss_amount = abs(entry_price - stop_loss) / entry_price
        win_amount = loss_amount * 1.5  # Assume 1.5:1 reward/risk
        
        # Kelly fraction
        kelly_fraction = (win_rate * win_amount - (1 - win_rate) * loss_amount) / win_amount
        kelly_fraction = max(kelly_fraction, 0)  # No negative positions
        kelly_fraction = min(kelly_fraction, self.config.max_position_size)  # Cap position
        
        target_value = portfolio_value * kelly_fraction
        return target_value / entry_price


class RiskManager:
    """Comprehensive risk manager for backtesting"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.position_sizer = PositionSizer(config)
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[float] = []
        self.returns: List[float] = []
        
        # Risk state tracking
        self.current_drawdown = 0.0
        self.max_historical_drawdown = 0.0
        self.risk_reduction_factor = 1.0
        
        logger.info("🛡️ Risk manager initialized")
        logger.info(f"   Max portfolio risk: {config.max_portfolio_risk:.1%}")
        logger.info(f"   Max position size: {config.max_position_size:.1%}")
        logger.info(f"   Position sizing: {config.position_size_method.value}")
    
    def can_open_position(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        current_positions: Optional[Dict[str, Position]] = None
    ) -> Tuple[bool, str]:
        """Check if position can be opened based on risk limits"""
        
        # Check maximum drawdown limit
        if self.current_drawdown >= self.config.max_drawdown_limit:
            return False, "Maximum drawdown limit exceeded"
        
        # Check position size limits
        position_pct = position_value / portfolio_value
        if position_pct > self.config.max_position_size:
            return False, f"Position size ({position_pct:.1%}) exceeds limit ({self.config.max_position_size:.1%})"
        
        # Check portfolio concentration
        current_positions = current_positions or self.positions
        total_exposure = sum(
            abs(pos.position_size * (pos.current_price or pos.entry_price)) 
            for pos in current_positions.values()
        )
        
        if (total_exposure + position_value) / portfolio_value > 1.0:
            return False, "Portfolio fully invested"
        
        return True, "Position approved"
    
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        direction: int,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate appropriate stop loss level"""
        
        base_stop = self.config.default_stop_loss
        
        # Adjust for volatility if available
        if volatility and self.config.adjust_for_volatility:
            # Use 2x volatility as stop distance
            vol_stop = volatility * 2
            base_stop = max(base_stop, vol_stop)
        
        if direction == 1:  # Long position
            return entry_price * (1 - base_stop)
        else:  # Short position
            return entry_price * (1 + base_stop)
    
    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        direction: int
    ) -> float:
        """Calculate take profit level based on risk/reward ratio"""
        
        stop_distance = abs(entry_price - stop_loss)
        reward_ratio = 2.0  # 2:1 reward/risk ratio
        
        if direction == 1:  # Long position
            return entry_price + (stop_distance * reward_ratio)
        else:  # Short position
            return entry_price - (stop_distance * reward_ratio)
    
    def update_portfolio_metrics(
        self,
        current_portfolio_value: float,
        timestamp: datetime
    ) -> Dict[str, float]:
        """Update portfolio metrics and risk state"""
        
        self.equity_curve.append(current_portfolio_value)
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]
            current_return = (current_portfolio_value - prev_value) / prev_value
            self.returns.append(current_return)
        
        # Update drawdown
        if self.equity_curve:
            peak_value = max(self.equity_curve)
            self.current_drawdown = (peak_value - current_portfolio_value) / peak_value
            self.max_historical_drawdown = max(self.max_historical_drawdown, self.current_drawdown)
        
        # Adjust risk based on performance
        self._adjust_risk_levels()
        
        # Calculate current metrics
        metrics = self._calculate_current_metrics()
        
        return metrics
    
    def _adjust_risk_levels(self) -> None:
        """Dynamically adjust risk levels based on performance"""
        
        if not self.config.adjust_for_drawdown:
            return
        
        # Reduce position sizing during drawdown periods
        if self.current_drawdown > 0.05:  # 5% drawdown threshold
            self.risk_reduction_factor = 1 - (self.current_drawdown * 2)
            self.risk_reduction_factor = max(self.risk_reduction_factor, 0.3)  # Minimum 30%
        else:
            self.risk_reduction_factor = 1.0
    
    def _calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current risk and performance metrics"""
        
        metrics = {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_historical_drawdown,
            'risk_reduction_factor': self.risk_reduction_factor,
            'portfolio_value': self.equity_curve[-1] if self.equity_curve else 0,
        }
        
        if len(self.returns) > 10:  # Need minimum data for meaningful metrics
            returns_series = pd.Series(self.returns)
            
            metrics.update({
                'sharpe_ratio': RiskMetrics.calculate_sharpe_ratio(returns_series),
                'sortino_ratio': RiskMetrics.calculate_sortino_ratio(returns_series), 
                'calmar_ratio': RiskMetrics.calculate_calmar_ratio(returns_series),
                'volatility': RiskMetrics.calculate_volatility(returns_series),
                'var_5pct': RiskMetrics.calculate_var(returns_series, 0.05)
            })
        
        return metrics
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        if not self.returns:
            return {'status': 'insufficient_data'}
        
        returns_series = pd.Series(self.returns)
        equity_series = pd.Series(self.equity_curve)
        
        max_dd, peak_date, trough_date = RiskMetrics.calculate_max_drawdown(equity_series)
        
        report = {
            'performance_metrics': {
                'total_return': (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100 if len(self.equity_curve) > 1 else 0,
                'annual_return': returns_series.mean() * 252 * 100,
                'volatility': RiskMetrics.calculate_volatility(returns_series) * 100,
                'sharpe_ratio': RiskMetrics.calculate_sharpe_ratio(returns_series),
                'sortino_ratio': RiskMetrics.calculate_sortino_ratio(returns_series),
                'calmar_ratio': RiskMetrics.calculate_calmar_ratio(returns_series)
            },
            'risk_metrics': {
                'max_drawdown': max_dd * 100,
                'current_drawdown': self.current_drawdown * 100,
                'var_5pct': RiskMetrics.calculate_var(returns_series) * 100,
                'worst_day': returns_series.min() * 100,
                'best_day': returns_series.max() * 100
            },
            'position_metrics': {
                'active_positions': len(self.positions),
                'risk_reduction_factor': self.risk_reduction_factor,
                'max_position_size': self.config.max_position_size * 100
            },
            'configuration': {
                'max_portfolio_risk': self.config.max_portfolio_risk * 100,
                'max_position_size': self.config.max_position_size * 100,
                'max_drawdown_limit': self.config.max_drawdown_limit * 100,
                'position_sizing_method': self.config.position_size_method.value,
                'risk_model': self.config.risk_model.value
            }
        }
        
        return report


# Demo function
def demo_risk_management() -> Dict[str, Any]:
    """Demonstrate risk management capabilities"""
    print("🛡️ SUPREME SYSTEM V5 - RISK MANAGEMENT DEMO")
    print("=" * 55)
    
    # Create risk configuration
    config = RiskConfig(
        max_portfolio_risk=0.02,
        max_position_size=0.08,
        position_size_method=PositionSizeMethod.VOLATILITY_ADJUSTED,
        target_volatility=0.15
    )
    
    # Initialize risk manager
    risk_manager = RiskManager(config)
    
    # Simulate portfolio performance
    initial_value = 100000
    portfolio_values = [initial_value]
    
    # Generate realistic portfolio performance
    np.random.seed(42)
    daily_returns = np.random.normal(0.0005, 0.015, 252)  # ~12% annual, 15% vol
    
    for i, daily_return in enumerate(daily_returns):
        new_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_value)
        
        # Update risk metrics
        timestamp = datetime.now()
        metrics = risk_manager.update_portfolio_metrics(new_value, timestamp)
    
    # Test position sizing
    test_positions = []
    for symbol, price, volatility in [('AAPL', 150, 0.25), ('TSLA', 200, 0.35), ('MSFT', 300, 0.20)]:
        position_size = risk_manager.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=price,
            stop_loss=price * 0.95,
            portfolio_value=portfolio_values[-1],
            volatility=volatility,
            signal_strength=1.2
        )
        
        can_open, reason = risk_manager.can_open_position(
            symbol, position_size * price, portfolio_values[-1]
        )
        
        test_positions.append({
            'symbol': symbol,
            'price': price,
            'position_size': position_size,
            'position_value': position_size * price,
            'can_open': can_open,
            'reason': reason
        })
    
    # Generate risk report
    risk_report = risk_manager.get_risk_report()
    
    print(f"\n📊 RISK MANAGEMENT RESULTS:")
    print(f"   Portfolio return: {risk_report['performance_metrics']['total_return']:.1f}%")
    print(f"   Sharpe ratio: {risk_report['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"   Max drawdown: {risk_report['risk_metrics']['max_drawdown']:.1f}%")
    print(f"   Volatility: {risk_report['performance_metrics']['volatility']:.1f}%")
    
    print(f"\n💰 POSITION SIZING EXAMPLES:")
    for pos in test_positions:
        print(f"   {pos['symbol']}: ${pos['position_value']:,.0f} ({pos['position_size']:.0f} shares) - {pos['reason']}")
    
    print(f"\n🛡️ RISK MANAGEMENT SYSTEM OPERATIONAL!")
    
    return {
        'risk_report': risk_report,
        'test_positions': test_positions,
        'portfolio_performance': portfolio_values
    }


if __name__ == "__main__":
    # Run risk management demo
    demo_risk_management()
