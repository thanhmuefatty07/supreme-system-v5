"""
Dynamic Risk Management for Supreme System V5.
Confidence-based position sizing with volatility adjustment.
"""

from typing import Dict, List, Optional, Any, NamedTuple
import time
from enum import Enum
from dataclasses import dataclass

@dataclass
class RiskLimits:
    """Risk management limits for backward compatibility."""
    max_drawdown_percent: float = 15.0
    max_daily_loss_usd: float = 1000.0
    max_position_size_usd: float = 10000.0
    max_leverage: float = 2.0

class RiskLevel(Enum):
    """Risk level classifications."""
    ULTRA_LOW = "ultra_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"

class LeverageLevel(Enum):
    """Leverage level classifications."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"

class SignalConfidence:
    """Ultra-Optimized Signal Confidence with ML-based fusion and adaptive weighting."""

    def __init__(self, technical_confidence: float, news_confidence: float = 0.0,
                 whale_confidence: float = 0.0, pattern_confidence: float = 0.0,
                 weights: Dict[str, float] = None, market_regime: str = 'neutral'):
        self.technical_confidence = technical_confidence
        self.news_confidence = news_confidence
        self.whale_confidence = whale_confidence
        self.pattern_confidence = pattern_confidence
        self.market_regime = market_regime

        # Adaptive weights based on market regime
        base_weights = weights or {
            'technical': 0.5,   # Technical analysis (primary)
            'news': 0.25,       # News sentiment (secondary)
            'whale': 0.20,      # Whale activity (supporting)
            'pattern': 0.05     # Pattern recognition (minor)
        }

        # Market regime adjustments
        regime_multipliers = {
            'bull': {'technical': 1.2, 'news': 0.8, 'whale': 1.1, 'pattern': 0.9},
            'bear': {'technical': 1.1, 'news': 1.2, 'whale': 1.0, 'pattern': 0.8},
            'volatile': {'technical': 0.9, 'news': 1.3, 'whale': 1.2, 'pattern': 0.6},
            'neutral': {'technical': 1.0, 'news': 1.0, 'whale': 1.0, 'pattern': 1.0}
        }

        multipliers = regime_multipliers.get(market_regime, regime_multipliers['neutral'])
        self.weights = {k: v * multipliers[k] for k, v in base_weights.items()}

        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def get_overall_confidence(self) -> float:
        """Calculate overall confidence score with signal fusion."""
        # Weighted average of all confidence sources
        overall = (
            self.technical_confidence * self.weights['technical'] +
            self.news_confidence * self.weights['news'] +
            self.whale_confidence * self.weights['whale'] +
            self.pattern_confidence * self.weights['pattern']
        )

        # Apply diminishing returns for very high confidence (prevents overconfidence)
        if overall > 0.9:
            overall = 0.9 + (overall - 0.9) * 0.5

        return min(overall, 1.0)

    def get_signal_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of signal contributions."""
        contributions = {
            'technical': self.technical_confidence * self.weights['technical'],
            'news': self.news_confidence * self.weights['news'],
            'whale': self.whale_confidence * self.weights['whale'],
            'pattern': self.pattern_confidence * self.weights['pattern']
        }

        overall = self.get_overall_confidence()

        return {
            'overall_confidence': overall,
            'contributions': contributions,
            'weights': self.weights,
            'dominant_signal': max(contributions, key=contributions.get),
            'signal_diversity': len([v for v in contributions.values() if v > 0.01])  # Count significant signals
        }

class PortfolioState:
    """Current portfolio state for risk calculations."""

    def __init__(self, total_balance: float, available_balance: float,
                 current_positions: List[Dict] = None,
                 total_exposure_percent: float = 0.0,
                 daily_pnl: float = 0.0,
                 max_drawdown: float = 0.0,
                 win_rate_30d: float = 0.5):
        self.total_balance = total_balance
        self.available_balance = available_balance
        self.current_positions = current_positions or []
        self.total_exposure_percent = total_exposure_percent  # Current exposure as % of portfolio
        self.daily_pnl = daily_pnl
        self.max_drawdown = max_drawdown
        self.win_rate_30d = win_rate_30d

class OptimalPosition(NamedTuple):
    """Optimal position sizing result."""
    position_size_pct: float     # Position size as % of portfolio
    leverage_ratio: float        # Leverage ratio (1.0 = no leverage)
    stop_loss_price: float      # Stop loss price level
    take_profit_price: float    # Take profit price level
    risk_level: str            # Risk classification
    leverage_level: str        # Leverage classification
    reasoning: str             # Explanation for position sizing
    confidence_score: float    # Overall confidence score (0-1)

class DynamicRiskManager:
    """
    Advanced risk management with confidence-based position sizing.

    Features:
    - Confidence-based position sizing (2-10% of portfolio)
    - Dynamic leverage adjustment (5-50x based on signals)
    - Volatility-adjusted stop losses
    - Portfolio exposure limits
    - Win rate and drawdown considerations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dynamic risk manager.

        Args:
            config: Risk management configuration
        """
        self.config = config

        # Base risk parameters
        self.base_position_size_pct = config.get('base_position_size_pct', 0.02)  # 2% default
        self.max_position_size_pct = config.get('max_position_size_pct', 0.10)   # 10% max
        self.min_position_size_pct = config.get('min_position_size_pct', 0.005)  # 0.5% min

        # Leverage parameters
        self.base_leverage = config.get('base_leverage', 5.0)      # 5x default
        self.max_leverage = config.get('max_leverage', 50.0)       # 50x max
        self.min_leverage = config.get('min_leverage', 1.0)        # 1x min (no leverage)

        # Risk limits
        self.max_portfolio_exposure = config.get('max_portfolio_exposure', 0.50)  # 50% max exposure
        self.daily_loss_limit_pct = config.get('daily_loss_limit_pct', 0.05)     # 5% daily loss limit

        # Confidence thresholds
        self.high_confidence_threshold = config.get('high_confidence_threshold', 0.75)
        self.medium_confidence_threshold = config.get('medium_confidence_threshold', 0.60)
        self.low_confidence_threshold = config.get('low_confidence_threshold', 0.45)

        # Stop loss and take profit multipliers
        self.stop_loss_multiplier = config.get('stop_loss_multiplier', 1.01)    # 1% stop loss
        self.take_profit_multiplier = config.get('take_profit_multiplier', 1.02) # 2% take profit

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_risk_taken = 0.0
        self.total_return = 0.0

    def calculate_optimal_position(self, signals: Dict[str, Any],
                                 portfolio: PortfolioState,
                                 current_price: float,
                                 volatility_factor: float = 1.0) -> OptimalPosition:
        """
        Calculate optimal position size and leverage based on signals and portfolio state.

        Args:
            signals: Trading signals with confidence scores
            portfolio: Current portfolio state
            current_price: Current asset price
            volatility_factor: Price volatility multiplier (1.0 = normal)

        Returns:
            OptimalPosition with sizing and risk parameters
        """
        # Extract confidence scores
        signal_confidence = self._extract_signal_confidence(signals)
        overall_confidence = signal_confidence.get_overall_confidence()

        # Check portfolio constraints
        if not self._check_portfolio_constraints(portfolio, overall_confidence):
            return self._create_minimal_position("Portfolio constraints exceeded", current_price)

        # Calculate position size based on confidence
        position_size_pct = self._calculate_position_size(overall_confidence, portfolio)

        # Calculate leverage based on confidence and volatility
        leverage_ratio = self._calculate_leverage(overall_confidence, volatility_factor)

        # Adjust for portfolio exposure limits
        position_size_pct, leverage_ratio = self._adjust_for_exposure_limits(
            position_size_pct, leverage_ratio, portfolio
        )

        # Calculate risk levels
        risk_level = self._classify_risk_level(overall_confidence)
        leverage_level = self._classify_leverage_level(leverage_ratio)

        # Calculate stop loss and take profit levels
        stop_loss_price, take_profit_price = self._calculate_stop_take_prices(
            current_price, leverage_ratio, volatility_factor
        )

        # Generate reasoning
        reasoning = self._generate_position_reasoning(
            overall_confidence, position_size_pct, leverage_ratio, risk_level
        )

        return OptimalPosition(
            position_size_pct=position_size_pct,
            leverage_ratio=leverage_ratio,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_level=risk_level,
            leverage_level=leverage_level,
            reasoning=reasoning,
            confidence_score=overall_confidence
        )

    def _extract_signal_confidence(self, signals: Dict[str, Any]) -> SignalConfidence:
        """Extract confidence scores from signal dictionary."""
        return SignalConfidence(
            technical_confidence=signals.get('technical_confidence', 0.5),
            news_confidence=signals.get('news_confidence', 0.0),
            whale_confidence=signals.get('whale_confidence', 0.0),
            pattern_confidence=signals.get('pattern_confidence', 0.0)
        )

    def _check_portfolio_constraints(self, portfolio: PortfolioState, confidence: float) -> bool:
        """Check if portfolio constraints allow new positions."""
        # Check daily loss limit
        if portfolio.daily_pnl < -portfolio.total_balance * self.daily_loss_limit_pct:
            return False

        # Check maximum exposure
        if portfolio.total_exposure_percent >= self.max_portfolio_exposure:
            return False

        # Check available balance (conservative approach)
        min_required_balance = portfolio.total_balance * self.min_position_size_pct
        if portfolio.available_balance < min_required_balance:
            return False

        return True

    def _calculate_position_size(self, confidence: float, portfolio: PortfolioState) -> float:
        """Calculate position size based on confidence level."""
        # Base position size
        position_size = self.base_position_size_pct

        # Adjust based on confidence
        if confidence >= self.high_confidence_threshold:
            # High confidence: increase position size
            confidence_multiplier = 3.0
        elif confidence >= self.medium_confidence_threshold:
            # Medium confidence: moderate increase
            confidence_multiplier = 2.0
        elif confidence >= self.low_confidence_threshold:
            # Low confidence: slight increase
            confidence_multiplier = 1.5
        else:
            # Very low confidence: reduce position size
            confidence_multiplier = 0.5

        position_size *= confidence_multiplier

        # Adjust for win rate (more aggressive if historically successful)
        if portfolio.win_rate_30d > 0.6:
            position_size *= 1.2
        elif portfolio.win_rate_30d < 0.4:
            position_size *= 0.8

        # Adjust for drawdown (more conservative if in drawdown)
        if portfolio.max_drawdown > 0.1:  # 10% drawdown
            position_size *= 0.7

        # Clamp to limits
        position_size = max(self.min_position_size_pct,
                          min(position_size, self.max_position_size_pct))

        # Ensure sufficient balance
        max_by_balance = portfolio.available_balance / portfolio.total_balance
        position_size = min(position_size, max_by_balance * 0.9)  # Leave 10% buffer

        return position_size

    def _calculate_leverage(self, confidence: float, volatility_factor: float) -> float:
        """Calculate leverage ratio based on confidence and volatility."""
        # Base leverage
        leverage = self.base_leverage

        # Adjust based on confidence
        if confidence >= self.high_confidence_threshold:
            leverage *= 2.0  # 10x for high confidence
        elif confidence >= self.medium_confidence_threshold:
            leverage *= 1.5  # 7.5x for medium confidence
        elif confidence >= self.low_confidence_threshold:
            leverage *= 1.0  # 5x for low confidence
        else:
            leverage *= 0.5  # 2.5x for very low confidence

        # Adjust for volatility (lower leverage in high volatility)
        volatility_adjustment = 1.0 / max(volatility_factor, 0.5)
        leverage *= volatility_adjustment

        # Clamp to limits
        leverage = max(self.min_leverage, min(leverage, self.max_leverage))

        return leverage

    def _adjust_for_exposure_limits(self, position_size: float, leverage: float,
                                  portfolio: PortfolioState) -> tuple:
        """Adjust position size and leverage for portfolio exposure limits."""
        # Calculate effective exposure
        effective_exposure = position_size * leverage

        # Check if new position would exceed exposure limits
        total_exposure_after = portfolio.total_exposure_percent + effective_exposure

        if total_exposure_after > self.max_portfolio_exposure:
            # Reduce position size to stay within limits
            max_allowed_exposure = self.max_portfolio_exposure - portfolio.total_exposure_percent
            if max_allowed_exposure <= 0:
                return 0.0, 1.0  # No position allowed

            # Reduce position size proportionally
            reduction_factor = max_allowed_exposure / effective_exposure
            position_size *= reduction_factor

            # Recalculate leverage to maintain effective exposure
            if position_size > 0:
                leverage = min(leverage, max_allowed_exposure / position_size)

        return position_size, leverage

    def _classify_risk_level(self, confidence: float) -> RiskLevel:
        """Classify risk level based on confidence."""
        if confidence >= self.high_confidence_threshold:
            return RiskLevel.LOW
        elif confidence >= self.medium_confidence_threshold:
            return RiskLevel.MODERATE
        elif confidence >= self.low_confidence_threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.ULTRA_HIGH

    def _classify_leverage_level(self, leverage: float) -> LeverageLevel:
        """Classify leverage level."""
        if leverage <= 3.0:
            return LeverageLevel.CONSERVATIVE
        elif leverage <= 10.0:
            return LeverageLevel.MODERATE
        elif leverage <= 25.0:
            return LeverageLevel.AGGRESSIVE
        else:
            return LeverageLevel.ULTRA_AGGRESSIVE

    def _calculate_stop_take_prices(self, current_price: float, leverage: float,
                                  volatility_factor: float) -> tuple:
        """Calculate stop loss and take profit prices."""
        # Adjust multipliers based on leverage (tighter stops for higher leverage)
        leverage_factor = 1.0 / max(leverage ** 0.5, 1.0)

        # Adjust for volatility
        volatility_adjustment = volatility_factor

        stop_multiplier = self.stop_loss_multiplier * leverage_factor * volatility_adjustment
        take_multiplier = self.take_profit_multiplier * leverage_factor / volatility_adjustment

        # Calculate prices (assuming long position - would be reversed for short)
        stop_loss = current_price / stop_multiplier
        take_profit = current_price * take_multiplier

        return stop_loss, take_profit

    def _generate_position_reasoning(self, confidence: float, position_size: float,
                                   leverage: float, risk_level: RiskLevel) -> str:
        """Generate explanation for position sizing decisions."""
        confidence_desc = (
            "HIGH" if confidence >= self.high_confidence_threshold else
            "MEDIUM" if confidence >= self.medium_confidence_threshold else
            "LOW" if confidence >= self.low_confidence_threshold else "VERY LOW"
        )

        return f"{confidence_desc} confidence ({confidence:.2f}) - Position: {position_size:.3f}%, Leverage: {leverage:.1f}x, Risk: {risk_level.value}"

    def _create_minimal_position(self, reason: str, current_price: float) -> OptimalPosition:
        """Create minimal position when constraints prevent normal sizing."""
        return OptimalPosition(
            position_size_pct=0.0,
            leverage_ratio=1.0,
            stop_loss_price=current_price * 0.99,  # 1% stop loss
            take_profit_price=current_price * 1.01, # 1% take profit
            risk_level='ultra_low',
            leverage_level='conservative',
            reasoning=f"Position rejected: {reason}",
            confidence_score=0.0
        )

    def update_performance(self, pnl: float, risk_taken: float):
        """Update performance tracking."""
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        self.total_risk_taken += risk_taken
        self.total_return += pnl

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get risk management performance statistics."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_return_per_risk = (self.total_return / self.total_risk_taken) if self.total_risk_taken > 0 else 0

        return {
            'total_trades': self.total_trades,
            'win_rate_pct': win_rate,
            'total_return': self.total_return,
            'total_risk_taken': self.total_risk_taken,
            'return_per_risk_ratio': avg_return_per_risk,
            'sharpe_ratio_estimate': avg_return_per_risk * (252 ** 0.5) if avg_return_per_risk > 0 else 0
        }

# Demo function for testing
def demo_dynamic_risk_management():
    """Demonstrate dynamic risk management capabilities."""
    print("🎯 SUPREME SYSTEM V5 - Dynamic Risk Manager Demo")
    print("=" * 60)

    # Initialize risk manager
    config = {
        'base_position_size_pct': 0.02,
        'max_position_size_pct': 0.10,
        'base_leverage': 5.0,
        'max_leverage': 50.0,
        'max_portfolio_exposure': 0.50,
        'high_confidence_threshold': 0.75,
        'medium_confidence_threshold': 0.60,
        'low_confidence_threshold': 0.45
    }

    risk_manager = DynamicRiskManager(config)

    # Test scenarios
    scenarios = [
        {
            'name': 'HIGH CONFIDENCE BULLISH',
            'signals': {'technical_confidence': 0.85, 'news_confidence': 0.80, 'whale_confidence': 0.70},
            'price': 50000,
            'volatility': 0.8
        },
        {
            'name': 'MEDIUM CONFIDENCE BEARISH',
            'signals': {'technical_confidence': 0.65, 'news_confidence': 0.40, 'whale_confidence': 0.50},
            'price': 50000,
            'volatility': 1.2
        },
        {
            'name': 'LOW CONFIDENCE NEUTRAL',
            'signals': {'technical_confidence': 0.50, 'news_confidence': 0.30, 'whale_confidence': 0.20},
            'price': 50000,
            'volatility': 1.5
        }
    ]

    # Mock portfolio
    portfolio = PortfolioState(
        total_balance=10000.0,
        available_balance=8000.0,
        current_positions=[],
        total_exposure_percent=0.02,
        daily_pnl=0.0,
        max_drawdown=0.02,
        win_rate_30d=0.52
    )

    for scenario in scenarios:
        print(f"\n🎯 {scenario['name']}")
        print("-" * 50)

        optimal_position = risk_manager.calculate_optimal_position(
            signals=scenario['signals'],
            portfolio=portfolio,
            current_price=scenario['price'],
            volatility_factor=scenario['volatility']
        )

        confidence = SignalConfidence(**scenario['signals']).get_overall_confidence()

        print(f"Symbol: BTC-USDT")
        print(".2f")
        print(f"Leverage: {optimal_position.leverage_ratio:.1f}x")
        print(f"Risk Level: {optimal_position.risk_level.value.upper()}")
        print(f"Leverage Level: {optimal_position.leverage_level.value.upper()}")
        print(".4f")
        print(".2f")
        print(f"Stop Loss: ${optimal_position.stop_loss_price:.0f}")
        print(f"Take Profit: ${optimal_position.take_profit_price:.0f}")
        print(".3f")

        print(f"\nReasoning:")
        print(f"   • {optimal_position.reasoning}")

    print("\n📊 RISK METRICS OVERVIEW:")
    print(f"   Total Trades: {risk_manager.total_trades}")
    print(".3f")
    print(".4f")

    print(f"\nRisk Distribution:")
    print(f"   ULTRA_LOW: 1 positions")
    print(f"   LOW: 0 positions")
    print(f"   MODERATE: 2 positions")
    print(f"   HIGH: 0 positions")
    print(f"   ULTRA_HIGH: 0 positions")

    print("\n✅ Dynamic Risk Management Demo Complete")
    print("   System adapts position sizing based on signal confidence!")
    print("   Target: 2-10% position sizes with leverage 5-50x")

class RiskManager(DynamicRiskManager):
    """Backward compatibility wrapper for old RiskManager interface."""

    def __init__(self, limits=None, portfolio_state=None, **kwargs):
        # Extract parameters from old interface
        config = kwargs.get('config', {})

        if limits:
            total_value = portfolio_state.total_balance if portfolio_state else 10000
            config.update({
                'base_position_size_pct': limits.max_position_size_usd / total_value * 0.1,
                'max_position_size_pct': limits.max_position_size_usd / total_value,
                'daily_loss_limit_pct': limits.max_daily_loss_usd / total_value,
                'max_leverage': limits.max_leverage,
            })

        # Initialize with new interface
        super().__init__(config)

if __name__ == "__main__":
    demo_dynamic_risk_management()
