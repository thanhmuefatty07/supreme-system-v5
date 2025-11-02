#!/usr/bin/env python3
"""
🚀 SUPREME SYSTEM V5 - Dynamic Risk Manager
Advanced risk management với confidence-based position sizing

Features:
- Dynamic position sizing (2-10% portfolio based on confidence)
- Adaptive leverage (10x-50x based on signal strength)
- Multi-signal risk assessment (technical + news + whale)
- Volatility-adjusted risk controls
- Memory-efficient processing for i3-4GB systems
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    ULTRA_LOW = "ultra_low"      # 0.5-1% position size
    LOW = "low"                  # 1-2% position size
    MODERATE = "moderate"        # 2-5% position size
    HIGH = "high"               # 5-8% position size
    ULTRA_HIGH = "ultra_high"   # 8-10% position size


class LeverageLevel(Enum):
    """Leverage level classifications"""
    CONSERVATIVE = "conservative"  # 5-10x
    MODERATE = "moderate"          # 10-25x
    AGGRESSIVE = "aggressive"      # 25-50x
    ULTRA_AGGRESSIVE = "ultra"     # 50x+ (with extreme caution)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for position"""
    position_size_percent: float
    leverage_ratio: float
    max_drawdown_limit: float
    stop_loss_percent: float
    take_profit_percent: float
    max_open_positions: int
    correlation_limit: float
    volatility_adjustment: float
    risk_reward_ratio: float
    expected_win_rate: float


@dataclass
class SignalConfidence:
    """Confidence scores from different signal sources"""
    technical_confidence: float = 0.5  # 0-1
    news_confidence: float = 0.5       # 0-1
    whale_confidence: float = 0.5      # 0-1
    pattern_confidence: float = 0.5    # 0-1
    overall_confidence: float = 0.5    # 0-1


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_balance: float
    available_balance: float
    current_positions: List[Dict[str, Any]] = field(default_factory=list)
    total_exposure_percent: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate_30d: float = 0.5


@dataclass
class OptimalPosition:
    """Optimal position calculation result"""
    symbol: str
    position_size_percent: float
    leverage_ratio: float
    risk_level: RiskLevel
    leverage_level: LeverageLevel
    expected_return: float
    risk_reward_ratio: float
    stop_loss_price: float
    take_profit_price: float
    confidence_score: float
    reasoning: List[str]


class DynamicRiskManager:
    """
    Advanced risk management với confidence-based position sizing
    Adapts position size và leverage based on signal strength
    Memory-efficient cho i3-4GB systems
    """

    def __init__(self, base_config: Dict[str, Any] = None):
        self.config = base_config or self._get_default_config()

        # Base risk parameters
        self.base_position_size = self.config.get('base_position_size', 0.02)  # 2% portfolio
        self.max_position_size = self.config.get('max_position_size', 0.10)    # 10% max
        self.base_leverage = self.config.get('base_leverage', 10)              # 10x base
        self.max_leverage = self.config.get('max_leverage', 50)                # 50x max

        # Risk limits
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)  # 5% max risk
        self.max_daily_loss = self.config.get('max_daily_loss', 0.02)         # 2% daily loss limit
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.10) # 10% drawdown limit

        # Confidence thresholds
        self.confidence_thresholds = {
            "very_high": 0.90,  # Allow max position size + leverage
            "high": 0.80,       # Allow increased position size
            "medium": 0.65,     # Base position size
            "low": 0.50,        # Reduced position size
            "very_low": 0.30    # Minimal position size
        }

        # Risk adjustment factors
        self.volatility_multipliers = {
            "low": 1.2,      # Increase risk in low volatility
            "normal": 1.0,   # Standard risk
            "high": 0.7,     # Reduce risk in high volatility
            "extreme": 0.3   # Minimal risk in extreme volatility
        }

        # Position history for correlation analysis
        self.position_history = []
        self.max_history_size = 1000

        logger.info("DynamicRiskManager initialized with config: %s", self.config)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk management configuration"""
        return {
            'base_position_size': 0.02,
            'max_position_size': 0.10,
            'base_leverage': 10,
            'max_leverage': 50,
            'max_portfolio_risk': 0.05,
            'max_daily_loss': 0.02,
            'max_drawdown_limit': 0.10,
            'enable_dynamic_leverage': True,
            'enable_volatility_adjustment': True,
            'enable_correlation_filter': True,
            'risk_reward_min_ratio': 1.5,
            'win_rate_target': 0.55
        }

    def calculate_optimal_position(self,
                                 signals: Dict[str, Any],
                                 portfolio: PortfolioState,
                                 current_price: float,
                                 volatility_factor: float = 1.0) -> OptimalPosition:
        """
        Calculate optimal position size và leverage based on all signals

        Args:
            signals: Dictionary containing all signal confidences
            portfolio: Current portfolio state
            current_price: Current market price
            volatility_factor: Current market volatility (1.0 = normal)

        Returns:
            OptimalPosition with complete risk-adjusted sizing
        """

        # Extract signal confidences
        signal_confidence = self._extract_signal_confidence(signals)

        # Calculate composite confidence
        composite_confidence = self._calculate_composite_confidence(signal_confidence)

        # Adjust for volatility
        volatility_adjusted_confidence = composite_confidence / volatility_factor

        # Check portfolio risk limits
        portfolio_risk_check = self._check_portfolio_risk_limits(portfolio, volatility_adjusted_confidence)

        if not portfolio_risk_check['allowed']:
            # Return minimal position if risk limits exceeded
            return self._create_minimal_position(signals.get('symbol', 'BTC-USDT'),
                                               current_price,
                                               f"Risk limits exceeded: {portfolio_risk_check['reason']}")

        # Calculate position sizing
        position_size_percent = self._calculate_position_size(volatility_adjusted_confidence,
                                                            portfolio,
                                                            volatility_factor)

        # Calculate leverage
        leverage_ratio = self._calculate_leverage_ratio(volatility_adjusted_confidence,
                                                       signals,
                                                       volatility_factor)

        # Determine risk and leverage levels
        risk_level = self._classify_risk_level(position_size_percent)
        leverage_level = self._classify_leverage_level(leverage_ratio)

        # Calculate risk metrics
        stop_loss_percent, take_profit_percent = self._calculate_stop_take_levels(
            volatility_adjusted_confidence, leverage_ratio, signals
        )

        # Calculate expected return and risk-reward
        expected_return = self._calculate_expected_return(
            composite_confidence, position_size_percent, leverage_ratio
        )
        risk_reward_ratio = self._calculate_risk_reward_ratio(
            stop_loss_percent, take_profit_percent, composite_confidence
        )

        # Calculate actual prices
        stop_loss_price = current_price * (1 - stop_loss_percent / 100)
        take_profit_price = current_price * (1 + take_profit_percent / 100)

        # Generate reasoning
        reasoning = self._generate_position_reasoning(
            signal_confidence, composite_confidence, volatility_factor,
            position_size_percent, leverage_ratio
        )

        position = OptimalPosition(
            symbol=signals.get('symbol', 'BTC-USDT'),
            position_size_percent=round(position_size_percent, 4),
            leverage_ratio=round(leverage_ratio, 1),
            risk_level=risk_level,
            leverage_level=leverage_level,
            expected_return=round(expected_return, 4),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            stop_loss_price=round(stop_loss_price, 2),
            take_profit_price=round(take_profit_price, 2),
            confidence_score=round(composite_confidence, 3),
            reasoning=reasoning
        )

        # Store position for correlation analysis
        self._store_position_history(position, signals)

        return position

    def _extract_signal_confidence(self, signals: Dict[str, Any]) -> SignalConfidence:
        """Extract confidence scores from signal dictionary"""
        return SignalConfidence(
            technical_confidence=signals.get('technical_confidence', 0.5),
            news_confidence=signals.get('news_confidence', 0.5),
            whale_confidence=signals.get('whale_confidence', 0.5),
            pattern_confidence=signals.get('pattern_confidence', 0.5)
        )

    def _calculate_composite_confidence(self, signal_confidence: SignalConfidence) -> float:
        """Calculate weighted composite confidence from all signals"""
        weights = {
            'technical': 0.40,  # Technical analysis most important
            'news': 0.30,       # News impact significant
            'whale': 0.20,      # Whale activity moderate importance
            'pattern': 0.10     # Pattern recognition supplementary
        }

        composite = (
            signal_confidence.technical_confidence * weights['technical'] +
            signal_confidence.news_confidence * weights['news'] +
            signal_confidence.whale_confidence * weights['whale'] +
            signal_confidence.pattern_confidence * weights['pattern']
        )

        signal_confidence.overall_confidence = composite
        return min(composite, 1.0)

    def _calculate_position_size(self, adjusted_confidence: float,
                               portfolio: PortfolioState,
                               volatility_factor: float) -> float:
        """Calculate optimal position size based on confidence"""

        # Base position size from confidence
        if adjusted_confidence >= self.confidence_thresholds["very_high"]:
            base_size = self.max_position_size  # 10%
        elif adjusted_confidence >= self.confidence_thresholds["high"]:
            base_size = self.base_position_size * 3  # 6%
        elif adjusted_confidence >= self.confidence_thresholds["medium"]:
            base_size = self.base_position_size * 2  # 4%
        elif adjusted_confidence >= self.confidence_thresholds["low"]:
            base_size = self.base_position_size  # 2%
        else:
            base_size = self.base_position_size * 0.5  # 1%

        # Volatility adjustment
        volatility_multiplier = self._get_volatility_multiplier(volatility_factor)
        adjusted_size = base_size * volatility_multiplier

        # Portfolio risk limits
        max_allowed_by_risk = self.max_portfolio_risk - portfolio.total_exposure_percent
        adjusted_size = min(adjusted_size, max_allowed_by_risk)

        # Available balance check
        max_allowed_by_balance = portfolio.available_balance / portfolio.total_balance
        adjusted_size = min(adjusted_size, max_allowed_by_balance)

        return max(0.005, min(adjusted_size, self.max_position_size))  # 0.5% to 10%

    def _calculate_leverage_ratio(self, adjusted_confidence: float,
                                signals: Dict[str, Any],
                                volatility_factor: float) -> float:
        """Calculate optimal leverage ratio"""

        if not self.config.get('enable_dynamic_leverage', True):
            return self.base_leverage

        # Base leverage from confidence
        if adjusted_confidence >= self.confidence_thresholds["very_high"]:
            base_leverage = self.max_leverage  # 50x
        elif adjusted_confidence >= self.confidence_thresholds["high"]:
            base_leverage = 30  # 30x
        elif adjusted_confidence >= self.confidence_thresholds["medium"]:
            base_leverage = self.base_leverage  # 10x
        elif adjusted_confidence >= self.confidence_thresholds["low"]:
            base_leverage = 5  # 5x
        else:
            base_leverage = 2  # 2x (minimal leverage)

        # Volatility reduction
        if volatility_factor > 1.5:
            base_leverage *= 0.7  # Reduce 30% in high volatility
        elif volatility_factor > 2.0:
            base_leverage *= 0.5  # Reduce 50% in extreme volatility

        # News sentiment adjustment
        news_sentiment = signals.get('news_sentiment', 0.0)
        if abs(news_sentiment) > 0.7:  # Strong sentiment
            sentiment_boost = abs(news_sentiment) * 0.2
            base_leverage *= (1 + sentiment_boost)

        return max(1, min(base_leverage, self.max_leverage))

    def _calculate_stop_take_levels(self, confidence: float, leverage: float,
                                  signals: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate dynamic stop loss và take profit levels"""

        # Base levels depend on leverage (higher leverage = tighter stops)
        if leverage >= 25:
            base_stop = 0.5  # 0.5% stop loss
            base_take = 1.5  # 1.5% take profit
        elif leverage >= 10:
            base_stop = 1.0  # 1.0% stop loss
            base_take = 3.0  # 3.0% take profit
        else:
            base_stop = 2.0  # 2.0% stop loss
            base_take = 6.0  # 6.0% take profit

        # Confidence adjustment (higher confidence = wider targets)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0

        stop_loss = base_stop * confidence_multiplier
        take_profit = base_take * confidence_multiplier

        # Ensure minimum risk-reward ratio of 1:2
        min_take_profit = stop_loss * 2
        take_profit = max(take_profit, min_take_profit)

        return stop_loss, take_profit

    def _calculate_expected_return(self, confidence: float, position_size: float, leverage: float) -> float:
        """Calculate expected return based on confidence và position parameters"""
        # Simplified expected return model
        base_return = confidence * 0.02  # 2% base return at max confidence
        leverage_boost = (leverage - 1) * 0.005  # 0.5% per leverage unit

        return (base_return + leverage_boost) * position_size * 100  # Convert to percentage

    def _calculate_risk_reward_ratio(self, stop_loss: float, take_profit: float, confidence: float) -> float:
        """Calculate risk-reward ratio"""
        if stop_loss <= 0:
            return 0.0

        ratio = take_profit / stop_loss

        # Adjust based on confidence (higher confidence can accept lower ratios)
        if confidence > 0.8:
            ratio *= 0.9  # Allow slightly lower ratios for high confidence

        return ratio

    def _check_portfolio_risk_limits(self, portfolio: PortfolioState, confidence: float) -> Dict[str, Any]:
        """Check if position is allowed under portfolio risk limits"""

        # Daily loss limit
        if portfolio.daily_pnl < -self.max_daily_loss * portfolio.total_balance:
            return {
                'allowed': False,
                'reason': f"Daily loss limit exceeded: {portfolio.daily_pnl:.2f}"
            }

        # Drawdown limit
        if portfolio.max_drawdown > self.max_drawdown_limit:
            return {
                'allowed': False,
                'reason': f"Drawdown limit exceeded: {portfolio.max_drawdown:.2%}"
            }

        # Maximum open positions
        max_positions = self.config.get('max_open_positions', 3)
        if len(portfolio.current_positions) >= max_positions:
            return {
                'allowed': False,
                'reason': f"Maximum open positions reached: {len(portfolio.current_positions)}"
            }

        return {'allowed': True}

    def _classify_risk_level(self, position_size_percent: float) -> RiskLevel:
        """Classify risk level based on position size"""
        if position_size_percent >= 0.08:
            return RiskLevel.ULTRA_HIGH
        elif position_size_percent >= 0.05:
            return RiskLevel.HIGH
        elif position_size_percent >= 0.02:
            return RiskLevel.MODERATE
        elif position_size_percent >= 0.01:
            return RiskLevel.LOW
        else:
            return RiskLevel.ULTRA_LOW

    def _classify_leverage_level(self, leverage_ratio: float) -> LeverageLevel:
        """Classify leverage level"""
        if leverage_ratio >= 50:
            return LeverageLevel.ULTRA_AGGRESSIVE
        elif leverage_ratio >= 25:
            return LeverageLevel.AGGRESSIVE
        elif leverage_ratio >= 10:
            return LeverageLevel.MODERATE
        else:
            return LeverageLevel.CONSERVATIVE

    def _get_volatility_multiplier(self, volatility_factor: float) -> float:
        """Get volatility adjustment multiplier"""
        if volatility_factor >= 2.0:
            return self.volatility_multipliers["extreme"]
        elif volatility_factor >= 1.5:
            return self.volatility_multipliers["high"]
        elif volatility_factor >= 0.7:
            return self.volatility_multipliers["normal"]
        else:
            return self.volatility_multipliers["low"]

    def _create_minimal_position(self, symbol: str, current_price: float, reason: str) -> OptimalPosition:
        """Create minimal position when risk limits prevent normal sizing"""
        return OptimalPosition(
            symbol=symbol,
            position_size_percent=0.005,  # 0.5% minimal position
            leverage_ratio=1.0,           # No leverage
            risk_level=RiskLevel.ULTRA_LOW,
            leverage_level=LeverageLevel.CONSERVATIVE,
            expected_return=0.01,
            risk_reward_ratio=1.0,
            stop_loss_price=current_price * 0.995,   # 0.5% stop
            take_profit_price=current_price * 1.01,   # 1% target
            confidence_score=0.1,
            reasoning=[f"MINIMAL POSITION: {reason}"]
        )

    def _generate_position_reasoning(self, signal_confidence: SignalConfidence,
                                   composite_confidence: float, volatility_factor: float,
                                   position_size: float, leverage: float) -> List[str]:
        """Generate human-readable reasoning for position sizing"""
        reasoning = []

        # Confidence assessment
        if composite_confidence >= self.confidence_thresholds["very_high"]:
            reasoning.append("VERY HIGH confidence - maximum position size allowed")
        elif composite_confidence >= self.confidence_thresholds["high"]:
            reasoning.append("HIGH confidence - increased position size")
        elif composite_confidence >= self.confidence_thresholds["medium"]:
            reasoning.append("MEDIUM confidence - standard position size")
        else:
            reasoning.append("LOW confidence - reduced position size")

        # Signal breakdown
        if signal_confidence.technical_confidence > 0.7:
            reasoning.append(".2f")
        if signal_confidence.news_confidence > 0.7:
            reasoning.append(".2f")
        if signal_confidence.whale_confidence > 0.7:
            reasoning.append(".2f")
        # Volatility assessment
        if volatility_factor > 1.5:
            reasoning.append(".2f")
        elif volatility_factor < 0.7:
            reasoning.append("Low volatility - increased position size allowed")

        # Leverage reasoning
        if leverage >= 25:
            reasoning.append("HIGH leverage due to strong signal confluence")
        elif leverage <= 5:
            reasoning.append("LOW leverage due to risk management constraints")

        return reasoning

    def _store_position_history(self, position: OptimalPosition, signals: Dict[str, Any]):
        """Store position for correlation analysis"""
        history_entry = {
            'timestamp': time.time(),
            'position': position,
            'signals': signals.copy(),
            'outcome': None  # To be filled later
        }

        self.position_history.append(history_entry)

        # Maintain history size limit
        if len(self.position_history) > self.max_history_size:
            self.position_history = self.position_history[-self.max_history_size:]

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics"""
        return {
            'total_positions_opened': len(self.position_history),
            'average_confidence': sum(p['position'].confidence_score for p in self.position_history) / max(len(self.position_history), 1),
            'average_position_size': sum(p['position'].position_size_percent for p in self.position_history) / max(len(self.position_history), 1),
            'average_leverage': sum(p['position'].leverage_ratio for p in self.position_history) / max(len(self.position_history), 1),
            'risk_distribution': self._analyze_risk_distribution(),
            'correlation_warnings': self._check_position_correlation()
        }

    def _analyze_risk_distribution(self) -> Dict[str, int]:
        """Analyze distribution of risk levels"""
        distribution = {level.value: 0 for level in RiskLevel}

        for entry in self.position_history:
            distribution[entry['position'].risk_level.value] += 1

        return distribution

    def _check_position_correlation(self) -> List[str]:
        """Check for correlated position warnings"""
        warnings = []

        # Simple correlation check - positions in same direction
        recent_positions = self.position_history[-10:]  # Last 10 positions

        if len(recent_positions) >= 5:
            bullish_positions = sum(1 for p in recent_positions
                                   if p['signals'].get('technical_confidence', 0) > 0.6)

            if bullish_positions >= 4:  # 80% bullish
                warnings.append("High bullish correlation detected - consider diversification")

            bearish_positions = sum(1 for p in recent_positions
                                   if p['signals'].get('technical_confidence', 0) < 0.4)

            if bearish_positions >= 4:  # 80% bearish
                warnings.append("High bearish correlation detected - consider hedging")

        return warnings


async def demo_dynamic_risk_management():
    """Demo dynamic risk management system"""
    print("🚀 SUPREME SYSTEM V5 - Dynamic Risk Manager Demo")
    print("=" * 60)

    # Initialize risk manager
    risk_manager = DynamicRiskManager()

    # Create sample portfolio state
    portfolio = PortfolioState(
        total_balance=10000.0,
        available_balance=8000.0,
        current_positions=[],
        total_exposure_percent=0.02,  # 2% currently exposed
        daily_pnl=-100.0,  # Small loss today
        max_drawdown=0.03,
        win_rate_30d=0.52
    )

    # Test different signal scenarios
    test_scenarios = [
        {
            'name': 'HIGH CONFIDENCE BULLISH',
            'signals': {
                'symbol': 'BTC-USDT',
                'technical_confidence': 0.9,
                'news_confidence': 0.8,
                'whale_confidence': 0.7,
                'pattern_confidence': 0.8,
                'news_sentiment': 0.8
            },
            'current_price': 50000.0,
            'volatility_factor': 1.2
        },
        {
            'name': 'MEDIUM CONFIDENCE BEARISH',
            'signals': {
                'symbol': 'BTC-USDT',
                'technical_confidence': 0.3,
                'news_confidence': 0.6,
                'whale_confidence': 0.5,
                'pattern_confidence': 0.4,
                'news_sentiment': -0.6
            },
            'current_price': 48000.0,
            'volatility_factor': 1.8
        },
        {
            'name': 'LOW CONFIDENCE NEUTRAL',
            'signals': {
                'symbol': 'BTC-USDT',
                'technical_confidence': 0.5,
                'news_confidence': 0.4,
                'whale_confidence': 0.5,
                'pattern_confidence': 0.5,
                'news_sentiment': 0.0
            },
            'current_price': 49000.0,
            'volatility_factor': 0.8
        }
    ]

    for scenario in test_scenarios:
        print(f"\n🎯 {scenario['name']}")
        print("-" * 40)

        # Calculate optimal position
        optimal_position = risk_manager.calculate_optimal_position(
            scenario['signals'],
            portfolio,
            scenario['current_price'],
            scenario['volatility_factor']
        )

        print(f"Symbol: {optimal_position.symbol}")
        print(".2f")
        print(f"Leverage: {optimal_position.leverage_ratio:.1f}x")
        print(f"Risk Level: {optimal_position.risk_level.value.upper()}")
        print(f"Leverage Level: {optimal_position.leverage_level.value.upper()}")
        print(".4f")
        print(".2f")
        print(f"Stop Loss: ${optimal_position.stop_loss_price:.0f}")
        print(f"Take Profit: ${optimal_position.take_profit_price:.0f}")
        print(".3f")
        print("\nReasoning:")
        for reason in optimal_position.reasoning:
            print(f"   • {reason}")

    print("\n📊 RISK METRICS OVERVIEW:")
    metrics = risk_manager.get_risk_metrics()
    print(f"   Total Positions: {metrics['total_positions_opened']}")
    print(".3f")
    print(".4f")
    print(".1f")
    print("\nRisk Distribution:")
    for level, count in metrics['risk_distribution'].items():
        print(f"   {level.upper()}: {count} positions")

    if metrics['correlation_warnings']:
        print("\n⚠️  CORRELATION WARNINGS:")
        for warning in metrics['correlation_warnings']:
            print(f"   • {warning}")

    print("\n✅ Dynamic Risk Management Demo Complete")
    print("   System adapts position sizing based on signal confidence!")
    print("   Target: 2-10% position sizes with leverage 5-50x")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_dynamic_risk_management())
