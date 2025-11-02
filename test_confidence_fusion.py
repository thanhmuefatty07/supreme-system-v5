#!/usr/bin/env python3
"""
Test Confidence Fusion in Dynamic Risk Manager
"""

from python.supreme_system_v5.risk import SignalConfidence, DynamicRiskManager, PortfolioState, OptimalPosition

def test_signal_confidence_fusion():
    """Test signal confidence fusion."""
    print("🎯 Signal Confidence Fusion Test")

    # Test default weights
    confidence = SignalConfidence(
        technical_confidence=0.8,
        news_confidence=0.7,
        whale_confidence=0.6,
        pattern_confidence=0.5
    )

    overall = confidence.get_overall_confidence()
    breakdown = confidence.get_signal_breakdown()

    print(f"  Technical: 0.8 (weight: {breakdown['weights']['technical']})")
    print(f"  News: 0.7 (weight: {breakdown['weights']['news']})")
    print(f"  Whale: 0.6 (weight: {breakdown['weights']['whale']})")
    print(f"  Pattern: 0.5 (weight: {breakdown['weights']['pattern']})")
    print(".3f")
    print(f"  Dominant signal: {breakdown['dominant_signal']}")
    print(f"  Signal diversity: {breakdown['signal_diversity']}")

    # Test custom weights
    custom_weights = {
        'technical': 0.6,
        'news': 0.2,
        'whale': 0.15,
        'pattern': 0.05
    }

    confidence_custom = SignalConfidence(
        technical_confidence=0.8,
        news_confidence=0.7,
        whale_confidence=0.6,
        pattern_confidence=0.5,
        weights=custom_weights
    )

    overall_custom = confidence_custom.get_overall_confidence()
    print(".3f")
    if abs(overall - overall_custom) < 0.01:
        print("  ✅ Custom weights validation passed")
    else:
        print("  ❌ Custom weights validation failed")
        return False

    return True

def test_dynamic_risk_manager():
    """Test dynamic risk manager with confidence fusion."""
    print("\n🎯 Dynamic Risk Manager Test")

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

    # Test scenarios
    scenarios = [
        {
            'name': 'High Confidence',
            'signals': {'technical_confidence': 0.85, 'news_confidence': 0.80, 'whale_confidence': 0.70},
            'expected_leverage': 10.0,  # 5x * 2x for high confidence
        },
        {
            'name': 'Medium Confidence',
            'signals': {'technical_confidence': 0.65, 'news_confidence': 0.40, 'whale_confidence': 0.50},
            'expected_leverage': 7.5,  # 5x * 1.5x for medium confidence
        },
        {
            'name': 'Low Confidence',
            'signals': {'technical_confidence': 0.50, 'news_confidence': 0.30, 'whale_confidence': 0.20},
            'expected_leverage': 5.0,  # 5x base for low confidence
        }
    ]

    for scenario in scenarios:
        print(f"\n  Testing {scenario['name']}:")
        print(f"    Signals: {scenario['signals']}")

        optimal_position = risk_manager.calculate_optimal_position(
            signals=scenario['signals'],
            portfolio=portfolio,
            current_price=50000,
            volatility_factor=1.0
        )

        print(".3f")
        print(f"    Leverage: {optimal_position.leverage_ratio:.1f}x")
        print(f"    Risk Level: {optimal_position.risk_level.value}")
        print(f"    Stop Loss: ${optimal_position.stop_loss_price:.0f}")
        print(f"    Take Profit: ${optimal_position.take_profit_price:.0f}")
        print(f"    Reasoning: {optimal_position.reasoning}")

        # Validate leverage adjustment
        assert abs(optimal_position.leverage_ratio - scenario['expected_leverage']) < 1.0, \
            f"Leverage should be ~{scenario['expected_leverage']}x"

    print("  ✅ Dynamic risk manager validation passed")
    return True

def main():
    """Run all confidence fusion tests."""
    print("Supreme System V5 - Confidence Fusion Tests")
    print("=" * 50)

    try:
        test_signal_confidence_fusion()
        test_dynamic_risk_manager()

        print("\n✅ All Confidence Fusion Tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    main()
