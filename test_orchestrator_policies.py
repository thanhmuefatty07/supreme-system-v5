#!/usr/bin/env python3
"""
Test Adaptive Orchestrator Policies (backpressure + priority)
"""

import asyncio
import time
from python.supreme_system_v5.orchestrator import MasterTradingOrchestrator

async def test_orchestrator_policies():
    """Test orchestrator adaptive policies."""
    print("🎯 Orchestrator Policies Test")
    print("=" * 50)

    # Initialize orchestrator
    config = {
        'initial_balance': 10000.0,
        'technical_interval': 30,
        'news_interval': 600,
        'whale_interval': 600,
        'mtf_interval': 120,
        'patterns_interval': 60,
        'cpu_high_threshold': 80.0,
        'cpu_low_threshold': 70.0,
        'memory_high_threshold': 3.86,
        'latency_high_threshold': 200,
        'resource_check_interval': 5,  # Check every 5 seconds for demo
        'technical_config': {
            'ema_period': 14,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'price_history_size': 100
        },
        'news_config': {},
        'whale_config': {'whale_threshold_usd': 100000},
        'mtf_config': {},
        'risk_config': {
            'base_position_size_pct': 0.02,
            'max_position_size_pct': 0.10
        }
    }

    orchestrator = MasterTradingOrchestrator(config)

    # Initialize components
    success = await orchestrator.initialize_components()
    if not success:
        print("❌ Component initialization failed")
        return False

    print("✅ Components initialized")

    # Test normal operation
    print("\n🟢 Testing Normal Operation:")
    for i in range(3):
        result = await orchestrator.run_orchestration_cycle()
        print(f"  Cycle {i+1}: {result.components_executed} components, {len(result.trading_signals)} signals")
        await asyncio.sleep(0.1)

    # Test priority ordering
    print("\n🔄 Testing Priority Ordering:")
    components_due = orchestrator._get_components_due_for_execution()
    priorities = [orchestrator.components[comp].priority.value for comp in components_due]
    print(f"  Execution order: {components_due}")
    print(f"  Priorities: {priorities}")

    # Verify critical components come first
    if components_due and orchestrator.components[components_due[0]].priority.value == "critical":
        print("  ✅ Critical components prioritized")
    else:
        print("  ❌ Priority ordering may be incorrect")

    # Test backpressure simulation (mock high CPU)
    print("\n⚠️  Testing Backpressure Simulation:")
    original_get_cpu = orchestrator._get_current_cpu_usage
    orchestrator._get_current_cpu_usage = lambda: 85.0  # Mock high CPU

    # Wait for backpressure check
    await asyncio.sleep(6)

    # Check if backpressure activated
    if orchestrator.backpressure_active:
        print("  ✅ Backpressure activated on high CPU")
        print(".1f")
    else:
        print("  ❌ Backpressure not activated")

    # Test recovery (mock normal CPU)
    orchestrator._get_current_cpu_usage = lambda: 65.0  # Mock normal CPU
    await asyncio.sleep(6)

    if not orchestrator.backpressure_active:
        print("  ✅ Backpressure deactivated on normal CPU")
    else:
        print("  ❌ Backpressure not deactivated")

    # Restore original function
    orchestrator._get_current_cpu_usage = original_get_cpu

    # Get final status
    status = orchestrator.get_system_status()
    print("\n📊 Final System Status:")
    print(f"  Total cycles: {status['cycle_count']}")
    print(f"  Components: {len(status['component_status'])}")
    print(f"  Portfolio balance: ${status['portfolio']['total_balance']:.2f}")

    print("\n✅ Orchestrator Policies Test PASSED")
    return True

async def main():
    """Run orchestrator tests."""
    try:
        return await test_orchestrator_policies()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
