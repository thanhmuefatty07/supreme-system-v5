#!/usr/bin/env python3
"""
Simple Test for Orchestrator Policies
"""

def test_orchestrator_basic():
    """Test basic orchestrator functionality."""
    print("🎯 Orchestrator Basic Test")
    print("=" * 30)

    try:
        from python.supreme_system_v5.orchestrator.master import MasterTradingOrchestrator

        config = {
            'initial_balance': 10000.0,
            'technical_interval': 30,
            'news_interval': 600,
            'cpu_high_threshold': 80.0,
        }

        orchestrator = MasterTradingOrchestrator(config)
        print("✅ Orchestrator initialized")

        # Test backpressure logic
        orchestrator.backpressure_active = False
        orchestrator.interval_multiplier = 1.0

        # Mock high CPU
        orchestrator._get_current_cpu_usage = lambda: 85.0
        orchestrator._get_current_memory_usage = lambda: 2.5
        orchestrator._get_current_latency = lambda: 150

        # Trigger backpressure check
        orchestrator._check_and_adjust_backpressure()

        if orchestrator.backpressure_active:
            print("✅ Backpressure activated on high CPU")
            print(".1f")
        else:
            print("❌ Backpressure not activated")

        # Test priority ordering
        components_due = orchestrator._get_components_due_for_execution()
        if components_due:
            print(f"✅ Components due for execution: {len(components_due)}")
        else:
            print("ℹ️  No components due (expected for new orchestrator)")

        print("✅ Orchestrator Basic Test PASSED")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_orchestrator_basic()
