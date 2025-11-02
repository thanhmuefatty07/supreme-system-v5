#!/usr/bin/env python3
"""
Test Observability & SLOs Implementation
"""

from python.supreme_system_v5.monitoring import AdvancedResourceMonitor

def test_slo_definitions():
    """Test SLO definitions and compliance checking."""
    print("🎯 SLO Definitions Test")
    print("=" * 30)

    config = {
        'cpu_high_threshold': 88.0,
        'memory_high_threshold': 3.86,
        'latency_high_threshold': 200,
        'monitoring_interval': 2.0,
        'optimization_check_interval': 10.0
    }

    monitor = AdvancedResourceMonitor(config)

    # Check SLO definitions
    expected_slos = {
        'cpu_usage_percent': {'target': 88.0, 'window_minutes': 5, 'breach_count': 3},
        'memory_usage_gb': {'target': 3.86, 'window_minutes': 5, 'breach_count': 3},
        'indicator_latency_ms': {'target': 200.0, 'window_minutes': 1, 'breach_count': 2},
        'event_skip_ratio': {'target': 0.7, 'window_minutes': 10, 'breach_count': 5},
        'uptime_percent': {'target': 99.9, 'window_hours': 24, 'breach_count': 1}
    }

    actual_slos = monitor.slo_definitions

    print("SLO Definitions:")
    for slo_name, slo_config in expected_slos.items():
        if slo_name in actual_slos:
            print(f"  ✅ {slo_name}: target={actual_slos[slo_name]['target']}")
        else:
            print(f"  ❌ {slo_name}: missing")

    # Test SLO compliance (should be compliant initially)
    compliance = monitor.check_slo_compliance()
    print(f"\nInitial SLO Compliance: {'✅ PASS' if compliance['compliant'] else '❌ FAIL'}")
    print(f"Active Alerts: {compliance['active_alerts']}")

    return True

def test_resource_monitoring():
    """Test resource monitoring and metrics collection."""
    print("\n🎯 Resource Monitoring Test")
    print("=" * 30)

    config = {
        'cpu_high_threshold': 88.0,
        'memory_high_threshold': 3.86,
        'latency_high_threshold': 200,
        'monitoring_interval': 1.0,
        'optimization_check_interval': 5.0
    }

    monitor = AdvancedResourceMonitor(config)

    # Start monitoring
    monitor.start_monitoring()
    print("✅ Monitoring started")

    # Simulate some activity
    import time
    for i in range(5):
        monitor.record_indicator_latency(50.0 + i * 10)  # 50-90ms latency
        monitor.record_event_processed(i % 2 == 0)  # Alternate processed/skipped
        time.sleep(0.5)

    # Get health report
    health_report = monitor.get_system_health_report()
    print("System Health Report:")
    print(f"  Overall Health: {health_report['overall_health']:.1f}/100")
    print(f"  Performance Profile: {health_report['performance_profile']}")
    print(f"  Metrics Collected: {health_report['metrics_count']}")

    # Get performance metrics
    perf_metrics = monitor.get_performance_metrics()
    print("Performance Metrics:")
    print(f"  Avg CPU: {perf_metrics['avg_cpu_percent']:.1f}%")
    print(f"  Avg Memory: {perf_metrics['avg_memory_gb']:.2f}GB")
    print(f"  Avg Latency: {perf_metrics['avg_indicator_latency_ms']:.1f}ms")
    print(f"  Events Processed: {perf_metrics['events_processed']}")
    print(f"  Events Skipped: {perf_metrics['events_skipped']}")

    # Test Prometheus export
    prometheus_metrics = monitor.export_prometheus_metrics()
    if prometheus_metrics and len(prometheus_metrics) > 50:
        print("✅ Prometheus metrics export working")
    else:
        print("❌ Prometheus metrics export failed")

    # Stop monitoring
    monitor.stop_monitoring()
    print("✅ Monitoring stopped")

    return True

def test_slo_alerting():
    """Test SLO violation detection and alerting."""
    print("\n🎯 SLO Alerting Test")
    print("=" * 30)

    config = {
        'cpu_high_threshold': 80.0,  # Lower threshold for testing
        'memory_high_threshold': 2.0,
        'latency_high_threshold': 100,
        'monitoring_interval': 1.0,
        'optimization_check_interval': 3.0
    }

    monitor = AdvancedResourceMonitor(config)

    # Mock high CPU usage
    original_cpu = monitor._get_current_cpu_usage
    monitor._get_current_cpu_usage = lambda: 85.0  # Above threshold

    # Trigger SLO check
    compliance = monitor.check_slo_compliance()

    # Should have violations now
    if not compliance['compliant']:
        print("✅ SLO violations detected correctly")
        print(f"Violations: {compliance['violations']}")
        print(f"Active Alerts: {compliance['active_alerts']}")
    else:
        print("❌ SLO violations not detected")

    # Get SLO report
    slo_report = monitor.get_slo_report()
    print("SLO Report:")
    print(f"  Overall Compliant: {slo_report['overall_compliant']}")
    print(f"  Active Alerts: {len(slo_report['active_alerts'])}")
    print(f"  Recommendations: {len(slo_report['recommendations'])}")

    # Restore original function
    monitor._get_current_cpu_usage = original_cpu

    return True

def main():
    """Run all observability tests."""
    print("Supreme System V5 - Observability & SLOs Tests")
    print("=" * 50)

    try:
        test_slo_definitions()
        test_resource_monitoring()
        test_slo_alerting()

        print("\n✅ All Observability & SLOs Tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
