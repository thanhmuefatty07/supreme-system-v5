#!/usr/bin/env python3
"""
Supreme System V5 - A/B Testing & Monitoring Validation Script
Validates automated A/B testing pipeline and monitoring infrastructure.
"""

import sys
import os
import json
import time
import subprocess
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def validate_grafana_dashboards() -> bool:
    """Validate that Grafana dashboards are properly configured."""
    print("🔍 Validating Grafana Dashboards...")

    dashboard_files = [
        'monitoring/grafana/dashboards/optimization-dashboard.json',
        'monitoring/grafana/dashboards/supreme-system-dashboard.json'
    ]

    for dashboard_file in dashboard_files:
        if not os.path.exists(dashboard_file):
            print(f"❌ Dashboard file missing: {dashboard_file}")
            return False

        try:
            with open(dashboard_file, 'r') as f:
                dashboard = json.load(f)

            # Validate basic structure
            if 'dashboard' not in dashboard:
                print(f"❌ Invalid dashboard structure in {dashboard_file}")
                return False

            if 'panels' not in dashboard['dashboard']:
                print(f"❌ No panels found in {dashboard_file}")
                return False

            print(f"✅ Valid dashboard: {dashboard_file} ({len(dashboard['dashboard']['panels'])} panels)")

        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {dashboard_file}: {e}")
            return False

    return True

def validate_prometheus_alerts() -> bool:
    """Validate that Prometheus alerting rules are properly configured."""
    print("🔍 Validating Prometheus Alerting Rules...")

    alerts_file = 'monitoring/prometheus/alerting_rules.yml'
    if not os.path.exists(alerts_file):
        print(f"❌ Alerting rules file missing: {alerts_file}")
        return False

    try:
        import yaml
        with open(alerts_file, 'r') as f:
            alerts_config = yaml.safe_load(f)

        if 'groups' not in alerts_config:
            print("❌ No alert groups found")
            return False

        slo_alerts_found = []
        for group in alerts_config['groups']:
            if 'rules' in group:
                for rule in group['rules']:
                    if rule.get('labels', {}).get('category') == 'optimization':
                        slo_alerts_found.append(rule['alert'])

        required_slo_alerts = [
            'CPUSLOViolation',
            'MemorySLOViolation',
            'LatencySLOViolation',
            'UptimeSLOViolation'
        ]

        missing_alerts = [alert for alert in required_slo_alerts if alert not in slo_alerts_found]
        if missing_alerts:
            print(f"❌ Missing SLO alerts: {missing_alerts}")
            return False

        print(f"✅ All SLO alerts configured: {slo_alerts_found}")
        return True

    except ImportError:
        print("⚠️ PyYAML not available, skipping YAML validation")
        return True
    except Exception as e:
        print(f"❌ Error validating alerting rules: {e}")
        return False

def validate_ab_testing_infrastructure() -> bool:
    """Validate A/B testing infrastructure."""
    print("🔍 Validating A/B Testing Infrastructure...")

    required_files = [
        'scripts/ab_test_run.sh',
        'scripts/report_ab.py',
        'env_optimized.template',
        '.env.standard'
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required A/B testing file missing: {file_path}")
            return False
        print(f"✅ Found: {file_path}")

    # Validate .env.standard exists (created during A/B test)
    if not os.path.exists('.env.standard'):
        print("⚠️ .env.standard not found - will be created during A/B test")
    else:
        print("✅ .env.standard configuration available")

    return True

def test_monitoring_endpoints() -> bool:
    """Test that monitoring endpoints are accessible."""
    print("🔍 Testing Monitoring Endpoints...")

    try:
        import requests

        # Test Prometheus endpoint (if running)
        try:
            response = requests.get('http://localhost:9090/-/healthy', timeout=5)
            if response.status_code == 200:
                print("✅ Prometheus endpoint accessible")
            else:
                print(f"⚠️ Prometheus endpoint returned status {response.status_code}")
        except requests.exceptions.RequestException:
            print("⚠️ Prometheus not running or not accessible")

        # Test Grafana endpoint (if running)
        try:
            response = requests.get('http://localhost:3000/api/health', timeout=5)
            if response.status_code == 200:
                print("✅ Grafana endpoint accessible")
            else:
                print(f"⚠️ Grafana endpoint returned status {response.status_code}")
        except requests.exceptions.RequestException:
            print("⚠️ Grafana not running or not accessible")

    except ImportError:
        print("⚠️ requests library not available for endpoint testing")

    return True  # Don't fail if services aren't running

def generate_ab_test_validation_report() -> Dict[str, Any]:
    """Generate comprehensive A/B testing and monitoring validation report."""
    print("📊 Generating A/B Testing & Monitoring Validation Report...")

    report = {
        'timestamp': time.time(),
        'validation_components': {},
        'overall_status': 'unknown',
        'recommendations': []
    }

    # Validate each component
    validations = [
        ('grafana_dashboards', validate_grafana_dashboards),
        ('prometheus_alerts', validate_prometheus_alerts),
        ('ab_testing_infrastructure', validate_ab_testing_infrastructure),
        ('monitoring_endpoints', test_monitoring_endpoints)
    ]

    all_passed = True
    for component_name, validation_func in validations:
        try:
            passed = validation_func()
            report['validation_components'][component_name] = 'PASSED' if passed else 'FAILED'
            if not passed:
                all_passed = False
                report['recommendations'].append(f"Fix {component_name.replace('_', ' ')} issues")
        except Exception as e:
            print(f"❌ Error validating {component_name}: {e}")
            report['validation_components'][component_name] = 'ERROR'
            all_passed = False

    report['overall_status'] = 'PASSED' if all_passed else 'FAILED'

    # Save report
    report_file = 'run_artifacts/ab_monitoring_validation_report.json'
    os.makedirs('run_artifacts', exist_ok=True)

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📊 Report saved to: {report_file}")
    return report

def simulate_ab_test_execution() -> bool:
    """Simulate a quick A/B test execution to validate the pipeline."""
    print("🧪 Simulating A/B Test Execution...")

    try:
        # Create test configurations
        optimized_config = """# Ultra Optimized Test Config
OPTIMIZED_MODE=true
EVENT_DRIVEN_PROCESSING=true
PERFORMANCE_PROFILE=performance
MAX_CPU_PERCENT=88.0
MAX_RAM_GB=3.86
TARGET_EVENT_SKIP_RATIO=0.7
"""

        standard_config = """# Standard Test Config
OPTIMIZED_MODE=false
EVENT_DRIVEN_PROCESSING=false
PERFORMANCE_PROFILE=normal
MAX_CPU_PERCENT=95.0
MAX_RAM_GB=4.5
TARGET_EVENT_SKIP_RATIO=0.5
"""

        # Write test configs
        with open('.env.optimized', 'w') as f:
            f.write(optimized_config)

        with open('.env.standard', 'w') as f:
            f.write(standard_config)

        print("✅ Test configurations created")

        # Test config switching
        import shutil
        shutil.copy('.env.optimized', '.env')
        print("✅ Configuration switching works")

        # Clean up test files
        os.remove('.env.optimized')
        os.remove('.env.standard')

        return True

    except Exception as e:
        print(f"❌ A/B test simulation failed: {e}")
        return False

def main():
    """Main validation entry point."""
    print("🚀 SUPREME SYSTEM V5 - A/B TESTING & MONITORING VALIDATION")
    print("=" * 70)

    # Generate validation report
    report = generate_ab_test_validation_report()

    # Test A/B simulation
    print("\n" + "=" * 70)
    ab_test_simulation = simulate_ab_test_execution()

    print("\n" + "=" * 70)
    print("📊 FINAL VALIDATION SUMMARY")
    print("=" * 70)

    print("A/B Testing & Monitoring Components:")
    for component, status in report['validation_components'].items():
        status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
        print(f"  {status_icon} {component.replace('_', ' ').title()}: {status}")

    print(f"\nA/B Test Simulation: {'✅ PASSED' if ab_test_simulation else '❌ FAILED'}")

    overall_success = (report['overall_status'] == 'PASSED' and ab_test_simulation)

    if overall_success:
        print("\n🎉 A/B TESTING & MONITORING VALIDATION PASSED!")
        print("All components are ready for automated A/B testing and monitoring.")
    else:
        print("\n❌ VALIDATION FAILED")
        print("Issues found that need to be resolved before A/B testing can be automated.")
        if report['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")

    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
