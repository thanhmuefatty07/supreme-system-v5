#!/usr/bin/env python3

"""

🏆 SUPREME SYSTEM V5 - REAL-TIME MONITORING DASHBOARD

File 2/4 - Live Monitoring & Alert System

Author: 10,000 Expert Team

Description: Real-time monitoring dashboard for continuous testing system

"""

import json
import time
import os
import logging
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from pathlib import Path

class RealTimeMonitoringDashboard:
    """
    🎯 REAL-TIME MONITORING DASHBOARD
    Live monitoring of continuous testing system with alerts and notifications
    """

    def __init__(self):
        self.dashboard_data = {}
        self.alert_history = []
        self.monitoring_active = True
        self.last_update = datetime.now()

        # Setup monitoring directories
        self.setup_monitoring_infrastructure()

        print("[INIT] REAL-TIME MONITORING DASHBOARD INITIALIZED")
        print("[MONITOR] Monitoring: Continuous Testing System")
        print("[ALERT] Alerts: Real-time notifications active")
        print("[DASHBOARD] Dashboard: Live performance tracking")

    def setup_monitoring_infrastructure(self):
        """Setup monitoring infrastructure"""
        os.makedirs('monitoring', exist_ok=True)
        os.makedirs('monitoring/alerts', exist_ok=True)
        os.makedirs('monitoring/dashboards', exist_ok=True)

        # Setup monitoring logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('monitoring/monitoring_dashboard.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_latest_testing_data(self) -> Dict[str, Any]:
        """
        Load latest testing data from reports and logs
        """
        try:
            # Check for daily reports
            report_files = list(Path('reports').glob('daily_report_day_*.json'))
            if report_files:
                latest_report = max(report_files, key=os.path.getctime)
                with open(latest_report, 'r') as f:
                    daily_data = json.load(f)
            else:
                daily_data = {}

            # Check for final report
            final_report_path = 'reports/continuous_testing_final_report.json'
            if os.path.exists(final_report_path):
                with open(final_report_path, 'r') as f:
                    final_data = json.load(f)
            else:
                final_data = {}

            # Load security logs
            security_logs = self.load_security_logs()

            # Load trading performance
            trading_performance = self.load_trading_performance()

            return {
                'daily_data': daily_data,
                'final_data': final_data,
                'security_logs': security_logs,
                'trading_performance': trading_performance,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error loading testing data: {str(e)}")
            return {}

    def load_security_logs(self) -> List[Dict[str, Any]]:
        """Load and parse security logs"""
        security_logs = []
        try:
            # Check if security logs exist in daily reports
            report_files = list(Path('reports').glob('daily_report_day_*.json'))
            for report_file in report_files:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                    if 'testing_summary' in report_data and 'security_validation' in report_data['testing_summary']:
                        security_logs.append(report_data['testing_summary']['security_validation'])
        except Exception as e:
            self.logger.warning(f"Could not load security logs: {str(e)}")

        return security_logs

    def load_trading_performance(self) -> Dict[str, Any]:
        """Load and aggregate trading performance data"""
        performance_data = {
            'total_trades': 0,
            'total_profit': 0,
            'average_roi': 0,
            'average_win_rate': 0,
            'symbols': {}
        }

        try:
            report_files = list(Path('reports').glob('daily_report_day_*.json'))
            all_roi = []
            all_win_rates = []

            for report_file in report_files:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)

                    if 'testing_summary' in report_data and 'performance_metrics' in report_data['testing_summary']:
                        metrics = report_data['testing_summary']['performance_metrics']
                        performance_data['total_trades'] += metrics.get('total_trades', 0)
                        performance_data['total_profit'] += metrics.get('total_profit', 0)
                        all_roi.append(metrics.get('overall_roi_percent', 0))
                        all_win_rates.append(metrics.get('average_win_rate', 0))

            if all_roi:
                performance_data['average_roi'] = np.mean(all_roi)
                performance_data['average_win_rate'] = np.mean(all_win_rates)

        except Exception as e:
            self.logger.warning(f"Could not load trading performance: {str(e)}")

        return performance_data

    def calculate_system_health(self, testing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall system health score
        """
        health_metrics = {
            'overall_score': 0,
            'trading_health': 0,
            'security_health': 0,
            'performance_health': 0,
            'recommendations': []
        }

        try:
            # Trading health (based on ROI and win rate)
            trading_performance = testing_data.get('trading_performance', {})
            avg_roi = trading_performance.get('average_roi', 0)
            avg_win_rate = trading_performance.get('average_win_rate', 0)

            trading_health = 0
            if avg_roi > 5:
                trading_health = 100
            elif avg_roi > 2:
                trading_health = 80
            elif avg_roi > 0:
                trading_health = 60
            else:
                trading_health = 40

            if avg_win_rate > 60:
                trading_health = min(100, trading_health + 20)
            elif avg_win_rate > 50:
                trading_health = min(100, trading_health + 10)

            health_metrics['trading_health'] = trading_health

            # Security health (based on security logs)
            security_logs = testing_data.get('security_logs', [])
            security_passes = sum(1 for log in security_logs if log.get('overall_status') == 'PASS')
            total_security_tests = len(security_logs)

            if total_security_tests > 0:
                security_health = (security_passes / total_security_tests) * 100
            else:
                security_health = 50  # Default if no tests

            health_metrics['security_health'] = security_health

            # Performance health (based on system stability)
            final_data = testing_data.get('final_data', {})
            if final_data.get('transition_status', {}).get('ready', False):
                performance_health = 90
            else:
                performance_health = 60

            health_metrics['performance_health'] = performance_health

            # Overall score (weighted average)
            health_metrics['overall_score'] = int(
                (trading_health * 0.4) +
                (security_health * 0.4) +
                (performance_health * 0.2)
            )

            # Generate recommendations
            health_metrics['recommendations'] = self.generate_health_recommendations(health_metrics)

        except Exception as e:
            self.logger.error(f"Error calculating system health: {str(e)}")
            health_metrics['overall_score'] = 0
            health_metrics['recommendations'] = ["Error calculating system health"]

        return health_metrics

    def generate_health_recommendations(self, health_metrics: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on metrics"""
        recommendations = []

        if health_metrics['trading_health'] < 70:
            recommendations.append("[CHART] Optimize trading strategy parameters")

        if health_metrics['security_health'] < 90:
            recommendations.append("[SECURE] Review and enhance security measures")

        if health_metrics['performance_health'] < 80:
            recommendations.append("[PERFORMANCE] Improve system performance and stability")

        if health_metrics['overall_score'] >= 90:
            recommendations.append("[LAUNCH] System ready for production deployment!")
        elif health_metrics['overall_score'] >= 70:
            recommendations.append("[OK] System performing well, minor optimizations needed")
        else:
            recommendations.append("[ACTION] System requires significant improvements")

        return recommendations

    def check_alerts(self, testing_data: Dict[str, Any], health_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for conditions that require alerts
        """
        alerts = []
        current_time = datetime.now()

        # Performance alerts
        trading_performance = testing_data.get('trading_performance', {})
        if trading_performance.get('average_roi', 0) < -5:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'PERFORMANCE',
                'message': f"Average ROI critically low: {trading_performance['average_roi']:.2f}%",
                'timestamp': current_time.isoformat(),
                'action': 'Review trading strategy immediately'
            })

        elif trading_performance.get('average_roi', 0) < 0:
            alerts.append({
                'type': 'WARNING',
                'category': 'PERFORMANCE',
                'message': f"Average ROI negative: {trading_performance['average_roi']:.2f}%",
                'timestamp': current_time.isoformat(),
                'action': 'Monitor performance closely'
            })

        # Security alerts
        security_logs = testing_data.get('security_logs', [])
        recent_failures = [log for log in security_logs[-5:] if log.get('overall_status') == 'FAIL']
        if len(recent_failures) >= 2:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'SECURITY',
                'message': f"Multiple security test failures: {len(recent_failures)} recent failures",
                'timestamp': current_time.isoformat(),
                'action': 'Investigate security issues immediately'
            })

        # System health alerts
        if health_metrics['overall_score'] < 50:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'SYSTEM_HEALTH',
                'message': f"System health critically low: {health_metrics['overall_score']}/100",
                'timestamp': current_time.isoformat(),
                'action': 'Comprehensive system review required'
            })

        elif health_metrics['overall_score'] < 70:
            alerts.append({
                'type': 'WARNING',
                'category': 'SYSTEM_HEALTH',
                'message': f"System health below optimal: {health_metrics['overall_score']}/100",
                'timestamp': current_time.isoformat(),
                'action': 'Address recommendations promptly'
            })

        # Data freshness alert
        last_updated_str = testing_data.get('last_updated')
        if last_updated_str:
            try:
                last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
                if current_time - last_updated > timedelta(hours=2):
                    alerts.append({
                        'type': 'WARNING',
                        'category': 'DATA',
                        'message': f"Data stale - last update: {last_updated.strftime('%Y-%m-%d %H:%M')}",
                        'timestamp': current_time.isoformat(),
                        'action': 'Check continuous testing system'
                    })
            except:
                pass

        return alerts

    def display_dashboard(self, testing_data: Dict[str, Any], health_metrics: Dict[str, Any], alerts: List[Dict[str, Any]]):
        """
        Display real-time monitoring dashboard
        """
        os.system('cls' if os.name == 'nt' else 'clear')

        print("\n" + "="*80)
        print("[AWARD] SUPREME SYSTEM V5 - REAL-TIME MONITORING DASHBOARD")
        print("="*80)
        print(f"[UPDATE] Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[MONITOR] Monitoring: Continuous Testing System")
        print("="*80)

        # System Health Section
        print("\n[HEALTH] SYSTEM HEALTH OVERVIEW")
        print("-" * 40)

        overall_score = health_metrics['overall_score']
        health_color = "[GREEN]" if overall_score >= 80 else "[YELLOW]" if overall_score >= 60 else "[RED]"

        print(f"{health_color} Overall Health: {overall_score}/100")
        print(f"[CHART] Trading Health: {health_metrics['trading_health']}/100")
        print(f"[SECURE] Security Health: {health_metrics['security_health']}/100")
        print(f"[PERFORMANCE] Performance Health: {health_metrics['performance_health']}/100")

        # Performance Metrics Section
        print("\n[RESULT] PERFORMANCE METRICS")
        print("-" * 40)

        trading_data = testing_data.get('trading_performance', {})
        print(f"[MONEY] Total Trades: {trading_data.get('total_trades', 0):,}")
        print(f"[CHART] Total Profit: ${trading_data.get('total_profit', 0):+.2f}")
        print(f"[TARGET] Average ROI: {trading_data.get('average_roi', 0):+.2f}%")
        print(f"[AWARD] Average Win Rate: {trading_data.get('average_win_rate', 0):.1f}%")

        # Security Status Section
        print("\n[SECURE] SECURITY STATUS")
        print("-" * 40)

        security_logs = testing_data.get('security_logs', [])
        recent_security = security_logs[-1] if security_logs else {}

        if recent_security:
            status_icon = "[PASS]" if recent_security.get('overall_status') == 'PASS' else "[FAIL]"
            print(f"{status_icon} Latest Security: {recent_security.get('overall_status', 'UNKNOWN')}")
            print(f"[DATE] Last Test: {recent_security.get('timestamp', 'Unknown')}")
        else:
            print("[EMPTY] No security data available")

        # Alerts Section
        print("\n[ALERT] ACTIVE ALERTS")
        print("-" * 40)

        if alerts:
            for i, alert in enumerate(alerts[:5], 1):  # Show max 5 alerts
                alert_type = alert['type']
                alert_icon = "[CRITICAL]" if alert_type == 'CRITICAL' else "[WARNING]" if alert_type == 'WARNING' else "[INFO]"
                print(f"{alert_icon} {alert['category']}: {alert['message']}")
                print(f"   [TIME] {alert['timestamp']} | [ACTION] {alert['action']}")
                if i < len(alerts[:5]):
                    print()
        else:
            print("[OK] No active alerts - System operating normally")

        # Recommendations Section
        print("\n[TIPS] RECOMMENDATIONS")
        print("-" * 40)

        recommendations = health_metrics.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                print(f"• {rec}")
        else:
            print("No recommendations at this time")

        # Testing Progress Section
        print("\n[PROGRESS] TESTING PROGRESS")
        print("-" * 40)

        final_data = testing_data.get('final_data', {})
        if final_data:
            transition_status = final_data.get('transition_status', {})
            if transition_status.get('ready', False):
                print("[COMPLETE] TESTING COMPLETE - READY FOR REAL TRADING!")
                print("[LAUNCH] Next: Deploy with real capital and exchange connections")
            else:
                print("[TEST] Continuous testing in progress...")
                print("[WAIT] Evaluating transition readiness")
        else:
            # Check daily reports for progress
            report_files = list(Path('reports').glob('daily_report_day_*.json'))
            if report_files:
                days_completed = len(report_files)
                print(f"[RESULT] Days Completed: {days_completed}/7")
                print(f"[CHART] Progress: {(days_completed/7)*100:.1f}%")
            else:
                print("[EMPTY] No testing data available")

        print("\n" + "="*80)
        print("[TIPS] Tips: Dashboard updates every 30 seconds. Check logs for detailed information.")
        print("="*80)

    def save_dashboard_data(self, testing_data: Dict[str, Any], health_metrics: Dict[str, Any], alerts: List[Dict[str, Any]]):
        """
        Save dashboard data for historical tracking
        """
        dashboard_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'testing_data': testing_data,
            'health_metrics': health_metrics,
            'alerts': alerts,
            'monitoring_uptime': str(datetime.now() - self.last_update)
        }

        # Save current dashboard state
        with open('monitoring/dashboards/current_dashboard.json', 'w') as f:
            json.dump(dashboard_snapshot, f, indent=2, default=str)

        # Save alert history
        self.alert_history.extend(alerts)
        with open('monitoring/alerts/alert_history.json', 'w') as f:
            json.dump(self.alert_history[-100:], f, indent=2, default=str)  # Keep last 100 alerts

        # Log significant alerts
        for alert in alerts:
            if alert['type'] in ['CRITICAL', 'WARNING']:
                self.logger.warning(f"ALERT: {alert['category']} - {alert['message']}")

    def send_notifications(self, alerts: List[Dict[str, Any]]):
        """
        Send notifications for critical alerts
        In production, this would integrate with email/SMS/telegram
        """
        for alert in alerts:
            if alert['type'] == 'CRITICAL':
                # In production: Send email/SMS/telegram notification
                self.logger.critical(f"CRITICAL ALERT REQUIRES IMMEDIATE ATTENTION: {alert['message']}")

                # Create alert file for external monitoring systems
                alert_file = f"monitoring/alerts/critical_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(alert_file, 'w') as f:
                    json.dump(alert, f, indent=2, default=str)

    def start_monitoring(self, update_interval: int = 30):
        """
        Start continuous monitoring
        """
        self.logger.info("[LAUNCH] STARTING REAL-TIME MONITORING DASHBOARD")
        self.logger.info(f"[RESULT] Update interval: {update_interval} seconds")

        try:
            while self.monitoring_active:
                # Load latest data
                testing_data = self.load_latest_testing_data()

                # Calculate system health
                health_metrics = self.calculate_system_health(testing_data)

                # Check for alerts
                alerts = self.check_alerts(testing_data, health_metrics)

                # Display dashboard
                self.display_dashboard(testing_data, health_metrics, alerts)

                # Save data and send notifications
                self.save_dashboard_data(testing_data, health_metrics, alerts)
                self.send_notifications(alerts)

                # Wait for next update
                time.sleep(update_interval)

        except KeyboardInterrupt:
            self.logger.info("Monitoring dashboard stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {str(e)}")

    def stop_monitoring(self):
        """Stop monitoring dashboard"""
        self.monitoring_active = False
        self.logger.info("Monitoring dashboard stopped")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("[AWARD] SUPREME SYSTEM V5 - REAL-TIME MONITORING DASHBOARD")
    print("[TARGET] Live Monitoring & Alert System")
    print("="*80)

    try:
        # Initialize monitoring dashboard
        dashboard = RealTimeMonitoringDashboard()

        print("\n[TIPS] Starting real-time monitoring...")
        print("[RESULT] Dashboard will update every 30 seconds")
        print("[ALERT] Alerts will be displayed for critical conditions")
        print("[STOP] Press Ctrl+C to stop monitoring\n")

        # Start monitoring
        dashboard.start_monitoring(update_interval=30)

        return 0

    except KeyboardInterrupt:
        print("\n[STOP] Monitoring dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"[ERROR] Error starting monitoring dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
