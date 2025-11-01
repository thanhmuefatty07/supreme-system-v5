#!/usr/bin/env python3

"""
Real-time performance monitoring for Supreme System V5

Monitors trading core vs dashboard resource usage
"""

import time
import psutil
import json
import os
from datetime import datetime
from typing import Dict, List
import docker



class PerformanceMonitor:

    def __init__(self):
        self.docker_client = docker.from_env()
        self.monitoring = True
        self.alerts = []

        # Performance thresholds
        self.thresholds = {
            'trading_memory_mb': 2048,  # 2GB max
            'dashboard_memory_mb': 512, # 512MB max
            'trading_cpu_percent': 75,   # 75% max
            'dashboard_cpu_percent': 25, # 25% max
            'total_memory_percent': 85,  # 85% system max
            'trading_latency_ms': 25     # 25ms max
        }

    def get_container_stats(self, container_name: str) -> Dict:
        """Get resource statistics for a container"""
        try:
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)

            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_mb = memory_usage / (1024**2)
            memory_percent = (memory_usage / memory_limit) * 100

            # Calculate CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0

            return {
                'container': container_name,
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'cpu_percent': cpu_percent,
                'status': container.status,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'container': container_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def check_performance_alerts(self, trading_stats: Dict, dashboard_stats: Dict) -> List[str]:
        """Check for performance threshold violations"""
        alerts = []

        # Trading memory check
        if trading_stats.get('memory_mb', 0) > self.thresholds['trading_memory_mb']:
            alerts.append(f"⚠️ Trading memory high: {trading_stats['memory_mb']:.1f}MB > {self.thresholds['trading_memory_mb']}MB")

        # Dashboard memory check
        if dashboard_stats.get('memory_mb', 0) > self.thresholds['dashboard_memory_mb']:
            alerts.append(f"⚠️ Dashboard memory high: {dashboard_stats['memory_mb']:.1f}MB > {self.thresholds['dashboard_memory_mb']}MB")

        # Trading CPU check
        if trading_stats.get('cpu_percent', 0) > self.thresholds['trading_cpu_percent']:
            alerts.append(f"⚠️ Trading CPU high: {trading_stats['cpu_percent']:.1f}% > {self.thresholds['trading_cpu_percent']}%")

        # Dashboard CPU check
        if dashboard_stats.get('cpu_percent', 0) > self.thresholds['dashboard_cpu_percent']:
            alerts.append(f"⚠️ Dashboard CPU high: {dashboard_stats['cpu_percent']:.1f}% > {self.thresholds['dashboard_cpu_percent']}%")

        # System memory check
        system_memory = psutil.virtual_memory()
        if system_memory.percent > self.thresholds['total_memory_percent']:
            alerts.append(f"🚨 System memory critical: {system_memory.percent:.1f}% > {self.thresholds['total_memory_percent']}%")

        return alerts

    def log_performance_data(self, data: Dict):
        """Log performance data to file"""
        log_file = '/app/logs/performance.jsonl'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        with open(log_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')

    def monitor_loop(self, interval: int = 30):
        """Main monitoring loop"""
        print(f"🔍 Starting performance monitoring (interval: {interval}s)")
        print("📊 Monitoring trading core vs dashboard resource usage")
        print("=" * 60)

        while self.monitoring:
            try:
                # Get container statistics
                trading_stats = self.get_container_stats('supreme-system-v5_supreme-trading_1')
                dashboard_stats = self.get_container_stats('supreme-system-v5_supreme-dashboard_1')

                # Get system statistics
                system_stats = {
                    'memory_percent': psutil.virtual_memory().percent,
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'swap_percent': psutil.swap_memory().percent,
                    'disk_usage_percent': psutil.disk_usage('/').percent
                }

                # Check for alerts
                alerts = self.check_performance_alerts(trading_stats, dashboard_stats)

                # Prepare monitoring data
                monitoring_data = {
                    'timestamp': datetime.now().isoformat(),
                    'trading': trading_stats,
                    'dashboard': dashboard_stats,
                    'system': system_stats,
                    'alerts': alerts
                }

                # Log data
                self.log_performance_data(monitoring_data)

                # Print status
                self.print_status(monitoring_data)

                # Handle alerts
                if alerts:
                    self.handle_alerts(alerts)

                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(5)

    def print_status(self, data: Dict):
        """Print current performance status"""
        timestamp = datetime.now().strftime('%H:%M:%S')

        trading = data.get('trading', {})
        dashboard = data.get('dashboard', {})
        system = data.get('system', {})

        print(f"\n[{timestamp}] Performance Status:")
        print(f"  🎯 Trading:   {trading.get('memory_mb', 0):.1f}MB ({trading.get('cpu_percent', 0):.1f}% CPU)")
        print(f"  📊 Dashboard: {dashboard.get('memory_mb', 0):.1f}MB ({dashboard.get('cpu_percent', 0):.1f}% CPU)")
        print(f"  💻 System:    {system.get('memory_percent', 0):.1f}% RAM, {system.get('cpu_percent', 0):.1f}% CPU")

        if data.get('alerts'):
            print(f"  🚨 Alerts: {len(data['alerts'])}")
            for alert in data['alerts']:
                print(f"     {alert}")

    def handle_alerts(self, alerts: List[str]):
        """Handle performance alerts"""
        for alert in alerts:
            # Log alert
            print(f"🚨 ALERT: {alert}")

            # Could implement additional alert handling here:
            # - Send notifications
            # - Auto-scale resources
            # - Trigger emergency procedures

        # Store alerts for dashboard
        self.alerts.extend(alerts)

        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]

    def generate_report(self) -> Dict:
        """Generate performance summary report"""
        try:
            # Read recent performance data
            log_file = '/app/logs/performance.jsonl'
            recent_data = []

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-100:]:  # Last 100 entries
                        try:
                            recent_data.append(json.loads(line))
                        except:
                            continue

            if not recent_data:
                return {'error': 'No performance data available'}

            # Calculate averages
            trading_memory_avg = sum(d['trading'].get('memory_mb', 0) for d in recent_data) / len(recent_data)
            dashboard_memory_avg = sum(d['dashboard'].get('memory_mb', 0) for d in recent_data) / len(recent_data)

            trading_cpu_avg = sum(d['trading'].get('cpu_percent', 0) for d in recent_data) / len(recent_data)
            dashboard_cpu_avg = sum(d['dashboard'].get('cpu_percent', 0) for d in recent_data) / len(recent_data)

            # Count recent alerts
            recent_alerts = sum(len(d.get('alerts', [])) for d in recent_data)

            return {
                'period_minutes': len(recent_data) * 0.5,  # Assuming 30s intervals
                'trading_memory_avg_mb': trading_memory_avg,
                'dashboard_memory_avg_mb': dashboard_memory_avg,
                'trading_cpu_avg_percent': trading_cpu_avg,
                'dashboard_cpu_avg_percent': dashboard_cpu_avg,
                'recent_alerts': recent_alerts,
                'memory_isolation_effective': trading_memory_avg < self.thresholds['trading_memory_mb'],
                'cpu_isolation_effective': trading_cpu_avg < self.thresholds['trading_cpu_percent'],
                'dashboard_within_limits': dashboard_memory_avg < self.thresholds['dashboard_memory_mb'],
                'performance_acceptable': recent_alerts < 5,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {'error': f'Report generation failed: {e}'}

def main():
    monitor = PerformanceMonitor()

    try:
        monitor.monitor_loop(interval=30)  # Monitor every 30 seconds
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
    finally:
        # Generate final report
        report = monitor.generate_report()
        print(f"\n📋 FINAL PERFORMANCE REPORT:")
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
