#!/usr/bin/env python3
"""
Phase 4: 24-Hour Monitoring & Optimization Script
Automated monitoring, health checks, and performance tracking for production system.
"""

import time
import json
import logging
import requests
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/24h_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Monitoring24H:
    """
    24-Hour Production Monitoring System
    
    Provides comprehensive monitoring, health checks, and performance tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize 24-hour monitoring system."""
        self.config = config or self._default_config()
        self.start_time = datetime.now()
        self.monitoring_duration = timedelta(hours=24)
        self.end_time = self.start_time + self.monitoring_duration
        
        # Monitoring state
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.metrics_collection_interval = self.config.get('metrics_collection_interval', 60)
        self.alert_check_interval = self.config.get('alert_check_interval', 300)
        
        # Health endpoints
        self.health_endpoints = {
            'live': self.config.get('health_live_endpoint', 'http://localhost:8000/health/live'),
            'ready': self.config.get('health_ready_endpoint', 'http://localhost:8000/health/ready'),
            'startup': self.config.get('health_startup_endpoint', 'http://localhost:8000/health/startup'),
            'metrics': self.config.get('metrics_endpoint', 'http://localhost:9090/metrics')
        }
        
        # Metrics storage
        self.metrics_history = []
        self.health_history = []
        self.incidents = []
        
        # Alert thresholds
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 0.02,
            'latency_p95': 1.0,
            'disk_usage': 70.0
        })
        
        logger.info("24-Hour Monitoring System Initialized")
        logger.info(f"Start Time: {self.start_time}")
        logger.info(f"End Time: {self.end_time}")
        logger.info(f"Duration: {self.monitoring_duration}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'health_check_interval': 30,  # seconds
            'metrics_collection_interval': 60,  # seconds
            'alert_check_interval': 300,  # seconds
            'health_live_endpoint': 'http://localhost:8000/health/live',
            'health_ready_endpoint': 'http://localhost:8000/health/ready',
            'metrics_endpoint': 'http://localhost:9090/metrics',
            'alert_thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'error_rate': 0.02,
                'latency_p95': 1.0,
                'disk_usage': 70.0
            }
        }
    
    def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health check results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'live': False,
            'ready': False,
            'startup': False,
            'metrics_available': False,
            'overall_status': 'UNKNOWN'
        }
        
        # Check liveness
        try:
            response = requests.get(self.health_endpoints['live'], timeout=5)
            results['live'] = response.status_code == 200
        except Exception as e:
            logger.warning(f"Liveness check failed: {e}")
            results['live'] = False
        
        # Check readiness
        try:
            response = requests.get(self.health_endpoints['ready'], timeout=5)
            results['ready'] = response.status_code == 200
        except Exception as e:
            logger.warning(f"Readiness check failed: {e}")
            results['ready'] = False
        
        # Check metrics endpoint
        try:
            response = requests.get(self.health_endpoints['metrics'], timeout=5)
            results['metrics_available'] = response.status_code == 200
        except Exception as e:
            logger.warning(f"Metrics check failed: {e}")
            results['metrics_available'] = False
        
        # Determine overall status
        if results['live'] and results['ready']:
            results['overall_status'] = 'HEALTHY'
        elif results['live']:
            results['overall_status'] = 'DEGRADED'
        else:
            results['overall_status'] = 'UNHEALTHY'
        
        self.health_history.append(results)
        return results
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect system and application metrics.
        
        Returns:
            Collected metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {},
            'application': {},
            'trading': {}
        }
        
        try:
            # Collect Prometheus metrics
            response = requests.get(self.health_endpoints['metrics'], timeout=10)
            if response.status_code == 200:
                metrics_text = response.text
                metrics['application'] = self._parse_prometheus_metrics(metrics_text)
        except Exception as e:
            logger.warning(f"Metrics collection failed: {e}")
        
        # Collect system metrics
        try:
            import psutil
            metrics['system'] = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'disk_percent': psutil.disk_usage('/').percent,
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
            }
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, Any]:
        """Parse Prometheus metrics text format."""
        parsed = {}
        
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            # Simple parsing (can be enhanced)
            if 'supreme_' in line:
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0]
                    try:
                        value = float(parts[1])
                        parsed[metric_name] = value
                    except ValueError:
                        pass
        
        return parsed
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for alert conditions.
        
        Returns:
            List of active alerts
        """
        alerts = []
        
        if not self.metrics_history:
            return alerts
        
        # Get latest metrics
        latest_metrics = self.metrics_history[-1]
        
        # Check CPU usage
        if 'system' in latest_metrics and 'cpu_percent' in latest_metrics['system']:
            cpu = latest_metrics['system']['cpu_percent']
            if cpu > self.alert_thresholds['cpu_usage']:
                alerts.append({
                    'severity': 'WARNING',
                    'metric': 'cpu_usage',
                    'value': cpu,
                    'threshold': self.alert_thresholds['cpu_usage'],
                    'message': f'High CPU usage: {cpu:.1f}%'
                })
        
        # Check memory usage
        if 'system' in latest_metrics and 'memory_percent' in latest_metrics['system']:
            memory = latest_metrics['system']['memory_percent']
            if memory > self.alert_thresholds['memory_usage']:
                alerts.append({
                    'severity': 'WARNING',
                    'metric': 'memory_usage',
                    'value': memory,
                    'threshold': self.alert_thresholds['memory_usage'],
                    'message': f'High memory usage: {memory:.1f}%'
                })
        
        # Check disk usage
        if 'system' in latest_metrics and 'disk_percent' in latest_metrics['system']:
            disk = latest_metrics['system']['disk_percent']
            if disk > self.alert_thresholds['disk_usage']:
                alerts.append({
                    'severity': 'INFO',
                    'metric': 'disk_usage',
                    'value': disk,
                    'threshold': self.alert_thresholds['disk_usage'],
                    'message': f'High disk usage: {disk:.1f}%'
                })
        
        # Check health status
        if self.health_history:
            latest_health = self.health_history[-1]
            if latest_health['overall_status'] == 'UNHEALTHY':
                alerts.append({
                    'severity': 'CRITICAL',
                    'metric': 'health_status',
                    'value': latest_health['overall_status'],
                    'message': 'Service is unhealthy!'
                })
        
        # Store incidents
        for alert in alerts:
            if alert['severity'] in ['CRITICAL', 'WARNING']:
                self.incidents.append({
                    'timestamp': datetime.now().isoformat(),
                    **alert
                })
                logger.warning(f"ALERT: {alert['message']}")
        
        return alerts
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate 24-hour monitoring report.
        
        Returns:
            Comprehensive monitoring report
        """
        elapsed = datetime.now() - self.start_time
        elapsed_hours = elapsed.total_seconds() / 3600
        
        # Calculate statistics
        if self.health_history:
            healthy_count = sum(1 for h in self.health_history if h['overall_status'] == 'HEALTHY')
            health_percentage = (healthy_count / len(self.health_history)) * 100
        else:
            health_percentage = 0
        
        # Calculate average metrics
        avg_metrics = {}
        if self.metrics_history:
            cpu_values = [m['system'].get('cpu_percent', 0) for m in self.metrics_history if 'system' in m]
            memory_values = [m['system'].get('memory_percent', 0) for m in self.metrics_history if 'system' in m]
            
            if cpu_values:
                avg_metrics['avg_cpu'] = sum(cpu_values) / len(cpu_values)
                avg_metrics['max_cpu'] = max(cpu_values)
            
            if memory_values:
                avg_metrics['avg_memory'] = sum(memory_values) / len(memory_values)
                avg_metrics['max_memory'] = max(memory_values)
        
        report = {
            'monitoring_period': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_hours': elapsed_hours,
                'target_duration_hours': 24
            },
            'health_summary': {
                'total_checks': len(self.health_history),
                'healthy_percentage': health_percentage,
                'incidents': len(self.incidents)
            },
            'metrics_summary': {
                'total_collections': len(self.metrics_history),
                'average_metrics': avg_metrics
            },
            'incidents': self.incidents,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on monitoring data."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        # Analyze metrics
        cpu_values = [m['system'].get('cpu_percent', 0) for m in self.metrics_history if 'system' in m]
        memory_values = [m['system'].get('memory_percent', 0) for m in self.metrics_history if 'system' in m]
        
        if cpu_values:
            avg_cpu = sum(cpu_values) / len(cpu_values)
            if avg_cpu > 70:
                recommendations.append("Consider optimizing CPU usage or scaling horizontally")
        
        if memory_values:
            avg_memory = sum(memory_values) / len(memory_values)
            if avg_memory > 80:
                recommendations.append("Consider optimizing memory usage or increasing resources")
        
        if len(self.incidents) > 10:
            recommendations.append("High number of incidents detected - review system stability")
        
        return recommendations
    
    def run_monitoring_loop(self):
        """Run continuous monitoring loop."""
        logger.info("Starting 24-hour monitoring loop...")
        
        last_health_check = time.time()
        last_metrics_collection = time.time()
        last_alert_check = time.time()
        
        try:
            while datetime.now() < self.end_time:
                current_time = time.time()
                
                # Health checks
                if current_time - last_health_check >= self.health_check_interval:
                    self.check_health()
                    last_health_check = current_time
                
                # Metrics collection
                if current_time - last_metrics_collection >= self.metrics_collection_interval:
                    self.collect_metrics()
                    last_metrics_collection = current_time
                
                # Alert checks
                if current_time - last_alert_check >= self.alert_check_interval:
                    alerts = self.check_alerts()
                    if alerts:
                        logger.warning(f"Active alerts: {len(alerts)}")
                    last_alert_check = current_time
                
                # Sleep briefly to avoid tight loop
                time.sleep(5)
        
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}", exc_info=True)
        finally:
            # Generate final report
            logger.info("Generating final monitoring report...")
            report = self.generate_report()
            
            # Save report
            report_file = Path('logs/24h_monitoring_report.json')
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Monitoring report saved to: {report_file}")
            logger.info(f"Total incidents: {len(self.incidents)}")
            logger.info(f"Health percentage: {report['health_summary']['healthy_percentage']:.1f}%")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="24-Hour Production Monitoring")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--duration', type=int, default=24, help='Monitoring duration in hours')
    parser.add_argument('--health-interval', type=int, default=30, help='Health check interval (seconds)')
    parser.add_argument('--metrics-interval', type=int, default=60, help='Metrics collection interval (seconds)')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with CLI args
    if args.duration:
        config['monitoring_duration_hours'] = args.duration
    if args.health_interval:
        config['health_check_interval'] = args.health_interval
    if args.metrics_interval:
        config['metrics_collection_interval'] = args.metrics_interval
    
    # Initialize and run monitoring
    monitor = Monitoring24H(config)
    monitor.run_monitoring_loop()


if __name__ == "__main__":
    main()

