"""
🚨 Supreme System V5 - Alert Management System
Alert threshold monitoring and notification system

Features:
- Configured alert thresholds for 8 Tier-1 metrics
- Multi-severity alert levels (warning, critical)
- Duration-based alert triggering
- Alert suppression and recovery
- Integration with monitoring dashboard
- WebSocket alert broadcasting
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger("supreme_alerts")


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert state tracking"""
    OK = "ok"
    TRIGGERED = "triggered"
    SUPPRESSED = "suppressed"
    RECOVERED = "recovered"


@dataclass
class AlertThreshold:
    """Alert threshold configuration"""
    metric_name: str
    severity: AlertSeverity
    threshold_value: float
    comparison: str = ">"
    duration_seconds: float = 60.0
    description: str = ""
    recovery_threshold: Optional[float] = None
    suppression_duration_seconds: float = 300.0  # 5 minutes


@dataclass
class Alert:
    """Active alert instance"""
    threshold: AlertThreshold
    triggered_at: datetime
    current_value: float
    state: AlertState = AlertState.TRIGGERED
    suppressed_until: Optional[datetime] = None
    recovery_count: int = 0
    last_notification: Optional[datetime] = None
    
    def is_suppressed(self) -> bool:
        """Check if alert is currently suppressed"""
        return (self.suppressed_until is not None and 
                datetime.utcnow() < self.suppressed_until)
    
    def should_recover(self, current_value: float) -> bool:
        """Check if alert should recover"""
        if self.threshold.recovery_threshold is None:
            return False
        
        if self.threshold.comparison == ">":
            return current_value < self.threshold.recovery_threshold
        elif self.threshold.comparison == "<":
            return current_value > self.threshold.recovery_threshold
        elif self.threshold.comparison == "==":
            return current_value != self.threshold.threshold_value
        
        return False


class AlertManager:
    """Alert management system with configured thresholds"""
    
    def __init__(self):
        self.thresholds: Dict[str, List[AlertThreshold]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_callbacks: List[Callable] = []
        
        # Metric value tracking for duration-based alerts
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_check = time.time()
        
        # Initialize default thresholds as per requirements
        self._setup_default_thresholds()
        
        logger.info("🚨 Alert manager initialized")
        logger.info(f"   Configured thresholds: {len(self.thresholds)} metrics")
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds as per requirements"""
        
        # API latency: >50ms for 3 minutes (warning), >100ms (critical)
        self.add_threshold(AlertThreshold(
            metric_name="api_latency_ms",
            severity=AlertSeverity.WARNING,
            threshold_value=50.0,
            comparison=">",
            duration_seconds=180.0,  # 3 minutes
            description="API latency high for 3 minutes",
            recovery_threshold=40.0
        ))
        
        self.add_threshold(AlertThreshold(
            metric_name="api_latency_ms",
            severity=AlertSeverity.CRITICAL,
            threshold_value=100.0,
            comparison=">",
            duration_seconds=60.0,  # 1 minute
            description="API latency critically high",
            recovery_threshold=80.0
        ))
        
        # Trading loop: >50ms for 1 minute
        self.add_threshold(AlertThreshold(
            metric_name="trading_loop_ms",
            severity=AlertSeverity.WARNING,
            threshold_value=50.0,
            comparison=">",
            duration_seconds=60.0,  # 1 minute
            description="Trading loop latency high for 1 minute",
            recovery_threshold=40.0
        ))
        
        # Exchange connectivity: down for >10 seconds
        self.add_threshold(AlertThreshold(
            metric_name="exchange_connectivity",
            severity=AlertSeverity.CRITICAL,
            threshold_value=0,
            comparison="==",
            duration_seconds=10.0,  # 10 seconds
            description="Exchange connection down for >10 seconds",
            recovery_threshold=1
        ))
        
        # Daily PnL: beyond max daily loss
        self.add_threshold(AlertThreshold(
            metric_name="pnl_daily",
            severity=AlertSeverity.WARNING,
            threshold_value=-500.0,  # -$500
            comparison="<",
            duration_seconds=30.0,
            description="Daily loss approaching limit",
            recovery_threshold=-400.0
        ))
        
        self.add_threshold(AlertThreshold(
            metric_name="pnl_daily",
            severity=AlertSeverity.CRITICAL,
            threshold_value=-1000.0,  # -$1000
            comparison="<",
            duration_seconds=10.0,
            description="Daily loss limit exceeded",
            recovery_threshold=-800.0
        ))
        
        # WebSocket clients: high connection count
        self.add_threshold(AlertThreshold(
            metric_name="websocket_clients",
            severity=AlertSeverity.WARNING,
            threshold_value=1000,
            comparison=">",
            duration_seconds=120.0,  # 2 minutes
            description="High WebSocket connection count",
            recovery_threshold=800
        ))
        
        # Max drawdown percentage
        self.add_threshold(AlertThreshold(
            metric_name="max_drawdown_pct",
            severity=AlertSeverity.WARNING,
            threshold_value=5.0,  # 5%
            comparison=">",
            duration_seconds=60.0,
            description="Drawdown approaching risk limit",
            recovery_threshold=4.0
        ))
        
        self.add_threshold(AlertThreshold(
            metric_name="max_drawdown_pct",
            severity=AlertSeverity.CRITICAL,
            threshold_value=10.0,  # 10%
            comparison=">",
            duration_seconds=30.0,
            description="Drawdown exceeds risk limit",
            recovery_threshold=8.0
        ))
    
    def add_threshold(self, threshold: AlertThreshold):
        """Add alert threshold"""
        if threshold.metric_name not in self.thresholds:
            self.thresholds[threshold.metric_name] = []
        
        self.thresholds[threshold.metric_name].append(threshold)
        logger.info(f"🚨 Alert threshold added: {threshold.metric_name} {threshold.comparison} {threshold.threshold_value} ({threshold.severity.value})")
    
    def update_metric_value(self, metric_name: str, value: float):
        """Update metric value and check thresholds"""
        timestamp = time.time()
        
        # Store metric value with timestamp
        self.metric_values[metric_name].append({
            "timestamp": timestamp,
            "value": value
        })
        
        # Check thresholds for this metric
        if metric_name in self.thresholds:
            for threshold in self.thresholds[metric_name]:
                self._check_threshold(threshold, value, timestamp)
    
    def _check_threshold(self, threshold: AlertThreshold, current_value: float, timestamp: float):
        """Check if threshold should trigger or recover"""
        alert_key = f"{threshold.metric_name}_{threshold.severity.value}"
        
        # Check if threshold condition is met
        condition_met = self._evaluate_threshold_condition(threshold, current_value)
        
        if condition_met:
            self._handle_threshold_breach(threshold, current_value, timestamp, alert_key)
        else:
            self._handle_threshold_recovery(threshold, current_value, alert_key)
    
    def _evaluate_threshold_condition(self, threshold: AlertThreshold, value: float) -> bool:
        """Evaluate if threshold condition is met"""
        if threshold.comparison == ">":
            return value > threshold.threshold_value
        elif threshold.comparison == "<":
            return value < threshold.threshold_value
        elif threshold.comparison == "==":
            return value == threshold.threshold_value
        elif threshold.comparison == ">=":
            return value >= threshold.threshold_value
        elif threshold.comparison == "<=":
            return value <= threshold.threshold_value
        
        return False
    
    def _handle_threshold_breach(self, threshold: AlertThreshold, value: float, timestamp: float, alert_key: str):
        """Handle threshold breach"""
        # Check if we need duration-based alerting
        if threshold.duration_seconds > 0:
            # Check if condition has been met for required duration
            metric_history = self.metric_values[threshold.metric_name]
            breach_start_time = None
            
            # Find when the breach started
            for entry in reversed(metric_history):
                if self._evaluate_threshold_condition(threshold, entry["value"]):
                    breach_start_time = entry["timestamp"]
                else:
                    break
            
            if breach_start_time is None:
                return
            
            # Check if breach has lasted long enough
            breach_duration = timestamp - breach_start_time
            if breach_duration < threshold.duration_seconds:
                return
        
        # Trigger or update alert
        if alert_key in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_key]
            alert.current_value = value
            alert.state = AlertState.TRIGGERED
        else:
            # Create new alert
            alert = Alert(
                threshold=threshold,
                triggered_at=datetime.utcnow(),
                current_value=value,
                state=AlertState.TRIGGERED
            )
            self.active_alerts[alert_key] = alert
            
            # Send notification
            asyncio.create_task(self._send_alert_notification(alert))
    
    def _handle_threshold_recovery(self, threshold: AlertThreshold, value: float, alert_key: str):
        """Handle threshold recovery"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            
            # Check if should recover
            if alert.should_recover(value):
                alert.state = AlertState.RECOVERED
                alert.recovery_count += 1
                
                # Send recovery notification
                asyncio.create_task(self._send_recovery_notification(alert))
                
                # Move to history and remove from active
                self.alert_history.append(alert)
                del self.active_alerts[alert_key]
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification to registered callbacks"""
        if alert.is_suppressed():
            return
        
        # Prepare alert data
        alert_data = {
            "metric_name": alert.threshold.metric_name,
            "severity": alert.threshold.severity.value,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold.threshold_value,
            "description": alert.threshold.description,
            "triggered_at": alert.triggered_at.isoformat(),
            "comparison": alert.threshold.comparison
        }
        
        # Send to registered callbacks
        for callback in self.notification_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"Alert notification callback failed: {e}")
        
        # Update last notification time
        alert.last_notification = datetime.utcnow()
        
        logger.warning(f"🚨 ALERT: {alert.threshold.metric_name} = {alert.current_value} {alert.threshold.comparison} {alert.threshold.threshold_value}")
    
    async def _send_recovery_notification(self, alert: Alert):
        """Send recovery notification"""
        recovery_data = {
            "metric_name": alert.threshold.metric_name,
            "severity": alert.threshold.severity.value,
            "recovered_at": datetime.utcnow().isoformat(),
            "final_value": alert.current_value,
            "description": f"Recovered: {alert.threshold.description}"
        }
        
        # Send to registered callbacks
        for callback in self.notification_callbacks:
            try:
                await callback(recovery_data)
            except Exception as e:
                logger.error(f"Recovery notification callback failed: {e}")
        
        logger.info(f"✅ RECOVERED: {alert.threshold.metric_name} alert recovered")
    
    def add_notification_callback(self, callback: Callable):
        """Add notification callback function"""
        self.notification_callbacks.append(callback)
        logger.info("🔔 Alert notification callback added")
    
    def suppress_alert(self, metric_name: str, severity: AlertSeverity, duration_minutes: float = 5.0):
        """Temporarily suppress alert"""
        alert_key = f"{metric_name}_{severity.value}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.suppressed_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
            alert.state = AlertState.SUPPRESSED
            
            logger.info(f"🔇 Alert suppressed: {alert_key} for {duration_minutes} minutes")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        active_alerts = []
        
        for alert_key, alert in self.active_alerts.items():
            alert_data = {
                "alert_key": alert_key,
                "metric_name": alert.threshold.metric_name,
                "severity": alert.threshold.severity.value,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold.threshold_value,
                "description": alert.threshold.description,
                "triggered_at": alert.triggered_at.isoformat(),
                "state": alert.state.value,
                "is_suppressed": alert.is_suppressed(),
                "recovery_count": alert.recovery_count
            }
            active_alerts.append(alert_data)
        
        return active_alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary"""
        active_count_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            if not alert.is_suppressed():
                active_count_by_severity[alert.threshold.severity.value] += 1
        
        return {
            "total_thresholds": sum(len(thresholds) for thresholds in self.thresholds.values()),
            "active_alerts": len(self.active_alerts),
            "suppressed_alerts": sum(1 for alert in self.active_alerts.values() if alert.is_suppressed()),
            "alerts_by_severity": dict(active_count_by_severity),
            "alert_history_count": len(self.alert_history),
            "last_check": datetime.fromtimestamp(self.last_check).isoformat(),
            "notification_callbacks": len(self.notification_callbacks)
        }


def configure_alert_thresholds() -> AlertManager:
    """Configure alert thresholds as per requirements"""
    alert_manager = AlertManager()
    
    # API latency thresholds
    alert_manager.add_threshold(AlertThreshold(
        metric_name="api_latency_ms",
        severity=AlertSeverity.WARNING,
        threshold_value=50.0,
        comparison=">",
        duration_seconds=180.0,  # 3 minutes
        description="API latency > 50ms for 3 minutes",
        recovery_threshold=40.0
    ))
    
    # Trading loop latency thresholds  
    alert_manager.add_threshold(AlertThreshold(
        metric_name="trading_loop_ms",
        severity=AlertSeverity.WARNING,
        threshold_value=50.0,
        comparison=">",
        duration_seconds=60.0,  # 1 minute
        description="Trading loop > 50ms for 1 minute",
        recovery_threshold=40.0
    ))
    
    # Exchange connectivity
    alert_manager.add_threshold(AlertThreshold(
        metric_name="exchange_connectivity",
        severity=AlertSeverity.CRITICAL,
        threshold_value=0,
        comparison="==",
        duration_seconds=10.0,  # 10 seconds
        description="Exchange connection down > 10 seconds",
        recovery_threshold=1
    ))
    
    # Daily PnL loss limit
    alert_manager.add_threshold(AlertThreshold(
        metric_name="pnl_daily",
        severity=AlertSeverity.CRITICAL,
        threshold_value=-1000.0,  # Max daily loss as configured
        comparison="<",
        duration_seconds=30.0,
        description="Daily PnL below max daily loss limit",
        recovery_threshold=-800.0
    ))
    
    # Additional thresholds for completeness
    alert_manager.add_threshold(AlertThreshold(
        metric_name="websocket_clients",
        severity=AlertSeverity.WARNING,
        threshold_value=1000,
        comparison=">",
        duration_seconds=120.0,
        description="High WebSocket connection count",
        recovery_threshold=800
    ))
    
    alert_manager.add_threshold(AlertThreshold(
        metric_name="max_drawdown_pct",
        severity=AlertSeverity.CRITICAL,
        threshold_value=10.0,  # 10% drawdown
        comparison=">",
        duration_seconds=60.0,
        description="Maximum drawdown exceeds 10%",
        recovery_threshold=8.0
    ))
    
    logger.info(f"✅ Alert thresholds configured: {len(alert_manager.thresholds)} metrics")
    return alert_manager


# Global alert manager instance
alert_manager = configure_alert_thresholds()


# Integration functions
async def check_metric_alerts(metric_name: str, value: float):
    """Check metric against configured alert thresholds"""
    alert_manager.update_metric_value(metric_name, value)


async def broadcast_alert_to_websocket(alert_data: Dict[str, Any]):
    """Broadcast alert to WebSocket clients"""
    try:
        from ..api.websocket import broadcast_system_alert
        
        severity = alert_data.get("severity", "info")
        message = alert_data.get("description", "System alert")
        
        await broadcast_system_alert(
            message=message,
            severity=severity,
            metric=alert_data.get("metric_name"),
            current_value=alert_data.get("current_value"),
            threshold=alert_data.get("threshold_value")
        )
    except Exception as e:
        logger.error(f"Failed to broadcast alert to WebSocket: {e}")


# Register WebSocket broadcast as default callback
alert_manager.add_notification_callback(broadcast_alert_to_websocket)


if __name__ == "__main__":
    # Demo alert system
    import asyncio
    
    async def demo():
        print("🚨 Supreme System V5 Alert System Demo")
        print("=" * 45)
        
        # Configure alerts
        alert_mgr = configure_alert_thresholds()
        
        # Simulate metric breaches
        print("📋 Testing alert thresholds...")
        
        # Simulate API latency breach
        alert_mgr.update_metric_value("api_latency_ms", 75.0)  # Above warning
        await asyncio.sleep(0.1)
        
        # Simulate PnL loss
        alert_mgr.update_metric_value("pnl_daily", -1200.0)  # Below critical
        await asyncio.sleep(0.1)
        
        # Simulate exchange disconnect
        alert_mgr.update_metric_value("exchange_connectivity", 0)  # Disconnected
        await asyncio.sleep(0.1)
        
        # Get alert summary
        summary = alert_mgr.get_alert_summary()
        print(f"📉 Alert Summary:")
        print(f"   Total thresholds: {summary['total_thresholds']}")
        print(f"   Active alerts: {summary['active_alerts']}")
        print(f"   Alerts by severity: {summary['alerts_by_severity']}")
        
        # Show active alerts
        active_alerts = alert_mgr.get_active_alerts()
        for alert in active_alerts:
            print(f"   ⚠️ {alert['severity'].upper()}: {alert['description']}")
        
        print("🚀 Alert system demo completed")
    
    asyncio.run(demo())