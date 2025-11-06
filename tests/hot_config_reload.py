#!/usr/bin/env python3
"""
Supreme System V5 - HOT CONFIGURATION RELOAD TEST
Critical validation of zero-downtime configuration updates

Tests dynamic configuration reloading with memory leak detection,
performance impact assessment, and rollback capabilities
"""

import asyncio
import json
import logging
import os
import psutil
import signal
import sys
import tempfile
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import argparse
import tracemalloc
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hot_config_reload_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Dynamic configuration manager with hot reload capabilities"""

    def __init__(self, config_file: str, reload_interval: float = 5.0):
        self.config_file = config_file
        self.reload_interval = reload_interval
        self.current_config: Dict[str, Any] = {}
        self.config_listeners: List[Callable[[Dict[str, Any], Dict[str, Any]], None]] = []
        self.reload_thread: Optional[threading.Thread] = None
        self.running = False
        self.last_modified = 0.0
        self.reload_count = 0
        self.failed_reloads = 0

        # Configuration validation rules
        self.validation_rules = {
            'memory_limit_mb': lambda x: isinstance(x, int) and 100 <= x <= 4000,
            'max_concurrent_trades': lambda x: isinstance(x, int) and 1 <= x <= 50,
            'risk_multiplier': lambda x: isinstance(x, float) and 0.1 <= x <= 5.0,
            'trading_enabled': lambda x: isinstance(x, bool),
            'log_level': lambda x: x in ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        }

    def start_auto_reload(self):
        """Start automatic configuration monitoring"""
        if self.running:
            return

        self.running = True
        self.reload_thread = threading.Thread(target=self._monitor_config_file, daemon=True)
        self.reload_thread.start()
        logger.info("Configuration auto-reload started")

    def stop_auto_reload(self):
        """Stop automatic configuration monitoring"""
        self.running = False
        if self.reload_thread:
            self.reload_thread.join(timeout=5.0)
        logger.info("Configuration auto-reload stopped")

    def _monitor_config_file(self):
        """Monitor configuration file for changes"""
        while self.running:
            try:
                if os.path.exists(self.config_file):
                    current_modified = os.path.getmtime(self.config_file)

                    if current_modified > self.last_modified:
                        logger.info("Configuration file changed, reloading...")
                        if self._reload_configuration():
                            self.last_modified = current_modified
                        else:
                            logger.error("Configuration reload failed, keeping current config")

                time.sleep(self.reload_interval)

            except Exception as e:
                logger.error(f"Configuration monitoring error: {e}")
                time.sleep(self.reload_interval)

    def _reload_configuration(self) -> bool:
        """Reload configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                new_config = json.load(f)

            # Validate configuration
            if not self._validate_configuration(new_config):
                logger.error("Configuration validation failed")
                self.failed_reloads += 1
                return False

            # Store old config for comparison
            old_config = self.current_config.copy()

            # Apply new configuration
            self.current_config = new_config
            self.reload_count += 1

            # Notify listeners
            self._notify_listeners(old_config, new_config.copy())

            logger.info(f"Configuration reloaded successfully (reload #{self.reload_count})")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            self.failed_reloads += 1
            return False
        except Exception as e:
            logger.error(f"Configuration reload error: {e}")
            self.failed_reloads += 1
            return False

    def _validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against rules"""
        for key, validator in self.validation_rules.items():
            if key in config:
                try:
                    if not validator(config[key]):
                        logger.error(f"Configuration validation failed for {key}: {config[key]}")
                        return False
                except Exception as e:
                    logger.error(f"Configuration validation error for {key}: {e}")
                    return False

        return True

    def _notify_listeners(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Notify all configuration listeners"""
        for listener in self.config_listeners:
            try:
                listener(old_config, new_config)
            except Exception as e:
                logger.error(f"Configuration listener error: {e}")

    def add_listener(self, listener: Callable[[Dict[str, Any], Dict[str, Any]], None]):
        """Add configuration change listener"""
        self.config_listeners.append(listener)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.current_config.get(key, default)

    def set_config_value(self, key: str, value: Any):
        """Set configuration value (for testing)"""
        self.current_config[key] = value

    def get_stats(self) -> Dict[str, Any]:
        """Get configuration manager statistics"""
        return {
            'reload_count': self.reload_count,
            'failed_reloads': self.failed_reloads,
            'active_listeners': len(self.config_listeners),
            'current_config_keys': len(self.current_config),
            'is_monitoring': self.running,
        }

class MockTradingComponent:
    """Mock trading component that responds to configuration changes"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.active_trades: List[Dict[str, Any]] = []
        self.memory_usage_history: List[float] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.config_change_count = 0

        # Register for configuration changes
        config_manager.add_listener(self.on_config_change)

        # Initialize with current config
        self.update_from_config({}, config_manager.current_config)

    def on_config_change(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Handle configuration changes"""
        logger.info("Trading component received configuration change")
        self.config_change_count += 1
        self.update_from_config(old_config, new_config)

    def update_from_config(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Update component based on configuration"""
        max_trades = new_config.get('max_concurrent_trades', 10)
        risk_multiplier = new_config.get('risk_multiplier', 1.0)
        trading_enabled = new_config.get('trading_enabled', True)

        # Adjust active trades based on new limits
        if len(self.active_trades) > max_trades:
            removed = self.active_trades[max_trades:]
            self.active_trades = self.active_trades[:max_trades]
            logger.info(f"Closed {len(removed)} trades due to new limit: {max_trades}")

        # Update risk parameters
        for trade in self.active_trades:
            trade['risk_multiplier'] = risk_multiplier

        # Enable/disable trading
        if not trading_enabled and self.active_trades:
            logger.warning("Trading disabled - closing all active trades")
            self.active_trades.clear()

        logger.info(f"Trading component updated - max_trades: {max_trades}, "
                   f"risk_multiplier: {risk_multiplier}, trading_enabled: {trading_enabled}")

    def simulate_trading_activity(self):
        """Simulate trading activity"""
        if not self.config_manager.get_config('trading_enabled', True):
            return

        max_trades = self.config_manager.get_config('max_concurrent_trades', 10)
        risk_multiplier = self.config_manager.get_config('risk_multiplier', 1.0)

        # Randomly open/close trades
        if len(self.active_trades) < max_trades and np.random.random() < 0.3:
            # Open new trade
            trade = {
                'id': f"trade_{len(self.active_trades) + 1}",
                'symbol': np.random.choice(['BTC-USD', 'ETH-USD', 'SOL-USD']),
                'quantity': np.random.uniform(0.1, 1.0),
                'entry_price': np.random.uniform(30000, 50000),
                'risk_multiplier': risk_multiplier,
                'opened_at': datetime.now().isoformat()
            }
            self.active_trades.append(trade)
            logger.debug(f"Opened trade: {trade['id']}")

        # Randomly close trades
        if self.active_trades and np.random.random() < 0.2:
            closed_trade = self.active_trades.pop(0)
            logger.debug(f"Closed trade: {closed_trade['id']}")

        # Record memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.memory_usage_history.append(memory_mb)

        # Keep only last 1000 readings
        if len(self.memory_usage_history) > 1000:
            self.memory_usage_history = self.memory_usage_history[-500:]

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics"""
        return {
            'active_trades': len(self.active_trades),
            'config_changes_handled': self.config_change_count,
            'avg_memory_mb': np.mean(self.memory_usage_history) if self.memory_usage_history else 0,
            'max_memory_mb': max(self.memory_usage_history) if self.memory_usage_history else 0,
            'memory_samples': len(self.memory_usage_history)
        }

class MemoryLeakDetector:
    """Memory leak detection for configuration reloads"""

    def __init__(self):
        self.baseline_snapshots: List[Any] = []
        self.reload_snapshots: List[Any] = []
        self.memory_growth_threshold_mb = 50.0  # 50MB growth threshold

    def take_baseline_snapshot(self):
        """Take baseline memory snapshot"""
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            self.baseline_snapshots.append(snapshot)
            logger.debug("Baseline memory snapshot taken")

    def take_reload_snapshot(self):
        """Take snapshot after configuration reload"""
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            self.reload_snapshots.append(snapshot)
            logger.debug("Reload memory snapshot taken")

    def analyze_memory_leaks(self) -> Dict[str, Any]:
        """Analyze memory leaks between snapshots"""
        if not self.baseline_snapshots or not self.reload_snapshots:
            return {'error': 'Insufficient snapshots for analysis'}

        # Compare most recent snapshots
        baseline = self.baseline_snapshots[-1]
        reload_snapshot = self.reload_snapshots[-1]

        # Calculate memory differences
        baseline_stats = baseline.statistics('filename')
        reload_stats = reload_snapshot.statistics('filename')

        # Find significant memory growth
        memory_growth = []
        total_growth = 0

        for reload_stat in reload_stats:
            # Find corresponding baseline stat
            baseline_stat = None
            for base_stat in baseline_stats:
                if base_stat.filename == reload_stat.filename:
                    baseline_stat = base_stat
                    break

            if baseline_stat:
                growth = reload_stat.size - baseline_stat.size
                if growth > 1024 * 1024:  # > 1MB growth
                    memory_growth.append({
                        'filename': reload_stat.filename,
                        'baseline_size': baseline_stat.size,
                        'reload_size': reload_stat.size,
                        'growth_bytes': growth,
                        'growth_mb': growth / (1024 * 1024)
                    })
                    total_growth += growth

        # Check for significant leaks
        has_memory_leak = total_growth > self.memory_growth_threshold_mb * 1024 * 1024

        return {
            'total_growth_mb': total_growth / (1024 * 1024),
            'has_memory_leak': has_memory_leak,
            'significant_growth_locations': len(memory_growth),
            'growth_details': memory_growth[:10],  # Top 10 growth locations
            'leak_threshold_mb': self.memory_growth_threshold_mb
        }

class HotConfigReloadTester:
    """Main hot configuration reload testing engine"""

    def __init__(self, config_changes: int = 50, test_duration_minutes: int = 30,
                 memory_limit_mb: int = 300):
        self.config_changes = config_changes
        self.test_duration_minutes = test_duration_minutes
        self.memory_limit_mb = memory_limit_mb

        # Core components
        self.temp_dir = tempfile.mkdtemp(prefix="config_test_")
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        self.config_manager = ConfigurationManager(self.config_file, reload_interval=2.0)
        self.trading_component = MockTradingComponent(self.config_manager)
        self.memory_detector = MemoryLeakDetector()

        # Test state
        self.is_running = False
        self.test_start_time: Optional[datetime] = None

        # Results tracking
        self.config_changes_applied: List[Dict[str, Any]] = []
        self.memory_stats: List[Dict[str, Any]] = []
        self.performance_impacts: List[Dict[str, Any]] = []

        # Results
        self.results = {
            'configuration': {
                'config_changes': config_changes,
                'test_duration_minutes': test_duration_minutes,
                'memory_limit_mb': memory_limit_mb
            },
            'config_changes_applied': [],
            'memory_analysis': {},
            'performance_impact': {},
            'success': False
        }

        logger.info(f"Hot config reload tester initialized - {config_changes} changes, "
                   f"{test_duration_minutes}min duration, {memory_limit_mb}MB memory limit")

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass

    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info("Shutdown signal received - stopping hot config reload test")
        self.is_running = False

    def generate_initial_config(self) -> Dict[str, Any]:
        """Generate initial test configuration"""
        return {
            'memory_limit_mb': 300,
            'max_concurrent_trades': 10,
            'risk_multiplier': 1.0,
            'trading_enabled': True,
            'log_level': 'INFO',
            'market_data_refresh_rate': 5.0,
            'position_size_limit': 1000.0,
            'stop_loss_percentage': 0.02,
            'take_profit_percentage': 0.05,
            'max_daily_trades': 50
        }

    def generate_config_change(self, change_number: int) -> Dict[str, Any]:
        """Generate a configuration change"""
        base_config = self.generate_initial_config()

        # Apply changes based on change number
        modifiers = {
            0: {'max_concurrent_trades': 5, 'description': 'Reduce concurrent trades'},
            1: {'max_concurrent_trades': 20, 'description': 'Increase concurrent trades'},
            2: {'risk_multiplier': 0.5, 'description': 'Reduce risk'},
            3: {'risk_multiplier': 2.0, 'description': 'Increase risk'},
            4: {'trading_enabled': False, 'description': 'Disable trading'},
            5: {'trading_enabled': True, 'description': 'Enable trading'},
            6: {'memory_limit_mb': 200, 'description': 'Reduce memory limit'},
            7: {'memory_limit_mb': 400, 'description': 'Increase memory limit'},
            8: {'log_level': 'DEBUG', 'description': 'Enable debug logging'},
            9: {'log_level': 'ERROR', 'description': 'Minimal logging'},
        }

        modifier = modifiers.get(change_number % len(modifiers), {})
        base_config.update(modifier)
        base_config['change_number'] = change_number
        base_config['change_description'] = modifier.get('description', f'Change #{change_number}')

        return base_config

    def apply_config_change(self, change_number: int):
        """Apply a configuration change"""
        new_config = self.generate_config_change(change_number)

        # Write to config file (this will trigger reload)
        with open(self.config_file, 'w') as f:
            json.dump(new_config, f, indent=2)

        # Record the change
        change_record = {
            'change_number': change_number,
            'timestamp': datetime.now().isoformat(),
            'config': new_config.copy(),
            'description': new_config.get('change_description', f'Change #{change_number}')
        }

        self.config_changes_applied.append(change_record)
        self.results['config_changes_applied'].append(change_record)

        logger.info(f"Applied configuration change #{change_number}: {new_config.get('change_description', 'Unknown')}")

    def monitor_memory_usage(self) -> Dict[str, Any]:
        """Monitor memory usage during test"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'timestamp': datetime.now().isoformat(),
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'memory_limit_exceeded': (memory_info.rss / (1024 ** 2)) > self.memory_limit_mb
        }

    async def run_hot_config_test(self) -> Dict[str, Any]:
        """Execute the hot configuration reload test"""
        self.test_start_time = datetime.now()
        end_time = self.test_start_time + timedelta(minutes=self.test_duration_minutes)
        self.is_running = True

        logger.info("🚀 STARTING HOT CONFIGURATION RELOAD TEST")
        logger.info(f"Duration: {self.test_duration_minutes} minutes")
        logger.info(f"Config Changes: {self.config_changes}")
        logger.info(f"Memory Budget: {self.memory_limit_mb}MB")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Initialize configuration
            initial_config = self.generate_initial_config()
            with open(self.config_file, 'w') as f:
                json.dump(initial_config, f, indent=2)

            # Start configuration monitoring
            self.config_manager.start_auto_reload()

            # Start memory leak detection
            tracemalloc.start()
            self.memory_detector.take_baseline_snapshot()

            # Wait for initial config load
            await asyncio.sleep(3.0)

            # Main test loop
            change_interval = max(1, self.test_duration_minutes * 60 // self.config_changes)
            next_change_time = time.time() + change_interval
            change_number = 0

            while self.is_running and datetime.now() < end_time and change_number < self.config_changes:
                current_time = time.time()

                # Apply configuration changes at regular intervals
                if current_time >= next_change_time and change_number < self.config_changes:
                    # Take memory snapshot before change
                    self.memory_detector.take_baseline_snapshot()

                    # Apply configuration change
                    self.apply_config_change(change_number)

                    # Wait for reload and stabilization
                    await asyncio.sleep(5.0)

                    # Take memory snapshot after change
                    self.memory_detector.take_reload_snapshot()

                    change_number += 1
                    next_change_time = current_time + change_interval

                # Monitor memory usage
                memory_stats = self.monitor_memory_usage()
                self.memory_stats.append(memory_stats)

                if memory_stats['memory_limit_exceeded']:
                    logger.error(f"Memory limit exceeded: {memory_stats['rss_mb']:.1f}MB > {self.memory_limit_mb}MB")
                    break

                # Simulate trading activity
                self.trading_component.simulate_trading_activity()

                # Small delay
                await asyncio.sleep(0.1)

            # Test completed
            if change_number >= self.config_changes:
                self.results['success'] = True
                logger.info("✅ HOT CONFIGURATION RELOAD TEST COMPLETED")

            # Analyze memory leaks
            memory_analysis = self.memory_detector.analyze_memory_leaks()
            self.results['memory_analysis'] = memory_analysis

            # Analyze performance impact
            performance_impact = self.analyze_performance_impact()
            self.results['performance_impact'] = performance_impact

        except Exception as e:
            error_msg = f"Hot config test failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.results['errors'] = [error_msg]

        finally:
            self.is_running = False
            self.config_manager.stop_auto_reload()
            tracemalloc.stop()

        return self.results

    def analyze_performance_impact(self) -> Dict[str, Any]:
        """Analyze performance impact of configuration changes"""
        if not self.config_changes_applied:
            return {'error': 'No configuration changes to analyze'}

        # Analyze memory usage patterns
        memory_usage = [s['rss_mb'] for s in self.memory_stats]
        memory_growth = max(memory_usage) - min(memory_usage) if memory_usage else 0

        # Analyze configuration change frequency
        change_times = [datetime.fromisoformat(c['timestamp']) for c in self.config_changes_applied]
        if len(change_times) > 1:
            time_differences = [(change_times[i+1] - change_times[i]).total_seconds()
                              for i in range(len(change_times)-1)]
            avg_change_interval = sum(time_differences) / len(time_differences)
        else:
            avg_change_interval = 0

        # Analyze trading component stats
        trading_stats = self.trading_component.get_stats()

        return {
            'total_config_changes': len(self.config_changes_applied),
            'successful_reloads': self.config_manager.reload_count,
            'failed_reloads': self.config_manager.failed_reloads,
            'reload_success_rate': self.config_manager.reload_count / max(self.config_manager.reload_count + self.config_manager.failed_reloads, 1),
            'memory_growth_mb': memory_growth,
            'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
            'max_memory_mb': max(memory_usage) if memory_usage else 0,
            'avg_change_interval_seconds': avg_change_interval,
            'trading_component_stats': trading_stats
        }

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save test results to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"hot_config_reload_test_results_{timestamp}.json"

        # Create serializable copy
        results_copy = self.results.copy()

        # Convert config changes to serializable format
        serializable_changes = []
        for change in self.config_changes_applied[:20]:  # Limit to first 20
            serializable_changes.append({
                'change_number': change['change_number'],
                'timestamp': change['timestamp'],
                'description': change['description'],
                'key_changes': {k: v for k, v in change['config'].items()
                              if k in ['max_concurrent_trades', 'risk_multiplier', 'trading_enabled', 'memory_limit_mb']}
            })
        results_copy['sample_config_changes'] = serializable_changes

        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("🔄 HOT CONFIGURATION RELOAD TEST RESULTS")
        print("=" * 70)

        if self.results['success']:
            print("✅ TEST PASSED - Hot configuration reload working correctly")

            changes = self.results['config_changes_applied']
            memory_analysis = self.results['memory_analysis']
            performance = self.results['performance_impact']

            print("📊 Configuration Changes:"            print(f"   Total Changes Applied: {len(changes)}")
            print(f"   Successful Reloads: {performance.get('successful_reloads', 0)}")
            print(f"   Failed Reloads: {performance.get('failed_reloads', 0)}")
            print(".1f"
            print("💾 Memory Analysis:"            print(f"   Memory Growth: {memory_analysis.get('total_growth_mb', 0):.1f}MB")
            print(f"   Memory Leak Detected: {'❌ YES' if memory_analysis.get('has_memory_leak', False) else '✅ NO'}")
            print(f"   Significant Growth Locations: {memory_analysis.get('significant_growth_locations', 0)}")

            print("⚡ Performance Impact:"            print(".1f"            print(".1f"            print(f"   Active Trades (final): {performance.get('trading_component_stats', {}).get('active_trades', 0)}")

        else:
            print("❌ TEST FAILED")
            for error in self.results.get('errors', []):
                print(f"🔴 {error}")

        criteria = {
            'zero_downtime': self.config_manager.failed_reloads == 0,
            'memory_compliance': not any(s['memory_limit_exceeded'] for s in self.memory_stats),
            'no_memory_leaks': not self.results.get('memory_analysis', {}).get('has_memory_leak', True),
            'sufficient_changes': len(self.config_changes_applied) >= 10
        }

        print("🎯 Success Criteria:"        print(f"   Zero Downtime: {'✅' if criteria['zero_downtime'] else '❌'}")
        print(f"   Memory Compliance: {'✅' if criteria['memory_compliance'] else '❌'}")
        print(f"   No Memory Leaks: {'✅' if criteria['no_memory_leaks'] else '❌'}")
        print(f"   Sufficient Changes: {'✅' if criteria['sufficient_changes'] else '❌'}")

        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Hot Configuration Reload Test for Supreme System V5')
    parser.add_argument('--config-changes', type=int, default=50,
                       help='Number of configuration changes to apply (default: 50)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Test duration in minutes (default: 30)')
    parser.add_argument('--memory-limit', type=int, default=300,
                       help='Memory limit in MB (default: 300)')
    parser.add_argument('--output', type=str,
                       help='Output file for results (default: auto-generated)')

    args = parser.parse_args()

    print("🔄 SUPREME SYSTEM V5 - HOT CONFIGURATION RELOAD TEST")
    print("=" * 58)
    print(f"Config Changes: {args.config_changes}")
    print(f"Duration: {args.duration} minutes")
    print(f"Memory Limit: {args.memory_limit}MB")

    # Run the test
    tester = HotConfigReloadTester(
        config_changes=args.config_changes,
        test_duration_minutes=args.duration,
        memory_limit_mb=args.memory_limit
    )

    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        tester.is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        results = asyncio.run(tester.run_hot_config_test())

        # Save results
        output_file = tester.save_results(args.output)

        # Print summary
        tester.print_summary()

        # Exit with appropriate code
        memory_analysis = results.get('memory_analysis', {})
        performance = results.get('performance_impact', {})

        criteria_met = (
            results['success'] and
            not memory_analysis.get('has_memory_leak', True) and
            not any(s['memory_limit_exceeded'] for s in tester.memory_stats) and
            performance.get('reload_success_rate', 0) >= 0.95
        )

        sys.exit(0 if criteria_met else 1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        tester.save_results(args.output)
        tester.cleanup()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical test failure: {e}", exc_info=True)
        tester.save_results(args.output)
        tester.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
