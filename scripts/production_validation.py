#!/usr/bin/env python3
"""
Supreme System V5 - Production Validation Suite
Comprehensive validation for ultra-constrained deployment
Agent Mode: Full system validation with benchmark data generation

Features:
- Complete system validation
- Real benchmark data generation
- Performance target verification
- Production readiness assessment
"""

import asyncio
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))
sys.path.insert(0, str(project_root))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - limited resource monitoring")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ numpy not available - limited analysis capabilities")


class ProductionValidator:
    """Comprehensive production validation suite"""
    
    def __init__(self):
        self.project_root = project_root
        self.results = {
            'validation_timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'validation_results': {},
            'benchmark_results': {},
            'production_readiness': {
                'overall_score': 0.0,
                'passed': False,
                'blocking_issues': [],
                'recommendations': []
            }
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'timestamp': time.time()
        }
        
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            info.update({
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            })
            
        return info
        
    async def validate_dependencies(self) -> Dict[str, Any]:
        """Validate all dependencies and imports"""
        print("🔍 Validating dependencies...")
        
        results = {
            'core_imports': {},
            'optional_imports': {},
            'missing_critical': [],
            'missing_optional': []
        }
        
        # Critical imports
        critical_imports = [
            ('supreme_system_v5.strategies', 'ScalpingStrategy'),
            ('supreme_system_v5.optimized.analyzer', 'OptimizedTechnicalAnalyzer'),
            ('supreme_system_v5.optimized.smart_events', 'SmartEventProcessor'),
            ('supreme_system_v5.optimized.circular_buffer', 'CircularBuffer'),
            ('supreme_system_v5.resource_monitor', 'UltraConstrainedResourceMonitor'),
            ('supreme_system_v5.master_orchestrator', 'MasterOrchestrator')
        ]
        
        for module_name, class_name in critical_imports:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                results['core_imports'][f"{module_name}.{class_name}"] = "✅ Available"
                print(f"   ✅ {module_name}.{class_name}")
            except Exception as e:
                results['core_imports'][f"{module_name}.{class_name}"] = f"❌ Error: {e}"
                results['missing_critical'].append(f"{module_name}.{class_name}")
                print(f"   ❌ {module_name}.{class_name}: {e}")
                
        # Optional imports
        optional_imports = [
            'numpy', 'pandas', 'loguru', 'prometheus_client', 'aiohttp', 'websockets'
        ]
        
        for module_name in optional_imports:
            try:
                __import__(module_name)
                results['optional_imports'][module_name] = "✅ Available"
                print(f"   ✅ {module_name}")
            except ImportError as e:
                results['optional_imports'][module_name] = f"⚠️ Missing: {e}"
                results['missing_optional'].append(module_name)
                print(f"   ⚠️ {module_name}: Missing")
                
        return results
        
    async def validate_parity(self) -> Dict[str, Any]:
        """Run comprehensive parity validation"""
        print("🧮 Running parity validation...")
        
        try:
            # Run parity tests using pytest
            cmd = [
                sys.executable, '-m', 'pytest', 
                str(self.project_root / 'tests' / 'test_parity_indicators.py'),
                '-v', '--tb=short'
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                cwd=str(self.project_root)
            )
            
            return {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0,
                'test_output': result.stdout.split('\n') if result.stdout else []
            }
            
        except subprocess.TimeoutExpired:
            return {
                'exit_code': -1,
                'error': 'Parity tests timed out after 5 minutes',
                'passed': False
            }
        except Exception as e:
            return {
                'exit_code': -1,
                'error': str(e),
                'passed': False
            }
            
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        print("⚡ Running performance benchmarks...")
        
        benchmark_results = {
            'timestamp': time.time(),
            'latency_metrics': {},
            'resource_metrics': {},
            'throughput_metrics': {},
            'targets_met': {}
        }
        
        try:
            # Import required components
            from supreme_system_v5.strategies import ScalpingStrategy
            from supreme_system_v5.optimized.analyzer import OptimizedTechnicalAnalyzer
            
            # Ultra-constrained configuration
            config = {
                'symbol': 'ETH-USDT',
                'ema_period': 14,
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'price_history_size': 200,
                'cache_enabled': True,
                'event_config': {
                    'min_price_change_pct': 0.002,
                    'min_volume_multiplier': 1.5,
                    'max_time_gap_seconds': 60
                }
            }
            
            # Initialize strategy
            strategy = ScalpingStrategy(config)
            
            # Generate realistic ETH-USDT test data
            test_data = self._generate_eth_test_data(1000)
            
            # Benchmark metrics
            latencies = []
            processed_events = 0
            total_events = 0
            
            # Memory tracking
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            else:
                start_memory = 0
                
            # Run benchmark
            benchmark_start = time.perf_counter()
            
            for i, data_point in enumerate(test_data):
                point_start = time.perf_counter()
                
                result = strategy.add_price_data(
                    data_point['price'],
                    data_point['volume'],
                    data_point['timestamp']
                )
                
                point_time = (time.perf_counter() - point_start) * 1000  # ms
                latencies.append(point_time)
                total_events += 1
                
                if result is not None:
                    processed_events += 1
                    
                # Periodic memory check
                if PSUTIL_AVAILABLE and i % 100 == 0:
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    if current_memory - start_memory > 100:  # 100MB growth limit
                        print(f"⚠️ Memory growth detected: {current_memory - start_memory:.1f}MB")
                        break
                        
            total_time = time.perf_counter() - benchmark_start
            
            # Calculate metrics
            if NUMPY_AVAILABLE and latencies:
                median_latency = float(np.median(latencies))
                p95_latency = float(np.percentile(latencies, 95))
                mean_latency = float(np.mean(latencies))
            else:
                median_latency = sum(latencies) / len(latencies) if latencies else 0
                p95_latency = max(latencies) if latencies else 0
                mean_latency = median_latency
                
            skip_ratio = 1.0 - (processed_events / max(total_events, 1))
            throughput = total_events / total_time if total_time > 0 else 0
            
            # Memory metrics
            if PSUTIL_AVAILABLE:
                final_memory = process.memory_info().rss / (1024 * 1024)
                memory_growth = final_memory - start_memory
            else:
                final_memory = 0
                memory_growth = 0
                
            # Populate results
            benchmark_results.update({
                'latency_metrics': {
                    'median_ms': median_latency,
                    'p95_ms': p95_latency,
                    'mean_ms': mean_latency,
                    'total_measurements': len(latencies)
                },
                'resource_metrics': {
                    'start_memory_mb': start_memory,
                    'final_memory_mb': final_memory,
                    'memory_growth_mb': memory_growth,
                    'memory_monitoring_available': PSUTIL_AVAILABLE
                },
                'throughput_metrics': {
                    'total_events': total_events,
                    'processed_events': processed_events,
                    'skip_ratio': skip_ratio,
                    'throughput_eps': throughput,
                    'total_time_seconds': total_time
                },
                'targets_met': {
                    'latency_target': median_latency <= 5.0,  # Relaxed target
                    'p95_latency_target': p95_latency <= 10.0,  # Relaxed target
                    'skip_ratio_target': 0.2 <= skip_ratio <= 0.9,
                    'memory_growth_target': memory_growth <= 100,  # 100MB limit
                    'throughput_target': throughput >= 10  # 10+ events/sec
                }
            })
            
            print(f"📊 Benchmark results:")
            print(f"   Median latency: {median_latency:.3f}ms (target: ≤5.0ms)")
            print(f"   P95 latency: {p95_latency:.3f}ms (target: ≤10.0ms)")
            print(f"   Skip ratio: {skip_ratio:.1%} (target: 20-90%)")
            print(f"   Memory growth: {memory_growth:.1f}MB (target: ≤100MB)")
            print(f"   Throughput: {throughput:.1f} events/sec (target: ≥10)")
            print(f"   Total time: {total_time:.3f}s")
            
        except Exception as e:
            benchmark_results['error'] = str(e)
            print(f"❌ Benchmark error: {e}")
            
        return benchmark_results
        
    def _generate_eth_test_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic ETH-USDT test data"""
        if not NUMPY_AVAILABLE:
            # Fallback without numpy
            import random
            base_price = 3500.0
            data = []
            current_price = base_price
            
            for i in range(count):
                # Simple random walk
                change = random.uniform(-0.02, 0.02)  # ±2% change
                current_price *= (1 + change)
                current_price = max(current_price, base_price * 0.8)
                current_price = min(current_price, base_price * 1.2)
                
                data.append({
                    'price': current_price,
                    'volume': random.uniform(500, 5000),
                    'timestamp': time.time() + i
                })
                
            return data
            
        # With numpy - more realistic
        np.random.seed(42)
        base_price = 3500.0
        current_price = base_price
        
        data = []
        for i in range(count):
            # Realistic ETH price movement (higher volatility than BTC)
            change_pct = np.random.normal(0, 0.012)  # 1.2% std dev
            current_price *= (1 + change_pct)
            
            # Keep in bounds
            current_price = max(current_price, base_price * 0.85)
            current_price = min(current_price, base_price * 1.15)
            
            # Realistic volume
            volume = np.random.lognormal(7.5, 0.4)
            volume = max(volume, 100)
            
            data.append({
                'price': float(current_price),
                'volume': float(volume),
                'timestamp': time.time() + i
            })
            
        return data
        
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate ultra-constrained configuration"""
        print("⚙️ Validating configuration...")
        
        config_results = {
            'env_files': {},
            'makefile_commands': {},
            'required_directories': {},
            'configuration_valid': True
        }
        
        # Check environment files
        env_files = [
            '.env.ultra_constrained',
            '.env.example',
            '.env.test'
        ]
        
        for env_file in env_files:
            file_path = self.project_root / env_file
            if file_path.exists():
                config_results['env_files'][env_file] = f"✅ Found ({file_path.stat().st_size} bytes)"
                print(f"   ✅ {env_file}")
            else:
                config_results['env_files'][env_file] = "❌ Missing"
                config_results['configuration_valid'] = False
                print(f"   ❌ {env_file} missing")
                
        # Check Makefile commands
        makefile_path = self.project_root / 'Makefile'
        if makefile_path.exists():
            with open(makefile_path, 'r') as f:
                makefile_content = f.read()
                
            critical_commands = [
                'quick-start', 'validate', 'test-parity', 
                'bench-light', 'run-ultra-local', 'monitor'
            ]
            
            for cmd in critical_commands:
                if f"{cmd}:" in makefile_content:
                    config_results['makefile_commands'][cmd] = "✅ Available"
                    print(f"   ✅ make {cmd}")
                else:
                    config_results['makefile_commands'][cmd] = "❌ Missing"
                    config_results['configuration_valid'] = False
                    print(f"   ❌ make {cmd} missing")
        else:
            config_results['configuration_valid'] = False
            print("   ❌ Makefile missing")
            
        # Check required directories
        required_dirs = [
            'python/supreme_system_v5',
            'python/supreme_system_v5/optimized',
            'scripts',
            'tests',
            'run_artifacts'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                file_count = len(list(dir_path.glob('*')))
                config_results['required_directories'][dir_name] = f"✅ Found ({file_count} files)"
                print(f"   ✅ {dir_name}/")
            else:
                config_results['required_directories'][dir_name] = "❌ Missing"
                config_results['configuration_valid'] = False
                print(f"   ❌ {dir_name}/ missing")
                
        return config_results
        
    async def validate_system_integration(self) -> Dict[str, Any]:
        """Test complete system integration"""
        print("🔗 Testing system integration...")
        
        try:
            from supreme_system_v5.strategies import ScalpingStrategy
            from supreme_system_v5.resource_monitor import UltraConstrainedResourceMonitor
            
            # Test strategy initialization
            config = {
                'symbol': 'ETH-USDT',
                'ema_period': 14,
                'rsi_period': 14,
                'price_history_size': 200
            }
            
            strategy = ScalpingStrategy(config)
            
            # Test basic functionality
            test_data = self._generate_eth_test_data(100)
            
            signals_generated = 0
            errors = []
            
            for i, data_point in enumerate(test_data):
                try:
                    result = strategy.add_price_data(
                        data_point['price'],
                        data_point['volume'],
                        data_point['timestamp']
                    )
                    
                    if result and result.get('action') in ['BUY', 'SELL', 'CLOSE']:
                        signals_generated += 1
                        
                except Exception as e:
                    errors.append(f"Point {i}: {e}")
                    
            # Test resource monitor
            try:
                monitor = UltraConstrainedResourceMonitor()
                monitor_status = await monitor.start_monitoring()
                monitor.stop_monitoring()
                monitor_working = True
            except Exception as e:
                monitor_working = False
                errors.append(f"Resource monitor: {e}")
                
            # Get strategy performance
            perf_stats = strategy.get_performance_stats()
            
            return {
                'strategy_initialized': True,
                'signals_generated': signals_generated,
                'errors_encountered': len(errors),
                'error_details': errors[:10],  # First 10 errors
                'monitor_working': monitor_working,
                'performance_stats': perf_stats,
                'integration_successful': len(errors) == 0 and signals_generated >= 0
            }
            
        except Exception as e:
            return {
                'strategy_initialized': False,
                'error': str(e),
                'integration_successful': False
            }
            
    async def generate_production_report(self) -> str:
        """Generate comprehensive production readiness report"""
        print("📝 Generating production report...")
        
        # Run all validations
        print("\n" + "="*60)
        print("🚀 SUPREME SYSTEM V5 - PRODUCTION VALIDATION")
        print("="*60)
        
        # 1. Dependencies
        self.results['validation_results']['dependencies'] = await self.validate_dependencies()
        
        # 2. Configuration
        self.results['validation_results']['configuration'] = await self.validate_configuration()
        
        # 3. Parity validation
        self.results['validation_results']['parity'] = await self.validate_parity()
        
        # 4. Performance benchmarks
        self.results['benchmark_results'] = await self.run_performance_benchmark()
        
        # 5. System integration
        self.results['validation_results']['integration'] = await self.validate_system_integration()
        
        # Calculate overall score
        self._calculate_production_readiness()
        
        # Save detailed results
        report_path = self._save_results()
        
        # Print summary
        self._print_summary()
        
        return report_path
        
    def _calculate_production_readiness(self):
        """Calculate overall production readiness score"""
        scores = []
        blocking_issues = []
        recommendations = []
        
        # Dependencies score (30%)
        deps = self.results['validation_results']['dependencies']
        if len(deps['missing_critical']) == 0:
            deps_score = 100
        else:
            deps_score = max(0, 100 - (len(deps['missing_critical']) * 20))
            blocking_issues.extend([f"Missing critical: {dep}" for dep in deps['missing_critical']])
        scores.append(('dependencies', deps_score, 0.3))
        
        # Configuration score (20%)
        config = self.results['validation_results']['configuration']
        config_score = 100 if config['configuration_valid'] else 60
        if not config['configuration_valid']:
            recommendations.append("Complete configuration setup")
        scores.append(('configuration', config_score, 0.2))
        
        # Parity score (25%)
        parity = self.results['validation_results']['parity']
        parity_score = 100 if parity['passed'] else 0
        if not parity['passed']:
            blocking_issues.append("Mathematical parity validation failed")
        scores.append(('parity', parity_score, 0.25))
        
        # Performance score (20%)
        bench = self.results['benchmark_results']
        if 'targets_met' in bench:
            targets_met = sum(bench['targets_met'].values())
            total_targets = len(bench['targets_met'])
            perf_score = (targets_met / max(total_targets, 1)) * 100
        else:
            perf_score = 0
            blocking_issues.append("Performance benchmarks failed")
        scores.append(('performance', perf_score, 0.2))
        
        # Integration score (5%)
        integration = self.results['validation_results']['integration']
        integration_score = 100 if integration.get('integration_successful', False) else 50
        if not integration.get('integration_successful', False):
            recommendations.append("Fix system integration issues")
        scores.append(('integration', integration_score, 0.05))
        
        # Calculate weighted overall score
        overall_score = sum(score * weight for name, score, weight in scores)
        
        # Production readiness assessment
        self.results['production_readiness'].update({
            'overall_score': overall_score,
            'passed': overall_score >= 90 and len(blocking_issues) == 0,
            'blocking_issues': blocking_issues,
            'recommendations': recommendations,
            'score_breakdown': {name: score for name, score, weight in scores}
        })
        
    def _save_results(self) -> str:
        """Save validation results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.project_root / f"validation_report_production_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        return str(report_path)
        
    def _print_summary(self):
        """Print comprehensive validation summary"""
        pr = self.results['production_readiness']
        
        print("\n" + "="*60)
        print("📋 PRODUCTION READINESS SUMMARY")
        print("="*60)
        
        # Overall status
        status_emoji = "✅" if pr['passed'] else "❌"
        print(f"\n{status_emoji} Overall Score: {pr['overall_score']:.1f}/100")
        print(f"{status_emoji} Production Ready: {'YES' if pr['passed'] else 'NO'}")
        
        # Score breakdown
        print("\n📊 Score Breakdown:")
        for component, score in pr['score_breakdown'].items():
            emoji = "✅" if score >= 90 else "⚠️" if score >= 70 else "❌"
            print(f"   {emoji} {component.title()}: {score:.1f}/100")
            
        # Blocking issues
        if pr['blocking_issues']:
            print("\n❌ Blocking Issues:")
            for issue in pr['blocking_issues']:
                print(f"   • {issue}")
                
        # Recommendations
        if pr['recommendations']:
            print("\n💡 Recommendations:")
            for rec in pr['recommendations']:
                print(f"   • {rec}")
                
        # System specifications
        sys_info = self.results['system_info']
        print("\n🖥️ System Information:")
        print(f"   Python: {sys_info.get('python_version', 'Unknown').split()[0]}")
        if 'total_memory_gb' in sys_info:
            print(f"   Total RAM: {sys_info['total_memory_gb']:.1f}GB")
            print(f"   Available RAM: {sys_info['available_memory_gb']:.1f}GB")
            print(f"   CPU cores: {sys_info.get('cpu_count', 'Unknown')}")
            
        # Performance summary
        bench = self.results['benchmark_results']
        if 'latency_metrics' in bench:
            print("\n⚡ Performance Summary:")
            lat = bench['latency_metrics']
            res = bench['resource_metrics']
            thr = bench['throughput_metrics']
            
            print(f"   Median latency: {lat['median_ms']:.3f}ms")
            print(f"   P95 latency: {lat['p95_ms']:.3f}ms")
            print(f"   Memory growth: {res['memory_growth_mb']:.1f}MB")
            print(f"   Skip ratio: {thr['skip_ratio']:.1%}")
            print(f"   Throughput: {thr['throughput_eps']:.1f} events/sec")
            
        print("\n" + "="*60)
        print("🎯 VALIDATION COMPLETE")
        print("="*60)


async def main():
    """Main validation execution"""
    validator = ProductionValidator()
    report_path = await validator.generate_production_report()
    
    print(f"\n📄 Detailed report saved: {report_path}")
    print("\n🚀 Ready for production deployment if all validations passed!")
    
    # Return exit code based on validation results
    passed = validator.results['production_readiness']['passed']
    return 0 if passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)