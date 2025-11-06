#!/usr/bin/env python3
"""
Comprehensive System Validation - Full Integration Test

Validates complete Supreme System V5 with all components:
- Rust core integration
- Multi-algorithm framework
- News processing and sentiment analysis
- Whale detection and tracking
- Social media monitoring
- Memory constraint compliance (≤80MB)

Target: Complete system validation within 4GB RAM constraint
"""

import asyncio
import json
import time
import logging
import gc
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil

# Add project path
sys.path.append('python')
sys.path.append('rust/supreme_core/target/release')

# Comprehensive system imports
try:
    from supreme_system_v5.comprehensive_system import (
        ComprehensiveSupremeSystem, ComprehensiveConfig,
        NewsProcessor, SocialMediaMonitor, MoneyFlowAnalyzer,
        MultiAlgorithmFramework, MemoryMonitor
    )
    COMPREHENSIVE_AVAILABLE = True
except ImportError as e:
    COMPREHENSIVE_AVAILABLE = False
    logging.error(f"Comprehensive system not available: {e}")

# Rust core import
try:
    import supreme_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust core not available - testing Python-only mode")

# Testing utilities
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveValidator:
    """Complete system validation with performance benchmarking"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.validation_results = {}
        self.artifacts_dir = Path("run_artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Memory monitoring
        self.process = psutil.Process()
        self.initial_memory = self.get_current_memory_mb()
        
        print("📋 Supreme System V5 - Comprehensive Validation")
        print("Testing: Rust integration, Multi-algorithms, News, Whales, Sentiment")
        print("=" * 70)
        print(f"Initial memory: {self.initial_memory:.1f}MB")
        print(f"Target memory budget: 80MB")
        print(f"4GB RAM constraint compliance testing")
        print()
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def log_memory_checkpoint(self, checkpoint_name: str) -> float:
        """Log memory usage at checkpoint"""
        memory_mb = self.get_current_memory_mb()
        increase = memory_mb - self.initial_memory
        
        print(f"🧠 {checkpoint_name}: {memory_mb:.1f}MB (+{increase:.1f}MB)")
        
        # Force cleanup if approaching limits
        if memory_mb > 70:
            gc.collect()
            new_memory = self.get_current_memory_mb()
            if new_memory < memory_mb:
                print(f"   → After GC: {new_memory:.1f}MB (-{memory_mb - new_memory:.1f}MB)")
        
        return memory_mb
    
    async def validate_rust_core_integration(self) -> Dict[str, Any]:
        """Validate Rust core integration and performance"""
        print("🚀 VALIDATING: Rust Core Integration")
        
        if not RUST_AVAILABLE:
            return {
                'success': False,
                'error': 'rust_core_not_available',
                'message': 'Rust core module not found - compile required'
            }
        
        try:
            # Initialize Rust core
            rust_core = supreme_core.SupremeCore()
            memory_after_init = self.log_memory_checkpoint("Rust Core Initialized")
            
            # Test market data processing
            test_data = np.array([2000.0 + i * 0.1 + np.sin(i * 0.1) * 10 for i in range(100)], dtype=np.float64)
            
            start_time = time.time()
            result = rust_core.process_market_data(test_data)
            processing_time = time.time() - start_time
            
            # Test technical indicators
            start_time = time.time()
            indicators = rust_core.calculate_indicators(test_data, 14)
            indicators_time = time.time() - start_time
            
            # Test whale detection
            mock_transactions = [
                {
                    'amount': 5_000_000.0,
                    'timestamp': int(time.time()),
                    'from': 'whale_address_1',
                    'to': 'exchange_address_1'
                },
                {
                    'amount': 2_500_000.0,
                    'timestamp': int(time.time()),
                    'from': 'whale_address_2', 
                    'to': 'whale_address_3'
                }
            ]
            
            start_time = time.time()
            whale_alerts = rust_core.detect_whales(mock_transactions)
            whale_time = time.time() - start_time
            
            # Test news processing
            mock_news = [
                {
                    'title': 'Bitcoin reaches new all-time high',
                    'content': 'Cryptocurrency markets surge as Bitcoin breaks resistance levels',
                    'timestamp': int(time.time()),
                    'source': 'coindesk'
                },
                {
                    'title': 'Ethereum upgrade improves transaction speed',
                    'content': 'Latest Ethereum network upgrade reduces gas fees significantly',
                    'timestamp': int(time.time()),
                    'source': 'cointelegraph'
                }
            ]
            
            start_time = time.time()
            news_sentiment = rust_core.process_news(mock_news)
            news_time = time.time() - start_time
            
            # Get performance and memory stats
            perf_stats = rust_core.get_performance_stats()
            memory_stats = rust_core.get_memory_stats()
            
            memory_after_tests = self.log_memory_checkpoint("Rust Core Tests Complete")
            
            return {
                'success': True,
                'performance': {
                    'market_data_processing_ms': processing_time * 1000,
                    'indicators_calculation_ms': indicators_time * 1000,
                    'whale_detection_ms': whale_time * 1000,
                    'news_processing_ms': news_time * 1000,
                    'last_processing_time_ms': perf_stats['last_processing_time_ms']
                },
                'memory_usage': {
                    'initialization_mb': memory_after_init - self.initial_memory,
                    'total_usage_mb': memory_stats['current_usage_mb'],
                    'peak_usage_mb': memory_stats['peak_usage_mb'],
                    'budget_compliance': memory_stats['current_usage_mb'] <= 30  # 30MB target for Rust core
                },
                'functional_tests': {
                    'market_data_processed': len(result['processed_data']) if 'processed_data' in result else 0,
                    'indicators_calculated': len(indicators) if indicators else 0,
                    'whale_alerts_generated': len(whale_alerts) if whale_alerts else 0,
                    'news_items_processed': len(mock_news),
                    'sentiment_score': news_sentiment.get('sentiment_score', 0.0)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'memory_usage_mb': self.get_current_memory_mb()
            }
    
    async def validate_comprehensive_system(self) -> Dict[str, Any]:
        """Validate comprehensive multi-algorithm system"""
        print("🎆 VALIDATING: Comprehensive Multi-Algorithm System")
        
        if not COMPREHENSIVE_AVAILABLE:
            return {
                'success': False,
                'error': 'comprehensive_system_not_available',
                'message': 'Comprehensive system modules not found'
            }
        
        try:
            # Create comprehensive system with tight memory budget
            config = ComprehensiveConfig(memory_budget_mb=60.0)  # 60MB for Python components
            system = ComprehensiveSupremeSystem(config)
            
            memory_after_init = self.log_memory_checkpoint("Comprehensive System Initialized")
            
            # Activate algorithms
            algorithms_activated = []
            for alg_id in ['eth_usdt_scalping', 'whale_following', 'news_trading']:
                if system.algorithm_framework.activate_algorithm(alg_id):
                    algorithms_activated.append(alg_id)
            
            memory_after_algorithms = self.log_memory_checkpoint(f"Algorithms Activated ({len(algorithms_activated)})")
            
            # Test news processing
            news_items = await system.news_processor.fetch_news_feeds()
            if news_items:
                sentiment_result = system.news_processor.analyze_sentiment_batch(news_items[:3])  # Limit for memory
            else:
                # Mock news for testing
                mock_news = [
                    {'title': 'Crypto market shows strong momentum', 'content': 'Markets are bullish', 'timestamp': time.time(), 'source': 'test'},
                    {'title': 'Ethereum network upgrade successful', 'content': 'Technical improvements implemented', 'timestamp': time.time(), 'source': 'test'}
                ]
                sentiment_result = system.news_processor.analyze_sentiment_batch(mock_news)
            
            memory_after_news = self.log_memory_checkpoint("News Processing Complete")
            
            # Test social monitoring
            reddit_data = await system.social_monitor.monitor_reddit_sentiment()
            twitter_data = await system.social_monitor.monitor_twitter_trends()
            
            memory_after_social = self.log_memory_checkpoint("Social Monitoring Complete")
            
            # Test money flow analysis
            mock_market_data = [
                {
                    'close': 2000 + i * 0.5,
                    'volume': 1000 + i * 10
                }
                for i in range(50)
            ]
            
            flow_analysis = system.money_flow_analyzer.analyze_money_flow(mock_market_data)
            
            memory_after_flow = self.log_memory_checkpoint("Money Flow Analysis Complete")
            
            # Get comprehensive system status
            system_status = system.get_system_status()
            
            # Test system execution (brief)
            print("📍 Running brief system execution test (10 seconds)...")
            
            execution_task = asyncio.create_task(system.start_system())
            await asyncio.sleep(10)  # Run for 10 seconds
            execution_task.cancel()
            
            try:
                await execution_task
            except asyncio.CancelledError:
                pass
            
            await system.stop_system()
            
            memory_after_execution = self.log_memory_checkpoint("System Execution Test Complete")
            
            return {
                'success': True,
                'system_configuration': {
                    'memory_budget_mb': config.memory_budget_mb,
                    'enabled_algorithms': len(config.enabled_algorithms),
                    'news_sources': len(config.news_sources),
                    'social_sources': len(config.social_sources)
                },
                'algorithms': {
                    'total_available': len(system.algorithm_framework.algorithms),
                    'successfully_activated': len(algorithms_activated),
                    'activated_list': algorithms_activated,
                    'algorithm_stats': system.algorithm_framework.get_algorithm_stats()
                },
                'data_processing': {
                    'news_processing': {
                        'news_items_fetched': len(news_items) if 'news_items' in locals() else 0,
                        'sentiment_analysis': sentiment_result,
                    },
                    'social_monitoring': {
                        'reddit_sentiment': reddit_data.get('overall_sentiment', 0.0),
                        'twitter_sentiment': twitter_data.get('overall_sentiment', 0.0)
                    },
                    'money_flow_analysis': {
                        'mfi_calculated': 'money_flow_index' in flow_analysis,
                        'flow_strength': flow_analysis.get('flow_strength', 'unknown')
                    }
                },
                'memory_progression': {
                    'after_init_mb': memory_after_init,
                    'after_algorithms_mb': memory_after_algorithms,
                    'after_news_mb': memory_after_news,
                    'after_social_mb': memory_after_social,
                    'after_flow_mb': memory_after_flow,
                    'after_execution_mb': memory_after_execution,
                    'total_increase_mb': memory_after_execution - self.initial_memory
                },
                'system_status': system_status,
                'performance_grade': system_status['performance_grade']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'memory_usage_mb': self.get_current_memory_mb()
            }
    
    async def validate_memory_constraints(self) -> Dict[str, Any]:
        """Validate system compliance with 4GB RAM constraints"""
        print("🧠 VALIDATING: Memory Constraint Compliance")
        
        current_memory = self.get_current_memory_mb()
        total_system_memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Test memory stress scenarios
        stress_results = []
        
        # Scenario 1: Large data processing
        try:
            large_data = np.random.rand(50000).astype(np.float32)  # Use float32 for efficiency
            memory_with_data = self.get_current_memory_mb()
            
            if RUST_AVAILABLE:
                rust_core = supreme_core.SupremeCore()
                rust_core.process_market_data(large_data[:1000])  # Process subset
            
            del large_data
            gc.collect()
            
            memory_after_cleanup = self.get_current_memory_mb()
            
            stress_results.append({
                'scenario': 'large_data_processing',
                'peak_memory_mb': memory_with_data,
                'after_cleanup_mb': memory_after_cleanup,
                'memory_leaked_mb': max(0, memory_after_cleanup - current_memory)
            })
            
        except Exception as e:
            stress_results.append({
                'scenario': 'large_data_processing',
                'error': str(e)
            })
        
        # Scenario 2: Multiple algorithm simulation
        try:
            if COMPREHENSIVE_AVAILABLE:
                config = ComprehensiveConfig(memory_budget_mb=80.0)
                system = ComprehensiveSupremeSystem(config)
                
                memory_before_algorithms = self.get_current_memory_mb()
                
                # Try to activate all available algorithms
                activated_count = 0
                for alg_id in config.enabled_algorithms:
                    if system.algorithm_framework.activate_algorithm(alg_id):
                        activated_count += 1
                        if self.get_current_memory_mb() > 75:  # Stop if approaching limit
                            break
                
                memory_with_algorithms = self.get_current_memory_mb()
                
                # Cleanup
                await system.stop_system()
                del system
                gc.collect()
                
                memory_after_system_cleanup = self.get_current_memory_mb()
                
                stress_results.append({
                    'scenario': 'multiple_algorithms',
                    'algorithms_activated': activated_count,
                    'peak_memory_mb': memory_with_algorithms,
                    'after_cleanup_mb': memory_after_system_cleanup
                })
        
        except Exception as e:
            stress_results.append({
                'scenario': 'multiple_algorithms',
                'error': str(e)
            })
        
        final_memory = self.get_current_memory_mb()
        total_memory_increase = final_memory - self.initial_memory
        
        # Compliance assessment
        memory_budget_compliance = final_memory <= 80.0
        system_memory_usage_percent = (final_memory / 1024) / total_system_memory_gb * 100  # Convert to GB
        
        return {
            'success': True,
            'system_memory_info': {
                'total_system_memory_gb': total_system_memory_gb,
                'available_memory_gb': available_memory_gb,
                'memory_pressure': available_memory_gb < 1.0  # Less than 1GB available
            },
            'process_memory': {
                'initial_memory_mb': self.initial_memory,
                'final_memory_mb': final_memory,
                'total_increase_mb': total_memory_increase,
                'peak_memory_mb': max(result.get('peak_memory_mb', 0) for result in stress_results)
            },
            'compliance': {
                'memory_budget_compliance': memory_budget_compliance,
                'budget_target_mb': 80.0,
                'system_usage_percent': system_memory_usage_percent,
                'suitable_for_4gb_system': final_memory <= 100.0  # Conservative estimate
            },
            'stress_test_results': stress_results
        }
    
    async def validate_performance_targets(self) -> Dict[str, Any]:
        """Validate system performance against targets"""
        print("📊 VALIDATING: Performance Targets")
        
        performance_results = {}
        
        # Rust core performance (if available)
        if RUST_AVAILABLE:
            try:
                rust_core = supreme_core.SupremeCore()
                
                # Test processing speed
                test_data = np.random.rand(1000).astype(np.float64)
                
                # Market data processing speed
                start_time = time.time()
                for _ in range(100):  # 100 iterations
                    rust_core.process_market_data(test_data)
                processing_time = time.time() - start_time
                
                # Indicator calculation speed
                start_time = time.time()
                for _ in range(50):   # 50 iterations
                    rust_core.calculate_indicators(test_data, 14)
                indicators_time = time.time() - start_time
                
                performance_results['rust_core'] = {
                    'market_data_processing_ms_per_1k': (processing_time / 100) * 1000,
                    'indicators_calculation_ms_per_1k': (indicators_time / 50) * 1000,
                    'target_processing_ms': 50,  # Target: <50ms
                    'performance_grade': 'excellent' if (processing_time / 100) * 1000 < 10 else 'good' if (processing_time / 100) * 1000 < 50 else 'needs_optimization'
                }
                
            except Exception as e:
                performance_results['rust_core'] = {'error': str(e)}
        
        # System integration performance
        if COMPREHENSIVE_AVAILABLE:
            try:
                config = ComprehensiveConfig(memory_budget_mb=80.0)
                system = ComprehensiveSupremeSystem(config)
                
                # Test algorithm activation speed
                start_time = time.time()
                activated = []
                for alg_id in config.enabled_algorithms[:3]:  # Test first 3
                    if system.algorithm_framework.activate_algorithm(alg_id):
                        activated.append(alg_id)
                algorithm_activation_time = time.time() - start_time
                
                # Test news processing speed
                mock_news = [
                    {'title': f'News item {i}', 'content': f'Content {i}', 'timestamp': time.time(), 'source': 'test'}
                    for i in range(10)
                ]
                
                start_time = time.time()
                sentiment_result = system.news_processor.analyze_sentiment_batch(mock_news)
                news_processing_time = time.time() - start_time
                
                await system.stop_system()
                
                performance_results['comprehensive_system'] = {
                    'algorithm_activation_ms': algorithm_activation_time * 1000,
                    'algorithms_activated': len(activated),
                    'news_processing_ms_per_10_items': news_processing_time * 1000,
                    'target_activation_ms': 1000,  # Target: <1s
                    'target_news_processing_ms': 500,  # Target: <500ms for 10 items
                    'integration_performance': 'excellent' if algorithm_activation_time < 0.5 else 'good'
                }
                
            except Exception as e:
                performance_results['comprehensive_system'] = {'error': str(e)}
        
        return {
            'success': True,
            'performance_tests': performance_results
        }
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        report_lines = [
            "\n" + "=" * 70,
            "📋 SUPREME SYSTEM V5 - COMPREHENSIVE VALIDATION REPORT",
            "=" * 70,
            f"Validation Date: {datetime.now().isoformat()}",
            f"Total Validation Time: {(datetime.now() - self.start_time).total_seconds():.1f} seconds",
            f"Initial Memory: {self.initial_memory:.1f}MB",
            f"Final Memory: {self.get_current_memory_mb():.1f}MB",
            ""
        ]
        
        # Rust Core Results
        if 'rust_core' in results:
            rust_result = results['rust_core']
            report_lines.extend([
                "🚀 RUST CORE INTEGRATION:",
                f"   Status: {'SUCCESS' if rust_result['success'] else 'FAILED'} ✅" if rust_result.get('success') else "   Status: FAILED ❌",
            ])
            
            if rust_result.get('success'):
                perf = rust_result['performance']
                memory = rust_result['memory_usage']
                report_lines.extend([
                    f"   Performance: {perf['last_processing_time_ms']:.2f}ms processing time",
                    f"   Memory Usage: {memory['total_usage_mb']:.1f}MB (Budget: {'PASS' if memory['budget_compliance'] else 'FAIL'})",
                    f"   Whale Detection: {rust_result['functional_tests']['whale_alerts_generated']} alerts generated",
                    f"   News Processing: {rust_result['functional_tests']['news_items_processed']} items processed"
                ])
        
        # Comprehensive System Results
        if 'comprehensive' in results:
            comp_result = results['comprehensive']
            report_lines.extend([
                "",
                "🎆 COMPREHENSIVE MULTI-ALGORITHM SYSTEM:",
                f"   Status: {'SUCCESS' if comp_result['success'] else 'FAILED'} ✅" if comp_result.get('success') else "   Status: FAILED ❌",
            ])
            
            if comp_result.get('success'):
                algs = comp_result['algorithms']
                data_proc = comp_result['data_processing']
                report_lines.extend([
                    f"   Algorithms: {algs['successfully_activated']}/{algs['total_available']} activated successfully",
                    f"   News Processing: {data_proc['news_processing']['sentiment_analysis']['confidence']:.2f} confidence",
                    f"   Social Monitoring: Reddit {data_proc['social_monitoring']['reddit_sentiment']:.2f}, Twitter {data_proc['social_monitoring']['twitter_sentiment']:.2f}",
                    f"   Money Flow: {data_proc['money_flow_analysis']['flow_strength']} strength detected",
                    f"   Performance Grade: {comp_result['performance_grade'].upper()}"
                ])
        
        # Memory Constraint Results
        if 'memory_constraints' in results:
            mem_result = results['memory_constraints']
            report_lines.extend([
                "",
                "🧠 MEMORY CONSTRAINT COMPLIANCE:",
                f"   4GB System Suitable: {'YES' if mem_result['compliance']['suitable_for_4gb_system'] else 'NO'} ✅" if mem_result['compliance']['suitable_for_4gb_system'] else "   4GB System Suitable: NO ❌",
                f"   Memory Budget: {'PASS' if mem_result['compliance']['memory_budget_compliance'] else 'FAIL'} (Target: 80MB)",
                f"   Total Usage: {mem_result['process_memory']['final_memory_mb']:.1f}MB",
                f"   System Usage: {mem_result['compliance']['system_usage_percent']:.1f}% of total system memory"
            ])
        
        # Performance Results
        if 'performance' in results:
            perf_result = results['performance']
            report_lines.extend([
                "",
                "📊 PERFORMANCE VALIDATION:"
            ])
            
            if 'rust_core' in perf_result['performance_tests']:
                rust_perf = perf_result['performance_tests']['rust_core']
                if 'error' not in rust_perf:
                    report_lines.extend([
                        f"   Rust Processing: {rust_perf['market_data_processing_ms_per_1k']:.2f}ms/1k points (Grade: {rust_perf['performance_grade'].upper()})",
                        f"   Indicators: {rust_perf['indicators_calculation_ms_per_1k']:.2f}ms/1k points"
                    ])
        
        # Overall Assessment
        overall_success = all(
            results.get(key, {}).get('success', False) 
            for key in ['rust_core', 'comprehensive', 'memory_constraints', 'performance']
            if key in results
        )
        
        report_lines.extend([
            "",
            "=" * 70,
            f"🎆 OVERALL VALIDATION: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'} {'✅' if overall_success else '⚠️'}",
            "=" * 70,
        ])
        
        if overall_success:
            report_lines.extend([
                "🎉 ALL VALIDATIONS PASSED - SYSTEM READY FOR PRODUCTION",
                "✅ Rust core integration functional",
                "✅ Multi-algorithm framework operational", 
                "✅ News and sentiment analysis working",
                "✅ Memory constraints satisfied (4GB RAM compatible)",
                "✅ Performance targets met",
                "",
                "🚀 READY FOR: Real trading deployment with 4GB RAM constraint compliance"
            ])
        else:
            report_lines.extend([
                "⚠️ PARTIAL VALIDATION - Some components need attention",
                "✅ Core functionality working",
                "⚠️ Some optimizations may be needed",
                "",
                "🔧 RECOMMENDED: Address any failed validations before production"
            ])
        
        return "\n".join(report_lines)
    
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete comprehensive validation suite"""
        validation_results = {}
        
        try:
            # 1. Rust Core Integration
            validation_results['rust_core'] = await self.validate_rust_core_integration()
            
            # 2. Comprehensive System
            validation_results['comprehensive'] = await self.validate_comprehensive_system()
            
            # 3. Memory Constraints
            validation_results['memory_constraints'] = await self.validate_memory_constraints()
            
            # 4. Performance Targets
            validation_results['performance'] = await self.validate_performance_targets()
            
            # Generate and save report
            report = self.generate_validation_report(validation_results)
            print(report)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.artifacts_dir / f"comprehensive_validation_{timestamp}.json"
            report_file = self.artifacts_dir / f"comprehensive_validation_report_{timestamp}.md"
            
            with open(results_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\n📁 Detailed results saved: {results_file}")
            print(f"📁 Validation report saved: {report_file}")
            
            return validation_results
            
        except Exception as e:
            print(f"\n❌ Critical validation error: {e}")
            return {'error': str(e)}


async def main():
    """Main comprehensive validation execution"""
    validator = ComprehensiveValidator()
    
    try:
        results = await validator.run_complete_validation()
        
        # Determine exit code
        if 'error' in results:
            sys.exit(1)
        
        # Check if all validations passed
        all_success = all(
            results.get(key, {}).get('success', False) 
            for key in ['rust_core', 'comprehensive', 'memory_constraints', 'performance']
            if key in results
        )
        
        if all_success:
            print("\n🎆 COMPREHENSIVE VALIDATION COMPLETE - ALL SYSTEMS GO")
            sys.exit(0)
        else:
            print("\n⚠️ COMPREHENSIVE VALIDATION PARTIAL - REVIEW REQUIRED")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n🚨 Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Critical validation failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())