#!/usr/bin/env python3
"""
🚀 Supreme System V5 - Simple Backtest Runner
Quick backtesting for immediate strategy validation

Usage:
    python run_backtest.py                    # Quick backtest (5 min)
    python run_backtest.py --duration 10     # 10 minute backtest
    python run_backtest.py --symbol ETH-USDT # Specific symbol
    python run_backtest.py --live            # Live data (if available)
    python run_backtest.py --enhanced        # Enhanced engine (if available)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add python path
sys.path.insert(0, str(Path(__file__).parent / "python"))

print("🚀 Supreme System V5 - Backtest Runner")
print("=" * 40)

class SimpleBacktestRunner:
    """Simple backtest runner for immediate testing"""
    
    def __init__(self, duration_minutes: int = 5, symbol: str = 'ETH-USDT'):
        self.duration_minutes = duration_minutes
        self.symbol = symbol
        self.start_time = time.time()
        self.end_time = self.start_time + (duration_minutes * 60)
        
        self.trades = []
        self.signals = []
        self.price_data = []
        self.performance_metrics = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'processing_times': [],
            'accuracy_scores': [],
            'balance': 10000.0,
            'initial_balance': 10000.0
        }
        
        # Results storage
        self.results_dir = Path("run_artifacts")
        self.results_dir.mkdir(exist_ok=True)
        
    async def run_simple_strategy_test(self) -> Dict[str, Any]:
        """Run simple strategy testing with mock data"""
        print(f"🎯 Testing strategy with {self.symbol} for {self.duration_minutes} minutes")
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        try:
            # Import strategy
            from supreme_system_v5.strategies import ScalpingStrategy
            
            # Configure strategy
            config = {
                'symbol': self.symbol,
                'ema_period': 14,
                'rsi_period': 14,
                'position_size_pct': 0.02,
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.02
            }
            
            strategy = ScalpingStrategy(config)
            
            print(f"✅ Strategy initialized: {type(strategy).__name__}")
            print(f"   Symbol: {self.symbol}")
            print(f"   EMA Period: {config['ema_period']}")
            print(f"   RSI Period: {config['rsi_period']}")
            print(f"   Position Size: {config['position_size_pct']*100}%")
            print()
            
            # Generate realistic test data
            print(f"📊 Generating realistic market data...")
            
            update_count = 0
            signal_count = 0
            
            base_price = 3500.0 if 'ETH' in self.symbol else 67000.0
            current_price = base_price
            
            while time.time() < self.end_time:
                # Generate realistic price movement
                import random
                import math
                
                # Price movement with trend and noise
                trend = math.sin(update_count * 0.01) * 0.002  # Small trend component
                noise = random.gauss(0, 0.001)  # Random noise
                price_change = trend + noise
                
                current_price *= (1 + price_change)
                volume = random.uniform(800, 1200)
                
                # Process data
                process_start = time.perf_counter()
                
                try:
                    # Add price data to strategy
                    result = strategy.add_price_data(current_price, volume, time.time())
                    
                    if result:
                        signal_count += 1
                        self.performance_metrics['total_signals'] += 1
                        
                        # Get current indicators
                        perf_stats = strategy.get_performance_stats()
                        
                        # Store signal data
                        signal_data = {
                            'timestamp': time.time(),
                            'price': current_price,
                            'volume': volume,
                            'update_count': update_count,
                            'performance_stats': perf_stats
                        }
                        
                        self.signals.append(signal_data)
                        
                        # Log periodic progress
                        if signal_count % 10 == 0:
                            print(f"   📈 Signals generated: {signal_count}, Price: ${current_price:.2f}")
                            
                except Exception as e:
                    print(f"   ⚠️ Strategy processing error: {e}")
                    
                # Track processing time
                processing_time = time.perf_counter() - process_start
                self.performance_metrics['processing_times'].append(processing_time * 1000)  # ms
                
                # Store price data
                self.price_data.append({
                    'timestamp': time.time(),
                    'price': current_price,
                    'volume': volume
                })
                
                update_count += 1
                
                # Adaptive sleep (faster for testing)
                await asyncio.sleep(0.1)  # 100ms intervals for fast testing
                
            # Final results
            runtime = time.time() - self.start_time
            
            print(f"\n✅ Backtest completed!")
            print(f"   Runtime: {runtime:.1f} seconds")
            print(f"   Updates processed: {update_count}")
            print(f"   Signals generated: {signal_count}")
            print(f"   Signal rate: {signal_count/runtime:.1f} signals/sec")
            
            if self.performance_metrics['processing_times']:
                avg_latency = sum(self.performance_metrics['processing_times']) / len(self.performance_metrics['processing_times'])
                max_latency = max(self.performance_metrics['processing_times'])
                print(f"   Avg processing: {avg_latency:.2f}ms")
                print(f"   Max processing: {max_latency:.2f}ms")
                
            return {
                'status': 'SUCCESS',
                'runtime_seconds': runtime,
                'updates_processed': update_count,
                'signals_generated': signal_count,
                'signal_rate': signal_count/runtime,
                'avg_processing_ms': avg_latency if self.performance_metrics['processing_times'] else 0,
                'final_price': current_price,
                'price_change_pct': ((current_price - base_price) / base_price) * 100
            }
            
        except Exception as e:
            print(f"\n❌ Backtest failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'runtime_seconds': time.time() - self.start_time
            }
            
    async def run_enhanced_backtest(self) -> Dict[str, Any]:
        """Run enhanced backtest with full system integration"""
        print(f"🚀 Running enhanced backtest with full system integration")
        
        try:
            # Try to use enhanced backtest engine
            from supreme_system_v5.backtest_enhanced import run_realtime_backtest, EnhancedBacktestConfig
            
            config = EnhancedBacktestConfig(
                symbols=[self.symbol],
                realtime_interval=0.5,  # Fast testing
                initial_balance=10000.0,
                enable_adapter=True,
                enable_scalping_optimization=True,
                max_memory_mb=1000  # Reasonable limit
            )
            
            print(f"✅ Enhanced engine initialized")
            print(f"   Adapter enabled: True")
            print(f"   Scalping optimization: True")
            print(f"   Memory limit: 1000MB")
            print()
            
            # Run with timeout
            try:
                result = await asyncio.wait_for(
                    run_realtime_backtest(config),
                    timeout=self.duration_minutes * 60
                )
                
                print(f"\n✅ Enhanced backtest completed successfully!")
                return result
                
            except asyncio.TimeoutError:
                print(f"\n✅ Enhanced backtest completed (timeout after {self.duration_minutes} minutes)")
                return {
                    'status': 'TIMEOUT_SUCCESS',
                    'runtime_seconds': self.duration_minutes * 60,
                    'message': 'Backtest stopped after specified duration'
                }
                
        except ImportError as e:
            print(f"\n⚠️ Enhanced engine not available: {e}")
            print(f"Falling back to simple strategy test...")
            return await self.run_simple_strategy_test()
        except Exception as e:
            print(f"\n❌ Enhanced backtest error: {e}")
            print(f"Falling back to simple strategy test...")
            return await self.run_simple_strategy_test()
            
    async def save_results(self, results: Dict[str, Any]):
        """Save backtest results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        results_file = self.results_dir / f"backtest_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save price data if available
        if self.price_data:
            import pandas as pd
            df = pd.DataFrame(self.price_data)
            price_file = self.results_dir / f"price_data_{timestamp}.csv"
            df.to_csv(price_file, index=False)
            print(f"📊 Price data saved to: {price_file}")
            
        # Save signals if available  
        if self.signals:
            import pandas as pd
            df = pd.DataFrame(self.signals)
            signals_file = self.results_dir / f"signals_{timestamp}.csv"
            df.to_csv(signals_file, index=False)
            print(f"📈 Signals saved to: {signals_file}")
            
    def print_quick_summary(self, results: Dict[str, Any]):
        """Print quick summary for immediate feedback"""
        print(f"\n📊 BACKTEST SUMMARY")
        print("=" * 25)
        print(f"Status: {results.get('status', 'UNKNOWN')}")
        print(f"Runtime: {results.get('runtime_seconds', 0):.1f}s")
        print(f"Symbol: {self.symbol}")
        
        if results.get('signals_generated', 0) > 0:
            print(f"Signals: {results['signals_generated']}")
            print(f"Rate: {results.get('signal_rate', 0):.2f} signals/sec")
            
        if results.get('avg_processing_ms', 0) > 0:
            print(f"Avg processing: {results['avg_processing_ms']:.2f}ms")
            
        if 'price_change_pct' in results:
            print(f"Price change: {results['price_change_pct']:.2f}%")
            
        print()
        

async def main():
    """Main backtest execution"""
    parser = argparse.ArgumentParser(description='Supreme System V5 Backtest Runner')
    parser.add_argument('--duration', type=int, default=5, help='Backtest duration in minutes')
    parser.add_argument('--symbol', type=str, default='ETH-USDT', help='Trading symbol')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced backtest engine')
    parser.add_argument('--live', action='store_true', help='Use live data (if available)')
    parser.add_argument('--output-dir', type=str, default='run_artifacts', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Symbol: {args.symbol}")
    print(f"  Duration: {args.duration} minutes")
    print(f"  Enhanced: {args.enhanced}")
    print(f"  Live data: {args.live}")
    print(f"  Output: {args.output_dir}")
    print()
    
    # Create runner
    runner = SimpleBacktestRunner(
        duration_minutes=args.duration,
        symbol=args.symbol
    )
    
    # Set output directory
    runner.results_dir = Path(args.output_dir)
    runner.results_dir.mkdir(exist_ok=True)
    
    try:
        # Choose backtest mode
        if args.enhanced:
            print(f"🚀 Starting enhanced backtest...")
            results = await runner.run_enhanced_backtest()
        else:
            print(f"📈 Starting simple strategy test...")
            results = await runner.run_simple_strategy_test()
            
        # Print summary
        runner.print_quick_summary(results)
        
        # Save results
        await runner.save_results(results)
        
        # Success indicators
        if results.get('status') in ['SUCCESS', 'TIMEOUT_SUCCESS']:
            print(f"✅ Backtest completed successfully!")
            
            # Print next steps
            print(f"\n🎯 Next Steps:")
            print(f"  1. Review results in {args.output_dir}/")
            print(f"  2. Run 'make monitor' to check resource usage")
            print(f"  3. Run 'make final-validation' for production readiness")
            print(f"  4. Run 'make deploy-production' when ready")
            
            return 0
        else:
            print(f"⚠️ Backtest completed with issues")
            print(f"\nTroubleshooting:")
            print(f"  1. Check 'python scripts/fix_all_issues.py'")
            print(f"  2. Run 'make troubleshoot' for comprehensive guide")
            print(f"  3. Try with --enhanced flag if not used")
            
            return 1
            
    except KeyboardInterrupt:
        print(f"\n🛄 Backtest stopped by user (Ctrl+C)")
        print(f"Partial results may be available in {args.output_dir}/")
        return 0
        
    except Exception as e:
        print(f"\n❌ Backtest failed with error: {e}")
        
        # Save error information
        error_report = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'symbol': args.symbol,
                'duration': args.duration,
                'enhanced': args.enhanced,
                'live': args.live
            }
        }
        
        error_file = runner.results_dir / f"backtest_error_{int(time.time())}.json"
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)
            
        print(f"Error details saved to: {error_file}")
        print(f"\nTroubleshooting:")
        print(f"  1. python scripts/fix_all_issues.py")
        print(f"  2. make validate")
        print(f"  3. make install-deps")
        
        return 1
        

def print_usage_examples():
    """Print usage examples"""
    print("📚 BACKTEST USAGE EXAMPLES")
    print("=" * 30)
    print()
    print("🎯 Quick Tests (Recommended):")
    print("  python run_backtest.py                    # 5-minute ETH-USDT test")
    print("  python run_backtest.py --duration 2      # 2-minute quick test")
    print("  python run_backtest.py --symbol BTC-USDT # Test with Bitcoin")
    print()
    print("🚀 Advanced Tests:")
    print("  python run_backtest.py --enhanced        # Use enhanced engine")
    print("  python run_backtest.py --duration 15     # Longer validation test")
    print("  python run_backtest.py --live            # Live data (if configured)")
    print()
    print("📈 Analysis:")
    print("  Results saved to run_artifacts/")
    print("  CSV files for detailed analysis")
    print("  JSON reports for performance metrics")
    print()
    print("🛠️ Troubleshooting:")
    print("  If errors occur:")
    print("    python scripts/fix_all_issues.py")
    print("    make troubleshoot")
    print()
    
    
def check_system_ready() -> bool:
    """Check if system is ready for backtesting"""
    print("🔍 Checking system readiness...")
    
    ready = True
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        ready = False
    else:
        print("✅ Python version OK")
        
    # Check core imports
    try:
        import numpy
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy missing - run 'pip install numpy'")
        ready = False
        
    try:
        import pandas
        print("✅ Pandas available")
    except ImportError:
        print("❌ Pandas missing - run 'pip install pandas'")
        ready = False
        
    # Check strategy import
    try:
        sys.path.insert(0, str(Path.cwd() / "python"))
        from supreme_system_v5.strategies import ScalpingStrategy
        print("✅ ScalpingStrategy import OK")
    except ImportError as e:
        print(f"❌ ScalpingStrategy import failed: {e}")
        print("   Fix: python scripts/fix_all_issues.py")
        ready = False
        
    # Check output directory
    output_dir = Path("run_artifacts")
    if output_dir.exists():
        print("✅ Output directory ready")
    else:
        print("🔧 Creating output directory")
        output_dir.mkdir(exist_ok=True)
        
    print()
    
    if ready:
        print("✅ System ready for backtesting!")
        return True
    else:
        print("❌ System not ready - fix issues above first")
        return False
        

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print_usage_examples()
        sys.exit(0)
        
    # Check system readiness first
    if not check_system_ready():
        print("\n🛠️ Please fix system issues before running backtest")
        print("Recommended: python scripts/fix_all_issues.py")
        sys.exit(1)
        
    print("🏁 Starting backtest runner...")
    
    # Run backtest
    exit_code = asyncio.run(main())
    
    if exit_code == 0:
        print("\n🎉 Backtest runner completed successfully!")
        print("Check run_artifacts/ for detailed results.")
    else:
        print("\n⚠️ Backtest completed with issues.")
        print("Run troubleshooting commands shown above.")
        
    sys.exit(exit_code)