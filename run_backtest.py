#!/usr/bin/env python3
"""
🏆 Supreme System V5 - Production Backtest Launcher
Comprehensive backtesting with full system integration

Usage:
    python run_backtest.py --mode fast
    python run_backtest.py --mode comprehensive --symbols AAPL,TSLA,MSFT
    python run_backtest.py --config backtest_config.json

Features:
- Command-line interface for easy backtesting
- Multiple execution modes
- Configuration file support
- Results export and visualization
- Performance reporting
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Import backtesting components
try:
    from src.backtesting.backtest_engine import (
        BacktestEngine,
        BacktestConfig,
        BacktestMode,
        StrategyType,
    )
    from src.backtesting.historical_data import TimeFrame
    from src.backtesting.risk_manager import RiskConfig
    
    BACKTEST_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import backtesting components: {e}")
    BACKTEST_AVAILABLE = False

# Import system components
try:
    from src.config.hardware_profiles import optimal_profile
    HARDWARE_DETECTION = True
except ImportError:
    HARDWARE_DETECTION = False
    optimal_profile = None


class BacktestLauncher:
    """Production backtest launcher with full configuration"""
    
    def __init__(self) -> None:
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
    
    def create_default_config(self, mode: str = "standard") -> BacktestConfig:
        """Create default backtest configuration"""
        # Base configuration
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year default
        
        # Mode-specific settings
        if mode == "fast":
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000.0,
                symbols=["AAPL"],
                strategies=[StrategyType.TECHNICAL_ANALYSIS],
                mode=BacktestMode.FAST,
                years_of_data=1
            )
        elif mode == "comprehensive":
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=250000.0,
                symbols=["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"],
                strategies=[
                    StrategyType.NEUROMORPHIC,
                    StrategyType.FOUNDATION_MODELS,
                    StrategyType.TECHNICAL_ANALYSIS,
                    StrategyType.MEAN_REVERSION,
                    StrategyType.MOMENTUM
                ],
                mode=BacktestMode.COMPREHENSIVE,
                use_real_ai_signals=True,
                signal_confidence_threshold=0.7,
                years_of_data=5
            )
        else:  # standard
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000.0,
                symbols=["AAPL", "TSLA", "MSFT"],
                strategies=[
                    StrategyType.NEUROMORPHIC,
                    StrategyType.TECHNICAL_ANALYSIS,
                    StrategyType.MEAN_REVERSION
                ],
                mode=BacktestMode.STANDARD,
                use_real_ai_signals=True,
                years_of_data=3
            )
        
        return config
    
    def load_config_from_file(self, config_path: str) -> BacktestConfig:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert string dates to datetime objects
        config_dict['start_date'] = datetime.fromisoformat(config_dict['start_date'])
        config_dict['end_date'] = datetime.fromisoformat(config_dict['end_date'])
        
        # Convert strategy strings to enums
        if 'strategies' in config_dict:
            config_dict['strategies'] = [
                StrategyType(strategy) for strategy in config_dict['strategies']
            ]
        
        # Convert mode string to enum
        if 'mode' in config_dict:
            config_dict['mode'] = BacktestMode(config_dict['mode'])
        
        # Convert timeframe string to enum
        if 'timeframe' in config_dict:
            config_dict['timeframe'] = TimeFrame(config_dict['timeframe'])
        
        return BacktestConfig(**config_dict)
    
    def save_config_template(self, filepath: str) -> None:
        """Save configuration template for user customization"""
        template_config = {
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-12-31T23:59:59",
            "initial_capital": 100000.0,
            "symbols": ["AAPL", "TSLA", "MSFT"],
            "timeframe": "1d",
            "strategies": ["neuromorphic", "technical_analysis", "mean_reversion"],
            "mode": "standard",
            "commission_pct": 0.001,
            "slippage_bps": 2.0,
            "use_real_ai_signals": True,
            "signal_confidence_threshold": 0.7,
            "years_of_data": 3,
            "risk_config": {
                "max_portfolio_risk": 0.02,
                "max_position_size": 0.1,
                "max_drawdown_limit": 0.15,
                "target_volatility": 0.15
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(template_config, f, indent=2)
        
        print(f"✅ Configuration template saved to {filepath}")
    
    async def run_backtest(self, config: BacktestConfig) -> None:
        """Run comprehensive backtest with full reporting"""
        if not BACKTEST_AVAILABLE:
            logger.error("❌ Backtesting components not available")
            return
        
        print("🏆 SUPREME SYSTEM V5 - PRODUCTION BACKTEST")
        print("=" * 60)
        
        # Hardware information
        if HARDWARE_DETECTION and optimal_profile:
            print(f"🔧 Hardware: {optimal_profile.processor_type.value}")
            print(f"💾 Memory: {optimal_profile.memory_profile.value}")
        
        # Configuration summary
        print(f"\n📊 Configuration:")
        print(f"   Period: {config.start_date.date()} to {config.end_date.date()}")
        print(f"   Capital: ${config.initial_capital:,.0f}")
        print(f"   Symbols: {', '.join(config.symbols)}")
        print(f"   Strategies: {', '.join([s.value for s in config.strategies])}")
        print(f"   Mode: {config.mode.value}")
        print(f"   AI Signals: {config.use_real_ai_signals}")
        
        # Run backtest
        engine = BacktestEngine(config)
        
        print(f"\n🚀 Starting backtest execution...")
        start_time = datetime.now()
        
        try:
            result = await engine.run_backtest()
            
            # Display results
            print(f"\n🏆 BACKTEST RESULTS SUMMARY:")
            self._display_results(result)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.results_dir / f"backtest_results_{timestamp}.json"
            engine.save_results(result, str(results_file))
            
            # Generate performance report
            self._generate_performance_report(result, timestamp)
            
            print(f"\n💾 Results saved:")
            print(f"   JSON: {results_file}")
            print(f"   Report: results/performance_report_{timestamp}.txt")
            
        except Exception as e:
            logger.error(f"❌ Backtest execution failed: {e}")
            raise
        
        execution_time = datetime.now() - start_time
        print(f"\n✅ Backtest completed in {execution_time.total_seconds():.1f} seconds")
        print("🏆 Supreme System V5 Backtesting Complete!")
    
    def _display_results(self, result) -> None:
        """Display formatted backtest results"""
        summary = result.get_performance_summary()
        
        print(f"\n📈 Returns:")
        print(f"   Total Return: {summary['returns']['total_return_pct']:.2f}%")
        print(f"   Annual Return: {summary['returns']['annual_return_pct']:.2f}%")
        print(f"   Alpha vs Benchmark: {summary['returns']['alpha']:.4f}")
        
        print(f"\n📉 Risk-Adjusted:")
        print(f"   Sharpe Ratio: {summary['risk_adjusted']['sharpe_ratio']:.3f}")
        print(f"   Sortino Ratio: {summary['risk_adjusted']['sortino_ratio']:.3f}")
        print(f"   Calmar Ratio: {summary['risk_adjusted']['calmar_ratio']:.3f}")
        print(f"   Max Drawdown: {summary['risk_adjusted']['max_drawdown_pct']:.2f}%")
        print(f"   Volatility: {summary['risk_adjusted']['volatility_annual']:.2f}%")
        
        print(f"\n💹 Trading Stats:")
        print(f"   Total Trades: {summary['trading_stats']['total_trades']}")
        print(f"   Win Rate: {summary['trading_stats']['win_rate_pct']:.1f}%")
        print(f"   Profit Factor: {summary['trading_stats']['profit_factor']:.2f}")
        print(f"   Avg Win: {summary['trading_stats']['avg_win_pct']:.2f}%")
        print(f"   Avg Loss: {summary['trading_stats']['avg_loss_pct']:.2f}%")
        
        print(f"\n🤖 AI Performance:")
        print(f"   Signal Accuracy: {summary['ai_performance']['ai_signal_accuracy']:.1%}")
        print(f"   Signals Generated: {summary['ai_performance']['ai_signal_count']:,}")
        print(f"   Strategies Tested: {summary['ai_performance']['strategy_count']}")
        
        print(f"\n⚡ Execution:")
        print(f"   Processing Time: {summary['execution']['execution_time_seconds']:.1f}s")
        print(f"   Data Quality: {summary['execution']['data_quality_score']:.1%}")
        print(f"   Data Points: {summary['execution']['data_points_processed']:,}")
    
    def _generate_performance_report(self, result, timestamp: str) -> None:
        """Generate detailed performance report"""
        report_file = self.results_dir / f"performance_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("SUPREME SYSTEM V5 - BACKTEST PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: {result.config.start_date.date()} to {result.config.end_date.date()}\n")
            f.write(f"Initial Capital: ${result.config.initial_capital:,.2f}\n")
            f.write(f"Symbols: {', '.join(result.config.symbols)}\n")
            f.write(f"Strategies: {', '.join([s.value for s in result.config.strategies])}\n\n")
            
            # Performance summary
            summary = result.get_performance_summary()
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Return: {summary['returns']['total_return_pct']:.2f}%\n")
            f.write(f"Annual Return: {summary['returns']['annual_return_pct']:.2f}%\n")
            f.write(f"Sharpe Ratio: {summary['risk_adjusted']['sharpe_ratio']:.3f}\n")
            f.write(f"Max Drawdown: {summary['risk_adjusted']['max_drawdown_pct']:.2f}%\n")
            f.write(f"Win Rate: {summary['trading_stats']['win_rate_pct']:.1f}%\n")
            f.write(f"Profit Factor: {summary['trading_stats']['profit_factor']:.2f}\n\n")
            
            # Strategy breakdown
            f.write("STRATEGY PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            for strategy, metrics in result.strategy_performance.items():
                f.write(f"{strategy.upper()}:\n")
                f.write(f"  Trades: {metrics['trade_count']}\n")
                f.write(f"  Win Rate: {metrics['win_rate']:.1f}%\n")
                f.write(f"  Total PnL: ${metrics['total_pnl']:,.2f}\n")
                if 'avg_confidence' in metrics:
                    f.write(f"  Avg Confidence: {metrics['avg_confidence']:.2f}\n")
                f.write("\n")
            
            # Risk analysis
            f.write("RISK ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Value at Risk (95%): {result.value_at_risk_95:.2f}%\n")
            f.write(f"Expected Shortfall: {result.expected_shortfall:.2f}%\n")
            f.write(f"Volatility (Annual): {summary['risk_adjusted']['volatility_annual']:.2f}%\n")
            f.write(f"Sortino Ratio: {summary['risk_adjusted']['sortino_ratio']:.3f}\n")
            f.write(f"Calmar Ratio: {summary['risk_adjusted']['calmar_ratio']:.3f}\n\n")
            
            # Execution metrics
            f.write("EXECUTION METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Processing Time: {summary['execution']['execution_time_seconds']:.1f} seconds\n")
            f.write(f"Data Quality Score: {summary['execution']['data_quality_score']:.1%}\n")
            f.write(f"Data Points Processed: {summary['execution']['data_points_processed']:,}\n")
            f.write(f"AI Signals Generated: {summary['ai_performance']['ai_signal_count']:,}\n")
            f.write(f"AI Signal Accuracy: {summary['ai_performance']['ai_signal_accuracy']:.1%}\n")


def main():
    """Main entry point for backtest launcher"""
    parser = argparse.ArgumentParser(
        description="Supreme System V5 - Production Backtest Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --mode fast
  python run_backtest.py --mode comprehensive --symbols AAPL,TSLA,NVDA
  python run_backtest.py --config my_backtest.json
  python run_backtest.py --template
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['fast', 'standard', 'comprehensive'],
        default='standard',
        help='Backtest execution mode'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., AAPL,TSLA,MSFT)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--template',
        action='store_true',
        help='Generate configuration template file'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        help='Initial capital amount'
    )
    
    args = parser.parse_args()
    
    launcher = BacktestLauncher()
    
    # Generate template if requested
    if args.template:
        launcher.save_config_template('backtest_config_template.json')
        return
    
    try:
        # Load or create configuration
        if args.config:
            config = launcher.load_config_from_file(args.config)
            print(f"✅ Configuration loaded from {args.config}")
        else:
            config = launcher.create_default_config(args.mode)
            
            # Apply command-line overrides
            if args.symbols:
                config.symbols = [s.strip() for s in args.symbols.split(',')]
            
            if args.start_date:
                config.start_date = datetime.fromisoformat(args.start_date)
            
            if args.end_date:
                config.end_date = datetime.fromisoformat(args.end_date)
            
            if args.capital:
                config.initial_capital = args.capital
        
        # Run backtest
        asyncio.run(launcher.run_backtest(config))
        
    except Exception as e:
        logger.error(f"❌ Backtest launcher failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
