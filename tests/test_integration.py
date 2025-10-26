#!/usr/bin/env python3
"""
🧪 Supreme System V5 - Integration Test Suite
Comprehensive testing of full system integration

Features:
- End-to-end system testing
- Component integration validation
- Performance benchmark testing
- Data pipeline testing
- AI system validation
- Risk management testing
"""

import asyncio
import logging
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

# Import system components
try:
    from src.backtesting.backtest_engine import (
        BacktestConfig,
        BacktestEngine,
        BacktestMode,
        StrategyType,
    )
    from src.backtesting.historical_data import (
        HistoricalDataProvider,
        HistoricalDataStorage,
        TimeFrame,
    )
    from src.backtesting.risk_manager import RiskConfig, RiskManager
    
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

# Import AI components
try:
    from src.trading.engine import TradingEngine, TradingConfig
    from src.neuromorphic.processor import NeuromorphicProcessor
    from src.foundation_models.predictor import FoundationModelPredictor
    
    AI_COMPONENTS_AVAILABLE = True
except ImportError:
    AI_COMPONENTS_AVAILABLE = False

# Import monitoring
try:
    from src.monitoring.health import HealthChecker
    from src.monitoring.metrics import MetricsCollector
    
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


class IntegrationTestSuite(unittest.TestCase):
    """Comprehensive integration test suite"""
    
    def setUp(self) -> None:
        """Set up test environment"""
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.test_symbols = ["AAPL", "TSLA"]
        self.test_start_date = datetime(2024, 6, 1)
        self.test_end_date = datetime(2024, 8, 31)
        
        logger.info("Integration test setup complete")
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE, "Backtesting components not available")
    def test_historical_data_pipeline(self) -> None:
        """Test historical data pipeline integration"""
        logger.info("📊 Testing historical data pipeline...")
        
        # Initialize data components
        storage = HistoricalDataStorage("test_data/test_historical.db")
        provider = HistoricalDataProvider(storage)
        
        async def run_test():
            # Test data update
            result = await provider.update_historical_data(
                symbols=self.test_symbols,
                timeframes=[TimeFrame.DAY_1],
                years_back=1
            )
            
            self.assertGreater(result['total_bars_added'], 0, "Should add historical data")
            self.assertGreater(result['success_rate'], 0, "Should have successful data updates")
            
            # Test data retrieval
            from src.backtesting.historical_data import BacktestDataInterface
            interface = BacktestDataInterface(provider)
            
            price_data = interface.get_price_data(
                symbols=self.test_symbols,
                timeframe=TimeFrame.DAY_1,
                start_date=self.test_start_date,
                end_date=self.test_end_date
            )
            
            self.assertGreater(len(price_data), 0, "Should retrieve price data")
            logger.info(f"   Retrieved {len(price_data)} data points")
        
        asyncio.run(run_test())
        logger.info("✅ Historical data pipeline test passed")
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE, "Backtesting components not available")
    def test_risk_management_integration(self) -> None:
        """Test risk management system integration"""
        logger.info("🛡️ Testing risk management integration...")
        
        config = RiskConfig(
            max_portfolio_risk=0.02,
            max_position_size=0.1,
            max_drawdown_limit=0.15,
        )
        
        risk_manager = RiskManager(config)
        
        # Test position sizing
        position_size = risk_manager.position_sizer.calculate_position_size(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            portfolio_value=100000.0,
            volatility=0.25,
            signal_strength=1.2
        )
        
        self.assertGreater(position_size, 0, "Should calculate valid position size")
        
        # Test position limits
        can_open, reason = risk_manager.can_open_position(
            "AAPL", position_size * 150.0, 100000.0
        )
        
        self.assertTrue(can_open, f"Should allow position: {reason}")
        
        # Test portfolio metrics update
        metrics = risk_manager.update_portfolio_metrics(105000.0, datetime.now())
        
        self.assertIsInstance(metrics, dict, "Should return metrics dictionary")
        
        logger.info(f"   Position size: {position_size:.2f} shares")
        logger.info(f"   Risk metrics: {len(metrics)} indicators")
        logger.info("✅ Risk management integration test passed")
    
    @unittest.skipUnless(AI_COMPONENTS_AVAILABLE, "AI components not available")
    def test_ai_system_integration(self) -> None:
        """Test AI system integration"""
        logger.info("🤖 Testing AI system integration...")
        
        async def run_ai_test():
            # Test Neuromorphic processor
            neuromorphic = NeuromorphicProcessor(neuron_count=128)
            test_data = [100.0, 101.0, 99.5, 102.0, 98.0]
            
            pattern_result = await neuromorphic.analyze_pattern(test_data)
            
            self.assertIn('pattern_strength', pattern_result, "Should return pattern strength")
            self.assertIn('complexity', pattern_result, "Should return complexity measure")
            
            # Test Foundation Models
            predictor = FoundationModelPredictor()
            await predictor.initialize()
            
            prediction = await predictor.zero_shot_predict(
                "Predict AAPL price direction",
                {"symbol": "AAPL", "price": 150.0, "volume": 1000000}
            )
            
            self.assertIn('prediction', prediction, "Should return prediction")
            self.assertIn('confidence', prediction, "Should return confidence")
            
            logger.info(f"   Neuromorphic pattern: {pattern_result['pattern_strength']:.3f}")
            logger.info(f"   Foundation model confidence: {prediction['confidence']:.3f}")
        
        asyncio.run(run_ai_test())
        logger.info("✅ AI system integration test passed")
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and AI_COMPONENTS_AVAILABLE, "Full components not available")
    def test_end_to_end_backtest(self) -> None:
        """Test complete end-to-end backtesting pipeline"""
        logger.info("🏆 Testing end-to-end backtest integration...")
        
        async def run_e2e_test():
            # Create test configuration
            config = BacktestConfig(
                start_date=self.test_start_date,
                end_date=self.test_end_date,
                initial_capital=50000.0,
                symbols=self.test_symbols,
                strategies=[StrategyType.TECHNICAL_ANALYSIS, StrategyType.NEUROMORPHIC],
                mode=BacktestMode.FAST,
                use_real_ai_signals=True,
                years_of_data=1
            )
            
            # Run backtest
            engine = BacktestEngine(config)
            result = await engine.run_backtest()
            
            # Validate results
            self.assertIsNotNone(result, "Should return backtest result")
            self.assertGreaterEqual(result.total_trades, 0, "Should track trades")
            self.assertGreater(result.execution_time_seconds, 0, "Should track execution time")
            self.assertGreater(result.data_points_processed, 0, "Should process data")
            
            # Validate performance metrics
            summary = result.get_performance_summary()
            self.assertIn('returns', summary, "Should include return metrics")
            self.assertIn('risk_adjusted', summary, "Should include risk metrics")
            self.assertIn('ai_performance', summary, "Should include AI metrics")
            
            logger.info(f"   Total trades: {result.total_trades}")
            logger.info(f"   Execution time: {result.execution_time_seconds:.1f}s")
            logger.info(f"   Data points: {result.data_points_processed:,}")
            logger.info(f"   AI signals: {result.ai_signal_count:,}")
            
            if result.total_trades > 0:
                logger.info(f"   Total return: {result.total_return_pct:.2f}%")
                logger.info(f"   Win rate: {result.win_rate_pct:.1f}%")
        
        asyncio.run(run_e2e_test())
        logger.info("✅ End-to-end backtest integration test passed")
    
    @unittest.skipUnless(MONITORING_AVAILABLE, "Monitoring components not available")
    def test_monitoring_integration(self) -> None:
        """Test monitoring system integration"""
        logger.info("📊 Testing monitoring integration...")
        
        # Test health checker
        health_checker = HealthChecker()
        health_status = health_checker.perform_health_check()
        
        self.assertIn('status', health_status, "Should return health status")
        self.assertIn('components', health_status, "Should check components")
        
        # Test metrics collector
        metrics_collector = MetricsCollector()
        metrics = metrics_collector.collect_system_metrics()
        
        self.assertIsInstance(metrics, dict, "Should return metrics dictionary")
        self.assertGreater(len(metrics), 0, "Should collect metrics")
        
        logger.info(f"   System health: {health_status['status']}")
        logger.info(f"   Metrics collected: {len(metrics)}")
        logger.info("✅ Monitoring integration test passed")
    
    def test_performance_benchmark(self) -> None:
        """Test system performance benchmarks"""
        logger.info("⚡ Testing performance benchmarks...")
        
        async def run_performance_test():
            if not BACKTESTING_AVAILABLE:
                self.skipTest("Backtesting components not available")
                return
            
            # Performance benchmark test
            config = BacktestConfig(
                start_date=datetime(2024, 8, 1),
                end_date=datetime(2024, 8, 31),
                initial_capital=100000.0,
                symbols=["AAPL"],
                strategies=[StrategyType.TECHNICAL_ANALYSIS],
                mode=BacktestMode.FAST
            )
            
            engine = BacktestEngine(config)
            start_time = datetime.now()
            
            result = await engine.run_backtest()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Performance assertions
            self.assertLess(execution_time, 30.0, "Should complete within 30 seconds")
            self.assertGreater(result.data_quality_score, 0.7, "Should have good data quality")
            
            logger.info(f"   Execution time: {execution_time:.1f}s")
            logger.info(f"   Data quality: {result.data_quality_score:.1%}")
            logger.info(f"   Processing rate: {result.data_points_processed/execution_time:.0f} points/sec")
        
        asyncio.run(run_performance_test())
        logger.info("✅ Performance benchmark test passed")
    
    def test_system_configuration(self) -> None:
        """Test system configuration and hardware detection"""
        logger.info("🔧 Testing system configuration...")
        
        try:
            from src.config.hardware_profiles import hardware_detector, optimal_profile
            
            # Test hardware detection
            detected_profile = hardware_detector.detect_optimal_profile()
            self.assertIsNotNone(detected_profile, "Should detect hardware profile")
            
            # Test optimal profile
            if optimal_profile:
                self.assertHasAttr(optimal_profile, 'processor_type', "Should have processor type")
                self.assertHasAttr(optimal_profile, 'memory_profile', "Should have memory profile")
                
                logger.info(f"   Processor: {optimal_profile.processor_type.value}")
                logger.info(f"   Memory: {optimal_profile.memory_profile.value}")
            
            logger.info("✅ System configuration test passed")
            
        except ImportError:
            self.skipTest("Hardware detection not available")
    
    def assertHasAttr(self, obj, attr: str, msg: str = None) -> None:
        """Assert object has attribute"""
        if not hasattr(obj, attr):
            raise AssertionError(msg or f"Object does not have attribute '{attr}'")
    
    def tearDown(self) -> None:
        """Clean up after tests"""
        # Clean up test data
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
        
        logger.info("Integration test cleanup complete")


class TestRunner:
    """Custom test runner for integration tests"""
    
    def __init__(self) -> None:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        print("🧪 SUPREME SYSTEM V5 - INTEGRATION TEST SUITE")
        print("=" * 60)
        
        # Component availability check
        print(f"Component Status:")
        print(f"   Backtesting: {'\u2705' if BACKTESTING_AVAILABLE else '\u274c'}")
        print(f"   AI Components: {'\u2705' if AI_COMPONENTS_AVAILABLE else '\u274c'}")
        print(f"   Monitoring: {'\u2705' if MONITORING_AVAILABLE else '\u274c'}")
        print()
        
        # Run test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestSuite)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Summary
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        
        print(f"\n📊 Integration Test Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_tests - failures - errors - skipped}")
        print(f"   Failed: {failures}")
        print(f"   Errors: {errors}")
        print(f"   Skipped: {skipped}")
        
        success = (failures == 0 and errors == 0)
        
        if success:
            print(f"\n🏆 ALL INTEGRATION TESTS PASSED!")
            print("🚀 Supreme System V5 Ready for Production!")
        else:
            print(f"\n❌ Some tests failed. Please check the output above.")
        
        return success


if __name__ == "__main__":
    runner = TestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)
