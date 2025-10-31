# tests/test_smoke.py
import unittest
import os
import sys

# Add python/supreme_system_v5 to path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

try:
    from supreme_system_v5.utils import get_logger, Config
    from supreme_system_v5.data import DataManager
    from supreme_system_v5.backtest import BacktestEngine
    from supreme_system_v5.strategies import Strategy
    from supreme_system_v5.risk import RiskManager

    MINIMAL_IMPORTS_OK = True
except ImportError as e:
    print(f"Failed to import minimal components: {e}")
    MINIMAL_IMPORTS_OK = False


class SmokeTestSuite(unittest.TestCase):
    """Smoke tests for the minimal hybrid system."""

    @unittest.skipUnless(MINIMAL_IMPORTS_OK, "Minimal Python imports not available")
    def test_minimal_python_components_init(self):
        """Test if minimal Python components can be initialized."""
        logger = get_logger("test_logger")
        self.assertIsNotNone(logger)

        config = Config()
        self.assertIsNotNone(config)

        data_manager = DataManager()
        self.assertIsNotNone(data_manager)

        backtest_engine = BacktestEngine()
        self.assertIsNotNone(backtest_engine)

        strategy = Strategy("TestStrategy")
        self.assertIsNotNone(strategy)

        risk_manager = RiskManager()
        self.assertIsNotNone(risk_manager)
        print(" Minimal Python components initialized successfully.")

    @unittest.skipUnless(MINIMAL_IMPORTS_OK, "Minimal Python imports not available")
    def test_data_manager_fetch_data(self):
        """Test data manager can fetch mock data."""
        data_manager = DataManager()
        data = data_manager.fetch_data("TEST")
        self.assertIn("symbol", data)
        self.assertEqual(data["symbol"], "TEST")
        print(" Data Manager fetched mock data successfully.")

    @unittest.skipUnless(MINIMAL_IMPORTS_OK, "Minimal Python imports not available")
    def test_backtest_engine_run_backtest(self):
        """Test backtest engine can run a mock backtest."""
        engine = BacktestEngine()
        config = {"start_date": "2020-01-01", "end_date": "2020-12-31"}
        result = engine.run_backtest(config)
        self.assertIn("result", result)
        self.assertEqual(result["result"], "mock_success")
        print(" Backtest Engine ran mock backtest successfully.")

    @unittest.skipUnless(MINIMAL_IMPORTS_OK, "Minimal Python imports not available")
    def test_strategy_generate_signal(self):
        """Test strategy can generate a mock signal."""
        strategy = Strategy("TestStrategy")
        data = {"price": 100}
        signal = strategy.generate_signal(data)
        self.assertIn("signal", signal)
        self.assertEqual(signal["signal"], "BUY")
        print(" Strategy generated mock signal successfully.")

    @unittest.skipUnless(MINIMAL_IMPORTS_OK, "Minimal Python imports not available")
    def test_risk_manager_evaluate_trade(self):
        """Test risk manager can evaluate a mock trade."""
        risk_manager = RiskManager()
        trade = {"symbol": "TEST", "amount": 10}
        result = risk_manager.evaluate_trade(trade)
        self.assertTrue(result)
        print(" Risk Manager evaluated mock trade successfully.")


if __name__ == "__main__":
    print(" Running Smoke Tests for Supreme System V5 Minimal Hybrid Architecture")
    print("=" * 70)
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
    print("=" * 70)
    print(" Smoke Tests Completed.")
