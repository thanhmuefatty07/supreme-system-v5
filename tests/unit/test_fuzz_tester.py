"""
Comprehensive tests for FuzzTester utility
"""

import pytest
import time
from unittest.mock import Mock, patch
from src.utils.fuzz_tester import (
    FuzzTester, FuzzGenerator, FuzzTestingManager,
    FuzzStrategy, FuzzInput, FuzzResult,
    fuzz_test_module, fuzz_test_function
)


class TestFuzzTester:
    """Comprehensive tests for FuzzTester class"""

    @pytest.fixture
    def fuzz_tester(self):
        """Fixture providing FuzzTester instance"""
        return FuzzTester()

    @pytest.fixture
    def fuzz_tester_random(self):
        """Fixture providing FuzzTester with random strategy"""
        return FuzzTester(strategy=FuzzStrategy.RANDOM)

    # ==================== INITIALIZATION TESTS ====================

    def test_initialization_defaults(self):
        """Test default initialization"""
        tester = FuzzTester()
        assert tester.strategy == FuzzStrategy.COVERAGE_GUIDED
        assert hasattr(tester, 'generator')
        assert hasattr(tester, 'coverage_tracker')
        assert isinstance(tester.generator, FuzzGenerator)

    def test_initialization_custom_strategy(self):
        """Test initialization with custom strategy"""
        tester = FuzzTester(strategy=FuzzStrategy.RANDOM)
        assert tester.strategy == FuzzStrategy.RANDOM
        assert isinstance(tester.generator, FuzzGenerator)

    def test_initialization_invalid_strategy(self):
        """Test initialization with invalid strategy should work (enum validation)"""
        # FuzzStrategy enum should handle this
        tester = FuzzTester(strategy=FuzzStrategy.COVERAGE_GUIDED)
        assert tester.strategy == FuzzStrategy.COVERAGE_GUIDED

    # ==================== FUZZ FUNCTION TESTS ====================

    def test_fuzz_function_basic(self, fuzz_tester):
        """Test basic fuzz_function execution"""
        def simple_func(x: int) -> int:
            return x * 2

        def input_generator():
            return [5]  # Simple input generator

        result = fuzz_tester.fuzz_function(
            simple_func, [input_generator],
            max_iterations=10
        )

        assert isinstance(result, FuzzResult)
        assert result.function_name == "simple_func"
        assert result.total_inputs >= 0
        assert result.execution_time >= 0
        assert isinstance(result.interesting_inputs, list)
        assert isinstance(result.error_patterns, dict)

    def test_fuzz_function_with_crashes(self, fuzz_tester):
        """Test fuzz_function that finds crashes"""
        def crashing_func(x: int) -> int:
            if x == 999:
                raise ValueError("Test crash")
            return x * 2

        def input_generator():
            return [999]  # This should cause a crash

        result = fuzz_tester.fuzz_function(
            crashing_func, [input_generator],
            max_iterations=5
        )

        assert result.crashes_found > 0
        assert result.unique_crashes >= 1
        assert "ValueError" in result.error_patterns

    def test_fuzz_function_timeout(self, fuzz_tester):
        """Test fuzz_function respects timeout"""
        def slow_func(x: int) -> int:
            time.sleep(0.1)  # Slow function
            return x

        def input_generator():
            return [1]

        start_time = time.time()
        result = fuzz_tester.fuzz_function(
            slow_func, [input_generator],
            max_iterations=100,
            timeout_seconds=0.5  # Short timeout
        )
        end_time = time.time()

        assert end_time - start_time < 1.0  # Should not take full time
        assert result.execution_time < 1.0

    def test_fuzz_function_no_inputs(self, fuzz_tester):
        """Test fuzz_function with no input generators"""
        def no_arg_func() -> str:
            return "hello"

        result = fuzz_tester.fuzz_function(
            no_arg_func, [],  # No input generators
            max_iterations=5
        )

        assert result.function_name == "no_arg_func"
        assert result.total_inputs >= 0

    def test_fuzz_function_max_iterations(self, fuzz_tester):
        """Test fuzz_function respects max_iterations"""
        call_count = 0

        def counting_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        def input_generator():
            return [1]

        result = fuzz_tester.fuzz_function(
            counting_func, [input_generator],
            max_iterations=5
        )

        assert call_count == 5
        assert result.total_inputs == 5

    # ==================== COVERAGE TRACKING TESTS ====================

    @patch('coverage.Coverage')
    def test_coverage_tracking_enabled(self, mock_coverage, fuzz_tester):
        """Test coverage tracking is enabled for COVERAGE_GUIDED strategy"""
        def simple_func(x: int) -> int:
            return x * 2

        def input_generator():
            return [1]

        fuzz_tester.fuzz_function(simple_func, [input_generator], max_iterations=1)

        # Coverage should be started and stopped
        mock_coverage.return_value.start.assert_called_once()
        mock_coverage.return_value.stop.assert_called_once()
        mock_coverage.return_value.save.assert_called_once()

    def test_coverage_tracking_disabled_random(self, fuzz_tester_random):
        """Test coverage tracking is disabled for RANDOM strategy"""
        def simple_func(x: int) -> int:
            return x * 2

        def input_generator():
            return [1]

        # Should not crash without coverage
        result = fuzz_tester_random.fuzz_function(
            simple_func, [input_generator],
            max_iterations=5
        )

        assert result.coverage_achieved == 0.0  # No coverage tracking

    # ==================== INTERESTING INPUT DETECTION ====================

    def test_interesting_input_detection(self, fuzz_tester):
        """Test detection of interesting inputs"""
        # Test various result types that should be considered interesting
        assert fuzz_tester._is_interesting_input(None, []) is True  # None result
        assert fuzz_tester._is_interesting_input(1e20, []) is True  # Extreme value
        assert fuzz_tester._is_interesting_input("x" * 2000, []) is True  # Very long string
        assert fuzz_tester._is_interesting_input([1] * 200, []) is True  # Large list

        # Test normal inputs
        assert fuzz_tester._is_interesting_input(42, []) is False  # Normal int
        assert fuzz_tester._is_interesting_input("hello", []) is False  # Normal string
        assert fuzz_tester._is_interesting_input([1, 2, 3], []) is False  # Normal list

    # ==================== ERROR HANDLING TESTS ====================

    def test_fuzz_function_handles_generator_errors(self, fuzz_tester):
        """Test fuzz_function handles input generator errors"""
        def simple_func(x: int) -> int:
            return x

        def bad_generator():
            raise RuntimeError("Generator failed")

        # The fuzz function should propagate generator errors (this is expected behavior)
        with pytest.raises(RuntimeError, match="Generator failed"):
            fuzz_tester.fuzz_function(
                simple_func, [bad_generator],
                max_iterations=3
            )

    def test_fuzz_function_handles_function_errors(self, fuzz_tester):
        """Test fuzz_function handles target function errors"""
        def error_func(x: int) -> int:
            raise TypeError(f"Invalid input: {x}")

        def input_generator():
            return [1]

        result = fuzz_tester.fuzz_function(
            error_func, [input_generator],
            max_iterations=3
        )

        assert result.crashes_found == 3
        assert result.unique_crashes == 1
        assert "TypeError" in result.error_patterns


class TestFuzzGenerator:
    """Tests for FuzzGenerator class"""

    @pytest.fixture
    def generator(self):
        """Fixture providing FuzzGenerator instance"""
        return FuzzGenerator()

    def test_initialization(self, generator):
        """Test FuzzGenerator initialization"""
        assert generator.strategy == FuzzStrategy.COVERAGE_GUIDED
        assert isinstance(generator.seed_corpus, list)
        assert isinstance(generator.coverage_data, dict)

    def test_generate_string_inputs(self, generator):
        """Test string input generation"""
        inputs = generator.generate_string_inputs(count=5)
        assert len(inputs) >= 5  # May include more due to predefined patterns
        assert all(isinstance(inp, FuzzInput) for inp in inputs)
        assert all(isinstance(inp.data, str) for inp in inputs)

    def test_generate_numeric_inputs(self, generator):
        """Test numeric input generation"""
        inputs = generator.generate_numeric_inputs(count=5)
        assert len(inputs) >= 5  # May include more due to predefined boundaries
        assert all(isinstance(inp, FuzzInput) for inp in inputs)
        assert all(isinstance(inp.data, (int, float)) for inp in inputs)

    def test_generate_list_inputs(self, generator):
        """Test list input generation"""
        inputs = generator.generate_list_inputs(count=3)
        assert len(inputs) >= 3  # May include more due to predefined patterns
        assert all(isinstance(inp, FuzzInput) for inp in inputs)
        assert all(isinstance(inp.data, list) for inp in inputs)

    def test_generate_string_inputs_edge_cases(self, generator):
        """Test string generation includes edge cases"""
        inputs = generator.generate_string_inputs(count=20)

        # Should include various edge cases
        string_data = [inp.data for inp in inputs]
        assert "" in string_data  # Empty string
        assert any(len(s) > 100 for s in string_data)  # Long strings
        assert any("special_chars" in s or "unicode" in s for s in string_data)  # Special chars

    def test_generate_numeric_inputs_boundaries(self, generator):
        """Test numeric generation includes boundary values"""
        inputs = generator.generate_numeric_inputs(count=20)

        numeric_data = [inp.data for inp in inputs]
        assert 0 in numeric_data
        assert any(isinstance(x, float) and str(x) == 'inf' for x in numeric_data)  # Infinity


class TestFuzzTestingManager:
    """Tests for FuzzTestingManager class"""

    @pytest.fixture
    def manager(self):
        """Fixture providing FuzzTestingManager instance"""
        return FuzzTestingManager()

    def test_initialization(self, manager):
        """Test FuzzTestingManager initialization"""
        assert hasattr(manager, 'tester')
        assert isinstance(manager.tester, FuzzTester)

    def test_fuzz_module_functions_empty_module(self, manager):
        """Test fuzzing an empty module"""
        # Create a mock empty module
        import types
        empty_module = types.ModuleType("empty_test_module")

        results = manager.fuzz_module_functions(empty_module, max_iterations_per_function=1)
        assert isinstance(results, dict)
        assert len(results) == 0  # No functions to fuzz

    def test_generate_fuzz_test_report(self, manager):
        """Test report generation"""
        # Create mock results
        mock_result = FuzzResult(
            function_name="test_func",
            total_inputs=100,
            crashes_found=5,
            unique_crashes=3,
            coverage_achieved=75.5,
            execution_time=10.2,
            interesting_inputs=[],
            error_patterns={"ValueError": 3, "TypeError": 2}
        )

        results = {"test_func": mock_result}
        report = manager.generate_fuzz_test_report(results)

        assert "Fuzz Testing Report" in report
        assert "Functions tested: 1" in report
        assert "Total crashes found: 5" in report
        assert "75.5%" in report
        assert "test_func" in report

    def test_generate_fuzz_test_report_empty(self, manager):
        """Test report generation with empty results"""
        results = {}
        report = manager.generate_fuzz_test_report(results)

        assert "Functions tested: 0" in report
        assert "Total crashes found: 0" in report


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_fuzz_test_function(self):
        """Test fuzz_test_function convenience function"""
        def simple_func(x: int) -> int:
            return x * 2

        def input_gen():
            return [5]

        result = fuzz_test_function(simple_func, [input_gen], max_iterations=3)

        assert isinstance(result, FuzzResult)
        assert result.function_name == "simple_func"
        assert result.total_inputs >= 0

    @patch('src.utils.fuzz_tester.FuzzTestingManager')
    def test_fuzz_test_module(self, mock_manager_class):
        """Test fuzz_test_module convenience function"""
        # Mock the manager
        mock_manager = Mock()
        mock_manager.fuzz_module_functions.return_value = {"func1": Mock()}
        mock_manager.generate_fuzz_test_report.return_value = "Test Report"
        mock_manager_class.return_value = mock_manager

        # Create a dummy module
        import types
        test_module = types.ModuleType("test_module")

        report = fuzz_test_module(test_module)

        assert report == "Test Report"
        mock_manager.fuzz_module_functions.assert_called_once()
        mock_manager.generate_fuzz_test_report.assert_called_once()


class TestFuzzInput:
    """Tests for FuzzInput dataclass"""

    def test_fuzz_input_creation(self):
        """Test FuzzInput creation"""
        input_obj = FuzzInput(data="test")
        assert input_obj.data == "test"
        assert input_obj.metadata == {}
        assert input_obj.coverage_paths == set()
        assert input_obj.execution_time == 0.0
        assert input_obj.crashed is False
        assert input_obj.error_message == ""

    def test_fuzz_input_custom_values(self):
        """Test FuzzInput with custom values"""
        input_obj = FuzzInput(
            data=[1, 2, 3],
            metadata={"test": "value"},
            execution_time=1.5,
            crashed=True,
            error_message="Test error"
        )
        assert input_obj.data == [1, 2, 3]
        assert input_obj.metadata == {"test": "value"}
        assert input_obj.execution_time == 1.5
        assert input_obj.crashed is True
        assert input_obj.error_message == "Test error"


class TestFuzzResult:
    """Tests for FuzzResult dataclass"""

    def test_fuzz_result_creation(self):
        """Test FuzzResult creation"""
        result = FuzzResult(
            function_name="test_func",
            total_inputs=100,
            crashes_found=5,
            unique_crashes=3,
            coverage_achieved=75.5,
            execution_time=10.2,
            interesting_inputs=[],
            error_patterns={}
        )

        assert result.function_name == "test_func"
        assert result.total_inputs == 100
        assert result.crashes_found == 5
        assert result.unique_crashes == 3
        assert result.coverage_achieved == 75.5
        assert result.execution_time == 10.2
        assert result.interesting_inputs == []
        assert result.error_patterns == {}


class TestFuzzStrategy:
    """Tests for FuzzStrategy enum"""

    def test_fuzz_strategy_values(self):
        """Test FuzzStrategy enum values"""
        assert FuzzStrategy.RANDOM.value == "random"
        assert FuzzStrategy.COVERAGE_GUIDED.value == "coverage_guided"
        assert FuzzStrategy.MUTATION_BASED.value == "mutation_based"
        assert FuzzStrategy.GENERATIONAL.value == "generational"

    def test_fuzz_strategy_count(self):
        """Test FuzzStrategy has expected number of values"""
        assert len(FuzzStrategy) == 4


# ==================== INTEGRATION TESTS ====================

class TestFuzzTestingIntegration:
    """Integration tests for complete fuzz testing workflow"""

    def test_complete_fuzz_workflow(self):
        """Test complete fuzz testing workflow"""
        # Create a function with known vulnerabilities
        def vulnerable_func(x: str, y: int) -> str:
            if y == 0:
                raise ZeroDivisionError("Division by zero")
            if len(x) > 100:
                raise ValueError("String too long")
            return x * y

        # Create input generators
        def string_gen():
            return ["short", "a" * 200, "normal"]

        def int_gen():
            return [1, 0, -1, 100]

        # Run fuzz testing
        tester = FuzzTester()
        result = tester.fuzz_function(
            vulnerable_func,
            [string_gen, int_gen],
            max_iterations=20
        )

        # Verify results
        assert result.function_name == "vulnerable_func"
        assert result.crashes_found > 0  # Should find the vulnerabilities
        assert result.total_inputs > 0
        assert isinstance(result.error_patterns, dict)

    def test_module_fuzz_integration(self):
        """Test fuzzing a complete module"""
        import types

        # Create a test module with multiple functions
        test_module = types.ModuleType("integration_test_module")

        def func1(x: int) -> int:
            return x * 2

        def func2(x: str) -> str:
            if x == "crash":
                raise ValueError("Test crash")
            return x.upper()

        def func3(x: float) -> float:
            return x ** 2

        # Add functions to module
        test_module.func1 = func1
        test_module.func2 = func2
        test_module.func3 = func3

        # Fuzz the module
        manager = FuzzTestingManager()
        results = manager.fuzz_module_functions(test_module, max_iterations_per_function=10)

        # Verify results
        assert len(results) == 3  # All functions should be tested
        assert all(isinstance(r, FuzzResult) for r in results.values())

        # func2 should have crashes
        func2_result = results.get("func2")
        if func2_result:
            assert func2_result.crashes_found > 0


# ==================== PERFORMANCE TESTS ====================

class TestFuzzTestingPerformance:
    """Performance tests for fuzz testing"""

    def test_fuzz_performance_large_iterations(self):
        """Test performance with large number of iterations"""
        def simple_func(x: int) -> int:
            return x + 1

        def input_gen():
            return [42]

        tester = FuzzTester(strategy=FuzzStrategy.RANDOM)  # No coverage overhead

        start_time = time.time()
        result = tester.fuzz_function(
            simple_func, [input_gen],
            max_iterations=1000
        )
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 10.0  # Less than 10 seconds
        assert result.total_inputs == 1000

    def test_fuzz_memory_efficiency(self):
        """Test memory efficiency with large inputs"""
        def memory_func(data: list) -> int:
            return len(data)

        def large_input_gen():
            return [list(range(10000))]  # Large list

        tester = FuzzTester()

        # Should handle large inputs without crashing
        result = tester.fuzz_function(
            memory_func, [large_input_gen],
            max_iterations=5
        )

        assert result.total_inputs == 5
        assert result.crashes_found == 0  # Should not crash on large inputs
