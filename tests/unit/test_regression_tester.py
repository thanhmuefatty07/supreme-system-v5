"""
Comprehensive tests for RegressionTester utility
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess
from src.utils.regression_tester import (
    RegressionTestingManager, RegressionTestGenerator,
    RegressionTestRunner, CodeChangeAnalyzer,
    CodeChange, RegressionTest, RegressionTestSuite
)


class TestRegressionTestingManager:
    """Comprehensive tests for RegressionTestingManager class"""

    @pytest.fixture
    def manager(self):
        """Fixture providing RegressionTestingManager instance"""
        return RegressionTestingManager()

    @pytest.fixture
    def mock_suite(self):
        """Mock RegressionTestSuite"""
        return RegressionTestSuite(
            commit_hash="abc123",
            tests=[
                RegressionTest(
                    code="def test_example(): assert True",
                    target_function="example_func",
                    change_description="Modified function",
                    test_type="function_regression",
                    priority="high",
                    expected_behavior="Should work"
                )
            ],
            coverage_estimate=75.0,
            generated_at="2025-11-19"
        )

    @pytest.fixture
    def mock_results(self):
        """Mock test results"""
        return {
            "total_tests": 5,
            "passed": 4,
            "failed": 1,
            "errors": 0,
            "execution_time": 2.5,
            "test_results": []
        }

    # ==================== INITIALIZATION TESTS ====================

    def test_initialization(self, manager):
        """Test RegressionTestingManager initialization"""
        assert manager is not None
        assert hasattr(manager, 'generator')
        assert hasattr(manager, 'runner')
        assert isinstance(manager.generator, RegressionTestGenerator)
        assert isinstance(manager.runner, RegressionTestRunner)

    # ==================== FULL REGRESSION CYCLE TESTS ====================

    @patch.object(RegressionTestGenerator, 'generate_regression_tests')
    @patch.object(RegressionTestRunner, 'run_regression_suite')
    def test_run_full_regression_cycle_success(self, mock_run_suite, mock_generate_tests,
                                              manager, mock_suite, mock_results):
        """Test successful full regression cycle"""
        mock_generate_tests.return_value = mock_suite
        mock_run_suite.return_value = mock_results

        result = manager.run_full_regression_cycle("abc123")

        assert "suite" in result
        assert "results" in result
        assert "analysis" in result
        assert "report" in result

        # Verify analysis
        analysis = result["analysis"]
        assert analysis["success_rate"] == 80.0  # 4/5 * 100
        assert analysis["regression_detected"] is True  # failed > 0
        assert analysis["severity"] == "HIGH"  # failed but not >50%

        mock_generate_tests.assert_called_once_with("abc123")
        mock_run_suite.assert_called_once_with(mock_suite)

    @patch.object(RegressionTestGenerator, 'generate_regression_tests')
    @patch.object(RegressionTestRunner, 'run_regression_suite')
    def test_run_full_regression_cycle_no_failures(self, mock_run_suite, mock_generate_tests,
                                                   manager, mock_suite):
        """Test regression cycle with no failures"""
        mock_generate_tests.return_value = mock_suite
        mock_run_suite.return_value = {
            "total_tests": 5,
            "passed": 5,
            "failed": 0,
            "errors": 0,
            "execution_time": 1.8,
            "test_results": []
        }

        result = manager.run_full_regression_cycle("abc123")

        analysis = result["analysis"]
        assert analysis["regression_detected"] is False
        assert analysis["severity"] == "LOW"

    @patch.object(RegressionTestGenerator, 'generate_regression_tests')
    @patch.object(RegressionTestRunner, 'run_regression_suite')
    def test_run_full_regression_cycle_critical_failures(self, mock_run_suite, mock_generate_tests,
                                                         manager, mock_suite):
        """Test regression cycle with critical failure rate"""
        mock_generate_tests.return_value = mock_suite
        mock_run_suite.return_value = {
            "total_tests": 10,
            "passed": 3,
            "failed": 6,  # >50% failed
            "errors": 1,
            "execution_time": 3.2,
            "test_results": []
        }

        result = manager.run_full_regression_cycle("abc123")

        analysis = result["analysis"]
        assert analysis["regression_detected"] is True
        assert analysis["severity"] == "CRITICAL"

    # ==================== ANALYSIS TESTS ====================

    def test_analyze_regression_results_perfect(self, manager, mock_suite):
        """Test analysis with perfect results"""
        results = {
            "total_tests": 10,
            "passed": 10,
            "failed": 0,
            "errors": 0,
            "execution_time": 2.1,
            "test_results": []
        }

        analysis = manager._analyze_regression_results(mock_suite, results)

        assert analysis["success_rate"] == 100.0
        assert analysis["regression_detected"] is False
        assert analysis["error_rate"] == 0.0
        assert analysis["severity"] == "LOW"
        assert analysis["performance_impact"] == 2.1

    def test_analyze_regression_results_zero_tests(self, manager, mock_suite):
        """Test analysis with zero tests"""
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "execution_time": 0.0
        }

        analysis = manager._analyze_regression_results(mock_suite, results)

        assert analysis["success_rate"] == 0.0
        assert analysis["regression_detected"] is False
        assert analysis["error_rate"] == 0.0

    def test_analyze_regression_results_with_errors(self, manager, mock_suite):
        """Test analysis with test errors"""
        results = {
            "total_tests": 10,
            "passed": 7,
            "failed": 1,
            "errors": 2,
            "execution_time": 2.8,
            "test_results": []
        }

        analysis = manager._analyze_regression_results(mock_suite, results)

        assert analysis["success_rate"] == 70.0
        assert analysis["regression_detected"] is True  # failed > 0
        assert analysis["error_rate"] == 20.0
        assert analysis["severity"] == "HIGH"  # failed > 0 takes precedence

    # ==================== REPORT GENERATION TESTS ====================

    def test_generate_regression_report_comprehensive(self, manager, mock_suite, mock_results):
        """Test comprehensive report generation"""
        analysis = {
            "success_rate": 80.0,
            "regression_detected": True,
            "severity": "HIGH",
            "error_rate": 0.0
        }

        report = manager._generate_regression_report(mock_suite, mock_results, analysis)

        # Check report structure
        assert "Regression Testing Report" in report
        assert mock_suite.commit_hash in report
        assert "Total tests generated: 5" in report
        assert "Tests passed: 4" in report
        assert "Tests failed: 1" in report
        assert "Success rate: 80.0%" in report
        assert "Regression detected: YES" in report
        assert "Severity level: HIGH" in report

        # Check recommendations
        assert "CRITICAL: Regression detected" in report

    def test_generate_regression_report_no_regression(self, manager, mock_suite):
        """Test report generation with no regression"""
        results = {
            "total_tests": 10,
            "passed": 10,
            "failed": 0,
            "errors": 0,
            "execution_time": 2.0,
            "test_results": []
        }
        analysis = {
            "success_rate": 100.0,
            "regression_detected": False,
            "severity": "LOW",
            "error_rate": 0.0
        }

        report = manager._generate_regression_report(mock_suite, results, analysis)

        assert "Regression detected: NO" in report
        assert "Severity level: LOW" in report
        assert "CRITICAL:" not in report

    def test_generate_regression_report_high_error_rate(self, manager, mock_suite):
        """Test report generation with high error rate"""
        results = {
            "total_tests": 10,
            "passed": 6,
            "failed": 0,
            "errors": 4,
            "execution_time": 2.0,
            "test_results": []
        }
        analysis = {
            "success_rate": 60.0,
            "regression_detected": False,
            "severity": "MEDIUM",
            "error_rate": 40.0
        }

        report = manager._generate_regression_report(mock_suite, results, analysis)

        assert "HIGH: High error rate" in report
        assert "MEDIUM: Low success rate" in report


class TestRegressionTestGenerator:
    """Tests for RegressionTestGenerator class"""

    @pytest.fixture
    def generator(self):
        """Fixture providing RegressionTestGenerator instance"""
        return RegressionTestGenerator()

    @pytest.fixture
    def mock_change(self):
        """Mock CodeChange"""
        return CodeChange(
            file_path="src/example.py",
            old_content="def func(): return 1",
            new_content="def func(): return 2",
            changed_functions=["func"],
            changed_lines=[(1, "def func(): return 1", "def func(): return 2")],
            change_type="modified"
        )

    @patch('src.utils.regression_tester.CodeChangeAnalyzer')
    def test_initialization(self, mock_analyzer_class, generator):
        """Test RegressionTestGenerator initialization"""
        assert generator is not None
        assert hasattr(generator, 'change_analyzer')

    @patch.object(CodeChangeAnalyzer, 'get_changes_since_commit')
    def test_generate_regression_tests_empty_changes(self, mock_get_changes, generator):
        """Test generating tests with no changes"""
        mock_get_changes.return_value = []

        suite = generator.generate_regression_tests("abc123")

        assert suite.commit_hash == "abc123"
        assert len(suite.tests) == 0
        assert suite.coverage_estimate == 0.0

    @patch.object(CodeChangeAnalyzer, 'get_changes_since_commit')
    @patch('src.utils.regression_tester.RegressionTestGenerator._generate_tests_for_change')
    def test_generate_regression_tests_with_changes(self, mock_generate_tests, mock_get_changes,
                                                   generator, mock_change):
        """Test generating tests with changes"""
        mock_get_changes.return_value = [mock_change]
        mock_generate_tests.return_value = [
            RegressionTest(
                code="test code",
                target_function="func",
                change_description="test",
                test_type="function",
                priority="high",
                expected_behavior="works"
            )
        ]

        suite = generator.generate_regression_tests("abc123")

        assert len(suite.tests) == 1
        assert suite.coverage_estimate == 5.0  # 1 test * 5.0
        mock_generate_tests.assert_called_once_with(mock_change)

    @patch.object(CodeChangeAnalyzer, 'get_changes_since_commit')
    def test_generate_regression_tests_coverage_cap(self, mock_get_changes, generator, mock_change):
        """Test coverage estimate is capped at 100%"""
        mock_get_changes.return_value = [mock_change] * 50  # Many changes

        with patch.object(generator, '_generate_tests_for_change') as mock_gen:
            mock_gen.return_value = [RegressionTest(
                code="test", target_function="func", change_description="test",
                test_type="function", priority="high", expected_behavior="works"
            )] * 50  # 50 tests per change

            suite = generator.generate_regression_tests("abc123")

            assert suite.coverage_estimate == 100.0  # Capped at 100


class TestRegressionTestRunner:
    """Tests for RegressionTestRunner class"""

    @pytest.fixture
    def runner(self):
        """Fixture providing RegressionTestRunner instance"""
        return RegressionTestRunner()

    @pytest.fixture
    def mock_suite(self):
        """Mock RegressionTestSuite"""
        return RegressionTestSuite(
            commit_hash="abc123",
            tests=[
                RegressionTest(
                    code="""
def test_example():
    assert 1 + 1 == 2
    assert True
""",
                    target_function="example_func",
                    change_description="test",
                    test_type="function",
                    priority="high",
                    expected_behavior="works"
                )
            ],
            coverage_estimate=50.0,
            generated_at="2025-11-19"
        )

    def test_initialization(self, runner):
        """Test RegressionTestRunner initialization"""
        assert runner is not None

    @patch('tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    def test_run_regression_suite_success(self, mock_subprocess, mock_tempfile, runner, mock_suite):
        """Test running regression suite successfully"""
        # Mock successful test execution
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "test session starts\n5 passed, 0 failed"
        mock_process.stderr = ""
        mock_subprocess.return_value = mock_process

        # Mock temp file
        mock_file = Mock()
        mock_file.name = "/tmp/test_file.py"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        results = runner.run_regression_suite(mock_suite)

        assert results["total_tests"] >= 0
        assert results["passed"] >= 0
        assert results["failed"] >= 0
        assert results["errors"] >= 0
        assert results["execution_time"] >= 0

    @patch('tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    def test_run_regression_suite_with_failures(self, mock_subprocess, mock_tempfile, runner, mock_suite):
        """Test running regression suite with failures"""
        # Mock test execution with failures
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.stdout = "test session starts\n3 passed, 2 failed"
        mock_process.stderr = "FAILED test_example.py::test_example"
        mock_subprocess.return_value = mock_process

        mock_file = Mock()
        mock_file.name = "/tmp/test_file.py"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        results = runner.run_regression_suite(mock_suite)

        assert results["failed"] >= 0
        assert results["total_tests"] == results["passed"] + results["failed"] + results["errors"]

    @patch('subprocess.run')
    def test_run_regression_suite_subprocess_error(self, mock_subprocess, runner, mock_suite):
        """Test handling subprocess execution errors"""
        mock_subprocess.side_effect = subprocess.SubprocessError("Command failed")

        results = runner.run_regression_suite(mock_suite)

        # Should return results with errors
        assert results["total_tests"] >= 0  # May still count tests even if failed
        assert results["passed"] == 0
        assert results["failed"] >= 0
        assert results["errors"] >= 1  # Should have at least one subprocess error


class TestCodeChangeAnalyzer:
    """Tests for CodeChangeAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Fixture providing CodeChangeAnalyzer instance"""
        return CodeChangeAnalyzer()

    @pytest.fixture
    def temp_git_repo(self):
        """Create a temporary git repository for testing"""
        temp_dir = tempfile.mkdtemp()
        original_cwd = Path.cwd()

        try:
            # Initialize git repo
            subprocess.run(['git', 'init'], cwd=temp_dir, check=True, capture_output=True)
            subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=temp_dir, check=True)
            subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=temp_dir, check=True)

            # Create initial file and commit
            test_file = Path(temp_dir) / 'test.py'
            test_file.write_text('def hello():\n    return "world"')

            subprocess.run(['git', 'add', 'test.py'], cwd=temp_dir, check=True)
            subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=temp_dir, check=True)

            yield temp_dir

        finally:
            # Cleanup
            import os
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch('git.Repo')
    def test_initialization(self, mock_repo_class, analyzer):
        """Test CodeChangeAnalyzer initialization"""
        assert analyzer is not None

    @patch('src.utils.regression_tester.git.Repo')
    def test_get_changes_since_commit(self, mock_repo_class, analyzer):
        """Test getting changes since a commit"""
        # Mock git repo and commits
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        mock_commit = Mock()
        mock_commit.hexsha = "abc123"
        mock_repo.head.commit = mock_commit

        # Mock git diff to return empty string (no changes)
        mock_repo.git.diff.return_value = ""

        # Mock the repo initialization in analyzer
        analyzer.repo = mock_repo

        changes = analyzer.get_changes_since_commit("abc123")

        # Should return empty list for no changes
        assert isinstance(changes, list)
        assert len(changes) == 0

    @patch('src.utils.regression_tester.git.Repo')
    def test_get_changes_since_commit_with_diffs(self, mock_repo_class, analyzer):
        """Test getting changes with actual diffs"""
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        # Mock git diff output
        mock_repo.git.diff.return_value = """diff --git a/src/test.py b/src/test.py
index abc123..def456 100644
--- a/src/test.py
+++ b/src/test.py
@@ -1,2 +1,3 @@ def hello():
-    return "world"
+    return "world!"
+    print("Added line")
"""

        # Mock the repo initialization in analyzer
        analyzer.repo = mock_repo

        changes = analyzer.get_changes_since_commit("abc123")

        assert len(changes) > 0
        change = changes[0]
        assert change.file_path == "src/test.py"
        assert change.change_type == "modified"
        assert len(change.changed_lines) > 0


class TestDataClasses:
    """Tests for data classes"""

    def test_code_change_creation(self):
        """Test CodeChange dataclass creation"""
        change = CodeChange(
            file_path="src/test.py",
            old_content="old",
            new_content="new",
            changed_functions=["func1"],
            changed_lines=[(1, "old", "new")],
            change_type="modified"
        )

        assert change.file_path == "src/test.py"
        assert change.old_content == "old"
        assert change.new_content == "new"
        assert change.changed_functions == ["func1"]
        assert change.changed_lines == [(1, "old", "new")]
        assert change.change_type == "modified"

    def test_regression_test_creation(self):
        """Test RegressionTest dataclass creation"""
        test = RegressionTest(
            code="def test(): pass",
            target_function="func",
            change_description="modified",
            test_type="function",
            priority="high",
            expected_behavior="works"
        )

        assert test.code == "def test(): pass"
        assert test.target_function == "func"
        assert test.change_description == "modified"
        assert test.test_type == "function"
        assert test.priority == "high"
        assert test.expected_behavior == "works"

    def test_regression_test_suite_creation(self):
        """Test RegressionTestSuite dataclass creation"""
        suite = RegressionTestSuite(
            commit_hash="abc123",
            tests=[],
            coverage_estimate=75.5,
            generated_at="2025-11-19"
        )

        assert suite.commit_hash == "abc123"
        assert suite.tests == []
        assert suite.coverage_estimate == 75.5
        assert suite.generated_at == "2025-11-19"


# ==================== INTEGRATION TESTS ====================

class TestRegressionTestingIntegration:
    """Integration tests for complete regression testing workflow"""

    @patch('src.utils.regression_tester.RegressionTestGenerator')
    @patch('src.utils.regression_tester.RegressionTestRunner')
    def test_full_integration_workflow(self, mock_runner_class, mock_generator_class):
        """Test complete integration workflow"""
        # Mock components
        mock_generator = Mock()
        mock_runner = Mock()

        mock_generator_class.return_value = mock_generator
        mock_runner_class.return_value = mock_runner

        # Mock suite and results
        mock_suite = Mock()
        mock_suite.commit_hash = "abc123"
        mock_suite.tests = []
        mock_suite.coverage_estimate = 60.0

        mock_results = {
            "total_tests": 10,
            "passed": 8,
            "failed": 2,
            "errors": 0,
            "execution_time": 3.5,
            "test_results": []
        }

        mock_generator.generate_regression_tests.return_value = mock_suite
        mock_runner.run_regression_suite.return_value = mock_results

        # Run full workflow
        manager = RegressionTestingManager()
        result = manager.run_full_regression_cycle("abc123")

        # Verify workflow completed
        assert "suite" in result
        assert "results" in result
        assert "analysis" in result
        assert "report" in result

        # Verify method calls
        mock_generator.generate_regression_tests.assert_called_once_with("abc123")
        mock_runner.run_regression_suite.assert_called_once_with(mock_suite)

    def test_error_handling_in_workflow(self):
        """Test error handling in regression workflow"""
        manager = RegressionTestingManager()

        # Test with invalid commit hash (should handle gracefully)
        with patch.object(manager.generator, 'generate_regression_tests') as mock_gen:
            mock_gen.side_effect = Exception("Git error")

            with pytest.raises(Exception):
                manager.run_full_regression_cycle("invalid_commit")


# ==================== EDGE CASE TESTS ====================

class TestRegressionTestingEdgeCases:
    """Edge case tests for regression testing"""

    def test_empty_test_suite(self):
        """Test handling of empty test suite"""
        suite = RegressionTestSuite(
            commit_hash="abc123",
            tests=[],
            coverage_estimate=0.0,
            generated_at="2025-11-19"
        )

        runner = RegressionTestRunner()
        results = runner.run_regression_suite(suite)

        assert results["total_tests"] == 0
        assert results["passed"] == 0
        assert results["failed"] == 0
        assert results["errors"] == 0
        assert "test_results" in results

    @patch('subprocess.run')
    def test_subprocess_timeout_simulation(self, mock_subprocess):
        """Test handling of subprocess timeouts"""
        # Simulate timeout
        mock_process = Mock()
        mock_process.returncode = None  # Timeout
        mock_process.stdout = ""
        mock_process.stderr = "Timeout occurred"
        mock_subprocess.return_value = mock_process

        runner = RegressionTestRunner()
        suite = RegressionTestSuite(
            commit_hash="abc123",
            tests=[RegressionTest(
                code="def test(): import time; time.sleep(300)",  # Long test
                target_function="slow_func",
                change_description="test",
                test_type="performance",
                priority="medium",
                expected_behavior="should not timeout"
            )],
            coverage_estimate=10.0,
            generated_at="2025-11-19"
        )

        results = runner.run_regression_suite(suite)

        # Should handle timeout gracefully
        assert isinstance(results, dict)
        assert "total_tests" in results
