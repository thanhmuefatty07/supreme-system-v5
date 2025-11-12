"""
Mutation Testing Framework for Supreme System V5

Advanced mutation testing to ensure test suite quality and identify weak test cases.
Generates mutants of the code and verifies that tests can detect the changes.
"""

import ast
import copy
import importlib
import inspect
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class MutationOperator(Enum):
    """Types of mutation operators."""
    ARITHMETIC_REPLACEMENT = "arithmetic_replacement"
    LOGICAL_REPLACEMENT = "logical_replacement"
    COMPARISON_REPLACEMENT = "comparison_replacement"
    CONSTANT_REPLACEMENT = "constant_replacement"
    STATEMENT_DELETION = "statement_deletion"
    RETURN_VALUE_REPLACEMENT = "return_value_replacement"
    VARIABLE_REPLACEMENT = "variable_replacement"
    METHOD_CALL_REPLACEMENT = "method_call_replacement"


@dataclass
class Mutant:
    """Represents a code mutant."""
    id: str
    file_path: str
    line_number: int
    operator: MutationOperator
    original_code: str
    mutated_code: str
    status: str = "alive"  # alive, killed, equivalent
    killed_by: List[str] = field(default_factory=list)


@dataclass
class MutationTestResult:
    """Results of mutation testing."""
    total_mutants: int
    killed_mutants: int
    alive_mutants: int
    equivalent_mutants: int
    mutation_score: float
    coverage_improvement: float
    weak_tests: List[str]
    strong_tests: List[str]


class MutationEngine:
    """Core mutation testing engine."""

    def __init__(self, source_dir: str = "src", test_command: str = "pytest"):
        self.source_dir = Path(source_dir)
        self.test_command = test_command
        self.mutants: List[Mutant] = []
        self.results: Dict[str, Any] = {}

    def run_mutation_testing(self, target_files: List[str] = None,
                           operators: List[MutationOperator] = None) -> MutationTestResult:
        """Run comprehensive mutation testing."""
        if operators is None:
            operators = list(MutationOperator)

        # Generate mutants
        self._generate_mutants(target_files or self._get_source_files(), operators)

        # Test mutants
        results = self._test_mutants()

        # Calculate mutation score
        mutation_score = self._calculate_mutation_score(results)

        # Identify weak tests
        weak_tests, strong_tests = self._analyze_test_strength(results)

        return MutationTestResult(
            total_mutants=len(self.mutants),
            killed_mutants=sum(1 for m in self.mutants if m.status == "killed"),
            alive_mutants=sum(1 for m in self.mutants if m.status == "alive"),
            equivalent_mutants=sum(1 for m in self.mutants if m.status == "equivalent"),
            mutation_score=mutation_score,
            coverage_improvement=self._estimate_coverage_improvement(),
            weak_tests=weak_tests,
            strong_tests=strong_tests
        )

    def _get_source_files(self) -> List[str]:
        """Get all Python source files."""
        return [str(f) for f in self.source_dir.rglob("*.py")
                if not f.name.startswith("test_") and f.name != "__init__.py"]

    def _generate_mutants(self, files: List[str], operators: List[MutationOperator]):
        """Generate mutants for target files."""
        for file_path in files:
            try:
                mutants = self._generate_mutants_for_file(file_path, operators)
                self.mutants.extend(mutants)
                logger.info(f"Generated {len(mutants)} mutants for {file_path}")
            except Exception as e:
                logger.error(f"Failed to generate mutants for {file_path}: {e}")

    def _generate_mutants_for_file(self, file_path: str, operators: List[MutationOperator]) -> List[Mutant]:
        """Generate mutants for a single file."""
        mutants = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        lines = content.split('\n')

        mutant_id = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and any(op in operators for op in [MutationOperator.ARITHMETIC_REPLACEMENT]):
                # Arithmetic operator replacement
                if isinstance(node.op, ast.Add):
                    mutants.extend(self._create_mutant(file_path, node.lineno, mutant_id,
                                                     "x + y", "x - y", MutationOperator.ARITHMETIC_REPLACEMENT, lines))
                    mutant_id += 1
                elif isinstance(node.op, ast.Sub):
                    mutants.extend(self._create_mutant(file_path, node.lineno, mutant_id,
                                                     "x - y", "x + y", MutationOperator.ARITHMETIC_REPLACEMENT, lines))
                    mutant_id += 1

            elif isinstance(node, ast.Compare) and MutationOperator.COMPARISON_REPLACEMENT in operators:
                # Comparison operator replacement
                if isinstance(node.ops[0], ast.Eq):
                    mutants.extend(self._create_mutant(file_path, node.lineno, mutant_id,
                                                     "x == y", "x != y", MutationOperator.COMPARISON_REPLACEMENT, lines))
                    mutant_id += 1

            elif isinstance(node, ast.Return) and MutationOperator.RETURN_VALUE_REPLACEMENT in operators:
                # Return value replacement
                mutants.extend(self._create_mutant(file_path, node.lineno, mutant_id,
                                                 "return x", "return None", MutationOperator.RETURN_VALUE_REPLACEMENT, lines))
                mutant_id += 1

            elif isinstance(node, ast.Constant) and MutationOperator.CONSTANT_REPLACEMENT in operators:
                # Constant replacement
                if isinstance(node.value, (int, float)) and node.value != 0:
                    mutants.extend(self._create_mutant(file_path, node.lineno, mutant_id,
                                                     str(node.value), "0", MutationOperator.CONSTANT_REPLACEMENT, lines))
                    mutant_id += 1

        return mutants

    def _create_mutant(self, file_path: str, line_no: int, mutant_id: int,
                      original: str, mutated: str, operator: MutationOperator, lines: List[str]) -> List[Mutant]:
        """Create a mutant instance."""
        line_content = lines[line_no - 1] if line_no <= len(lines) else ""

        # Simple text replacement (in production, use AST transformation)
        if original in line_content:
            mutated_line = line_content.replace(original, mutated, 1)
            return [Mutant(
                id=f"{Path(file_path).stem}_{mutant_id}",
                file_path=file_path,
                line_number=line_no,
                operator=operator,
                original_code=line_content,
                mutated_code=mutated_line
            )]
        return []

    def _test_mutants(self) -> Dict[str, Any]:
        """Test all generated mutants."""
        results = {}

        # Use thread pool for parallel testing
        with ThreadPoolExecutor(max_workers=min(4, len(self.mutants))) as executor:
            future_to_mutant = {
                executor.submit(self._test_single_mutant, mutant): mutant
                for mutant in self.mutants
            }

            for future in as_completed(future_to_mutant):
                mutant = future_to_mutant[future]
                try:
                    result = future.result()
                    results[mutant.id] = result
                except Exception as e:
                    logger.error(f"Failed to test mutant {mutant.id}: {e}")
                    results[mutant.id] = {"status": "error", "killed_by": []}

        return results

    def _test_single_mutant(self, mutant: Mutant) -> Dict[str, Any]:
        """Test a single mutant."""
        # Create temporary mutated file
        with tempfile.TemporaryDirectory() as temp_dir:
            mutated_file = Path(temp_dir) / Path(mutant.file_path).name

            # Read original file
            with open(mutant.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Apply mutation
            if mutant.line_number <= len(lines):
                lines[mutant.line_number - 1] = mutant.mutated_code + '\n'

                # Write mutated file
                with open(mutated_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                # Copy to source directory temporarily
                temp_source = Path(temp_dir) / "src"
                temp_source.mkdir()
                import shutil
                for file in Path(self.source_dir).rglob("*.py"):
                    rel_path = file.relative_to(self.source_dir)
                    dest = temp_source / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if str(rel_path) == Path(mutant.file_path).relative_to(self.source_dir):
                        shutil.copy(mutated_file, dest)
                    else:
                        shutil.copy(file, dest)

                # Run tests
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minutes timeout
                    )

                    if result.returncode != 0:
                        # Tests failed - mutant killed
                        killed_by = self._extract_failed_tests(result.stdout + result.stderr)
                        return {"status": "killed", "killed_by": killed_by}
                    else:
                        # Tests passed - mutant alive
                        return {"status": "alive", "killed_by": []}

                except subprocess.TimeoutExpired:
                    logger.warning(f"Mutant {mutant.id} test timed out")
                    return {"status": "alive", "killed_by": []}
                except Exception as e:
                    logger.error(f"Error testing mutant {mutant.id}: {e}")
                    return {"status": "error", "killed_by": []}

        return {"status": "equivalent", "killed_by": []}

    def _extract_failed_tests(self, output: str) -> List[str]:
        """Extract names of failed tests from pytest output."""
        failed_tests = []
        lines = output.split('\n')

        for line in lines:
            if line.startswith('FAILED') or line.startswith('ERROR'):
                # Extract test name from line like "FAILED tests/test_example.py::test_function - ..."
                match = re.search(r'FAILED\s+([^:\s]+)::([^:\s]+)', line)
                if match:
                    failed_tests.append(f"{match.group(1)}::{match.group(2)}")

        return failed_tests

    def _calculate_mutation_score(self, results: Dict[str, Any]) -> float:
        """Calculate mutation score."""
        total_tested = len(results)
        if total_tested == 0:
            return 0.0

        killed = sum(1 for r in results.values() if r.get("status") == "killed")
        return (killed / total_tested) * 100

    def _analyze_test_strength(self, results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze which tests are weak vs strong."""
        test_effectiveness = {}

        for mutant_result in results.values():
            for test in mutant_result.get("killed_by", []):
                if test not in test_effectiveness:
                    test_effectiveness[test] = 0
                test_effectiveness[test] += 1

        # Sort tests by effectiveness
        sorted_tests = sorted(test_effectiveness.items(), key=lambda x: x[1], reverse=True)

        # Top 25% are strong, bottom 25% are weak
        split_point = len(sorted_tests) // 4
        strong_tests = [test for test, _ in sorted_tests[:split_point]]
        weak_tests = [test for test, _ in sorted_tests[-split_point:]]

        return weak_tests, strong_tests

    def _estimate_coverage_improvement(self) -> float:
        """Estimate coverage improvement needed based on mutation results."""
        # Simplified estimation - in production, correlate with actual coverage
        alive_ratio = sum(1 for m in self.mutants if m.status == "alive") / len(self.mutants)
        return alive_ratio * 15  # Rough estimate


class MutationTestManager:
    """High-level manager for mutation testing campaigns."""

    def __init__(self):
        self.engine = MutationEngine()

    def run_comprehensive_analysis(self, target_modules: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive mutation analysis."""
        logger.info("Starting comprehensive mutation testing analysis")

        # Run mutation testing
        result = self.engine.run_mutation_testing(target_files=target_modules)

        # Generate recommendations
        recommendations = self._generate_recommendations(result)

        return {
            "mutation_results": result,
            "recommendations": recommendations,
            "analysis_summary": self._create_analysis_summary(result)
        }

    def _generate_recommendations(self, result: MutationTestResult) -> List[str]:
        """Generate recommendations based on mutation results."""
        recommendations = []

        if result.mutation_score < 70:
            recommendations.append("CRITICAL: Mutation score below 70%. Test suite quality is inadequate.")

        if len(result.weak_tests) > 0:
            recommendations.append(f"Improve {len(result.weak_tests)} weak tests: {', '.join(result.weak_tests[:5])}")

        if result.alive_mutants > result.total_mutants * 0.3:
            recommendations.append("Too many alive mutants. Add more comprehensive test cases.")

        if result.coverage_improvement > 10:
            recommendations.append(".1f")

        return recommendations

    def _create_analysis_summary(self, result: MutationTestResult) -> str:
        """Create human-readable analysis summary."""
        summary = f"""
Mutation Testing Analysis Summary
=================================

Total Mutants: {result.total_mutants}
Killed Mutants: {result.killed_mutants}
Alive Mutants: {result.alive_mutants}
Equivalent Mutants: {result.equivalent_mutants}
Mutation Score: {result.mutation_score:.1f}%

Coverage Improvement Estimate: {result.coverage_improvement:.1f}%

Test Suite Quality Assessment:
"""

        if result.mutation_score >= 80:
            summary += "- EXCELLENT: High-quality test suite with strong mutation resistance"
        elif result.mutation_score >= 70:
            summary += "- GOOD: Adequate test coverage with room for improvement"
        elif result.mutation_score >= 60:
            summary += "- FAIR: Test suite needs enhancement"
        else:
            summary += "- POOR: Test suite requires significant improvement"

        return summary


# Convenience functions
def run_mutation_analysis(target_file: str = None) -> MutationTestResult:
    """Run mutation analysis on target file or entire codebase."""
    manager = MutationTestManager()

    if target_file:
        result = manager.engine.run_mutation_testing(target_files=[target_file])
    else:
        result = manager.engine.run_mutation_testing()

    return result


def analyze_test_weaknesses() -> Dict[str, List[str]]:
    """Analyze and return weak vs strong tests."""
    manager = MutationTestManager()
    result = manager.engine.run_mutation_testing()

    return {
        "weak_tests": result.weak_tests,
        "strong_tests": result.strong_tests
    }


if __name__ == "__main__":
    # Example usage
    manager = MutationTestManager()
    analysis = manager.run_comprehensive_analysis()

    print(analysis["analysis_summary"])
    print("\nRecommendations:")
    for rec in analysis["recommendations"]:
        print(f"- {rec}")

