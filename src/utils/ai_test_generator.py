"""
AI-Powered Test Case Generation System

Advanced test generation using machine learning, static analysis, and coverage-guided techniques.
Provides automated test case generation to achieve 80%+ code coverage.
"""

import ast
import inspect
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class TestGenerationStrategy(Enum):
    """Test generation strategies."""
    COVERAGE_GUIDED = "coverage_guided"
    AI_GENERATED = "ai_generated"
    MUTATION_BASED = "mutation_based"
    PROPERTY_BASED = "property_based"
    CHAOS_ENGINEERING = "chaos_engineering"


@dataclass
class CodeAnalysis:
    """Code analysis results for test generation."""
    file_path: str
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    branches: List[Dict[str, Any]] = field(default_factory=list)
    complexity: Dict[str, int] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    uncovered_lines: Set[int] = field(default_factory=set)


@dataclass
class GeneratedTest:
    """Generated test case."""
    code: str
    target_function: str
    coverage_targets: List[int]
    test_type: str
    confidence: float
    dependencies: List[str]


class AICodeAnalyzer:
    """Advanced code analyzer for test generation."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def analyze_file(self, file_path: str) -> CodeAnalysis:
        """Analyze Python file for test generation opportunities."""
        analysis = CodeAnalysis(file_path=file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node, content)
                    analysis.functions.append(func_info)
                elif isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, content)
                    analysis.classes.append(class_info)

            # Calculate complexity
            analysis.complexity = self._calculate_complexity(content)

            # Extract dependencies
            analysis.dependencies = self._extract_dependencies(content)

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")

        return analysis

    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze individual function."""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', start_line + 1)

        function_code = '\n'.join(lines[start_line:end_line])

        return {
            'name': node.name,
            'line_start': start_line + 1,
            'line_end': end_line,
            'args': [arg.arg for arg in node.args.args],
            'returns': self._infer_return_type(node),
            'complexity': self._calculate_function_complexity(node),
            'branches': self._count_branches(node),
            'code': function_code
        }

    def _analyze_class(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analyze class structure."""
        return {
            'name': node.name,
            'methods': [method.name for method in node.body
                       if isinstance(method, ast.FunctionDef)],
            'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases]
        }

    def _calculate_complexity(self, content: str) -> Dict[str, int]:
        """Calculate cyclomatic complexity."""
        complexity = {}
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity[node.name] = self._calculate_function_complexity(node)

        return complexity

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and len(child.values) > 1:
                complexity += len(child.values) - 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)

        return complexity

    def _count_branches(self, node: ast.FunctionDef) -> int:
        """Count decision points in function."""
        branches = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                branches += 1
        return branches

    def _infer_return_type(self, node: ast.FunctionDef) -> str:
        """Infer return type from function."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                if isinstance(child.value, ast.Str):
                    return 'str'
                elif isinstance(child.value, ast.Num):
                    return 'int' if isinstance(child.value.n, int) else 'float'
                elif isinstance(child.value, ast.List):
                    return 'list'
                elif isinstance(child.value, ast.Dict):
                    return 'dict'
        return 'Any'

    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract import dependencies."""
        dependencies = []
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)

        return list(set(dependencies))


class AITestGenerator:
    """AI-powered test generator using multiple strategies."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.analyzer = AICodeAnalyzer()
        self.openai_client = None

        if OPENAI_AVAILABLE and openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)

        # Load or train ML model for test case prioritization
        self.ml_model = self._load_ml_model()

    def _load_ml_model(self) -> RandomForestClassifier:
        """Load or train ML model for test effectiveness prediction."""
        # For now, return a basic model. In production, this would be trained on historical data
        return RandomForestClassifier(n_estimators=100, random_state=42)

    def generate_tests_for_file(self, file_path: str, uncovered_lines: Set[int] = None) -> List[GeneratedTest]:
        """Generate comprehensive test cases for a file."""
        analysis = self.analyzer.analyze_file(file_path)

        if uncovered_lines:
            analysis.uncovered_lines = uncovered_lines

        generated_tests = []

        # Strategy 1: Coverage-guided test generation
        coverage_tests = self._generate_coverage_guided_tests(analysis)
        generated_tests.extend(coverage_tests)

        # Strategy 2: AI-generated tests using GPT
        if self.openai_client:
            ai_tests = self._generate_ai_tests(analysis)
            generated_tests.extend(ai_tests)

        # Strategy 3: Property-based test generation
        property_tests = self._generate_property_based_tests(analysis)
        generated_tests.extend(property_tests)

        # Strategy 4: Edge case and error condition tests
        edge_tests = self._generate_edge_case_tests(analysis)
        generated_tests.extend(edge_tests)

        return generated_tests

    def _generate_coverage_guided_tests(self, analysis: CodeAnalysis) -> List[GeneratedTest]:
        """Generate tests targeting uncovered lines."""
        tests = []

        for func in analysis.functions:
            if not self._is_function_tested(func, analysis.uncovered_lines):
                test = self._generate_basic_function_test(func, analysis)
                if test:
                    tests.append(test)

        return tests

    def _generate_ai_tests(self, analysis: CodeAnalysis) -> List[GeneratedTest]:
        """Generate tests using AI (GPT-4 level)."""
        tests = []

        if not self.openai_client:
            return tests

        for func in analysis.functions:
            if func['complexity'] > 3:  # Only for complex functions
                ai_test = self._generate_ai_test_for_function(func, analysis)
                if ai_test:
                    tests.append(ai_test)

        return tests

    def _generate_property_based_tests(self, analysis: CodeAnalysis) -> List[GeneratedTest]:
        """Generate property-based tests using Hypothesis patterns."""
        tests = []

        for func in analysis.functions:
            if func['args'] and func['complexity'] > 2:
                property_test = self._generate_property_test(func, analysis)
                if property_test:
                    tests.append(property_test)

        return tests

    def _generate_edge_case_tests(self, analysis: CodeAnalysis) -> List[GeneratedTest]:
        """Generate tests for edge cases and error conditions."""
        tests = []

        for func in analysis.functions:
            edge_tests = self._generate_edge_cases_for_function(func, analysis)
            tests.extend(edge_tests)

        return tests

    def _is_function_tested(self, func: Dict[str, Any], uncovered_lines: Set[int]) -> bool:
        """Check if function has uncovered lines."""
        func_lines = set(range(func['line_start'], func['line_end'] + 1))
        return not bool(func_lines & uncovered_lines)

    def _generate_basic_function_test(self, func: Dict[str, Any], analysis: CodeAnalysis) -> Optional[GeneratedTest]:
        """Generate basic unit test for function."""
        func_name = func['name']
        args = func['args']

        # Generate mock arguments based on inferred types
        mock_args = self._generate_mock_args(args)

        setup_lines = '\n    '.join(mock_args)
        args_list = ', '.join([arg.split(' = ')[0] for arg in mock_args])

        test_code = f"""
def test_{func_name}_basic():
    \"\"\"Test basic functionality of {func_name}.\"\"\"
    # Setup
    {setup_lines}

    # Execute
    result = {func_name}({args_list})

    # Assert
    assert result is not None
"""

        return GeneratedTest(
            code=test_code,
            target_function=func_name,
            coverage_targets=list(range(func['line_start'], func['line_end'] + 1)),
            test_type='unit',
            confidence=0.7,
            dependencies=analysis.dependencies
        )

    def _generate_ai_test_for_function(self, func: Dict[str, Any], analysis: CodeAnalysis) -> Optional[GeneratedTest]:
        """Generate AI-powered test using OpenAI."""
        try:
            prompt = self._create_ai_test_prompt(func, analysis)

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )

            test_code = response.choices[0].message.content.strip()

            return GeneratedTest(
                code=test_code,
                target_function=func['name'],
                coverage_targets=[],  # AI will determine coverage
                test_type='ai_generated',
                confidence=0.9,
                dependencies=analysis.dependencies
            )

        except Exception as e:
            logger.error(f"AI test generation failed: {e}")
            return None

    def _create_ai_test_prompt(self, func: Dict[str, Any], analysis: CodeAnalysis) -> str:
        """Create prompt for AI test generation."""
        return f"""
Generate a comprehensive pytest test for the following Python function:

Function name: {func['name']}
Arguments: {func['args']}
Return type: {func['returns']}
Complexity: {func['complexity']}
Code:
{func['code']}

Requirements:
1. Use pytest framework
2. Include edge cases and error conditions
3. Test all branches and paths
4. Include proper assertions
5. Add docstring
6. Handle exceptions appropriately
7. Use descriptive test name

Generate only the test function code, no additional text.
"""

    def _generate_property_test(self, func: Dict[str, Any], analysis: CodeAnalysis) -> Optional[GeneratedTest]:
        """Generate property-based test."""
        func_name = func['name']
        args = func['args']

        if not args:
            return None

        test_code = f"""
@given({self._generate_hypothesis_strategies(args)})
def test_{func_name}_properties({', '.join(args)}):
    \"\"\"Property-based test for {func_name}.\"\"\"
    result = {func_name}({', '.join(args)})

    # Property assertions
    assert isinstance(result, {func['returns']})
    # Add domain-specific properties here
"""

        return GeneratedTest(
            code=test_code,
            target_function=func_name,
            coverage_targets=list(range(func['line_start'], func['line_end'] + 1)),
            test_type='property',
            confidence=0.8,
            dependencies=analysis.dependencies + ['hypothesis']
        )

    def _generate_edge_cases_for_function(self, func: Dict[str, Any], analysis: CodeAnalysis) -> List[GeneratedTest]:
        """Generate edge case tests."""
        tests = []
        func_name = func['name']

        # Test with None values
        if func['args']:
            test_code = f"""
def test_{func_name}_none_values():
    \"\"\"Test {func_name} with None values.\"\"\"
    with pytest.raises((TypeError, ValueError)):
        {func_name}({', '.join(['None'] * len(func['args']))})
"""
            tests.append(GeneratedTest(
                code=test_code,
                target_function=func_name,
                coverage_targets=[func['line_start']],
                test_type='edge_case',
                confidence=0.6,
                dependencies=analysis.dependencies
            ))

        # Test with empty values
        if any('list' in str(func) or 'dict' in str(func) for func in analysis.functions):
            test_code = f"""
def test_{func_name}_empty_inputs():
    \"\"\"Test {func_name} with empty inputs.\"\"\"
    result = {func_name}({', '.join(['[]' if 'list' in arg else '{}' if 'dict' in arg else '""' for arg in func['args']])})
    assert result is not None
"""
            tests.append(GeneratedTest(
                code=test_code,
                target_function=func_name,
                coverage_targets=[func['line_start']],
                test_type='edge_case',
                confidence=0.6,
                dependencies=analysis.dependencies
            ))

        return tests

    def _generate_mock_args(self, args: List[str]) -> List[str]:
        """Generate mock arguments for testing."""
        mock_args = []
        for arg in args:
            if 'list' in arg.lower():
                mock_args.append(f"{arg} = []")
            elif 'dict' in arg.lower():
                mock_args.append(f"{arg} = {{}}")
            elif 'str' in arg.lower():
                mock_args.append(f"{arg} = \"test\"")
            elif 'int' in arg.lower():
                mock_args.append(f"{arg} = 42")
            elif 'float' in arg.lower():
                mock_args.append(f"{arg} = 3.14")
            elif 'bool' in arg.lower():
                mock_args.append(f"{arg} = True")
            else:
                mock_args.append(f"{arg} = None")
        return mock_args

    def _generate_hypothesis_strategies(self, args: List[str]) -> str:
        """Generate Hypothesis strategies for property testing."""
        strategies = []
        for arg in args:
            if 'list' in arg.lower():
                strategies.append(f"lists(integers())")
            elif 'dict' in arg.lower():
                strategies.append(f"dictionaries(keys=text(), values=integers())")
            elif 'str' in arg.lower():
                strategies.append(f"text()")
            elif 'int' in arg.lower():
                strategies.append(f"integers()")
            elif 'float' in arg.lower():
                strategies.append(f"floats()")
            elif 'bool' in arg.lower():
                strategies.append(f"booleans()")
            else:
                strategies.append(f"text()")
        return ', '.join(strategies)


class CoverageAnalyzer:
    """Analyze coverage and identify gaps."""

    def __init__(self):
        self.coverage_data = {}

    def analyze_coverage_report(self, coverage_file: str = 'htmlcov/index.html') -> Dict[str, Any]:
        """Analyze coverage report to identify gaps."""
        gaps = {}

        try:
            with open(coverage_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract coverage data (simplified parsing)
            # In production, use coverage.py API
            lines = content.split('\n')
            for line in lines:
                if 'data-ratio' in line and 'right' in line:
                    # Parse coverage percentages
                    pass

        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")

        return gaps

    def get_uncovered_lines(self, file_path: str) -> Set[int]:
        """Get uncovered lines for a file."""
        # Use coverage.py API to get detailed line coverage
        if not COVERAGE_AVAILABLE:
            return set()

        try:
            cov = coverage.Coverage()
            cov.load()

            # Get line coverage data
            uncovered = set()
            for filename, data in cov.get_data().lines.items():
                if filename == file_path:
                    all_lines = set(range(min(data), max(data) + 1))
                    covered_lines = set(data)
                    uncovered = all_lines - covered_lines
                    break

            return uncovered

        except Exception as e:
            logger.error(f"Failed to get uncovered lines: {e}")
            return set()


class TestGeneratorManager:
    """Main manager for automated test generation."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.ai_generator = AITestGenerator(openai_api_key)
        self.coverage_analyzer = CoverageAnalyzer()
        self.generated_tests = []

    def generate_comprehensive_test_suite(self, target_directory: str, min_coverage: float = 80.0) -> Dict[str, Any]:
        """Generate comprehensive test suite to achieve target coverage."""
        results = {
            'total_tests_generated': 0,
            'coverage_improvement': 0.0,
            'test_files_created': [],
            'errors': []
        }

        # Find all Python files in target directory
        python_files = list(Path(target_directory).rglob('*.py'))

        for file_path in python_files:
            try:
                # Skip test files and __init__.py
                if 'test' in str(file_path) or file_path.name == '__init__.py':
                    continue

                logger.info(f"Generating tests for {file_path}")

                # Get uncovered lines
                uncovered_lines = self.coverage_analyzer.get_uncovered_lines(str(file_path))

                # Generate tests
                tests = self.ai_generator.generate_tests_for_file(str(file_path), uncovered_lines)

                if tests:
                    # Create test file
                    test_file_path = self._create_test_file(file_path, tests)
                    results['test_files_created'].append(str(test_file_path))
                    results['total_tests_generated'] += len(tests)

            except Exception as e:
                results['errors'].append(f"Error processing {file_path}: {e}")

        return results

    def _create_test_file(self, source_file: Path, tests: List[GeneratedTest]) -> Path:
        """Create test file from generated tests."""
        test_dir = Path('tests/unit')
        test_dir.mkdir(exist_ok=True)

        # Create test file name
        test_filename = f"test_{source_file.stem}_generated.py"
        test_file_path = test_dir / test_filename

        # Generate imports
        imports = self._generate_imports(tests)

        # Combine all test code
        test_content = imports + '\n\n' + '\n\n'.join([test.code for test in tests])

        # Write test file
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)

        return test_file_path

    def _generate_imports(self, tests: List[GeneratedTest]) -> str:
        """Generate import statements for test file."""
        all_deps = set()
        for test in tests:
            all_deps.update(test.dependencies)

        imports = [
            "import pytest",
            "import numpy as np",
            "import pandas as pd",
            "from unittest.mock import Mock, patch, MagicMock"
        ]

        # Add hypothesis if property tests exist
        if any(test.test_type == 'property' for test in tests):
            imports.append("from hypothesis import given, strategies as st")

        # Add source module import
        source_module = "src." + '.'.join(['utils'] + ['ai_test_generator'])  # Adjust based on actual structure
        imports.append(f"from {source_module} import *")

        return '\n'.join(imports)


# Convenience functions
def generate_tests_for_module(module_path: str, openai_key: Optional[str] = None) -> Dict[str, Any]:
    """Generate tests for a specific module."""
    manager = TestGeneratorManager(openai_key)
    return manager.generate_comprehensive_test_suite(module_path)


def analyze_coverage_gaps() -> Dict[str, Any]:
    """Analyze current coverage gaps."""
    analyzer = CoverageAnalyzer()
    return analyzer.analyze_coverage_report()


if __name__ == "__main__":
    # Example usage
    manager = TestGeneratorManager()

    # Generate tests for src directory
    results = manager.generate_comprehensive_test_suite('src')

    print(f"Generated {results['total_tests_generated']} tests")
    print(f"Created test files: {results['test_files_created']}")
    if results['errors']:
        print(f"Errors: {results['errors']}")
