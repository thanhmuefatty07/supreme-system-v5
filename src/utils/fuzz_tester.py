"""
Fuzz Testing Framework for Supreme System V5

Advanced fuzz testing to discover edge cases and vulnerabilities through
coverage-guided fuzzing and input generation.
"""

import logging
import random
import string
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import coverage
import ast
import inspect

logger = logging.getLogger(__name__)


class FuzzStrategy(Enum):
    """Fuzzing strategies."""
    RANDOM = "random"
    COVERAGE_GUIDED = "coverage_guided"
    MUTATION_BASED = "mutation_based"
    GENERATIONAL = "generational"


@dataclass
class FuzzInput:
    """Represents a fuzz test input."""
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    coverage_paths: Set[str] = field(default_factory=set)
    execution_time: float = 0.0
    crashed: bool = False
    error_message: str = ""


@dataclass
class FuzzResult:
    """Results of fuzz testing."""
    function_name: str
    total_inputs: int
    crashes_found: int
    unique_crashes: int
    coverage_achieved: float
    execution_time: float
    interesting_inputs: List[FuzzInput]
    error_patterns: Dict[str, int]


class FuzzGenerator:
    """Generates fuzz inputs for different data types."""

    def __init__(self, strategy: FuzzStrategy = FuzzStrategy.COVERAGE_GUIDED):
        self.strategy = strategy
        self.coverage_data = {}
        self.seed_corpus: List[FuzzInput] = []

    def generate_string_inputs(self, count: int = 100) -> List[FuzzInput]:
        """Generate fuzz inputs for string parameters."""
        inputs = []

        # Basic string patterns
        patterns = [
            "",  # Empty string
            "a",  # Single character
            "normal_string",
            "A" * 1000,  # Long string
            "special_chars: !@#$%^&*()",
            "unicode: 你好世界 🌍",
            "sql_injection: ' OR '1'='1",
            "path_traversal: ../../../etc/passwd",
            "command_injection: ; rm -rf /",
            "xml_injection: <![CDATA[<script>alert(1)</script>]]>",
            "json_injection: {\"__proto__\": {\"isAdmin\": true}}",
            "regex_special: .*+?^$()[]{}|\\",
            "null_bytes: \x00\x01\x02",
            "overflow: " + "x" * 100000,  # Very long string
        ]

        # Generate additional random strings
        for _ in range(count - len(patterns)):
            length = random.randint(0, 1000)
            chars = string.ascii_letters + string.digits + string.punctuation + " \t\n\r"
            random_string = ''.join(random.choices(chars, k=length))

            # Occasionally add unicode
            if random.random() < 0.1:
                random_string += "".join(chr(random.randint(0x80, 0xFFFF)) for _ in range(10))

            patterns.append(random_string)

        for pattern in patterns:
            inputs.append(FuzzInput(data=pattern))

        return inputs

    def generate_numeric_inputs(self, count: int = 100) -> List[FuzzInput]:
        """Generate fuzz inputs for numeric parameters."""
        inputs = []

        # Boundary values
        boundaries = [
            0, 1, -1,
            2**31 - 1, 2**31, 2**32 - 1,  # 32-bit boundaries
            2**63 - 1, 2**63, 2**64 - 1,  # 64-bit boundaries
            float('inf'), float('-inf'), float('nan'),
            1e-100, 1e100,  # Very small/large floats
            0.0, -0.0, 1.0, -1.0,
            3.14159, 2.71828,  # Common constants
        ]

        # Random values
        for _ in range(count - len(boundaries)):
            if random.random() < 0.5:
                # Integer
                value = random.randint(-2**63, 2**63)
            else:
                # Float
                value = random.uniform(-1e100, 1e100)
                if random.random() < 0.1:
                    value = float('inf') if random.random() < 0.5 else float('-inf')

            boundaries.append(value)

        for value in boundaries:
            inputs.append(FuzzInput(data=value))

        return inputs

    def generate_list_inputs(self, count: int = 50) -> List[FuzzInput]:
        """Generate fuzz inputs for list parameters."""
        inputs = []

        # Various list patterns
        patterns = [
            [],  # Empty list
            [1],  # Single element
            [1, 2, 3],  # Normal list
            [None] * 1000,  # Large list of None
            ["a"] * 1000,  # Large list of strings
            list(range(100000)),  # Very large list
            [{}] * 100,  # List of dicts
            [[1, 2], [3, 4]],  # Nested lists
            [float('inf')] * 10,  # Special floats
            [""] * 1000,  # Empty strings
        ]

        # Generate additional random lists
        for _ in range(count - len(patterns)):
            length = random.randint(0, 1000)
            if random.random() < 0.3:
                # Homogeneous list
                element_type = random.choice(['int', 'str', 'float', 'dict'])
                if element_type == 'int':
                    random_list = [random.randint(-1000, 1000) for _ in range(length)]
                elif element_type == 'str':
                    random_list = [''.join(random.choices(string.ascii_letters, k=random.randint(0, 100)))
                                 for _ in range(length)]
                elif element_type == 'float':
                    random_list = [random.uniform(-1000, 1000) for _ in range(length)]
                else:
                    random_list = [{} for _ in range(length)]
            else:
                # Heterogeneous list
                random_list = []
                for _ in range(length):
                    type_choice = random.choice(['int', 'str', 'float', 'dict', 'list'])
                    if type_choice == 'int':
                        random_list.append(random.randint(-100, 100))
                    elif type_choice == 'str':
                        random_list.append(''.join(random.choices(string.ascii_letters, k=random.randint(0, 10))))
                    elif type_choice == 'float':
                        random_list.append(random.uniform(-100, 100))
                    elif type_choice == 'dict':
                        random_list.append({})
                    else:
                        random_list.append([])

            patterns.append(random_list)

        for pattern in patterns:
            inputs.append(FuzzInput(data=pattern))

        return inputs

    def generate_dict_inputs(self, count: int = 50) -> List[FuzzInput]:
        """Generate fuzz inputs for dict parameters."""
        inputs = []

        # Various dict patterns
        patterns = [
            {},  # Empty dict
            {"key": "value"},  # Simple dict
            {"__proto__": {"isAdmin": True}},  # Prototype pollution
            {"__class__": "EvilClass"},  # Class pollution
            {None: None, 0: 0, "": ""},  # Edge keys
            {"x" * 1000: "y" * 1000},  # Large keys/values
            {i: i**2 for i in range(1000)},  # Large dict
            {"nested": {"deeply": {"nested": "value"}}},  # Deep nesting
            {"circular_ref": None},  # Placeholder for circular refs
        ]

        # Generate additional random dicts
        for _ in range(count - len(patterns)):
            size = random.randint(0, 100)
            random_dict = {}

            for _ in range(size):
                # Generate random keys
                key_type = random.choice(['str', 'int', 'tuple'])
                if key_type == 'str':
                    key = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 20)))
                elif key_type == 'int':
                    key = random.randint(-100, 100)
                else:
                    key = tuple(random.randint(-10, 10) for _ in range(random.randint(1, 3)))

                # Generate random values
                value_type = random.choice(['str', 'int', 'float', 'list', 'dict'])
                if value_type == 'str':
                    value = ''.join(random.choices(string.ascii_letters, k=random.randint(0, 50)))
                elif value_type == 'int':
                    value = random.randint(-1000, 1000)
                elif value_type == 'float':
                    value = random.uniform(-1000, 1000)
                elif value_type == 'list':
                    value = [random.randint(-10, 10) for _ in range(random.randint(0, 10))]
                else:
                    value = {}

                random_dict[key] = value

            patterns.append(random_dict)

        for pattern in patterns:
            inputs.append(FuzzInput(data=pattern))

        return inputs

    def mutate_input(self, input_data: FuzzInput) -> FuzzInput:
        """Mutate an existing input to create new test cases."""
        mutated_data = self._mutate_value(input_data.data)
        return FuzzInput(
            data=mutated_data,
            metadata={"mutated_from": input_data.metadata.get("id", "unknown")}
        )

    def _mutate_value(self, value: Any) -> Any:
        """Mutate a value to create fuzz input."""
        if isinstance(value, str):
            return self._mutate_string(value)
        elif isinstance(value, (int, float)):
            return self._mutate_number(value)
        elif isinstance(value, list):
            return self._mutate_list(value)
        elif isinstance(value, dict):
            return self._mutate_dict(value)
        else:
            # For other types, try basic mutations
            return self._mutate_generic(value)

    def _mutate_string(self, s: str) -> str:
        """Mutate a string value."""
        mutations = [
            lambda x: x + "A" * random.randint(1, 1000),  # Append
            lambda x: "A" * random.randint(1, 1000) + x,  # Prepend
            lambda x: x[::-1],  # Reverse
            lambda x: x.upper(),  # Uppercase
            lambda x: x.lower(),  # Lowercase
            lambda x: x.replace(random.choice(x) if x else "A", chr(random.randint(0, 255))),  # Replace char
            lambda x: x + chr(random.randint(0, 0xFFFF)),  # Add unicode
            lambda x: x + "\x00" * random.randint(1, 10),  # Add null bytes
        ]

        mutation = random.choice(mutations)
        return mutation(s)

    def _mutate_number(self, n: float) -> float:
        """Mutate a numeric value."""
        mutations = [
            lambda x: x + random.randint(-1000, 1000),
            lambda x: x * random.uniform(0.1, 10),
            lambda x: -x,
            lambda x: float('inf') if x == 0 else x,
            lambda x: float('nan'),
            lambda x: 0 if x != 0 else 1,
        ]

        mutation = random.choice(mutations)
        return mutation(n)

    def _mutate_list(self, lst: list) -> list:
        """Mutate a list."""
        if not lst:
            return [random.randint(-10, 10)]

        mutations = [
            lambda x: x + [random.randint(-10, 10)] * random.randint(1, 100),  # Extend
            lambda x: x[::-1],  # Reverse
            lambda x: x[random.randint(0, len(x)-1):] if x else x,  # Slice
            lambda x: [item * random.randint(2, 10) if isinstance(item, (int, float)) else item for item in x],  # Modify elements
            lambda x: [],  # Empty
        ]

        mutation = random.choice(mutations)
        return mutation(lst.copy())

    def _mutate_dict(self, d: dict) -> dict:
        """Mutate a dictionary."""
        if not d:
            return {random.choice(string.ascii_letters): random.randint(-10, 10)}

        mutations = [
            lambda x: {**x, random.choice(string.ascii_letters): random.randint(-10, 10)},  # Add key
            lambda x: {k: v * 2 if isinstance(v, (int, float)) else v for k, v in x.items()},  # Modify values
            lambda x: {},  # Empty
            lambda x: {k: {} if isinstance(v, dict) else v for k, v in x.items()},  # Nest dicts
        ]

        mutation = random.choice(mutations)
        return mutation(d.copy())

    def _mutate_generic(self, value: Any) -> Any:
        """Generic mutation for unsupported types."""
        return None  # Represent None/missing values


class FuzzTester:
    """Main fuzz testing engine."""

    def __init__(self, strategy: FuzzStrategy = FuzzStrategy.COVERAGE_GUIDED):
        self.strategy = strategy
        self.generator = FuzzGenerator(strategy)
        self.coverage_tracker = None

    def fuzz_function(self, func: Callable, input_generators: List[Callable],
                     max_iterations: int = 1000, timeout_seconds: int = 30) -> FuzzResult:
        """Fuzz test a function with generated inputs."""
        logger.info(f"Starting fuzz testing for {func.__name__}")

        start_time = time.time()
        inputs_tested = 0
        crashes_found = 0
        unique_crashes = set()
        interesting_inputs = []
        error_patterns = {}

        # Setup coverage tracking
        if self.strategy == FuzzStrategy.COVERAGE_GUIDED:
            self.coverage_tracker = coverage.Coverage()
            self.coverage_tracker.start()

        try:
            for iteration in range(max_iterations):
                if time.time() - start_time > timeout_seconds:
                    break

                # Generate inputs
                inputs = []
                for generator in input_generators:
                    fuzz_input = generator()
                    if isinstance(fuzz_input, list):
                        inputs.append(random.choice(fuzz_input))
                    else:
                        inputs.append(fuzz_input)

                # Test the function
                try:
                    execution_start = time.time()
                    result = func(*inputs)
                    execution_time = time.time() - execution_start

                    inputs_tested += 1

                    # Check if this is an interesting input (e.g., new coverage)
                    if self._is_interesting_input(result, inputs):
                        interesting_inputs.append(FuzzInput(
                            data=inputs,
                            execution_time=execution_time,
                            metadata={"iteration": iteration}
                        ))

                except Exception as e:
                    crashes_found += 1
                    error_msg = str(e)
                    error_type = type(e).__name__

                    if error_msg not in unique_crashes:
                        unique_crashes.add(error_msg)

                    error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

                    interesting_inputs.append(FuzzInput(
                        data=inputs,
                        execution_time=time.time() - execution_start,
                        crashed=True,
                        error_message=error_msg,
                        metadata={"iteration": iteration, "error_type": error_type}
                    ))

                    logger.debug(f"Crash found in iteration {iteration}: {error_msg}")

                # Progress reporting
                if iteration % 100 == 0:
                    logger.info(f"Fuzz testing progress: {iteration}/{max_iterations} iterations")

        finally:
            if self.coverage_tracker:
                self.coverage_tracker.stop()
                self.coverage_tracker.save()

        # Calculate coverage
        coverage_achieved = self._calculate_coverage_achieved(func)

        execution_time = time.time() - start_time

        return FuzzResult(
            function_name=func.__name__,
            total_inputs=inputs_tested,
            crashes_found=crashes_found,
            unique_crashes=len(unique_crashes),
            coverage_achieved=coverage_achieved,
            execution_time=execution_time,
            interesting_inputs=interesting_inputs,
            error_patterns=error_patterns
        )

    def _is_interesting_input(self, result: Any, inputs: List[Any]) -> bool:
        """Determine if an input is interesting (e.g., new behavior)."""
        # Simple heuristics - can be enhanced with coverage analysis
        if result is None:
            return True
        if isinstance(result, (int, float)) and not (-1e10 < result < 1e10):
            return True  # Extreme values
        if isinstance(result, str) and len(result) > 1000:
            return True  # Very long strings
        if isinstance(result, (list, dict)) and len(result) > 100:
            return True  # Large data structures

        return False

    def _calculate_coverage_achieved(self, func: Callable) -> float:
        """Calculate code coverage achieved for the function."""
        if not self.coverage_tracker:
            return 0.0

        try:
            # Get coverage data for the function's file
            source_file = inspect.getfile(func)
            analysis = self.coverage_tracker._analyze(source_file)

            if analysis:
                covered_lines = len(analysis.executed)
                total_lines = len(analysis.executable)
                return (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

        except Exception as e:
            logger.error(f"Coverage calculation failed: {e}")

        return 0.0


class FuzzTestingManager:
    """Manager for comprehensive fuzz testing campaigns."""

    def __init__(self):
        self.tester = FuzzTester()

    def fuzz_module_functions(self, module, max_iterations_per_function: int = 500) -> Dict[str, FuzzResult]:
        """Fuzz test all public functions in a module."""
        results = {}

        # Get all functions from the module
        functions = []
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith('_'):
                functions.append((name, obj))

        logger.info(f"Found {len(functions)} functions to fuzz test in {module.__name__}")

        for func_name, func in functions:
            try:
                logger.info(f"Fuzz testing function: {func_name}")

                # Generate input generators based on function signature
                input_generators = self._create_input_generators_for_function(func)

                if input_generators:
                    result = self.tester.fuzz_function(
                        func, input_generators,
                        max_iterations=max_iterations_per_function
                    )
                    results[func_name] = result

                    logger.info(f"Function {func_name}: {result.crashes_found} crashes, "
                              f"{result.coverage_achieved:.1f}% coverage")
                else:
                    logger.warning(f"Could not create input generators for {func_name}")

            except Exception as e:
                logger.error(f"Fuzz testing failed for {func_name}: {e}")

        return results

    def _create_input_generators_for_function(self, func: Callable) -> List[Callable]:
        """Create input generators based on function signature."""
        try:
            sig = inspect.signature(func)
            generators = []

            for param_name, param in sig.parameters.items():
                if param.annotation == str or 'str' in str(param.annotation).lower():
                    generators.append(lambda: self.tester.generator.generate_string_inputs(10))
                elif param.annotation in (int, float) or 'int' in str(param.annotation).lower() or 'float' in str(param.annotation).lower():
                    generators.append(lambda: self.tester.generator.generate_numeric_inputs(10))
                elif param.annotation == list or 'list' in str(param.annotation).lower():
                    generators.append(lambda: self.tester.generator.generate_list_inputs(5))
                elif param.annotation == dict or 'dict' in str(param.annotation).lower():
                    generators.append(lambda: self.tester.generator.generate_dict_inputs(5))
                else:
                    # Default to mixed inputs
                    generators.append(lambda: self.tester.generator.generate_string_inputs(5) +
                                    self.tester.generator.generate_numeric_inputs(5))

            return generators

        except Exception as e:
            logger.error(f"Failed to create input generators: {e}")
            return []

    def generate_fuzz_test_report(self, results: Dict[str, FuzzResult]) -> str:
        """Generate a comprehensive fuzz testing report."""
        total_functions = len(results)
        total_crashes = sum(r.crashes_found for r in results.values())
        total_unique_crashes = sum(r.unique_crashes for r in results.values())
        avg_coverage = sum(r.coverage_achieved for r in results.values()) / total_functions if total_functions > 0 else 0

        report = f"""
Fuzz Testing Report
===================

Summary:
- Functions tested: {total_functions}
- Total crashes found: {total_crashes}
- Unique crash types: {total_unique_crashes}
- Average coverage achieved: {avg_coverage:.1f}%

Detailed Results:
"""

        for func_name, result in results.items():
            report += f"""
{func_name}:
  - Inputs tested: {result.total_inputs}
  - Crashes found: {result.crashes_found}
  - Coverage: {result.coverage_achieved:.1f}%
  - Execution time: {result.execution_time:.2f}s
  - Error patterns: {result.error_patterns}
"""

        # Recommendations
        recommendations = []
        if total_crashes > 0:
            recommendations.append("CRITICAL: Crashes found - fix input validation and error handling")
        if avg_coverage < 70:
            recommendations.append("Coverage below target - add more comprehensive test cases")
        if len([r for r in results.values() if r.crashes_found > 0]) > total_functions * 0.5:
            recommendations.append("High crash rate across functions - implement global error handling")

        if recommendations:
            report += "\nRecommendations:\n" + "\n".join(f"- {rec}" for rec in recommendations)

        return report


# Convenience functions
def fuzz_test_module(module) -> str:
    """Fuzz test an entire module and return report."""
    manager = FuzzTestingManager()
    results = manager.fuzz_module_functions(module)
    return manager.generate_fuzz_test_report(results)


def fuzz_test_function(func: Callable, input_generators: List[Callable],
                      max_iterations: int = 1000) -> FuzzResult:
    """Fuzz test a single function."""
    tester = FuzzTester()
    return tester.fuzz_function(func, input_generators, max_iterations)


if __name__ == "__main__":
    # Example usage - fuzz test some utility functions
    import src.utils.data_utils as data_utils

    manager = FuzzTestingManager()
    results = manager.fuzz_module_functions(data_utils, max_iterations_per_function=100)

    print(manager.generate_fuzz_test_report(results))

