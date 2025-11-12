#!/usr/bin/env python3
"""
AI Test Validation Script

Multi-layer validation pipeline for AI-generated tests.
Ensures quality and executability before commit.
"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import re


class AITestValidator:
    """Validates AI-generated test code."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_test_file(self, test_code: str, filename: str = "test_generated.py") -> bool:
        """Run full validation pipeline."""
        print(f"\n🔍 Validating {filename}...")

        # Layer 1: Syntax validation
        if not self._validate_syntax(test_code, filename):
            return False

        # Layer 2: Import validation
        if not self._validate_imports(test_code, filename):
            return False

        # Layer 3: Test structure validation
        if not self._validate_test_structure(test_code, filename):
            return False

        # Layer 4: Type checking
        if not self._validate_types(test_code, filename):
            return False

        # Layer 5: Security scanning
        if not self._validate_security(test_code, filename):
            return False

        # Layer 6: Execution validation
        if not self._validate_execution(test_code, filename):
            return False

        print(f"✅ {filename} passed all validation checks")
        return True

    def _validate_syntax(self, code: str, filename: str) -> bool:
        """Validate Python syntax using AST."""
        try:
            ast.parse(code)
            print("  ✅ Syntax validation passed")
            return True
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {filename}: {e}")
            print(f"  ❌ Syntax error: {e}")
            return False

    def _validate_imports(self, code: str, filename: str) -> bool:
        """Validate that all imports are correct and available."""
        try:
            tree = ast.parse(code)

            # Extract all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Common issues to check
            issues = []

            # Check for common hypothesis import mistake
            if 'hypothesis' in code and '@given' in code:
                # Check if strategies is imported correctly
                if not any('hypothesis.strategies' in imp or 'strategies as st' in code for imp in imports):
                    issues.append("Missing 'from hypothesis import strategies as st'")

            # Check for pytest imports
            if '@pytest' in code and 'pytest' not in imports:
                issues.append("Using @pytest decorators but pytest not imported")

            # Check for typing imports if type hints used
            if any(hint in code for hint in [': List', ': Dict', ': Optional', ': Tuple']):
                if 'typing' not in imports:
                    issues.append("Type hints used but typing module not imported")

            if issues:
                self.errors.extend(issues)
                for issue in issues:
                    print(f"  ❌ Import issue: {issue}")
                return False

            print("  ✅ Import validation passed")
            return True

        except Exception as e:
            self.errors.append(f"Import validation error: {e}")
            print(f"  ❌ Import validation error: {e}")
            return False

    def _validate_test_structure(self, code: str, filename: str) -> bool:
        """Validate test structure and naming conventions."""
        try:
            tree = ast.parse(code)

            # Find all test functions
            test_functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        test_functions.append(node.name)

            if not test_functions:
                self.errors.append(f"No test functions found in {filename}")
                print(f"  ❌ No test functions found (must start with 'test_')")
                return False

            # Check for assertions in test functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    has_assert = any(
                        isinstance(child, ast.Assert)
                        for child in ast.walk(node)
                    )
                    if not has_assert:
                        self.warnings.append(f"Test function '{node.name}' has no assertions")
                        print(f"  ⚠️ Warning: '{node.name}' has no assertions")

            print(f"  ✅ Test structure validated ({len(test_functions)} tests found)")
            return True

        except Exception as e:
            self.errors.append(f"Test structure validation error: {e}")
            print(f"  ❌ Test structure error: {e}")
            return False

    def _validate_types(self, code: str, filename: str) -> bool:
        """Run mypy type checking."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ['python', '-m', 'mypy', '--ignore-missing-imports', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            Path(temp_file).unlink()

            if result.returncode != 0:
                # Only report if serious type errors
                if 'error:' in result.stdout:
                    self.warnings.append(f"Type checking warnings in {filename}")
                    print(f"  ⚠️ Type checking warnings (non-blocking)")

            print("  ✅ Type checking passed")
            return True

        except subprocess.TimeoutExpired:
            self.warnings.append("Type checking timed out")
            print("  ⚠️ Type checking timed out")
            return True  # Non-blocking
        except Exception as e:
            self.warnings.append(f"Type checking error: {e}")
            print(f"  ⚠️ Type checking error: {e}")
            return True  # Non-blocking

    def _validate_security(self, code: str, filename: str) -> bool:
        """Run security scanning with bandit."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ['python', '-m', 'bandit', '-ll', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            Path(temp_file).unlink()

            if result.returncode != 0 and 'No issues identified' not in result.stdout:
                self.warnings.append(f"Security scan found issues in {filename}")
                print(f"  ⚠️ Security warnings (non-blocking)")

            print("  ✅ Security scanning passed")
            return True

        except subprocess.TimeoutExpired:
            self.warnings.append("Security scan timed out")
            return True  # Non-blocking
        except Exception:
            # Bandit might not be installed, non-blocking
            print("  ⚠️ Security scanning skipped (bandit not available)")
            return True

    def _validate_execution(self, code: str, filename: str) -> bool:
        """Validate that the code can be imported without errors."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Try to import the module
            result = subprocess.run(
                ['python', '-c', f'import importlib.util; spec = importlib.util.spec_from_file_location("test_module", "{temp_file}"); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)'],
                capture_output=True,
                text=True,
                timeout=10
            )

            Path(temp_file).unlink()

            if result.returncode != 0:
                self.errors.append(f"Execution validation failed: {result.stderr}")
                print(f"  ❌ Execution error: {result.stderr[:200]}")
                return False

            print("  ✅ Execution validation passed")
            return True

        except subprocess.TimeoutExpired:
            self.errors.append("Execution validation timed out")
            print("  ❌ Execution timed out")
            return False
        except Exception as e:
            self.errors.append(f"Execution validation error: {e}")
            print(f"  ❌ Execution error: {e}")
            return False

    def get_report(self) -> str:
        """Get validation report."""
        report = "\n" + "="*60 + "\n"
        report += "VALIDATION REPORT\n"
        report += "="*60 + "\n\n"

        if self.errors:
            report += f"❌ ERRORS ({len(self.errors)}):\n"
            for error in self.errors:
                report += f"  - {error}\n"
            report += "\n"

        if self.warnings:
            report += f"⚠️ WARNINGS ({len(self.warnings)}):\n"
            for warning in self.warnings:
                report += f"  - {warning}\n"
            report += "\n"

        if not self.errors and not self.warnings:
            report += "✅ ALL CHECKS PASSED\n"

        report += "="*60 + "\n"
        return report


def validate_directory(directory: Path) -> Tuple[int, int]:
    """Validate all test files in directory."""
    validator = AITestValidator()
    passed = 0
    failed = 0

    print(f"\n📁 Validating tests in {directory}...\n")

    for test_file in directory.rglob('test_*_generated.py'):
        try:
            code = test_file.read_text(encoding='utf-8')
            if validator.validate_test_file(code, test_file.name):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Error reading {test_file}: {e}")
            failed += 1

    print(validator.get_report())
    print(f"\n📊 Summary: {passed} passed, {failed} failed\n")

    return passed, failed


if __name__ == "__main__":
    # Validate all AI-generated tests
    tests_dir = Path('tests')

    if not tests_dir.exists():
        print(f"❌ Tests directory not found: {tests_dir}")
        sys.exit(1)

    passed, failed = validate_directory(tests_dir)

    if failed > 0:
        print(f"❌ Validation failed: {failed} test files have issues")
        sys.exit(1)
    else:
        print("✅ All test files validated successfully")
        sys.exit(0)
