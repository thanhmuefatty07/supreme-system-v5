#!/usr/bin/env python3
"""
Pre-commit Validation Script

Validates code quality, tests, and coverage before commits
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json
import re
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommitValidator:
    """Validate commits for quality and standards"""

    def __init__(self):
        self.issues = []
        self.passed_checks = []

    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        logger.info("Starting pre-commit validation")

        checks = [
            self.check_code_formatting,
            self.check_imports,
            self.check_tests_pass,
            self.check_coverage_threshold,
            self.check_security_issues,
            self.check_commit_message,
            self.check_file_sizes,
            self.check_todos_completed
        ]

        all_passed = True

        for check in checks:
            try:
                passed, message = check()
                if passed:
                    self.passed_checks.append(message)
                    logger.info(f"✅ {message}")
                else:
                    self.issues.append(message)
                    logger.error(f"❌ {message}")
                    all_passed = False
            except Exception as e:
                error_msg = f"Check failed with exception: {e}"
                self.issues.append(error_msg)
                logger.error(f"❌ {error_msg}")
                all_passed = False

        return all_passed

    def check_code_formatting(self) -> Tuple[bool, str]:
        """Check code formatting with black and isort"""
        try:
            # Check if black formatting is correct
            result = subprocess.run([
                sys.executable, "-m", "black", "--check", "--diff", "src/", "tests/"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return False, "Code formatting issues detected. Run: black src/ tests/"

            # Check import sorting
            result = subprocess.run([
                sys.executable, "-m", "isort", "--check-only", "--diff", "src/", "tests/"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return False, "Import sorting issues detected. Run: isort src/ tests/"

            return True, "Code formatting and imports are correct"

        except FileNotFoundError:
            return False, "black/isort not installed. Run: pip install black isort"

    def check_imports(self) -> Tuple[bool, str]:
        """Check for import issues"""
        try:
            # Try importing all modules
            import_errors = []

            for py_file in Path("src").rglob("*.py"):
                if "__init__.py" in str(py_file):
                    continue

                try:
                    module_path = str(py_file.relative_to(Path.cwd())).replace("/", ".").replace("\\", ".").replace(".py", "")
                    __import__(module_path)
                except Exception as e:
                    import_errors.append(f"{py_file}: {e}")

            if import_errors:
                return False, f"Import errors detected: {len(import_errors)} files"

            return True, "All modules import successfully"

        except Exception as e:
            return False, f"Import check failed: {e}"

    def check_tests_pass(self) -> Tuple[bool, str]:
        """Run tests and check they pass"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/",
                "--tb=no", "-q", "--maxfail=5"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                failed_count = len(re.findall(r'FAILED|ERROR', result.stdout + result.stderr))
                return False, f"Tests failing: {failed_count} failures detected"

            return True, "All tests pass"

        except subprocess.TimeoutExpired:
            return False, "Tests timed out (120s limit)"

    def check_coverage_threshold(self) -> Tuple[bool, str]:
        """Check coverage meets minimum threshold"""
        try:
            coverage_target = int(os.getenv('COVERAGE_TARGET', 40))

            result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/",
                "--cov=src", "--cov-report=json", "--cov-fail-under=0"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return False, "Coverage analysis failed"

            # Parse coverage from output
            coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', result.stdout)
            if coverage_match:
                coverage = int(coverage_match.group(1))
                if coverage < coverage_target:
                    return False, f"Coverage {coverage}% below target {coverage_target}%"

                return True, f"Coverage {coverage}% meets target {coverage_target}%"

            return False, "Could not parse coverage results"

        except Exception as e:
            return False, f"Coverage check failed: {e}"

    def check_security_issues(self) -> Tuple[bool, str]:
        """Check for security issues with bandit"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", "src/",
                "-f", "json", "-o", "/dev/null"
            ], capture_output=True, text=True)

            # Parse JSON output for high/critical issues
            if result.returncode > 0:
                # Bandit returns non-zero for issues found
                return False, "Security issues detected by bandit"

            return True, "No security issues detected"

        except FileNotFoundError:
            return False, "bandit not installed. Run: pip install bandit"
        except Exception as e:
            return False, f"Security check failed: {e}"

    def check_commit_message(self) -> Tuple[bool, str]:
        """Check commit message format"""
        try:
            # Get the latest commit message
            result = subprocess.run([
                "git", "log", "--oneline", "-1"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return True, "No git repository or commits (skipping message check)"

            message = result.stdout.strip()

            # Check message format (basic validation)
            if len(message.split()) < 3:
                return False, "Commit message too short (minimum 3 words)"

            if not re.match(r'^[A-Z]', message):
                return False, "Commit message should start with capital letter"

            return True, "Commit message format is valid"

        except Exception as e:
            return True, f"Commit message check skipped: {e}"

    def check_file_sizes(self) -> Tuple[bool, str]:
        """Check for unusually large files"""
        max_size = 1024 * 1024  # 1MB

        large_files = []
        for file_path in Path("src").rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > max_size:
                large_files.append(str(file_path))

        if large_files:
            return False, f"Large files detected: {len(large_files)} files > 1MB"

        return True, "All files are within size limits"

    def check_todos_completed(self) -> Tuple[bool, str]:
        """Check for unresolved TODO comments"""
        todo_patterns = [
            r'# TODO:',
            r'# FIXME:',
            r'# XXX:',
            r'# HACK:'
        ]

        urgent_todos = []

        for py_file in Path("src").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        for pattern in todo_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                urgent_todos.append(f"{py_file}:{i}")
            except Exception:
                continue

        if len(urgent_todos) > 10:  # Allow some TODOs but not too many
            return False, f"Too many unresolved TODOs: {len(urgent_todos)} items"

        return True, f"TODO count acceptable ({len(urgent_todos)} items)"

    def generate_report(self) -> str:
        """Generate validation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        status = "PASSED" if not self.issues else "FAILED"

        report = f"""# Pre-commit Validation Report

**Status:** {status}  
**Timestamp:** {timestamp}  
**Checks Passed:** {len(self.passed_checks)}  
**Issues Found:** {len(self.issues)}

"""

        if self.passed_checks:
            report += "## ✅ Passed Checks\n\n"
            for check in self.passed_checks:
                report += f"- {check}\n"

        if self.issues:
            report += "\n## ❌ Issues Found\n\n"
            for issue in self.issues:
                report += f"- {issue}\n"

        report += "\n## Next Steps\n\n"
        if self.issues:
            report += "- Fix the issues listed above\n"
            report += "- Re-run validation: `python scripts/commit_validator.py`\n"
            report += "- Commit again after fixes\n"
        else:
            report += "- All checks passed! Ready to commit.\n"

        return report

    def save_report(self, report: str):
        """Save validation report"""
        report_path = Path("pre_commit_validation_report.md")
        report_path.write_text(report, encoding='utf-8')
        return report_path


def main():
    """Main execution"""
    print("=" * 60)
    print("  SUPREME SYSTEM V5 - PRE-COMMIT VALIDATOR")
    print("=" * 60)
    print()

    validator = CommitValidator()

    print("🔍 Running validation checks...")
    print("-" * 60)

    passed = validator.run_all_checks()

    print()
    print("=" * 60)

    if passed:
        print("  ✅ ALL CHECKS PASSED")
        print("  Ready to commit!")
    else:
        print("  ❌ VALIDATION FAILED")
        print(f"  Issues found: {len(validator.issues)}")
    print("=" * 60)

    # Generate and save report
    report = validator.generate_report()
    report_path = validator.save_report(report)

    print(f"\n📋 Detailed report: {report_path}")

    if validator.issues:
        print("\n🔧 Issues to fix:")
        for i, issue in enumerate(validator.issues[:5], 1):
            print(f"   {i}. {issue}")
        if len(validator.issues) > 5:
            print(f"   ... and {len(validator.issues) - 5} more")

    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())



