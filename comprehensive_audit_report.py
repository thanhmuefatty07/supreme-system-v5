#!/usr/bin/env python3
"""
COMPREHENSIVE AUDIT REPORT - REAL-TIME VERIFICATION
Week 2 Achievements, Compatibility, Synchronization, Coverage & Functionality
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class ComprehensiveAudit:
    """Comprehensive audit system for all project aspects"""

    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()

    def log(self, message: str, level: str = "INFO"):
        """Log audit messages with timestamps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def check_git_synchronization(self):
        """Check git repository synchronization status"""
        self.log("🔍 Checking Git synchronization...")

        try:
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True, cwd='.')
            uncommitted_changes = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

            # Check ahead/behind status
            result = subprocess.run(['git', 'rev-list', '--count', '--left-right', 'HEAD...origin/main'],
                                  capture_output=True, text=True, cwd='.')
            ahead_behind = result.stdout.strip().split('\t') if result.stdout.strip() else ['0', '0']

            self.results['git_sync'] = {
                'uncommitted_changes': uncommitted_changes,
                'commits_ahead': int(ahead_behind[0]) if len(ahead_behind) > 0 else 0,
                'commits_behind': int(ahead_behind[1]) if len(ahead_behind) > 1 else 0,
                'synchronized': uncommitted_changes == 0 and ahead_behind == ['0', '0']
            }

            status = "✅ SYNCHRONIZED" if self.results['git_sync']['synchronized'] else "❌ OUT OF SYNC"
            self.log(f"Git Status: {status} (Ahead: {self.results['git_sync']['commits_ahead']}, Behind: {self.results['git_sync']['commits_behind']}, Uncommitted: {uncommitted_changes})")

        except Exception as e:
            self.log(f"Git check failed: {e}", "ERROR")
            self.results['git_sync'] = {'error': str(e)}

    def check_coverage_accuracy(self):
        """Check actual coverage metrics"""
        self.log("📊 Checking coverage accuracy...")

        try:
            # Read the latest coverage data
            coverage_file = 'coverage.json'
            if not os.path.exists(coverage_file):
                self.log("Coverage file not found", "ERROR")
                self.results['coverage'] = {'error': 'coverage.json not found'}
                return

            with open(coverage_file, 'r') as f:
                data = json.load(f)

            totals = data['totals']
            coverage_percent = totals['percent_covered']
            covered_lines = totals['covered_lines']
            missing_lines = totals['missing_lines']
            total_lines = covered_lines + missing_lines

            self.results['coverage'] = {
                'percentage': coverage_percent,
                'covered_lines': covered_lines,
                'missing_lines': missing_lines,
                'total_lines': total_lines,
                'files_covered': len(data['files'])
            }

            self.log(f"Coverage: {coverage_percent:.2f}% ({covered_lines}/{total_lines} lines)")

        except Exception as e:
            self.log(f"Coverage check failed: {e}", "ERROR")
            self.results['coverage'] = {'error': str(e)}

    def check_test_execution(self):
        """Check test execution status"""
        self.log("🧪 Checking test execution...")

        try:
            # Run a subset of tests to check basic functionality
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/unit/test_constants_gen.py',
                'tests/unit/test_enterprise_concurrency.py',
                '--tb=no', '-q', '--disable-warnings'
            ], capture_output=True, text=True, cwd='.')

            # Parse results
            output_lines = result.stdout.strip().split('\n')
            summary_line = [line for line in output_lines if 'passed' in line and 'failed' in line]

            if summary_line:
                summary = summary_line[-1]
                # Extract numbers from summary like "44 passed, 2 skipped, 10 warnings in 21.77s"
                parts = summary.split(',')
                passed = int(parts[0].split()[0]) if len(parts) > 0 else 0
                skipped = int(parts[1].split()[0]) if len(parts) > 1 else 0
                failed = 0  # Assume 0 if not mentioned

                self.results['tests'] = {
                    'passed': passed,
                    'failed': failed,
                    'skipped': skipped,
                    'total': passed + failed + skipped,
                    'exit_code': result.returncode
                }

                status = "✅ PASSING" if result.returncode == 0 else "❌ FAILING"
                self.log(f"Tests: {status} ({passed} passed, {failed} failed, {skipped} skipped)")

            else:
                self.results['tests'] = {'error': 'Could not parse test results'}
                self.log("Could not parse test results", "ERROR")

        except Exception as e:
            self.log(f"Test check failed: {e}", "ERROR")
            self.results['tests'] = {'error': str(e)}

    def check_file_compatibility(self):
        """Check file and import compatibility"""
        self.log("🔧 Checking file compatibility...")

        issues = []

        # Check for problematic mocking approaches
        problematic_files = []
        for root, dirs, files in os.walk('tests'):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if 'sys.modules[' in content and 'pybit' in content:
                                problematic_files.append(filepath)
                    except:
                        continue

        if problematic_files:
            issues.append(f"❌ {len(problematic_files)} files using problematic sys.modules mocking")
            for file in problematic_files[:3]:  # Show first 3
                issues.append(f"   - {file}")

        # Check for PyTorch import issues
        try:
            import torch
            torch_available = True
        except (ImportError, OSError):
            torch_available = False
            issues.append("❌ PyTorch not available - causing fatal import errors")

        # Check for other common import issues
        try:
            import pybit
            pybit_available = True
        except ImportError:
            pybit_available = False
            issues.append("❌ PyBit not available")

        self.results['compatibility'] = {
            'torch_available': torch_available,
            'pybit_available': pybit_available,
            'problematic_mocking': len(problematic_files),
            'issues': issues
        }

        if issues:
            for issue in issues:
                self.log(issue)
        else:
            self.log("✅ No major compatibility issues detected")

    def check_functionality(self):
        """Check core functionality"""
        self.log("⚙️ Checking core functionality...")

        functionality_status = {}

        # Check if main app can be imported
        try:
            from src.supreme_system_app import SupremeSystemApp, ApplicationConfig
            functionality_status['main_app_import'] = True
        except Exception as e:
            functionality_status['main_app_import'] = False
            functionality_status['main_app_error'] = str(e)

        # Check if core components can be imported
        core_modules = [
            'src.config.config',
            'src.utils.constants',
            'src.enterprise.concurrency',
            'src.enterprise.memory'
        ]

        for module in core_modules:
            try:
                __import__(module)
                functionality_status[f'{module}_import'] = True
            except Exception as e:
                functionality_status[f'{module}_import'] = False
                functionality_status[f'{module}_error'] = str(e)

        self.results['functionality'] = functionality_status

        working_modules = sum(1 for k, v in functionality_status.items() if k.endswith('_import') and v)
        total_modules = sum(1 for k in functionality_status.keys() if k.endswith('_import'))

        self.log(f"Functionality: {working_modules}/{total_modules} core modules working")

        for k, v in functionality_status.items():
            if k.endswith('_error') and v:
                self.log(f"   Error in {k.replace('_error', '')}: {v}", "ERROR")

    def generate_audit_report(self):
        """Generate comprehensive audit report"""
        self.log("📋 Generating comprehensive audit report...")

        report = f"""
{'='*80}
COMPREHENSIVE AUDIT REPORT - Week 2 Achievements
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Audit Duration: {(datetime.now() - self.start_time).total_seconds():.2f}s
{'='*80}

"""

        # Git Synchronization
        git = self.results.get('git_sync', {})
        if git.get('synchronized'):
            report += "✅ GIT SYNCHRONIZATION: Fully synchronized\n"
        else:
            report += f"❌ GIT SYNCHRONIZATION: Out of sync (Ahead: {git.get('commits_ahead', 0)}, Behind: {git.get('commits_behind', 0)}, Uncommitted: {git.get('uncommitted_changes', 0)})\n"

        # Coverage Status
        cov = self.results.get('coverage', {})
        if 'percentage' in cov:
            report += f"📊 COVERAGE STATUS: {cov['percentage']:.2f}% ({cov['covered_lines']}/{cov['total_lines']} lines, {cov['files_covered']} files)\n"
            if cov['percentage'] < 10:
                report += "   ⚠️  CRITICAL: Coverage extremely low - indicates major testing gaps\n"
        else:
            report += f"❌ COVERAGE STATUS: {cov.get('error', 'Unknown error')}\n"

        # Test Execution
        tests = self.results.get('tests', {})
        if 'passed' in tests:
            total_tests = tests['passed'] + tests['failed'] + tests['skipped']
            report += f"🧪 TEST EXECUTION: {tests['passed']}/{total_tests} passing ({tests['failed']} failed, {tests['skipped']} skipped)\n"
            if tests.get('exit_code', 1) != 0:
                report += "   ⚠️  Tests failing - functionality issues present\n"
        else:
            report += f"❌ TEST EXECUTION: {tests.get('error', 'Unknown error')}\n"

        # Compatibility Issues
        compat = self.results.get('compatibility', {})
        if compat.get('issues'):
            report += f"🔧 COMPATIBILITY: {len(compat['issues'])} issues detected\n"
            for issue in compat['issues'][:5]:  # Show first 5
                report += f"   {issue}\n"
        else:
            report += "✅ COMPATIBILITY: No major issues detected\n"

        # Functionality Status
        func = self.results.get('functionality', {})
        working = sum(1 for k, v in func.items() if k.endswith('_import') and v)
        total = sum(1 for k in func.keys() if k.endswith('_import'))
        report += f"⚙️  FUNCTIONALITY: {working}/{total} core modules functional\n"

        # Overall Assessment
        report += "\n" + "="*80 + "\n"
        report += "OVERALL ASSESSMENT:\n"

        critical_issues = 0

        if not git.get('synchronized'):
            critical_issues += 1
            report += "❌ CRITICAL: Repository out of synchronization\n"

        if cov.get('percentage', 0) < 10:
            critical_issues += 1
            report += "❌ CRITICAL: Coverage below 10% - major testing gaps\n"

        if tests.get('exit_code', 1) != 0:
            critical_issues += 1
            report += "❌ CRITICAL: Tests failing - broken functionality\n"

        if compat.get('issues'):
            critical_issues += 1
            report += "❌ CRITICAL: Compatibility issues detected\n"

        if working < total:
            critical_issues += 1
            report += "❌ CRITICAL: Core functionality broken\n"

        if critical_issues == 0:
            report += "✅ SYSTEM HEALTHY: All critical metrics passing\n"
        else:
            report += f"❌ SYSTEM UNHEALTHY: {critical_issues} critical issues requiring immediate attention\n"

        report += "="*80 + "\n"

        return report

    def run_full_audit(self):
        """Run complete audit suite"""
        self.log("🚀 Starting comprehensive audit...")

        self.check_git_synchronization()
        self.check_coverage_accuracy()
        self.check_test_execution()
        self.check_file_compatibility()
        self.check_functionality()

        report = self.generate_audit_report()

        # Save report
        with open('AUDIT_REPORT.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("\n" + report)
        self.log("✅ Audit complete - report saved to AUDIT_REPORT.md")


if __name__ == "__main__":
    audit = ComprehensiveAudit()
    audit.run_full_audit()

