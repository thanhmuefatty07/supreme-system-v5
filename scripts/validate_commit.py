#!/usr/bin/env python3
"""
Supreme System V5 - Commit Validation Protocol
Enforces comprehensive commit standards with data-backed reporting.
"""

import sys
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CommitValidationResult:
    """Result of commit validation."""
    valid: bool
    issues: List[str]
    recommendations: List[str]
    score: float  # 0.0 to 1.0

class CommitValidator:
    """Enforce Supreme System V5 commit standards."""

    def __init__(self, commit_hash: Optional[str] = None):
        self.commit_hash = commit_hash or 'HEAD'
        self.project_root = Path(__file__).parent.parent

    def get_commit_info(self) -> Dict[str, Any]:
        """Get comprehensive commit information."""
        try:
            # Get commit message
            result = subprocess.run(
                ['git', 'log', '--format=%H%n%s%n%b%n---', '-n', '1', self.commit_hash],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )

            lines = result.stdout.strip().split('\n')
            commit_hash = lines[0]
            subject = lines[1] if len(lines) > 1 else ""
            body = []

            for line in lines[2:]:
                if line == '---':
                    break
                body.append(line)

            # Get changed files
            result = subprocess.run(
                ['git', 'show', '--name-only', '--format=', self.commit_hash],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            changed_files = [f for f in result.stdout.strip().split('\n') if f]

            # Get diff stats
            result = subprocess.run(
                ['git', 'show', '--stat', '--format=', self.commit_hash],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            diff_stats = result.stdout.strip()

            return {
                'hash': commit_hash,
                'subject': subject,
                'body': '\n'.join(body),
                'changed_files': changed_files,
                'diff_stats': diff_stats,
                'full_message': f"{subject}\n{chr(10).join(body)}"
            }

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to get commit info: {e}")
            return {}

    def validate_commit_message(self, commit_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate commit message quality and completeness."""
        issues = []
        recommendations = []
        score = 0.0

        message = commit_info.get('full_message', '')
        subject = commit_info.get('subject', '')

        # Check subject line format
        if not subject:
            issues.append("Missing commit subject")
        elif len(subject) > 72:
            issues.append("Subject line too long (>72 characters)")
        elif not subject[0].isupper():
            issues.append("Subject should start with capital letter")
        else:
            score += 0.2

        # Check for roadmap milestone indicators
        roadmap_indicators = ['ROADMAP', 'MILESTONE', 'TASK', 'COMPLETE', 'IMPLEMENT']
        has_roadmap_reference = any(indicator in message.upper() for indicator in roadmap_indicators)
        if has_roadmap_reference:
            score += 0.2
        else:
            recommendations.append("Consider referencing roadmap task if applicable")

        # Check for technical content
        technical_indicators = [
            r'\b(file|line|function|class|method)\b',
            r'\b(added|removed|modified|refactored|optimized)\b',
            r'\b(CPU|RAM|latency|performance|benchmark)\b',
            r'\b(\d+\.\d+|\d+%)',  # Numbers/metrics
            r'\b(before|after|improved|reduced)\b'
        ]

        technical_score = 0
        for pattern in technical_indicators:
            if re.search(pattern, message, re.IGNORECASE):
                technical_score += 0.1

        if technical_score >= 0.3:
            score += 0.3
        elif technical_score >= 0.1:
            score += 0.1
            recommendations.append("Consider adding more technical details")
        else:
            recommendations.append("Add technical details (file references, metrics, changes)")

        # Check for test/performance data
        if any(keyword in message.upper() for keyword in ['TEST', 'BENCHMARK', 'PERFORMANCE', 'METRIC']):
            score += 0.2
        else:
            recommendations.append("Include test results or performance metrics")

        # Check message length
        if len(message) < 20:
            issues.append("Commit message too short")
        elif len(message) > 500:
            issues.append("Commit message too long (>500 characters)")
        else:
            score += 0.1

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'score': min(score, 1.0)
        }

    def validate_changed_files(self, commit_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate changed files and their relevance."""
        issues = []
        recommendations = []
        score = 0.0

        changed_files = commit_info.get('changed_files', [])
        diff_stats = commit_info.get('diff_stats', '')

        if not changed_files:
            issues.append("No files changed in commit")
            return {'valid': False, 'issues': issues, 'recommendations': recommendations, 'score': 0.0}

        # Check for test artifacts
        test_artifacts = [f for f in changed_files if 'test' in f.lower() or 'benchmark' in f.lower()]
        if test_artifacts:
            score += 0.3
        else:
            recommendations.append("Consider including test results or benchmark artifacts")

        # Check for documentation updates
        docs_files = [f for f in changed_files if 'readme' in f.lower() or 'doc' in f.lower()]
        if docs_files:
            score += 0.2

        # Check diff size (not too large, not empty)
        lines_changed = 0
        for line in diff_stats.split('\n'):
            if '|' in line and len(line.split('|')) >= 2:
                try:
                    lines_changed += int(line.split('|')[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        if lines_changed == 0:
            issues.append("No lines changed (empty commit)")
        elif lines_changed > 1000:
            recommendations.append("Large commit detected - consider splitting into smaller commits")
        else:
            score += 0.2

        # Check for critical files
        critical_files = ['requirements.txt', 'setup.py', 'pyproject.toml', 'Makefile']
        critical_changed = [f for f in changed_files if any(cf in f for cf in critical_files)]
        if critical_changed:
            score += 0.2

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'score': min(score, 1.0)
        }

    def validate_overall_quality(self, message_validation: Dict, files_validation: Dict) -> CommitValidationResult:
        """Calculate overall commit quality score."""
        message_score = message_validation['score']
        files_score = files_validation['score']

        overall_score = (message_score + files_score) / 2

        all_issues = message_validation['issues'] + files_validation['issues']
        all_recommendations = message_validation['recommendations'] + files_validation['recommendations']

        # Determine if commit meets minimum standards
        meets_minimum = (
            message_score >= 0.6 and  # Good message quality
            files_score >= 0.4 and    # Reasonable file changes
            len([i for i in all_issues if 'critical' in i.lower()]) == 0  # No critical issues
        )

        return CommitValidationResult(
            valid=meets_minimum,
            issues=all_issues,
            recommendations=all_recommendations,
            score=overall_score
        )

    def validate_commit(self) -> CommitValidationResult:
        """Run complete commit validation."""
        commit_info = self.get_commit_info()

        if not commit_info:
            return CommitValidationResult(
                valid=False,
                issues=["Could not retrieve commit information"],
                recommendations=["Ensure you're in a git repository and commit exists"],
                score=0.0
            )

        message_validation = self.validate_commit_message(commit_info)
        files_validation = self.validate_changed_files(commit_info)

        return self.validate_overall_quality(message_validation, files_validation)

    def print_validation_report(self, result: CommitValidationResult):
        """Print comprehensive validation report."""
        print("\n" + "="*80)
        print("📋 SUPREME SYSTEM V5 - COMMIT VALIDATION REPORT")
        print("="*80)
        print(f"Overall Score: {result.score:.2f}/1.0 ({'✅ PASS' if result.valid else '❌ FAIL'})")
        print()

        if result.issues:
            print("🚨 CRITICAL ISSUES:")
            for issue in result.issues:
                print(f"  ❌ {issue}")
            print()

        if result.recommendations:
            print("💡 RECOMMENDATIONS:")
            for rec in result.recommendations:
                print(f"  💡 {rec}")
            print()

        print("📊 QUALITY BREAKDOWN:")
        print(f"  Message Quality: Good commit messages include:")
        print(f"    • Technical details (file/line references)")
        print(f"    • Performance metrics (CPU/RAM/latency)")
        print(f"    • Before/after comparisons")
        print(f"    • Test results and artifacts")
        print(f"  File Changes: Meaningful, focused changes with:")
        print(f"    • Test coverage for new features")
        print(f"    • Documentation updates")
        print(f"    • Reasonable diff size (<1000 lines)")
        print()

        if result.valid:
            print("🎉 Commit meets Supreme System V5 standards!")
        else:
            print("⚠️  Commit needs improvement to meet standards.")
            print("   See CONTRIBUTING.md for detailed guidelines.")
        print("="*80)

def main():
    """Main validation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate Supreme System V5 commit quality')
    parser.add_argument('--commit', default='HEAD', help='Commit hash to validate')
    parser.add_argument('--strict', action='store_true', help='Enforce strict validation')

    args = parser.parse_args()

    validator = CommitValidator(args.commit)
    result = validator.validate_commit()
    validator.print_validation_report(result)

    # Exit with appropriate code
    if args.strict and not result.valid:
        sys.exit(1)
    elif not result.valid:
        sys.exit(1)  # Fail on validation issues

    sys.exit(0)

if __name__ == "__main__":
    main()
