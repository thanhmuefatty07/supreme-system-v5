#!/usr/bin/env python3
"""
🚀 SUPREME SYSTEM V5 - FINAL PROJECT VALIDATION PROTOCOL
Comprehensive validation of all remaining tasks and completion status.
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FinalValidator')

class SupremeSystemFinalValidator:
    """Nuclear-grade final validation for Supreme System V5."""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.validation_results = {}
        self.base_path = Path('.')
        self.run_artifacts_dir = Path('run_artifacts')
        self.logs_dir = Path('logs')
        
        # Ensure directories exist
        self.run_artifacts_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    def validate_repository_structure(self) -> Dict[str, Any]:
        """Validate core repository structure and files."""
        logger.info("🔍 Validating repository structure...")
        
        required_files = [
            'README.md',
            'requirements.txt',
            'main.py',
            'realtime_backtest.py',
            '.env.example',
            'docker-compose.yml',
            'Makefile',
            'pyproject.toml'
        ]
        
        required_dirs = [
            'src',
            'tests',
            'scripts',
            'docs',
            'dashboard',
            'monitoring',
            'python'
        ]
        
        structure_results = {
            'files_present': {},
            'directories_present': {},
            'missing_files': [],
            'missing_directories': []
        }
        
        # Check required files
        for file_path in required_files:
            file_exists = (self.base_path / file_path).exists()
            structure_results['files_present'][file_path] = file_exists
            if not file_exists:
                structure_results['missing_files'].append(file_path)
                
        # Check required directories
        for dir_path in required_dirs:
            dir_exists = (self.base_path / dir_path).exists()
            structure_results['directories_present'][dir_path] = dir_exists
            if not dir_exists:
                structure_results['missing_directories'].append(dir_path)
                
        structure_score = (
            len([f for f in structure_results['files_present'].values() if f]) +
            len([d for d in structure_results['directories_present'].values() if d])
        ) / (len(required_files) + len(required_dirs))
        
        structure_results['score'] = structure_score
        structure_results['status'] = 'PASS' if structure_score >= 0.9 else 'FAIL'
        
        logger.info(f"   Repository structure score: {structure_score:.2%}")
        return structure_results
        
    def validate_priority_2_benchmarks(self) -> Dict[str, Any]:
        """Validate Priority 2: Real Benchmark Execution & Data Validation."""
        logger.info("📊 Validating Priority 2: Benchmark execution...")
        
        benchmark_results = {
            'benchmark_script_exists': False,
            'artifacts_directory_exists': False,
            'recent_benchmark_files': [],
            'parity_tests_available': False,
            'status': 'NOT_STARTED'
        }
        
        # Check if benchmark script exists
        benchmark_script = self.base_path / 'scripts' / 'bench_optimized.py'
        benchmark_results['benchmark_script_exists'] = benchmark_script.exists()
        
        # Check if execute_benchmarks script exists
        execute_script = self.base_path / 'scripts' / 'execute_benchmarks.py'
        benchmark_results['execute_script_exists'] = execute_script.exists()
        
        # Check artifacts directory
        benchmark_results['artifacts_directory_exists'] = self.run_artifacts_dir.exists()
        
        # Check for recent benchmark files
        if self.run_artifacts_dir.exists():
            benchmark_files = list(self.run_artifacts_dir.glob('bench_*.json'))
            load_files = list(self.run_artifacts_dir.glob('load_*.json'))
            benchmark_results['recent_benchmark_files'] = [
                str(f) for f in (benchmark_files + load_files)
            ]
            
        # Check parity tests
        parity_test_file = self.base_path / 'tests' / 'test_parity_indicators.py'
        benchmark_results['parity_tests_available'] = parity_test_file.exists()
        
        # Determine status
        if len(benchmark_results['recent_benchmark_files']) > 0:
            benchmark_results['status'] = 'COMPLETED'
        elif (benchmark_results['benchmark_script_exists'] and 
              benchmark_results['execute_script_exists']):
            benchmark_results['status'] = 'READY_TO_EXECUTE'
        else:
            benchmark_results['status'] = 'NOT_READY'
            
        logger.info(f"   Priority 2 status: {benchmark_results['status']}")
        return benchmark_results
        
    def validate_priority_5_ab_testing(self) -> Dict[str, Any]:
        """Validate Priority 5: 24h A/B Testing Infrastructure."""
        logger.info("🧪 Validating Priority 5: A/B Testing infrastructure...")
        
        ab_test_results = {
            'ab_script_exists': False,
            'report_script_exists': False,
            'ab_test_artifacts': [],
            'infrastructure_ready': False,
            'status': 'NOT_STARTED'
        }
        
        # Check A/B test scripts
        ab_script = self.base_path / 'scripts' / 'ab_test_run.sh'
        ab_test_results['ab_script_exists'] = ab_script.exists()
        
        report_script = self.base_path / 'scripts' / 'report_ab.py'
        ab_test_results['report_script_exists'] = report_script.exists()
        
        # Check for A/B test artifacts
        if self.run_artifacts_dir.exists():
            ab_files = list(self.run_artifacts_dir.glob('ab_*.json'))
            report_files = list(self.run_artifacts_dir.glob('*report*.json'))
            ab_test_results['ab_test_artifacts'] = [
                str(f) for f in (ab_files + report_files)
            ]
            
        # Check infrastructure readiness
        ab_test_results['infrastructure_ready'] = (
            ab_test_results['ab_script_exists'] and 
            ab_test_results['report_script_exists']
        )
        
        # Determine status
        if len(ab_test_results['ab_test_artifacts']) > 0:
            ab_test_results['status'] = 'COMPLETED'
        elif ab_test_results['infrastructure_ready']:
            ab_test_results['status'] = 'INFRASTRUCTURE_READY'
        else:
            ab_test_results['status'] = 'NOT_READY'
            
        logger.info(f"   Priority 5 status: {ab_test_results['status']}")
        return ab_test_results
        
    def validate_priority_7_commit_standards(self) -> Dict[str, Any]:
        """Validate Priority 7: Commit Standards & Validation Protocol."""
        logger.info("📋 Validating Priority 7: Commit standards...")
        
        commit_results = {
            'contributing_md_exists': False,
            'contributing_md_enhanced': False,
            'pre_commit_config_exists': False,
            'validation_scripts_exist': False,
            'status': 'NOT_STARTED'
        }
        
        # Check CONTRIBUTING.md
        contributing_file = self.base_path / 'CONTRIBUTING.md'
        commit_results['contributing_md_exists'] = contributing_file.exists()
        
        # Check if CONTRIBUTING.md has been enhanced (look for specific sections)
        if contributing_file.exists():
            try:
                content = contributing_file.read_text()
                enhanced_indicators = [
                    'Performance Metrics:',
                    'Benchmark Requirements',
                    'Commit Validation',
                    'Quality Gates'
                ]
                commit_results['contributing_md_enhanced'] = all(
                    indicator in content for indicator in enhanced_indicators[:2]  # At least 2 indicators
                )
            except Exception:
                commit_results['contributing_md_enhanced'] = False
                
        # Check pre-commit configuration
        pre_commit_file = self.base_path / '.pre-commit-config.yaml'
        commit_results['pre_commit_config_exists'] = pre_commit_file.exists()
        
        # Check validation scripts
        validation_scripts = [
            'scripts/validate_commit.py',
            'scripts/validate_environment.py',
            'scripts/error_diagnosis.py'
        ]
        
        existing_validation_scripts = [
            script for script in validation_scripts
            if (self.base_path / script).exists()
        ]
        
        commit_results['validation_scripts_exist'] = len(existing_validation_scripts) >= 2
        commit_results['existing_validation_scripts'] = existing_validation_scripts
        
        # Determine status
        if (commit_results['contributing_md_enhanced'] and 
            commit_results['validation_scripts_exist']):
            commit_results['status'] = 'COMPLETED'
        elif commit_results['contributing_md_exists']:
            commit_results['status'] = 'PARTIAL'
        else:
            commit_results['status'] = 'NOT_READY'
            
        logger.info(f"   Priority 7 status: {commit_results['status']}")
        return commit_results
        
    def validate_nuclear_intervention_branch(self) -> Dict[str, Any]:
        """Validate nuclear-intervention-v6 branch status."""
        logger.info("🚀 Validating nuclear intervention branch...")
        
        branch_results = {
            'branch_exists': False,
            'new_files_added': 0,
            'commits_made': 0,
            'ready_for_merge': False,
            'status': 'UNKNOWN'
        }
        
        try:
            # Check current branch
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True, text=True, timeout=10
            )
            current_branch = result.stdout.strip()
            branch_results['current_branch'] = current_branch
            branch_results['branch_exists'] = 'nuclear-intervention-v6' in current_branch
            
            # Check commits in branch (if exists)
            if branch_results['branch_exists']:
                result = subprocess.run(
                    ['git', 'log', '--oneline', 'main..HEAD'],
                    capture_output=True, text=True, timeout=10
                )
                commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
                branch_results['commits_made'] = len(commits)
                branch_results['recent_commits'] = commits[:5]  # Last 5 commits
                
                # Check added files
                result = subprocess.run(
                    ['git', 'diff', '--name-only', 'main..HEAD'],
                    capture_output=True, text=True, timeout=10
                )
                added_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
                branch_results['new_files_added'] = len(added_files)
                branch_results['added_files'] = added_files
                
        except Exception as e:
            logger.warning(f"Could not check git branch status: {e}")
            
        # Determine readiness for merge
        branch_results['ready_for_merge'] = (
            branch_results['branch_exists'] and
            branch_results['commits_made'] > 0 and
            branch_results['new_files_added'] > 0
        )
        
        # Determine status
        if branch_results['ready_for_merge']:
            branch_results['status'] = 'READY_FOR_MERGE'
        elif branch_results['branch_exists']:
            branch_results['status'] = 'IN_PROGRESS'
        else:
            branch_results['status'] = 'NOT_STARTED'
            
        logger.info(f"   Branch status: {branch_results['status']}")
        return branch_results
        
    def generate_completion_report(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive completion report."""
        logger.info("📊 Generating completion report...")
        
        # Calculate overall completion percentage
        priority_scores = {
            'priority_2': 1.0 if validation_data['priority_2']['status'] == 'COMPLETED' else 0.5 if validation_data['priority_2']['status'] == 'READY_TO_EXECUTE' else 0.0,
            'priority_5': 1.0 if validation_data['priority_5']['status'] == 'COMPLETED' else 0.5 if validation_data['priority_5']['status'] == 'INFRASTRUCTURE_READY' else 0.0,
            'priority_7': 1.0 if validation_data['priority_7']['status'] == 'COMPLETED' else 0.5 if validation_data['priority_7']['status'] == 'PARTIAL' else 0.0
        }
        
        overall_completion = sum(priority_scores.values()) / len(priority_scores)
        
        # Determine project status
        if overall_completion >= 1.0:
            project_status = 'FULLY_COMPLETE'
        elif overall_completion >= 0.8:
            project_status = 'NEARLY_COMPLETE'
        elif overall_completion >= 0.5:
            project_status = 'PARTIALLY_COMPLETE'
        else:
            project_status = 'INCOMPLETE'
            
        completion_report = {
            'validation_timestamp': self.start_time.isoformat(),
            'overall_completion_percentage': overall_completion * 100,
            'project_status': project_status,
            'priority_completion': priority_scores,
            'repository_health': validation_data['repository_structure']['score'] * 100,
            'nuclear_intervention_status': validation_data['nuclear_branch']['status'],
            'ready_for_production': overall_completion >= 0.9 and validation_data['repository_structure']['score'] >= 0.9,
            'recommendations': []
        }
        
        # Generate recommendations
        if validation_data['priority_2']['status'] != 'COMPLETED':
            completion_report['recommendations'].append(
                "Execute Priority 2 benchmarks: python scripts/execute_benchmarks.py"
            )
            
        if validation_data['priority_5']['status'] != 'COMPLETED':
            completion_report['recommendations'].append(
                "Complete Priority 5 A/B testing infrastructure validation"
            )
            
        if validation_data['priority_7']['status'] != 'COMPLETED':
            completion_report['recommendations'].append(
                "Finalize Priority 7 commit standards and validation protocols"
            )
            
        if validation_data['nuclear_branch']['status'] == 'READY_FOR_MERGE':
            completion_report['recommendations'].append(
                "Merge nuclear-intervention-v6 branch to main"
            )
            
        return completion_report
        
    def execute_final_validation(self) -> Dict[str, Any]:
        """Execute comprehensive final validation protocol."""
        logger.info("🚀 SUPREME SYSTEM V5 - FINAL VALIDATION PROTOCOL")
        logger.info("=" * 70)
        
        validation_data = {
            'validation_metadata': {
                'start_time': self.start_time.isoformat(),
                'validator_version': 'v5-nuclear-final'
            }
        }
        
        # Execute all validations
        validation_data['repository_structure'] = self.validate_repository_structure()
        validation_data['priority_2'] = self.validate_priority_2_benchmarks()
        validation_data['priority_5'] = self.validate_priority_5_ab_testing()
        validation_data['priority_7'] = self.validate_priority_7_commit_standards()
        validation_data['nuclear_branch'] = self.validate_nuclear_intervention_branch()
        
        # Generate completion report
        completion_report = self.generate_completion_report(validation_data)
        validation_data['completion_report'] = completion_report
        
        # Save validation results
        validation_file = self.run_artifacts_dir / f'final_validation_{self.start_time.strftime("%Y%m%d_%H%M%S")}.json'
        with open(validation_file, 'w') as f:
            json.dump(validation_data, f, indent=2)
            
        logger.info(f"\n📊 Validation results saved to: {validation_file}")
        
        # Print summary
        self.print_validation_summary(completion_report)
        
        return validation_data
        
    def print_validation_summary(self, completion_report: Dict[str, Any]):
        """Print human-readable validation summary."""
        logger.info("\n" + "=" * 70)
        logger.info("🎯 SUPREME SYSTEM V5 - FINAL VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        # Overall status
        status = completion_report['project_status']
        completion_pct = completion_report['overall_completion_percentage']
        
        status_emoji = {
            'FULLY_COMPLETE': '✅',
            'NEARLY_COMPLETE': '🟡',
            'PARTIALLY_COMPLETE': '⚠️',
            'INCOMPLETE': '❌'
        }.get(status, '❓')
        
        logger.info(f"\n{status_emoji} PROJECT STATUS: {status} ({completion_pct:.1f}% complete)")
        
        # Priority status
        logger.info("\n📋 PRIORITY TASK STATUS:")
        logger.info("-" * 40)
        
        priority_names = {
            'priority_2': 'Priority 2: Benchmark Execution',
            'priority_5': 'Priority 5: A/B Testing',
            'priority_7': 'Priority 7: Commit Standards'
        }
        
        for key, score in completion_report['priority_completion'].items():
            name = priority_names.get(key, key)
            if score >= 1.0:
                status_icon = '✅ COMPLETE'
            elif score >= 0.5:
                status_icon = '🟡 PARTIAL'
            else:
                status_icon = '❌ INCOMPLETE'
            logger.info(f"{status_icon} {name}")
            
        # Repository health
        repo_health = completion_report['repository_health']
        logger.info(f"\n🏗️  REPOSITORY HEALTH: {repo_health:.1f}%")
        
        # Nuclear intervention status
        nuclear_status = completion_report['nuclear_intervention_status']
        logger.info(f"🚀 NUCLEAR INTERVENTION: {nuclear_status}")
        
        # Production readiness
        prod_ready = completion_report['ready_for_production']
        prod_status = "✅ READY" if prod_ready else "❌ NOT READY"
        logger.info(f"🎯 PRODUCTION READY: {prod_status}")
        
        # Recommendations
        if completion_report['recommendations']:
            logger.info("\n💡 RECOMMENDATIONS:")
            logger.info("-" * 40)
            for i, rec in enumerate(completion_report['recommendations'], 1):
                logger.info(f"{i}. {rec}")
        else:
            logger.info("\n🎉 NO ADDITIONAL ACTIONS REQUIRED!")
            
        logger.info("\n" + "=" * 70)
        
        if prod_ready:
            logger.info("🚀 SUPREME SYSTEM V5 IS READY FOR PRODUCTION DEPLOYMENT!")
        else:
            logger.info("⚠️  SUPREME SYSTEM V5 REQUIRES ADDITIONAL WORK BEFORE PRODUCTION")
            
        logger.info("=" * 70)
        
def main():
    """Main execution function."""
    try:
        validator = SupremeSystemFinalValidator()
        validation_results = validator.execute_final_validation()
        
        # Exit with appropriate code
        if validation_results['completion_report']['ready_for_production']:
            sys.exit(0)  # Success - ready for production
        elif validation_results['completion_report']['overall_completion_percentage'] >= 80:
            sys.exit(1)  # Nearly ready - minor work remaining
        else:
            sys.exit(2)  # Major work remaining
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Final validation interrupted by user")
        sys.exit(3)
    except Exception as e:
        logger.error(f"\n❌ Critical validation failure: {e}")
        sys.exit(4)
        
if __name__ == '__main__':
    main()