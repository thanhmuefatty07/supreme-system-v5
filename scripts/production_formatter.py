#!/usr/bin/env python3
"""
Production Formatter - Automated Code Quality Enforcement
Ultra-professional formatting system for production readiness
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class FormattingResult:
    """Result of formatting operation"""
    tool: str
    success: bool
    files_processed: int
    files_changed: int
    execution_time: float
    errors: List[str]
    warnings: List[str]

class ProductionFormatter:
    """Professional-grade code formatting system"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.results: List[FormattingResult] = []
        self.target_files = self._identify_target_files()
    
    def _identify_target_files(self) -> List[str]:
        """Identify files that need formatting"""
        target_patterns = [
            "python/**/*.py",
            "scripts/**/*.py", 
            "tests/**/*.py",
            "*.py"  # Root level Python files
        ]
        
        files = []
        for pattern in target_patterns:
            files.extend(self.project_root.glob(pattern))
        
        # Exclude src/ directory legacy Python files
        files = [f for f in files if not str(f).startswith(str(self.project_root / "src"))]
        
        return [str(f.relative_to(self.project_root)) for f in files]
    
    def execute_black_formatting(self) -> FormattingResult:
        """Execute Black formatting with production settings"""
        print("🖨 Executing Black formatting...")
        start_time = time.time()
        
        cmd = [
            sys.executable, "-m", "black",
            "--line-length", "88",
            "--target-version", "py311",
            "--exclude", "src/.*\.py",
            "--verbose",
            "."
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # Parse output for statistics
            stdout_lines = result.stdout.split("\n")
            files_changed = len([line for line in stdout_lines if "reformatted" in line])
            files_processed = len(self.target_files)
            
            formatting_result = FormattingResult(
                tool="black",
                success=result.returncode == 0,
                files_processed=files_processed,
                files_changed=files_changed,
                execution_time=execution_time,
                errors=[result.stderr] if result.stderr else [],
                warnings=[]
            )
            
            if formatting_result.success:
                print(f"✅ Black formatting completed: {files_changed} files reformatted")
            else:
                print(f"❌ Black formatting failed: {result.stderr}")
            
            self.results.append(formatting_result)
            return formatting_result
            
        except Exception as e:
            error_result = FormattingResult(
                tool="black",
                success=False,
                files_processed=0,
                files_changed=0,
                execution_time=time.time() - start_time,
                errors=[str(e)],
                warnings=[]
            )
            self.results.append(error_result)
            return error_result
    
    def execute_ruff_linting(self) -> FormattingResult:
        """Execute Ruff linting and auto-fix"""
        print("🔧 Executing Ruff linting and auto-fix...")
        start_time = time.time()
        
        # First, try to auto-fix issues
        fix_cmd = [
            sys.executable, "-m", "ruff", "check",
            "--fix", "--unsafe-fixes",
            "--exclude", "src/*.py",
            "python/", "scripts/", "tests/", "*.py"
        ]
        
        try:
            fix_result = subprocess.run(
                fix_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Then check final status
            check_cmd = [
                sys.executable, "-m", "ruff", "check",
                "--exclude", "src/*.py",
                "python/", "scripts/", "tests/", "*.py"
            ]
            
            check_result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            ruff_result = FormattingResult(
                tool="ruff",
                success=check_result.returncode == 0,
                files_processed=len(self.target_files),
                files_changed=len(fix_result.stdout.split("\n")) if fix_result.stdout else 0,
                execution_time=execution_time,
                errors=[check_result.stderr] if check_result.stderr else [],
                warnings=[fix_result.stdout] if fix_result.stdout else []
            )
            
            if ruff_result.success:
                print("✅ Ruff linting completed successfully")
            else:
                print(f"❌ Ruff linting issues remain: {check_result.stdout}")
            
            self.results.append(ruff_result)
            return ruff_result
            
        except Exception as e:
            error_result = FormattingResult(
                tool="ruff",
                success=False,
                files_processed=0,
                files_changed=0,
                execution_time=time.time() - start_time,
                errors=[str(e)],
                warnings=[]
            )
            self.results.append(error_result)
            return error_result
    
    def execute_isort_imports(self) -> FormattingResult:
        """Execute isort for import organization"""
        print("📄 Executing isort for import organization...")
        start_time = time.time()
        
        cmd = [
            sys.executable, "-m", "isort",
            "--profile", "black",
            "--line-length", "88",
            "--skip", "src",
            "python/", "scripts/", "tests/", "*.py"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            isort_result = FormattingResult(
                tool="isort",
                success=result.returncode == 0,
                files_processed=len(self.target_files),
                files_changed=result.stdout.count("Fixing") if result.stdout else 0,
                execution_time=execution_time,
                errors=[result.stderr] if result.stderr else [],
                warnings=[]
            )
            
            if isort_result.success:
                print("✅ Import sorting completed successfully")
            else:
                print(f"❌ Import sorting failed: {result.stderr}")
            
            self.results.append(isort_result)
            return isort_result
            
        except Exception as e:
            error_result = FormattingResult(
                tool="isort",
                success=False,
                files_processed=0,
                files_changed=0,
                execution_time=time.time() - start_time,
                errors=[str(e)],
                warnings=[]
            )
            self.results.append(error_result)
            return error_result
    
    def validate_syntax(self) -> FormattingResult:
        """Validate Python syntax for all target files"""
        print("🔍 Validating Python syntax...")
        start_time = time.time()
        
        errors = []
        files_processed = 0
        
        for file_path in self.target_files:
            if file_path.endswith('.py'):
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "py_compile", file_path
                    ], capture_output=True, text=True, cwd=self.project_root)
                    
                    files_processed += 1
                    
                    if result.returncode != 0:
                        errors.append(f"{file_path}: {result.stderr}")
                        
                except Exception as e:
                    errors.append(f"{file_path}: {str(e)}")
        
        execution_time = time.time() - start_time
        
        syntax_result = FormattingResult(
            tool="py_compile",
            success=len(errors) == 0,
            files_processed=files_processed,
            files_changed=0,
            execution_time=execution_time,
            errors=errors,
            warnings=[]
        )
        
        if syntax_result.success:
            print(f"✅ Syntax validation passed: {files_processed} files")
        else:
            print(f"❌ Syntax validation failed: {len(errors)} errors")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    {error}")
        
        self.results.append(syntax_result)
        return syntax_result
    
    def execute_full_formatting(self) -> bool:
        """Execute complete formatting pipeline"""
        print("🚀 PRODUCTION FORMATTER - COMPREHENSIVE EXECUTION")
        print("=" * 70)
        print(f"Target files: {len(self.target_files)} files")
        print("Scope: python/, scripts/, tests/, *.py (excluding src/)")
        print("=" * 70)
        
        # Execute formatting steps
        black_result = self.execute_black_formatting()
        ruff_result = self.execute_ruff_linting()
        isort_result = self.execute_isort_imports()
        syntax_result = self.validate_syntax()
        
        # Generate comprehensive report
        self.generate_report()
        
        # Determine overall success
        all_success = all([
            black_result.success,
            ruff_result.success, 
            isort_result.success,
            syntax_result.success
        ])
        
        return all_success
    
    def generate_report(self):
        """Generate comprehensive formatting report"""
        print("\n" + "=" * 70)
        print("📊 PRODUCTION FORMATTING REPORT")
        print("=" * 70)
        
        total_execution_time = sum(r.execution_time for r in self.results)
        total_files_changed = sum(r.files_changed for r in self.results)
        
        print(f"Total execution time: {total_execution_time:.2f}s")
        print(f"Total files changed: {total_files_changed}")
        print(f"Target files processed: {len(self.target_files)}")
        
        print("\n📈 Tool Performance:")
        for result in self.results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"  {result.tool:12} {status:12} {result.execution_time:6.2f}s {result.files_changed:3d} changes")
        
        # Show errors if any
        all_errors = []
        for result in self.results:
            all_errors.extend(result.errors)
        
        if all_errors:
            print("\n❌ ERRORS ENCOUNTERED:")
            for i, error in enumerate(all_errors[:10], 1):  # Show first 10 errors
                print(f"  {i}. {error}")
            if len(all_errors) > 10:
                print(f"  ... and {len(all_errors) - 10} more errors")
        
        # Success summary
        successful_tools = sum(1 for r in self.results if r.success)
        total_tools = len(self.results)
        
        print(f"\n🎯 OVERALL SUCCESS RATE: {successful_tools}/{total_tools} tools successful")
        
        if successful_tools == total_tools:
            print("🏆 ALL FORMATTING TOOLS COMPLETED SUCCESSFULLY!")
            print("✅ Production code quality standards achieved")
            print("🚀 System ready for immediate deployment")
        else:
            print(f"⚠️ PARTIAL SUCCESS: {total_tools - successful_tools} tools need attention")
            print("🔧 Manual intervention may be required")

def main():
    """Execute production formatting with comprehensive reporting"""
    formatter = ProductionFormatter()
    
    print("🏳 PRODUCTION CODE FORMATTING INITIATED")
    print("Objective: Achieve enterprise-grade code quality")
    print("Standard: Zero tolerance for formatting deviations")
    print("\n")
    
    # Execute full formatting pipeline
    success = formatter.execute_full_formatting()
    
    # Final status and next steps
    print("\n" + "=" * 70)
    if success:
        print("🎉 PRODUCTION FORMATTING COMPLETED SUCCESSFULLY!")
        print("\n🚀 IMMEDIATE NEXT STEPS:")
        print("  1. git add .")
        print("  2. git commit -m 'Production formatting: Black + Ruff + isort applied'")
        print("  3. git push origin main")
        print("  4. Monitor CI for green status")
        print("  5. Execute production deployment")
        sys.exit(0)
    else:
        print("⚠️ PRODUCTION FORMATTING REQUIRES ATTENTION")
        print("\n🔧 IMMEDIATE NEXT STEPS:")
        print("  1. Review errors reported above")
        print("  2. Fix critical issues manually")
        print("  3. Re-run this script")
        print("  4. Escalate to senior engineering team if needed")
        sys.exit(1)

if __name__ == "__main__":
    main()
