#!/usr/bin/env python3
"""
Error Diagnosis and Resilience Script for Supreme System V5
Comprehensive error handling with auto-diagnosis and recovery suggestions
"""

import sys
import os
import traceback
import platform
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

class ErrorDiagnosticSystem:
    """Comprehensive error diagnosis and recovery system"""
    
    EXIT_CODES = {
        0: "SUCCESS",
        1: "VALIDATION_FAILED",
        2: "CONFIG_ERROR", 
        3: "DEPENDENCY_ERROR",
        4: "RUNTIME_ERROR",
        5: "IMPORT_ERROR",
        6: "PERMISSION_ERROR",
        7: "RESOURCE_ERROR"
    }
    
    def __init__(self):
        self.system_info = self._collect_system_info()
        self.diagnosis_results = []
        self.recovery_suggestions = []
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        info = {
            'timestamp': time.time(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'environment_variables': dict(os.environ),
            'path': sys.path
        }
        
        # Add memory info if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent': memory.percent
            })
        except ImportError:
            info['psutil_available'] = False
            
        return info
    
    def diagnose_import_error(self, error: ImportError, module_name: str) -> Dict[str, Any]:
        """Diagnose import errors with specific solutions"""
        diagnosis = {
            'error_type': 'ImportError',
            'module': module_name,
            'error_message': str(error),
            'likely_causes': [],
            'solutions': []
        }
        
        # Check if module is in standard library
        try:
            import importlib.util
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                diagnosis['likely_causes'].append("Module not installed")
                diagnosis['solutions'].extend([
                    f"pip install {module_name}",
                    f"pip install -r requirements.txt"
                ])
            else:
                diagnosis['likely_causes'].append("Module installed but import failed")
                diagnosis['solutions'].extend([
                    "Check for circular imports",
                    "Verify module integrity: pip install --force-reinstall {module_name}"
                ])
        except Exception:
            diagnosis['likely_causes'].append("Unable to check module status")
        
        # Check common Supreme System V5 import issues
        if 'supreme_system_v5' in module_name:
            diagnosis['likely_causes'].append("Supreme System V5 module path issue")
            diagnosis['solutions'].extend([
                "Ensure you're in the project root directory",
                "Check PYTHONPATH includes python/ directory",
                "Run from project root: python -m supreme_system_v5.core"
            ])
        
        return diagnosis
    
    def diagnose_config_error(self, error: Exception) -> Dict[str, Any]:
        """Diagnose configuration-related errors"""
        diagnosis = {
            'error_type': 'ConfigError',
            'error_message': str(error),
            'likely_causes': [
                "Missing .env file",
                "Invalid .env format", 
                "Missing required configuration keys"
            ],
            'solutions': [
                "Copy .env.optimized to .env",
                "Run: python scripts/validate_environment.py",
                "Check .env file format (KEY=value, no spaces around =)"
            ]
        }
        
        # Check for specific config files
        project_root = Path.cwd()
        env_files = ['.env', '.env.optimized', '.env.example']
        existing_files = [f for f in env_files if (project_root / f).exists()]
        
        diagnosis['config_files_found'] = existing_files
        
        if not existing_files:
            diagnosis['solutions'].insert(0, "Create .env file with required keys")
        
        return diagnosis
    
    def diagnose_runtime_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Diagnose runtime errors"""
        diagnosis = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            'likely_causes': [],
            'solutions': []
        }
        
        error_msg = str(error).lower()
        
        # Memory-related errors
        if 'memory' in error_msg or 'memoryerror' in error_msg:
            diagnosis['likely_causes'].extend([
                "Insufficient RAM",
                "Memory leak in algorithm",
                "Large price history buffer"
            ])
            diagnosis['solutions'].extend([
                "Reduce PRICE_HISTORY_SIZE in config",
                "Enable CircularBuffer with smaller size",
                "Increase system swap space"
            ])
        
        # CPU-related errors
        elif 'timeout' in error_msg or 'performance' in error_msg:
            diagnosis['likely_causes'].extend([
                "CPU overload",
                "Inefficient algorithm",
                "Too many concurrent processes"
            ])
            diagnosis['solutions'].extend([
                "Reduce PROCESS_INTERVAL_SECONDS",
                "Enable SmartEventProcessor filtering",
                "Lower tick processing rate"
            ])
        
        # Network-related errors
        elif 'connection' in error_msg or 'network' in error_msg:
            diagnosis['likely_causes'].extend([
                "Network connectivity issues",
                "API rate limiting",
                "Firewall blocking connections"
            ])
            diagnosis['solutions'].extend([
                "Check internet connection",
                "Verify API credentials",
                "Check firewall settings"
            ])
        
        # Permission errors
        elif 'permission' in error_msg or 'access' in error_msg:
            diagnosis['likely_causes'].extend([
                "Insufficient file permissions",
                "Read-only filesystem",
                "User access restrictions"
            ])
            diagnosis['solutions'].extend([
                "Run with appropriate permissions",
                "Check file ownership: ls -la",
                "Ensure write access to project directory"
            ])
        
        return diagnosis
    
    def diagnose_exit_code(self, exit_code: int) -> Dict[str, Any]:
        """Diagnose system exit codes"""
        diagnosis = {
            'exit_code': exit_code,
            'exit_meaning': self.EXIT_CODES.get(exit_code, "UNKNOWN_ERROR"),
            'likely_causes': [],
            'solutions': []
        }
        
        if exit_code == 1:
            diagnosis['likely_causes'] = ["Environment validation failed"]
            diagnosis['solutions'] = ["Run: python scripts/validate_environment.py"]
        elif exit_code == 2:
            diagnosis['likely_causes'] = ["Configuration error"]
            diagnosis['solutions'] = ["Check .env file", "Validate config keys"]
        elif exit_code == 3:
            diagnosis['likely_causes'] = ["Missing dependencies"]
            diagnosis['solutions'] = ["Run: pip install -r requirements.txt"]
        elif exit_code == 4:
            diagnosis['likely_causes'] = ["Runtime error during execution"]
            diagnosis['solutions'] = ["Check logs for specific error", "Run with debug mode"]
        elif exit_code == 5:
            diagnosis['likely_causes'] = ["Import error"]
            diagnosis['solutions'] = ["Check Python path", "Verify module installation"]
        elif exit_code == -1:
            diagnosis['likely_causes'] = ["PowerShell/Terminal process terminated"]
            diagnosis['solutions'] = [
                "Check for syntax errors in Python code",
                "Run from command line instead of IDE terminal",
                "Check system resources (RAM/CPU)"
            ]
        
        return diagnosis
    
    def run_comprehensive_diagnosis(self, 
                                  error: Optional[Exception] = None, 
                                  exit_code: Optional[int] = None,
                                  context: str = "") -> Dict[str, Any]:
        """Run complete diagnostic analysis"""
        
        print("🔍 Supreme System V5 - Error Diagnosis")
        print("=" * 50)
        
        report = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'diagnostics': [],
            'recovery_plan': []
        }
        
        # Diagnose specific error if provided
        if error:
            if isinstance(error, ImportError):
                module_name = str(error).split("'")[1] if "'" in str(error) else "unknown"
                diagnosis = self.diagnose_import_error(error, module_name)
            elif 'config' in str(error).lower():
                diagnosis = self.diagnose_config_error(error)
            else:
                diagnosis = self.diagnose_runtime_error(error, context)
            
            report['diagnostics'].append(diagnosis)
            self._print_diagnosis(diagnosis)
        
        # Diagnose exit code if provided
        if exit_code is not None:
            exit_diagnosis = self.diagnose_exit_code(exit_code)
            report['diagnostics'].append(exit_diagnosis)
            self._print_diagnosis(exit_diagnosis)
        
        # Generate comprehensive recovery plan
        recovery_plan = self._generate_recovery_plan(report['diagnostics'])
        report['recovery_plan'] = recovery_plan
        
        print("\n🎥 RECOVERY PLAN")
        print("=" * 50)
        for i, step in enumerate(recovery_plan, 1):
            print(f"{i}. {step}")
        
        return report
    
    def _print_diagnosis(self, diagnosis: Dict[str, Any]):
        """Print formatted diagnosis"""
        print(f"\n🐛 {diagnosis['error_type']}: {diagnosis.get('error_message', '')[:100]}")
        
        if diagnosis.get('likely_causes'):
            print("\n🔍 Likely Causes:")
            for cause in diagnosis['likely_causes']:
                print(f"   • {cause}")
        
        if diagnosis.get('solutions'):
            print("\n🛠️ Solutions:")
            for solution in diagnosis['solutions']:
                print(f"   • {solution}")
    
    def _generate_recovery_plan(self, diagnostics: List[Dict[str, Any]]) -> List[str]:
        """Generate prioritized recovery plan"""
        plan = []
        
        # Basic system checks (always first)
        plan.extend([
            "Run environment validation: python scripts/validate_environment.py",
            "Check Python version >= 3.10",
            "Verify you're in project root directory"
        ])
        
        # Add specific solutions from diagnostics
        for diagnosis in diagnostics:
            if diagnosis.get('solutions'):
                for solution in diagnosis['solutions'][:2]:  # Top 2 solutions
                    if solution not in plan:
                        plan.append(solution)
        
        # Common recovery steps
        plan.extend([
            "Install/update dependencies: pip install -r requirements.txt",
            "Copy .env.optimized to .env if missing",
            "Run simple test: python -c 'import sys; print(sys.version)'",
            "If all else fails: restart terminal/IDE and try again"
        ])
        
        return plan
    
    def save_diagnosis_report(self, report: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Save diagnosis report to JSON file"""
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"error_diagnosis_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Diagnosis report saved to: {output_path}")
        return output_path


def diagnose_error(error: Exception = None, exit_code: int = None, context: str = ""):
    """Main error diagnosis function"""
    diagnostic_system = ErrorDiagnosticSystem()
    report = diagnostic_system.run_comprehensive_diagnosis(error, exit_code, context)
    diagnostic_system.save_diagnosis_report(report)
    return report


def main():
    """CLI entry point for error diagnosis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Supreme System V5 Error Diagnosis")
    parser.add_argument('--exit-code', type=int, help="Exit code to diagnose")
    parser.add_argument('--context', default="", help="Additional context")
    
    args = parser.parse_args()
    
    diagnostic_system = ErrorDiagnosticSystem()
    
    if args.exit_code:
        report = diagnostic_system.run_comprehensive_diagnosis(exit_code=args.exit_code, context=args.context)
    else:
        # Interactive mode
        print("Interactive Error Diagnosis Mode")
        print("Enter error details (or press Enter to skip):")
        error_msg = input("Error message: ").strip()
        
        if error_msg:
            # Create mock exception for diagnosis
            error = Exception(error_msg)
            report = diagnostic_system.run_comprehensive_diagnosis(error=error, context=args.context)
        else:
            report = diagnostic_system.run_comprehensive_diagnosis(context=args.context)
    
    diagnostic_system.save_diagnosis_report(report)
    print("\n🏁 Diagnosis completed successfully!")


if __name__ == "__main__":
    main()
