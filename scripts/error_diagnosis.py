#!/usr/bin/env python3
"""
Supreme System V5 - Production Error Diagnosis & Recovery
Comprehensive error handling for all entry points with fast triage capabilities.
"""

import sys
import os
import traceback
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

@dataclass
class ErrorContext:
    """Comprehensive error context for diagnosis."""
    error_type: str
    error_message: str
    traceback: str
    system_info: Dict[str, Any]
    environment_info: Dict[str, Any]
    dependency_status: Dict[str, Any]
    recommendations: List[str]

class SupremeErrorHandler:
    """Production-grade error handler for Supreme System V5."""

    def __init__(self):
        self.error_contexts: List[ErrorContext] = []

    def capture_system_info(self) -> Dict[str, Any]:
        """Capture comprehensive system information."""
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'environment_variables': {k: v for k, v in os.environ.items() if not k.startswith('_')},
        }

        # Try to get additional system info
        try:
            import psutil
            info['cpu_count'] = psutil.cpu_count()
            info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
            info['disk_free_gb'] = psutil.disk_usage('/').free / (1024**3)
        except ImportError:
            info['system_monitoring'] = 'psutil not available'

        return info

    def check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies and their status."""
        dependencies = {
            'core': [
                'fastapi', 'uvicorn', 'pydantic', 'aiohttp', 'numpy', 'pandas',
                'psutil', 'prometheus_client', 'loguru', 'python_dotenv'
            ],
            'technical_analysis': ['ta', 'finta'],
            'optional': ['matplotlib', 'plotly', 'scikit-learn']
        }

        status = {}

        for category, deps in dependencies.items():
            status[category] = {}
            for dep in deps:
                try:
                    __import__(dep.replace('-', '_'))
                    status[category][dep] = 'available'
                except ImportError:
                    status[category][dep] = 'missing'

        return status

    def diagnose_import_error(self, error: ImportError) -> List[str]:
        """Diagnose import errors and provide solutions."""
        recommendations = []
        error_msg = str(error).lower()

        if 'supreme_system_v5' in error_msg:
            recommendations.extend([
                "Ensure you're running from the supreme-system-v5 project root",
                "Check that python/ directory exists and contains supreme_system_v5/",
                "Run: pip install -e . to install the package in development mode"
            ])

        if 'rust' in error_msg or 'supreme_engine_rs' in error_msg:
            recommendations.extend([
                "Rust engine is optional - system will use Python fallbacks",
                "To build Rust engine: pip install maturin && maturin develop",
                "This is not a blocking error for basic functionality"
            ])

        missing_modules = ['fastapi', 'uvicorn', 'pydantic', 'aiohttp', 'psutil']
        for module in missing_modules:
            if module in error_msg:
                recommendations.append(f"Install missing dependency: pip install {module}")

        if not recommendations:
            recommendations.append("Check requirements.txt and run: pip install -r requirements.txt")

        return recommendations

    def diagnose_config_error(self, error: Exception) -> List[str]:
        """Diagnose configuration-related errors."""
        recommendations = []
        error_msg = str(error).lower()

        if 'config' in error_msg or 'env' in error_msg:
            recommendations.extend([
                "Create .env file from template: cp env_optimized.template .env",
                "Check that all required environment variables are set",
                "Validate .env file syntax and paths",
                "Run: python scripts/validate_environment.py to check setup"
            ])

        if 'file not found' in error_msg or 'no such file' in error_msg:
            recommendations.extend([
                "Ensure all required configuration files exist",
                "Check file paths in .env configuration",
                "Verify working directory is the project root"
            ])

        return recommendations

    def diagnose_connection_error(self, error: Exception) -> List[str]:
        """Diagnose network and connection errors."""
        recommendations = []
        error_msg = str(error).lower()

        if 'connection' in error_msg or 'network' in error_msg:
            recommendations.extend([
                "Check internet connectivity",
                "Verify firewall settings allow outbound connections",
                "Test DNS resolution: ping google.com",
                "Check proxy settings if behind corporate firewall"
            ])

        if 'api' in error_msg or 'key' in error_msg:
            recommendations.extend([
                "Verify API keys are correctly set in .env file",
                "Check API key permissions and validity",
                "Test API endpoints manually with curl",
                "Review API documentation for authentication requirements"
            ])

        return recommendations

    def diagnose_general_error(self, error: Exception) -> List[str]:
        """Provide general error diagnosis."""
        recommendations = [
            "Check system logs for more detailed error information",
            "Run diagnostics: python scripts/validate_environment.py",
            "Check available disk space and memory",
            "Try restarting the application",
            "Review recent system changes or updates"
        ]

        # Add OS-specific advice
        if platform.system() == 'Windows':
            recommendations.append("On Windows: Ensure antivirus is not blocking the application")
        elif platform.system() == 'Linux':
            recommendations.append("On Linux: Check SELinux/AppArmor policies if applicable")

        return recommendations

    def handle_error(self, error: Exception, context: str = "general") -> ErrorContext:
        """Handle and diagnose an error comprehensively."""
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()

        # Gather diagnostic information
        system_info = self.capture_system_info()
        dependency_status = self.check_dependencies()

        # Generate recommendations based on error type
        recommendations = []

        if isinstance(error, ImportError):
            recommendations.extend(self.diagnose_import_error(error))
        elif isinstance(error, (FileNotFoundError, OSError)) and 'config' in error_message.lower():
            recommendations.extend(self.diagnose_config_error(error))
        elif isinstance(error, (ConnectionError, TimeoutError)):
            recommendations.extend(self.diagnose_connection_error(error))
        else:
            recommendations.extend(self.diagnose_general_error(error))

        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            traceback=error_traceback,
            system_info=system_info,
            environment_info={'context': context, 'python_path': sys.path},
            dependency_status=dependency_status,
            recommendations=recommendations
        )

        self.error_contexts.append(error_context)
        return error_context

    def print_error_report(self, error_context: ErrorContext):
        """Print a comprehensive error report."""
        print("\n" + "="*80)
        print("🚨 SUPREME SYSTEM V5 - ERROR DIAGNOSIS REPORT")
        print("="*80)
        print(f"Error Type: {error_context.error_type}")
        print(f"Error Message: {error_context.error_message}")
        print(f"Context: {error_context.environment_info.get('context', 'unknown')}")
        print()

        print("🔍 SYSTEM INFORMATION:")
        print(f"  Platform: {error_context.system_info.get('platform', 'unknown')}")
        print(f"  Python: {error_context.system_info.get('python_version', 'unknown')}")
        print(f"  Working Directory: {error_context.system_info.get('working_directory', 'unknown')}")
        print()

        print("📦 DEPENDENCY STATUS:")
        for category, deps in error_context.dependency_status.items():
            print(f"  {category.title()}:")
            for dep, status in deps.items():
                status_icon = "✅" if status == 'available' else "❌"
                print(f"    {status_icon} {dep}: {status}")
        print()

        print("💡 RECOMMENDED SOLUTIONS:")
        for i, rec in enumerate(error_context.recommendations, 1):
            print(f"  {i}. {rec}")
        print()

        print("📋 FULL TRACEBACK:")
        print(error_context.traceback)
        print("="*80)

    def save_error_report(self, error_context: ErrorContext, filename: Optional[str] = None):
        """Save error report to file."""
        if filename is None:
            timestamp = error_context.system_info.get('timestamp', 'unknown')
            filename = f"error_report_{timestamp}.json"

        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        filepath = Path("logs") / filename

        report_data = {
            'error_context': {
                'type': error_context.error_type,
                'message': error_context.error_message,
                'traceback': error_context.traceback,
                'context': error_context.environment_info.get('context')
            },
            'system_info': error_context.system_info,
            'environment_info': error_context.environment_info,
            'dependency_status': error_context.dependency_status,
            'recommendations': error_context.recommendations
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"📊 Error report saved to: {filepath}")

# Global error handler instance
error_handler = SupremeErrorHandler()

def robust_main_wrapper(main_function, context: str = "application"):
    """
    Wrap main functions with comprehensive error handling.

    Args:
        main_function: The main function to wrap
        context: Context description for error reporting
    """
    def wrapped_main():
        try:
            return main_function()
        except KeyboardInterrupt:
            print("\n⚠️ Application interrupted by user")
            sys.exit(130)
        except Exception as e:
            error_context = error_handler.handle_error(e, context)
            error_handler.print_error_report(error_context)
            error_handler.save_error_report(error_context)
            print(f"\n❌ Application failed with exit code 1")
            print("📞 Check logs/ directory for detailed error report")
            sys.exit(1)

    return wrapped_main

# Convenience functions for common entry points
def create_robust_entry_point(main_function, context: str = "application"):
    """Create a robust entry point with error handling."""
    return robust_main_wrapper(main_function, context)

# Test the error handler
if __name__ == "__main__":
    print("🧪 Testing Supreme System V5 Error Diagnosis...")

    # Test import error
    try:
        import nonexistent_module
    except ImportError as e:
        error_context = error_handler.handle_error(e, "test_import")
        error_handler.print_error_report(error_context)

    print("✅ Error diagnosis system operational")
