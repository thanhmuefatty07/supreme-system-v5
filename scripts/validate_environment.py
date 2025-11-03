#!/usr/bin/env python3
"""
Supreme System V5 - Environment Validation Script
Automatically detects and resolves import/package issues for Windows and Linux.
Validates all runtime and test dependencies.
"""

import sys
import os
import platform
import subprocess
import importlib.util
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

class EnvironmentValidator:
    """Comprehensive environment validation for Supreme System V5."""

    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.issues = []
        self.warnings = []
        self.successes = []

    def log_issue(self, message: str):
        """Log a critical issue that prevents operation."""
        self.issues.append(message)
        print(f"❌ {message}")

    def log_warning(self, message: str):
        """Log a warning that may impact performance."""
        self.warnings.append(message)
        print(f"⚠️  {message}")

    def log_success(self, message: str):
        """Log a successful validation."""
        self.successes.append(message)
        print(f"✅ {message}")

    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        if self.python_version >= (3, 10):
            self.log_success(f"Python {self.python_version.major}.{self.python_version.minor} - Compatible")
            return True
        else:
            self.log_issue(f"Python {self.python_version.major}.{self.python_version.minor} - Requires Python 3.10+")
            return False

    def check_import(self, module_name: str, description: str = "", required: bool = True) -> bool:
        """Check if a module can be imported."""
        try:
            importlib.import_module(module_name)
            self.log_success(f"{module_name} - Available ({description})")
            return True
        except ImportError as e:
            if required:
                self.log_issue(f"{module_name} - Missing ({description}): {e}")
                return False
            else:
                self.log_warning(f"{module_name} - Optional ({description}): {e}")
                return False

    def check_core_dependencies(self) -> bool:
        """Check all core runtime dependencies."""
        print("\n🔧 CORE DEPENDENCIES")
        print("=" * 50)

        core_deps = [
            ('fastapi', 'Web framework'),
            ('uvicorn', 'ASGI server'),
            ('pydantic', 'Data validation'),
            ('aiohttp', 'Async HTTP client'),
            ('websockets', 'WebSocket support'),
            ('numpy', 'Numerical computing'),
            ('pandas', 'Data manipulation'),
            ('psutil', 'System monitoring'),
            ('prometheus_client', 'Metrics collection'),
            ('loguru', 'Advanced logging'),
            ('rich', 'Console output'),
            ('python_dotenv', 'Environment variables'),
        ]

        all_good = True
        for module, desc in core_deps:
            if not self.check_import(module, desc):
                all_good = False

        return all_good

    def check_technical_analysis(self) -> bool:
        """Check technical analysis libraries."""
        print("\n📊 TECHNICAL ANALYSIS")
        print("=" * 50)

        ta_libs = [
            ('ta', 'Technical Analysis (Python)', False),  # Optional
            ('finta', 'Financial Technical Analysis', False),  # Optional
        ]

        # At least one TA library should be available
        ta_available = False
        for module, desc, required in ta_libs:
            if self.check_import(module, desc, required):
                ta_available = True

        if not ta_available:
            self.log_warning("No technical analysis library available - using fallback implementations")

        return True  # Not blocking since we have fallbacks

    def check_supreme_system_imports(self) -> bool:
        """Check all Supreme System V5 imports."""
        print("\n🚀 SUPREME SYSTEM V5 IMPORTS")
        print("=" * 50)

        supreme_imports = [
            ('supreme_system_v5.core', 'Core system'),
            ('supreme_system_v5.strategies', 'Trading strategies'),
            ('supreme_system_v5.optimized.analyzer', 'Optimized analyzer'),
            ('supreme_system_v5.optimized.smart_events', 'Smart event processor'),
            ('supreme_system_v5.optimized.circular_buffer', 'Circular buffer'),
            ('supreme_system_v5.risk', 'Risk management'),
            ('supreme_system_v5.monitoring.resource_monitor', 'Resource monitor'),
        ]

        all_good = True
        for module, desc in supreme_imports:
            if not self.check_import(module, desc):
                all_good = False

        return all_good

    def check_system_resources(self) -> bool:
        """Check system resources and provide recommendations."""
        print("\n💻 SYSTEM RESOURCES")
        print("=" * 50)

        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)

            print(f"   CPU Cores: {cpu_count}")
            print(f"   Total RAM: {memory_gb:.1f}GB")

            # Check if system meets minimum requirements
            if cpu_count >= 2 and memory_gb >= 4:
                self.log_success("System meets minimum requirements (2+ cores, 4GB+ RAM)")
                return True
            elif cpu_count >= 1 and memory_gb >= 2:
                self.log_warning("System meets basic requirements but may be slow (1+ core, 2GB+ RAM)")
                return True
            else:
                self.log_issue("System does not meet minimum requirements (<1 core, <2GB RAM)")
                return False

        except ImportError:
            self.log_warning("psutil not available - cannot check system resources")
            return True

    def check_network_connectivity(self) -> bool:
        """Check basic network connectivity."""
        print("\n🌐 NETWORK CONNECTIVITY")
        print("=" * 50)

        try:
            import socket
            # Try to resolve a well-known host
            socket.gethostbyname('google.com')
            self.log_success("Network connectivity - OK")
            return True
        except Exception as e:
            self.log_warning(f"Network connectivity check failed: {e}")
            return False

    def generate_installation_commands(self) -> List[str]:
        """Generate installation commands for missing dependencies."""
        commands = []

        if self.system == 'windows':
            commands.extend([
                "# Windows Installation Commands:",
                "python -m pip install --upgrade pip setuptools wheel",
                "pip install -r requirements.txt --no-cache-dir",
            ])
        else:
            commands.extend([
                "# Linux/macOS Installation Commands:",
                "python3 -m pip install --upgrade pip setuptools wheel",
                "pip3 install -r requirements.txt --no-cache-dir",
            ])

        return commands

    def run_validation(self) -> bool:
        """Run complete environment validation."""
        print("🚀 SUPREME SYSTEM V5 - ENVIRONMENT VALIDATION")
        print("=" * 60)
        print(f"System: {self.system}")
        print(f"Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print()

        checks = [
            self.check_python_version,
            self.check_core_dependencies,
            self.check_technical_analysis,
            self.check_supreme_system_imports,
            self.check_system_resources,
            self.check_network_connectivity,
        ]

        all_passed = True
        for check in checks:
            if not check():
                all_passed = False

        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)

        if all_passed and not self.issues:
            print("🎉 ALL CHECKS PASSED - Environment ready for Supreme System V5!")
            return True
        else:
            print(f"❌ VALIDATION FAILED")
            print(f"   Critical Issues: {len(self.issues)}")
            print(f"   Warnings: {len(self.warnings)}")
            print(f"   Successful Checks: {len(self.successes)}")

            if self.issues:
                print("\n🔧 TO FIX CRITICAL ISSUES:")
                for cmd in self.generate_installation_commands():
                    print(f"   {cmd}")

            return False

def main():
    """Main validation entry point."""
    validator = EnvironmentValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
