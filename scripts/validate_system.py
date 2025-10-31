#!/usr/bin/env python3
"""
Supreme System V5 - Minimal Hybrid System Validation

Validates minimal Python + Rust hybrid architecture.
Optimized for i3-4GB systems with clean dependencies.
"""

import sys
import platform
import subprocess
from typing import Dict, List, Tuple
from pathlib import Path

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    from supreme_system_v5.utils import get_logger

    logger = get_logger(__name__)
except ImportError:
    # Fallback logging
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 11:
        return True, version_str
    else:
        return False, version_str


def check_rust_installed() -> bool:
    """Check if Rust is installed."""
    try:
        result = subprocess.run(
            ["cargo", "--version"], capture_output=True, text=True, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_system_info() -> Dict[str, str]:
    """Collect basic system information."""
    info = {}
    try:
        cpu_count = platform.processor()
        info["cpu"] = cpu_count if cpu_count else "Unknown"
    except Exception:
        info["cpu"] = "Unknown"

    try:
        import psutil

        memory = psutil.virtual_memory()
        info["memory_gb"] = f"{round(memory.total / (1024**3), 2)}"
    except ImportError:
        info["memory_gb"] = "psutil not available"
    except Exception:
        info["memory_gb"] = "Unknown"

    info["python_version"] = sys.version
    info["os"] = platform.system()
    info["os_release"] = platform.release()
    return info


def validate_minimal_imports() -> bool:
    """Validate minimal Python package imports."""
    try:
        from supreme_system_v5.utils import get_logger, Config
        from supreme_system_v5.data import DataManager
        from supreme_system_v5.backtest import BacktestEngine
        from supreme_system_v5.strategies import Strategy
        from supreme_system_v5.risk import RiskManager

        return True
    except ImportError as e:
        print(f"Minimal import validation failed: {e}")
        return False


def main():
    """Main validation function."""
    logger.info("Starting Supreme System V5 minimal validation")
    print("=" * 50)
    print("Supreme System V5 - Minimal Hybrid System Validation")
    print("=" * 50)

    system_info = get_system_info()
    print(f"System Info: {system_info}")

    all_passed = True

    # Check Python version
    python_ok, version = check_python_version()
    status = "" if python_ok else ""
    print(f"{status} Python Version (>=3.11): {version}")
    if not python_ok:
        all_passed = False

    # Check Rust toolchain (optional for minimal setup)
    rust_ok = check_rust_installed()
    status = "" if rust_ok else ""
    print(
        f"{status} Rust Toolchain (cargo): {'Installed' if rust_ok else 'Not Installed (optional)'}"
    )

    # Validate minimal Python imports
    imports_ok = validate_minimal_imports()
    status = "" if imports_ok else ""
    print(
        f"{status} Minimal Python Package Imports: {'OK' if imports_ok else 'Failed'}"
    )
    if not imports_ok:
        all_passed = False

    print("=" * 50)
    if all_passed:
        print(" Minimal system validation PASSED!")
        print(" Ready for hybrid Python + Rust development")
        print("   - Run: make build-rust  (if Rust installed)")
        print("   - Run: python -m pytest tests/")
        sys.exit(0)
    else:
        print(" Minimal system validation FAILED. Please address the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
