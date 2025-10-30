import sys
import platform
import subprocess
from typing import Dict, List, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from python.supreme_system_v5.utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

def check_python_version() -> Tuple[bool, str]:
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if version.major == 3 and version.minor >= 8:
        return True, version_str
    else:
        return False, version_str

def check_rust_installed() -> bool:
    try:
        result = subprocess.run(["cargo", "--version"], capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        logger.info(f"Rust version: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Rust/Cargo not found")
        return False

def check_dependencies() -> Dict[str, bool]:
    dependencies = ["numpy", "pandas", "fastapi", "uvicorn", "pydantic", "asyncio", "typing"]
    results = {}
    for dep in dependencies:
        try:
            __import__(dep)
            results[dep] = True
            logger.info(f"Dependency '{dep}' available")
        except ImportError:
            results[dep] = False
            logger.warning(f"Dependency '{dep}' missing")
    return results

def check_hardware() -> Dict[str, str]:
    info = {}
    try:
        cpu_count = platform.processor()
        info["cpu"] = cpu_count if cpu_count else "Unknown"
    except Exception:
        info["cpu"] = "Unknown"
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["memory_gb"] = ".1f"
    except ImportError:
        info["memory_gb"] = "Unknown (psutil not available)"
    return info

def run_basic_tests() -> bool:
    try:
        from python.supreme_system_v5 import utils, data, backtest, strategies, risk
        config = utils.Config()
        assert config.max_cpu_usage > 0
        data_provider = data.get_data_provider()
        assert data_provider.name == "stub"
        strategy = strategies.get_strategy("sma_crossover")
        assert strategy.name == "sma_crossover"
        risk_manager = risk.RiskManager()
        assert risk_manager.max_positions == 5
        logger.info("All basic tests passed")
        return True
    except Exception as e:
        logger.error(f"Basic tests failed: {e}")
        return False

def main() -> int:
    logger.info("Starting Supreme System V5 validation")
    print("=" * 50)
    print("Supreme System V5 - System Validation")
    print("=" * 50)

    all_passed = True

    python_ok, version = check_python_version()
    status = "" if python_ok else ""
    print(f"Python 3.8+ compatibility: {status} (version {version})")
    if not python_ok:
        all_passed = False

    rust_ok = check_rust_installed()
    status = "" if rust_ok else ""
    print(f"Rust/Cargo installation: {status}")
    if not rust_ok:
        print("  Note: Rust required for full functionality")

    print("\n Checking Python dependencies...")
    deps = check_dependencies()
    missing_deps = [dep for dep, available in deps.items() if not available]

    if missing_deps:
        print("  Missing dependencies:")
        for dep in missing_deps:
            print(f"    - {dep}")
        all_passed = False
    else:
        print("   All dependencies available")

    print("\n  Hardware information:")
    hw_info = check_hardware()
    for key, value in hw_info.items():
        print(f"    {key}: {value}")

    print("\n Running basic tests...")
    tests_ok = run_basic_tests()
    status = "" if tests_ok else ""
    print(f"    Basic functionality tests: {status}")
    if not tests_ok:
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print(" System validation PASSED")
        print("Ready to run Supreme System V5")
        return 0
    else:
        print(" System validation FAILED")
        print("Please fix the issues above before running")
        return 1

if __name__ == "__main__":
    sys.exit(main())
