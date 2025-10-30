import subprocess
import sys
import os

print(" Supreme System V5 - Dependency Cleanup")

# Remove bloat packages
bloat = ["matplotlib", "seaborn", "plotly", "bokeh", "scikit-learn", "torch", "transformers", "tensorflow", "jupyter", "ipython", "sphinx", "fastapi", "uvicorn", "pydantic", "scipy", "pyarrow", "requests", "yfinance", "ccxt"]

removed = 0
for pkg in bloat:
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg], capture_output=True, check=True)
        print(f"Removed: {pkg}")
        removed += 1
    except:
        pass

print(f" Removed {removed} packages")

# Install minimal requirements
minimal = [
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.1.0,<3.0.0", 
    "psutil>=5.9.0,<6.0.0",
    "aiohttp>=3.9.0,<4.0.0",
    "websockets>=12.0,<13.0.0",
    "ta>=0.10.2,<0.11.0",
    "finta>=1.3,<2.0.0",
    "prometheus-client>=0.17.0,<0.18.0",
    "loguru>=0.7.2,<0.8.0",
    "rich>=13.7.0,<14.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
    "click>=8.1.7,<9.0.0",
    "python-dateutil>=2.8.2,<3.0.0",
    "pytz>=2023.3,<2024.0",
    "pytest>=7.4.0,<8.0.0",
    "pytest-asyncio>=0.21.0,<0.22.0",
    "ruff>=0.1.0,<0.2.0",
    "black>=23.9.0,<24.0.0",
]

installed = 0
for pkg in minimal:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--no-cache-dir"], capture_output=True, check=True)
        print(f"Installed: {pkg}")
        installed += 1
    except Exception as e:
        print(f"Failed {pkg}: {e}")

print(f" Installed {installed} packages")

# Verify
try:
    import numpy, pandas, psutil, aiohttp, websockets, ta, finta
    print(" Core imports successful")
except ImportError as e:
    print(f" Import error: {e}")

print(" Cleanup completed!")
