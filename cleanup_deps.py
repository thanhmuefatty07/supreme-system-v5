import subprocess
import sys

print(" Cleaning Supreme System V5 Dependencies...")

# List of bloat packages to remove
bloat_packages = [
    "matplotlib",
    "seabsl-py",
    "plotly",
    "bokeh",
    "scikit-learn",
    "torch",
    "transformers",
    "tensorflow",
    "jupyter",
    "ipython",
    "sphinx",
    "sphinx-rtd-theme",
    "fastapi",
    "uvicorn",
    "pydantic",
    "scipy",
    "pyarrow",
    "requests",
    "yfinance",
    "ccxt",
    "flask",
    "django",
    "sqlalchemy",
    "opencv-python",
    "pillow",
    "tqdm",
    "absl-py",
    "alabaster",
    "alembic",
    "altair",
    "anyio",
    "appdirs",
    "argon2-cffi",
    "arrow",
    "ast-comments",
    "astroid",
    "async-timeout",
    "attrs",
    "babel",
    "backcall",
    "beautifulsoup4",
    "bleach",
    "blinker",
    "brotli",
    "cachetools",
    "certifi",
    "cffi",
    "chardet",
    "charset-normalizer",
    "click",
    "colorama",
    "cryptography",
    "cycler",
    "debugpy",
    "decorator",
    "defusedxml",
    "entry-points",
    "et-xmlfile",
    "executing",
    "filelock",
    "fonttools",
    "frozenlist",
    "fsspec",
    "greenlet",
    "h11",
    "h5py",
    "hpack",
    "httpcore",
    "httpx",
    "idna",
    "imageio",
    "imagesize",
    "importlib-metadata",
    "importlib-resources",
    "ipykernel",
    "ipython-genutils",
    "ipywidgets",
    "jedi",
    "jinja2",
    "joblib",
    "json5",
    "jsonschema",
    "jupyter-client",
    "jupyter-console",
    "jupyter-core",
    "jupyter-server",
    "jupyterlab",
    "jupyterlab-pygments",
    "jupyterlab-server",
    "jupyterlab-widgets",
    "kiwisolver",
    "latexcodec",
    "lxml",
    "markupsafe",
    "mistune",
    "mpmath",
    "multidict",
    "nbclient",
    "nbconvert",
    "nbformat",
    "nbval",
    "nest-asyncio",
    "networkx",
    "notebook",
    "numba",
    "numexpr",
    "openpyxl",
    "packaging",
    "pandocfilters",
    "parso",
    "pexpect",
    "pickleshare",
    "platformdirs",
    "prometheus-client",
    "prompt-toolkit",
    "psutil",
    "ptyprocess",
    "pure-eval",
    "py",
    "pyasn1",
    "pyasn1-modules",
    "pycparser",
    "pyct",
    "pydot",
    "pyerfa",
    "pygments",
    "pyjwt",
    "pymongo",
    "pyparsing",
    "pyrsistent",
    "pysocks",
    "python-dateutil",
    "python-json-logger",
    "pytz",
    "pywin32",
    "pywinpty",
    "pyzmq",
    "qtconsole",
    "qtpy",
    "regex",
    "requests-oauthlib",
    "retrying",
    "rfc3339-validator",
    "rfc3986-validator",
    "rich",
    "rsa",
    "ruamel.yaml",
    "scikit-image",
    "scipy",
    "send2trash",
    "setuptools",
    "shapely",
    "six",
    "sniffio",
    "soupsieve",
    "stack-data",
    "statsmodels",
    "sympy",
    "tables",
    "tensorboard",
    "tensorflow-estimator",
    "termcolor",
    "terminado",
    "testpath",
    "threadpoolctl",
    "tinycss2",
    "toml",
    "tomli",
    "toolz",
    "tornado",
    "tqdm",
    "traitlets",
    "typing-extensions",
    "uri-template",
    "urllib3",
    "wcwidth",
    "webcolors",
    "webencodings",
    "websocket-client",
    "widgetsnbextension",
    "xlrd",
    "xlsxwriter",
    "xlwt",
    "yarl",
    "zipp",
    "zope.event",
    "zope.interface",
]

removed = 0
for pkg in bloat_packages:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
            capture_output=True,
            check=True,
        )
        print(f" Removed: {pkg}")
        removed += 1
    except subprocess.CalledProcessError:
        pass  # Package not installed

print(f"\\n Removed {removed} bloat packages")

# Install minimal requirements
minimal_packages = [
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
for pkg in minimal_packages:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "--no-cache-dir"],
            capture_output=True,
            check=True,
        )
        print(f" Installed: {pkg}")
        installed += 1
    except subprocess.CalledProcessError as e:
        print(f" Failed to install {pkg}: {e}")

print(f"\\n Installed {installed} essential packages")

# Verify core imports
try:
    import numpy
    import pandas
    import psutil
    import aiohttp
    import websockets
    import ta
    import finta

    print("\\n Core imports successful!")
except ImportError as e:
    print(f"\\n Import error: {e}")

print("\\n Dependency cleanup completed!")
