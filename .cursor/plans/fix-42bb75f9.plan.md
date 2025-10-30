<!-- 42bb75f9-cfdb-42e5-8a04-5d36e66a9363 006bc806-6a95-473f-a1db-6a18620e1b51 -->
# Hybrid Python + Rust Clean Rebuild

## Objective

- Remove all error-prone legacy Python scripts.
- Keep only the minimal Python orchestrator, bridges, and Rust core.
- Rebuild a clean Python interface layer as stub modules, no f-string without placeholders, fully ruff/mypy/pytest-compliant.
- Standardize to pyproject.toml and Makefile for build.
- Ensure CI is green with ruff, mypy, pytest.

## Plan Steps

### 1. Remove Error Sources (Commit 1)

- Delete legacy and error files:
    - phase2_main.py
    - scripts/validate_system.py (to be replaced clean)
    - src/api/* (all Python)
    - src/backtesting/* (all Python; Rust core stays)
    - src/data_sources/* (all Python)
    - src/foundation_models/*, src/mamba_ssm/*, src/neuromorphic/*, src/monitoring/* (all Python)
    - tests/* (all Python)
- Preserve existing: Cargo.toml, src/lib.rs, src/indicators.rs, src/backtesting.rs, python/supreme_system_v5/**init**.py, core.py

### 2. Add Minimal Stubs (Commit 2)

- Create/replace with clean, minimal modules:
    - python/supreme_system_v5/utils.py: logger, Config dataclass, no f-strings
    - python/supreme_system_v5/data.py: stub only
    - python/supreme_system_v5/backtest.py: Python bridge to Rust backtesting
    - python/supreme_system_v5/strategies.py: base class stub
    - python/supreme_system_v5/risk.py: minimal wrapper
    - scripts/validate_system.py: clean system check script
    - tests/test_smoke.py: smoke test imports & Rust call

### 3. Standardize Tooling (Commit 3)

- Move pyproject_hybrid.toml -> pyproject.toml (update config)
- Move Makefile_hybrid -> Makefile (ensure build works)
- Add/update ruff/black/mypy config to block F541, F821, F811

### 4. Add Green CI Test (Commit 4)

- Ensure ruff check, pytest, mypy all pass
- Tag as release draft v5.0.0-clean-hybrid

### 5. Report/Document

- Output CI status/logs and guidance for running "make dev-setup && make build-rust && make validate"
- Output new commit hashes

## Todos

- remove-legacy
Remove all listed error-prone legacy Python files
- add-python-stubs
Add clean stub modules: utils.py, data.py, backtest.py, strategies.py, risk.py, validate_system.py, test_smoke.py
- standardize-build
Rename and update pyproject.toml, Makefile; enforce strict linting in CI
- green-ci
Run all checks, ensure green status, tag release
- report-result
Output CI logs, commit hashes, and full usage guidance

### To-dos

- [ ] Remove all listed error-prone legacy Python files
- [ ] Add clean stub modules: utils.py, data.py, backtest.py, strategies.py, risk.py, validate_system.py, test_smoke.py
- [ ] Rename and update pyproject.toml, Makefile; enforce strict linting in CI
- [ ] Run all checks, ensure green status, tag release
- [ ] Output CI logs, commit hashes, and full usage guidance