# 📊 Metrics Verification Guide

**Purpose:** Verify actual test count and coverage metrics to ensure README accuracy.

---

## 🚀 Quick Start

### Step 1: Run Verification

```powershell
.\verify_metrics.ps1
```

**What it does:**
- Collects actual test count using `pytest --collect-only`
- Runs all tests with coverage reporting
- Parses coverage percentage from JSON report
- Compares with README claims
- Generates `verification_results.json`

**Time:** 2-5 minutes (depending on test suite size)

**Output files:**
- `test_results.txt` - Full pytest output
- `coverage.json` - Coverage data in JSON format
- `htmlcov/index.html` - HTML coverage report
- `verification_results.json` - Verification summary

---

### Step 2: Review Results

Open `verification_results.json` to see:

```json
{
  "timestamp": "2025-11-17 22:45:00",
  "actual": {
    "tests": 450,
    "coverage": 28.5,
    "passed": 445,
    "failed": 5,
    "skipped": 10,
    "errors": 0
  },
  "claimed": {
    "tests": 474,
    "coverage": 27
  },
  "differences": {
    "tests": -24,
    "coverage": 1.5
  },
  "status": {
    "tests": "OVERCLAIMED",
    "coverage": "BETTER"
  }
}
```

---

### Step 3: Update README (if needed)

If metrics don't match:

```powershell
.\update_readme.ps1
```

**What it does:**
- Creates backup: `README.md.backup`
- Updates all metric references in README.md
- Updates badges
- Optionally commits and pushes changes

**Before running:**
- Make sure `verification_results.json` exists
- Review changes before committing

---

### Step 4: Create Tracking Issue (optional)

```powershell
.\create_issue.ps1
```

**What it does:**
- Creates GitHub Issue with verification results
- Adds labels: `metrics`, `verification`
- Opens issue in browser

**Requirements:**
- GitHub CLI (`gh`) installed
- Authenticated with GitHub

---

## 📋 Script Details

### `verify_metrics.ps1`

**Features:**
- ✅ Test count collection
- ✅ Coverage calculation
- ✅ Result comparison
- ✅ JSON export
- ✅ Error handling

**Dependencies:**
- pytest
- pytest-cov
- PowerShell 5.1+

---

### `update_readme.ps1`

**Features:**
- ✅ Automatic README updates
- ✅ Badge updates
- ✅ Backup creation
- ✅ Git integration

**What gets updated:**
- Test count badges
- Coverage badges
- Text references to metrics
- Architecture diagram comments

---

### `create_issue.ps1`

**Features:**
- ✅ GitHub Issue creation
- ✅ Automatic formatting
- ✅ Label assignment
- ✅ Browser opening

**Requirements:**
- GitHub CLI installed
- Repository access

---

## 🎯 Expected Workflow

```
1. Run verify_metrics.ps1
   ↓
2. Review verification_results.json
   ↓
3. If discrepancies found:
   → Run update_readme.ps1
   → Review changes
   → Commit & push
   ↓
4. Optional: create_issue.ps1
   → Track in GitHub
```

---

## ⚠️ Troubleshooting

### Issue: pytest not found

**Solution:**
```powershell
pip install pytest pytest-cov
```

### Issue: Coverage report not generated

**Solution:**
```powershell
# Check pytest.ini exists
Test-Path pytest.ini

# Run manually
pytest tests/ --cov=src --cov-report=json
```

### Issue: GitHub CLI not found

**Solution:**
```powershell
winget install GitHub.cli
gh auth login
```

---

## 📊 Interpreting Results

### Test Count Status

- **MATCH:** Exact match with README
- **BETTER:** More tests than claimed ✅
- **OVERCLAIMED:** Fewer tests than claimed ⚠️

### Coverage Status

- **MATCH:** Within 1% of claimed
- **BETTER:** Higher than claimed ✅
- **OVERCLAIMED:** Lower than claimed ⚠️

---

## 🔄 Regular Verification

**Recommended schedule:**
- After major feature additions
- Before releases
- Monthly maintenance

**Automation:**
Consider adding to CI/CD pipeline:
```yaml
- name: Verify Metrics
  run: .\verify_metrics.ps1
```

---

## 📝 Notes

- Scripts preserve original files (backups created)
- All scripts are idempotent (safe to run multiple times)
- Results are saved in JSON for programmatic access
- HTML coverage report provides detailed line-by-line analysis

---

**Last Updated:** 2025-11-17  
**Scripts Version:** 1.0



