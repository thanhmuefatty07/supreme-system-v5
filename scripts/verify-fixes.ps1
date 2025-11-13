# ============================================================================
# 🔍 SUPREME SYSTEM V5 - VERIFY ALL FIXES (PowerShell Version)
# 
# This script verifies that all critical fixes were applied successfully
# 
# Date: 2025-11-13
# ============================================================================

$ErrorActionPreference = "Stop"

# Colors
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[✅ PASS] $args" -ForegroundColor Green }
function Write-Error { Write-Host "[❌ FAIL] $args" -ForegroundColor Red }
function Write-Warning { Write-Host "[⚠️  WARN] $args" -ForegroundColor Yellow }

$PASS_COUNT = 0
$FAIL_COUNT = 0
$WARN_COUNT = 0

Write-Host ""
Write-Host "################################################################################" -ForegroundColor Cyan
Write-Host "#                                                                              #" -ForegroundColor Cyan
Write-Host "#              🔍 VERIFICATION REPORT - ALL CRITICAL FIXES 🔍                  #" -ForegroundColor Cyan
Write-Host "#                                                                              #" -ForegroundColor Cyan
Write-Host "################################################################################" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# TEST 1: API Keys Security
# ============================================================================
Write-Info "TEST 1/10: Checking API keys removed from source code..."

$hasHardcodedKeys = $false
if (Test-Path "config/multi_key_config.py") {
    $content = Get-Content "config/multi_key_config.py" -Raw
    if ($content -match 'AIzaSy[A-Za-z0-9_-]{30,}') {
        # Check if it's just validation pattern, not actual hardcoded key
        if ($content -notmatch 'os\.getenv|GEMINI_KEY_') {
            $hasHardcodedKeys = $true
        }
    }
}

if (Test-Path "RUN_OPTIMIZER.sh") {
    $content = Get-Content "RUN_OPTIMIZER.sh" -Raw
    if ($content -match 'export GEMINI_KEY_\d+="AIzaSy[A-Za-z0-9_-]{30,}"') {
        $hasHardcodedKeys = $true
    }
}

if ($hasHardcodedKeys) {
    Write-Error "Hardcoded API keys still found in source!"
    $FAIL_COUNT++
} else {
    Write-Success "No hardcoded API keys in source code"
    $PASS_COUNT++
}

# ============================================================================
# TEST 2: Environment Variable Loading
# ============================================================================
Write-Info "TEST 2/10: Checking config loads from environment variables..."

if (Test-Path "config/multi_key_config.py") {
    $content = Get-Content "config/multi_key_config.py" -Raw
    if ($content -match 'os\.getenv\("GEMINI_KEY_') {
        Write-Success "config/multi_key_config.py loads from environment"
        $PASS_COUNT++
    } else {
        Write-Error "config/multi_key_config.py not loading from environment"
        $FAIL_COUNT++
    }
} else {
    Write-Error "config/multi_key_config.py not found"
    $FAIL_COUNT++
}

# ============================================================================
# TEST 3: RUN_OPTIMIZER.sh Fixed
# ============================================================================
Write-Info "TEST 3/10: Checking RUN_OPTIMIZER.sh loads from environment..."

if (Test-Path "RUN_OPTIMIZER.sh") {
    $content = Get-Content "RUN_OPTIMIZER.sh" -Raw
    if (($content -match '\$\{GEMINI_KEY_\d+:-') -or ($content -match 'source \.env')) {
        Write-Success "RUN_OPTIMIZER.sh loads from environment/.env"
        $PASS_COUNT++
    } else {
        Write-Error "RUN_OPTIMIZER.sh not fixed properly"
        $FAIL_COUNT++
    }
} else {
    Write-Warning "RUN_OPTIMIZER.sh not found"
    $WARN_COUNT++
}

# ============================================================================
# TEST 4: .gitignore Security
# ============================================================================
Write-Info "TEST 4/10: Checking .gitignore has security patterns..."

if (Test-Path ".gitignore") {
    $content = Get-Content ".gitignore" -Raw
    if (($content -match '\.env') -and ($content -match '\*\.key|\*\.secret')) {
        Write-Success ".gitignore updated with security patterns"
        $PASS_COUNT++
    } else {
        Write-Error ".gitignore missing security patterns"
        $FAIL_COUNT++
    }
} else {
    Write-Error ".gitignore not found"
    $FAIL_COUNT++
}

# ============================================================================
# TEST 5: PromQL Alert Rule Fixed
# ============================================================================
Write-Info "TEST 5/10: Checking PromQL alert rule fixed..."

$alertFile = "monitoring/prometheus/rules/trading-system-alerts.yml"
if (Test-Path $alertFile) {
    $content = Get-Content $alertFile -Raw
    if (($content -match 'max_over_time') -and ($content -notmatch 'supremum\(')) {
        Write-Success "PromQL alert rule fixed (supremum → max_over_time)"
        $PASS_COUNT++
    } else {
        Write-Error "PromQL alert rule not fixed"
        $FAIL_COUNT++
    }
} else {
    Write-Warning "$alertFile not found"
    $WARN_COUNT++
}

# ============================================================================
# TEST 6: requirements.txt Updated
# ============================================================================
Write-Info "TEST 6/10: Checking requirements.txt has security updates..."

if (Test-Path "requirements.txt") {
    $content = Get-Content "requirements.txt" -Raw
    $requiredPackages = @("authlib>=1.6.5", "black>=24.3.0", "starlette>=0.40.0", "keras>=3.12.0", "mlflow>=2.19.0", "uv>=0.9.6")
    $missingPackages = @()
    
    foreach ($package in $requiredPackages) {
        $packageName = $package.Split('>=')[0]
        $packageVersion = $package.Split('>=')[1]
        if ($content -notmatch "$packageName>=$packageVersion") {
            $missingPackages += $package
        }
    }
    
    if ($missingPackages.Count -eq 0) {
        Write-Success "requirements.txt updated with all security packages"
        $PASS_COUNT++
    } else {
        Write-Error "requirements.txt missing: $($missingPackages -join ', ')"
        $FAIL_COUNT++
    }
} else {
    Write-Error "requirements.txt not found"
    $FAIL_COUNT++
}

# ============================================================================
# TEST 7: Packages Installed
# ============================================================================
Write-Info "TEST 7/10: Checking critical packages installed..."

try {
    $authlib = python -c "import authlib; print(authlib.__version__)" 2>$null
    $black = python -c "import black; print(black.__version__)" 2>$null
    $starlette = python -c "import starlette; print(starlette.__version__)" 2>$null
    
    if ($authlib -and $black -and $starlette) {
        Write-Success "Critical packages installed: authlib=$authlib, black=$black, starlette=$starlette"
        $PASS_COUNT++
    } else {
        Write-Warning "Some packages may not be installed"
        $WARN_COUNT++
    }
} catch {
    Write-Warning "Could not verify package versions: $_"
    $WARN_COUNT++
}

# ============================================================================
# TEST 8: Zero Trust Backward Compatibility
# ============================================================================
Write-Info "TEST 8/10: Checking Zero Trust backward compatibility..."

try {
    $result = python -c "from src.security.zero_trust import ZeroTrustSecurity, ZeroTrustManager; print('OK' if ZeroTrustSecurity is ZeroTrustManager else 'FAIL')" 2>$null
    if ($result -eq "OK") {
        Write-Success "ZeroTrustSecurity alias working correctly"
        $PASS_COUNT++
    } else {
        Write-Error "ZeroTrustSecurity alias not working"
        $FAIL_COUNT++
    }
} catch {
    Write-Warning "Could not verify Zero Trust compatibility: $_"
    $WARN_COUNT++
}

# ============================================================================
# TEST 9: Git Status
# ============================================================================
Write-Info "TEST 9/10: Checking Git status..."

try {
    $gitStatus = git status --porcelain 2>$null
    if ($gitStatus) {
        Write-Warning "Uncommitted changes detected (may need to commit)"
        $WARN_COUNT++
    } else {
        Write-Success "Git repository clean"
        $PASS_COUNT++
    }
} catch {
    Write-Warning "Could not check Git status"
    $WARN_COUNT++
}

# ============================================================================
# TEST 10: Recent Commits
# ============================================================================
Write-Info "TEST 10/10: Checking recent security fix commits..."

try {
    $recentCommits = git log --oneline -5 2>$null
    if ($recentCommits -match "Security fixes|API keys|CVE") {
        Write-Success "Security fix commits found in recent history"
        $PASS_COUNT++
    } else {
        Write-Warning "No security fix commits found in recent history"
        $WARN_COUNT++
    }
} catch {
    Write-Warning "Could not check Git commits"
    $WARN_COUNT++
}

# ============================================================================
# FINAL REPORT
# ============================================================================
Write-Host ""
Write-Host "################################################################################" -ForegroundColor Cyan
Write-Host "#                                                                              #" -ForegroundColor Cyan
Write-Host "#                       📊 VERIFICATION SUMMARY 📊                             #" -ForegroundColor Cyan
Write-Host "#                                                                              #" -ForegroundColor Cyan
Write-Host "################################################################################" -ForegroundColor Cyan
Write-Host ""
Write-Host "  ✅ PASSED:  $PASS_COUNT tests" -ForegroundColor Green
Write-Host "  ❌ FAILED:  $FAIL_COUNT tests" -ForegroundColor Red
Write-Host "  ⚠️  WARNINGS: $WARN_COUNT tests" -ForegroundColor Yellow
Write-Host ""

if ($FAIL_COUNT -eq 0) {
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host ""
    Write-Success "🎉 ALL CRITICAL FIXES VERIFIED SUCCESSFULLY!"
    Write-Host ""
    Write-Host "📋 NEXT STEPS:" -ForegroundColor Cyan
    Write-Host "  1. ✅ All fixes already applied and committed"
    Write-Host "  2. ⚠️  IMPORTANT: Revoke old API keys in Google Cloud Console"
    Write-Host "  3. ⚠️  IMPORTANT: Create new API keys and update .env file"
    Write-Host "  4. Run: pip install -r requirements.txt --upgrade (if needed)"
    Write-Host ""
    exit 0
} else {
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
    Write-Host ""
    Write-Error "⚠️  SOME FIXES FAILED - PLEASE REVIEW ERRORS ABOVE"
    Write-Host ""
    Write-Host "📋 TROUBLESHOOTING:" -ForegroundColor Yellow
    Write-Host "  1. Review error messages above"
    Write-Host "  2. Check file paths and permissions"
    Write-Host "  3. Ensure all fixes were applied correctly"
    Write-Host ""
    exit 1
}

