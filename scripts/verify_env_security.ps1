# Verify .env Security Configuration
# Supreme System V5 - Security Verification
# Usage: .\scripts\verify_env_security.ps1

Write-Host "Security Verification: .env Configuration" -ForegroundColor Cyan
Write-Host ""

$issues = @()
$warnings = @()

# Check 1: .env in .gitignore
Write-Host "Check 1: Verifying .env in .gitignore..." -ForegroundColor Cyan
$gitignoreContent = Get-Content ".gitignore" -ErrorAction SilentlyContinue
if ($gitignoreContent -match "^\.env$") {
    Write-Host "  [OK] .env is in .gitignore" -ForegroundColor Green
} else {
    $issues += ".env is NOT in .gitignore"
    Write-Host "  [FAIL] .env is NOT in .gitignore" -ForegroundColor Red
}

# Check 2: .env tracked by git
Write-Host "Check 2: Checking if .env is tracked by git..." -ForegroundColor Cyan
$trackedFiles = git ls-files | Select-String -Pattern "^\.env$"
if (-not $trackedFiles) {
    Write-Host "  [OK] .env is NOT tracked by git" -ForegroundColor Green
} else {
    $issues += ".env is tracked by git"
    Write-Host "  [FAIL] .env IS tracked by git" -ForegroundColor Red
    Write-Host "     Run: git rm --cached .env" -ForegroundColor Yellow
}

# Check 3: .env in git history
Write-Host "Check 3: Checking if .env exists in git history..." -ForegroundColor Cyan
$history = git log --all --full-history --oneline -- .env 2>$null
if (-not $history) {
    Write-Host "  [OK] .env NOT found in git history" -ForegroundColor Green
} else {
    $warnings += ".env found in git history (needs cleanup)"
    Write-Host "  [WARN] .env found in git history" -ForegroundColor Yellow
    Write-Host "     Commits:" -ForegroundColor Yellow
    $history | ForEach-Object { Write-Host "       $_" -ForegroundColor Yellow }
    Write-Host "     Run: .\scripts\remove_env_from_history.ps1" -ForegroundColor Yellow
}

# Check 4: .env.example exists
Write-Host "Check 4: Checking for .env.example..." -ForegroundColor Cyan
if (Test-Path ".env.example") {
    Write-Host "  [OK] .env.example exists" -ForegroundColor Green
} else {
    $warnings += ".env.example missing"
    Write-Host "  [WARN] .env.example missing" -ForegroundColor Yellow
}

# Check 5: .env exists locally (warning only)
Write-Host "Check 5: Checking if .env exists locally..." -ForegroundColor Cyan
if (Test-Path ".env") {
    Write-Host "  [INFO] .env exists locally (this is OK if gitignored)" -ForegroundColor Cyan
} else {
    Write-Host "  [INFO] .env does not exist locally" -ForegroundColor Cyan
}

# Summary
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan

if ($issues.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host "[SUCCESS] All security checks passed!" -ForegroundColor Green
} else {
    if ($issues.Count -gt 0) {
        Write-Host "[CRITICAL] ISSUES:" -ForegroundColor Red
        $issues | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
    }
    
    if ($warnings.Count -gt 0) {
        Write-Host "[WARN] WARNINGS:" -ForegroundColor Yellow
        $warnings | ForEach-Object { Write-Host "   - $_" -ForegroundColor Yellow }
    }
}

Write-Host ""
Write-Host "For detailed security guide, see: SECURITY_KEY_ROTATION_GUIDE.md" -ForegroundColor Cyan

