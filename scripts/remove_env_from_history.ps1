# PowerShell Script to Remove .env from Git History
# Supreme System V5 - Security Cleanup
# Usage: .\scripts\remove_env_from_history.ps1

Write-Host "SECURITY CLEANUP: Removing .env from Git History" -ForegroundColor Red
Write-Host ""

# Check if .env exists
if (Test-Path ".env") {
    Write-Host "[WARN] WARNING: .env file exists locally" -ForegroundColor Yellow
    Write-Host "   This script will remove it from git history but keep local file" -ForegroundColor Yellow
    Write-Host ""
}

# Check if git filter-branch is available
$gitVersion = git --version
Write-Host "Git Version: $gitVersion" -ForegroundColor Cyan
Write-Host ""

# Confirm action
$confirmation = Read-Host "This will rewrite git history. Continue? (yes/no)"
if ($confirmation -ne "yes") {
    Write-Host "Aborted." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Step 1: Removing .env from git history..." -ForegroundColor Cyan

# Remove .env from entire history
git filter-branch --force --index-filter `
    "git rm --cached --ignore-unmatch .env" `
    --prune-empty --tag-name-filter cat -- --all

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Error: git filter-branch failed" -ForegroundColor Red
    exit 1
}

    Write-Host "[OK] Step 1 Complete" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Cleaning up reflog..." -ForegroundColor Cyan
git reflog expire --expire=now --all

    Write-Host "[OK] Step 2 Complete" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Garbage collection..." -ForegroundColor Cyan
git gc --prune=now --aggressive

    Write-Host "[OK] Step 3 Complete" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4: Verifying .env is removed..." -ForegroundColor Cyan
$envInHistory = git log --all --full-history --oneline -- .env
if ($envInHistory) {
    Write-Host "[WARN] WARNING: .env still found in history:" -ForegroundColor Yellow
    Write-Host $envInHistory
    Write-Host ""
    Write-Host "Consider using BFG Repo-Cleaner for more thorough cleanup" -ForegroundColor Yellow
} else {
    Write-Host "[OK] .env removed from git history" -ForegroundColor Green
}

Write-Host ""
Write-Host "[CRITICAL] IMPORTANT NEXT STEPS:" -ForegroundColor Red
Write-Host "1. Force push: git push origin --force --all" -ForegroundColor Yellow
Write-Host "2. Force push tags: git push origin --force --tags" -ForegroundColor Yellow
Write-Host "3. Notify all collaborators to re-clone repository" -ForegroundColor Yellow
Write-Host "4. Rotate all API keys immediately" -ForegroundColor Yellow
Write-Host "5. Review SECURITY_KEY_ROTATION_GUIDE.md" -ForegroundColor Yellow
Write-Host ""

Write-Host "[OK] Script Complete" -ForegroundColor Green

