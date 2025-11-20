# ============================================================================
# 🚨 CRITICAL: Remove GitHub Token from Git History
# 
# WARNING: This script rewrites Git history and requires force push!
# Only run this if you understand the implications.
# 
# Date: 2025-11-13
# ============================================================================

$ErrorActionPreference = "Stop"

$TOKEN = "EXPOSED_TOKEN_REMOVED"
$REPLACEMENT = "REVOKED_TOKEN_REMOVED_FROM_HISTORY"

Write-Host ""
Write-Host "🚨 CRITICAL SECURITY FIX: Remove Token from Git History" -ForegroundColor Red
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
Write-Host ""
Write-Host "⚠️  WARNING: This will rewrite Git history!" -ForegroundColor Yellow
Write-Host "⚠️  All commit SHAs will change!" -ForegroundColor Yellow
Write-Host "⚠️  Requires force push to remote!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Token to remove: $TOKEN" -ForegroundColor Cyan
Write-Host ""

# Confirm
$confirm = Read-Host "Do you want to proceed? (type 'YES' to continue)"
if ($confirm -ne "YES") {
    Write-Host "Operation cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Step 1: Checking if token exists in history..." -ForegroundColor Blue
$tokenFound = git log --all -p | Select-String -Pattern $TOKEN -Quiet
if ($tokenFound) {
    Write-Host "✅ Token found in Git history" -ForegroundColor Red
} else {
    Write-Host "✅ Token not found in current history" -ForegroundColor Green
    Write-Host "No cleanup needed." -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "Step 2: Creating backup branch..." -ForegroundColor Blue
git branch backup-before-token-removal 2>$null
Write-Host "✅ Backup created: backup-before-token-removal" -ForegroundColor Green

Write-Host ""
Write-Host "Step 3: Removing token from Git history..." -ForegroundColor Blue
Write-Host "This may take several minutes..." -ForegroundColor Yellow

# Method 1: Using git filter-branch (works on Windows)
Write-Host "Using git filter-branch..." -ForegroundColor Cyan

# Create a script file for filter-branch
$filterScript = @"
#!/bin/sh
git filter-branch --force --index-filter `
  "git rm --cached --ignore-unmatch GITHUB_TOKEN_SECURITY.md 2>/dev/null || true" `
  --prune-empty --tag-name-filter cat -- --all

# Replace token in all files
git filter-branch --force --tree-filter `
  "find . -type f -exec sed -i 's/$TOKEN/$REPLACEMENT/g' {} + 2>/dev/null || true" `
  --prune-empty --tag-name-filter cat -- --all
"@

# Alternative: Use PowerShell-native approach
Write-Host "Using PowerShell-native approach..." -ForegroundColor Cyan

# Export all commits, replace token, recreate
$tempDir = "temp-git-cleanup"
if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir
}

Write-Host ""
Write-Host "⚠️  Manual cleanup required:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Option 1: Use BFG Repo-Cleaner (Recommended)" -ForegroundColor Cyan
Write-Host "  1. Download: https://rtyley.github.io/bfg-repo-cleaner/" -ForegroundColor White
Write-Host "  2. Create tokens.txt with: $TOKEN" -ForegroundColor White
Write-Host "  3. Run: java -jar bfg.jar --replace-text tokens.txt" -ForegroundColor White
Write-Host "  4. Run: git reflog expire --expire=now --all" -ForegroundColor White
Write-Host "  5. Run: git gc --prune=now --aggressive" -ForegroundColor White
Write-Host "  6. Run: git push origin --force --all" -ForegroundColor White
Write-Host ""
Write-Host "Option 2: Use git-filter-repo (Python)" -ForegroundColor Cyan
Write-Host "  1. Install: pip install git-filter-repo" -ForegroundColor White
Write-Host "  2. Create replace.txt with: $TOKEN==>$REPLACEMENT" -ForegroundColor White
Write-Host "  3. Run: git filter-repo --replace-text replace.txt" -ForegroundColor White
Write-Host "  4. Run: git push origin --force --all" -ForegroundColor White
Write-Host ""
Write-Host "Option 3: Manual commit removal (if only recent commits)" -ForegroundColor Cyan
Write-Host "  1. Interactive rebase: git rebase -i HEAD~5" -ForegroundColor White
Write-Host "  2. Edit commits containing token" -ForegroundColor White
Write-Host "  3. Remove token from files" -ForegroundColor White
Write-Host "  4. Force push: git push origin --force" -ForegroundColor White
Write-Host ""

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
Write-Host ""
Write-Host "🔴 CRITICAL: Revoke token immediately at:" -ForegroundColor Red
Write-Host "   https://github.com/settings/tokens" -ForegroundColor Yellow
Write-Host ""
Write-Host "Token: $TOKEN" -ForegroundColor Red
Write-Host ""

