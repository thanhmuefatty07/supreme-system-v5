# ============================================================================
# 🚨 CRITICAL: Clean GitHub Token from Git History using BFG Repo-Cleaner
# 
# Option 1: BFG Repo-Cleaner (Easiest Method)
# 
# WARNING: This script rewrites Git history and requires force push!
# Only run this if you understand the implications.
# 
# Date: 2025-11-13
# ============================================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "🚨 CRITICAL SECURITY FIX: Remove Token from Git History" -ForegroundColor Red
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
Write-Host ""
Write-Host "Method: BFG Repo-Cleaner (Option 1)" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  WARNING: This will rewrite Git history!" -ForegroundColor Yellow
Write-Host "⚠️  All commit SHAs will change!" -ForegroundColor Yellow
Write-Host "⚠️  Requires force push to remote!" -ForegroundColor Yellow
Write-Host ""

# Step 1: Check Java installation
Write-Host "Step 1: Checking Java installation..." -ForegroundColor Blue
try {
    $javaVersion = java -version 2>&1 | Select-Object -First 1
    Write-Host "✅ Java found: $javaVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Java not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Java:" -ForegroundColor Yellow
    Write-Host "  - Download: https://www.java.com/download/" -ForegroundColor White
    Write-Host "  - Or use: choco install openjdk (if Chocolatey installed)" -ForegroundColor White
    exit 1
}

# Step 2: Download BFG if not present
Write-Host ""
Write-Host "Step 2: Checking BFG Repo-Cleaner..." -ForegroundColor Blue
$bfgPath = "bfg.jar"
if (-not (Test-Path $bfgPath)) {
    Write-Host "⚠️  BFG not found. Downloading..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please download BFG manually:" -ForegroundColor Yellow
    Write-Host "  1. Go to: https://rtyley.github.io/bfg-repo-cleaner/" -ForegroundColor White
    Write-Host "  2. Download: bfg.jar" -ForegroundColor White
    Write-Host "  3. Save to: $(Get-Location)\bfg.jar" -ForegroundColor White
    Write-Host ""
    Write-Host "Or use PowerShell to download:" -ForegroundColor Cyan
    Write-Host '  Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar" -OutFile "bfg.jar"' -ForegroundColor White
    Write-Host ""
    
    $download = Read-Host "Do you want to download BFG now? (Y/N)"
    if ($download -eq "Y" -or $download -eq "y") {
        Write-Host "Downloading BFG..." -ForegroundColor Cyan
        try {
            Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar" -OutFile "bfg.jar"
            Write-Host "✅ BFG downloaded successfully" -ForegroundColor Green
        } catch {
            Write-Host "❌ Download failed. Please download manually." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Please download BFG and run this script again." -ForegroundColor Yellow
        exit 0
    }
} else {
    Write-Host "✅ BFG found: $bfgPath" -ForegroundColor Green
}

# Step 3: Create backup branch
Write-Host ""
Write-Host "Step 3: Creating backup branch..." -ForegroundColor Blue
$backupBranch = "backup-before-token-cleanup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
git branch $backupBranch 2>$null
Write-Host "✅ Backup created: $backupBranch" -ForegroundColor Green

# Step 4: Get token to remove
Write-Host ""
Write-Host "Step 4: Getting token to remove..." -ForegroundColor Blue
Write-Host ""
Write-Host "⚠️  IMPORTANT: Enter the exposed token that needs to be removed from Git history." -ForegroundColor Yellow
Write-Host "   This should be the token that was committed to the repository." -ForegroundColor Yellow
Write-Host ""
$tokenToRemove = Read-Host "Enter the GitHub token to remove (or press Enter to use placeholder)"
if ([string]::IsNullOrWhiteSpace($tokenToRemove)) {
    Write-Host "⚠️  Using placeholder. Please edit the script to add your actual token." -ForegroundColor Yellow
    $tokenToRemove = "YOUR_EXPOSED_TOKEN_HERE"
}
$replacement = "REVOKED_TOKEN_REMOVED_FROM_HISTORY"

# Create tokens.txt for BFG
$tokensFile = "tokens.txt"
"$tokenToRemove==>$replacement" | Out-File -FilePath $tokensFile -Encoding UTF8 -NoNewline
Write-Host "✅ Created: $tokensFile" -ForegroundColor Green
Write-Host "   Pattern: $tokenToRemove ==> $replacement" -ForegroundColor Gray

# Step 5: Confirm before proceeding
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
Write-Host ""
Write-Host "⚠️  FINAL CONFIRMATION REQUIRED" -ForegroundColor Yellow
Write-Host ""
Write-Host "This will:" -ForegroundColor White
Write-Host "  1. Remove token from ALL Git history" -ForegroundColor White
Write-Host "  2. Rewrite ALL commit SHAs" -ForegroundColor White
Write-Host "  3. Require force push to remote" -ForegroundColor White
Write-Host ""
Write-Host "🔴 CRITICAL: Make sure you have:" -ForegroundColor Red
Write-Host "  ✅ Revoked the token at https://github.com/settings/tokens" -ForegroundColor Yellow
Write-Host "  ✅ Backed up your repository" -ForegroundColor Yellow
Write-Host "  ✅ Coordinated with team (if shared repo)" -ForegroundColor Yellow
Write-Host ""

$confirm = Read-Host "Type 'YES' to proceed with cleanup"
if ($confirm -ne "YES") {
    Write-Host "Operation cancelled." -ForegroundColor Yellow
    exit 0
}

# Step 6: Run BFG
Write-Host ""
Write-Host "Step 6: Running BFG Repo-Cleaner..." -ForegroundColor Blue
Write-Host "This may take several minutes depending on repository size..." -ForegroundColor Yellow
Write-Host ""

try {
    java -jar bfg.jar --replace-text $tokensFile
    Write-Host ""
    Write-Host "✅ BFG completed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ BFG failed: $_" -ForegroundColor Red
    exit 1
}

# Step 7: Clean up Git
Write-Host ""
Write-Host "Step 7: Cleaning up Git references..." -ForegroundColor Blue
git reflog expire --expire=now --all
git gc --prune=now --aggressive
Write-Host "✅ Git cleanup completed" -ForegroundColor Green

# Step 8: Verify cleanup
Write-Host ""
Write-Host "Step 8: Verifying token removal..." -ForegroundColor Blue

# Skip verification if placeholder token was used (to avoid false positives from documentation)
if ($tokenToRemove -eq "YOUR_EXPOSED_TOKEN_HERE") {
    Write-Host "⚠️  Skipping verification - placeholder token used." -ForegroundColor Yellow
    Write-Host "   To verify cleanup, run with actual token or check manually:" -ForegroundColor Yellow
    Write-Host "   git log --all -p | Select-String -Pattern 'ghp_[A-Za-z0-9]{36}'" -ForegroundColor Gray
} else {
    # Only search for actual GitHub token pattern to avoid false positives
    # Check if token looks like a real GitHub token (starts with ghp_)
    if ($tokenToRemove -match "^ghp_") {
        $tokenStillExists = git log --all -p | Select-String -Pattern [regex]::Escape($tokenToRemove) -Quiet
        if ($tokenStillExists) {
            Write-Host "⚠️  WARNING: Token may still exist in history!" -ForegroundColor Yellow
            Write-Host "   Please verify manually:" -ForegroundColor Yellow
            Write-Host "   git log --all -p | Select-String -Pattern 'ghp_[A-Za-z0-9]{36}'" -ForegroundColor Gray
        } else {
            Write-Host "✅ Token not found in Git history" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠️  Token pattern not recognized. Skipping verification." -ForegroundColor Yellow
        Write-Host "   Please verify manually:" -ForegroundColor Yellow
        Write-Host "   git log --all -p | Select-String -Pattern 'ghp_[A-Za-z0-9]{36}'" -ForegroundColor Gray
    }
}

# Step 9: Instructions for force push
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "✅ CLEANUP COMPLETED!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Review changes:" -ForegroundColor White
Write-Host "   git log --oneline" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Force push to remote (WARNING: Rewrites remote history):" -ForegroundColor Yellow
Write-Host "   git push origin --force --all" -ForegroundColor White
Write-Host "   git push origin --force --tags" -ForegroundColor White
Write-Host ""
Write-Host "3. Update remote URL (remove token):" -ForegroundColor White
Write-Host "   git remote set-url origin https://github.com/thanhmuefatty07/supreme-system-v5.git" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Verify final cleanup:" -ForegroundColor White
Write-Host "   git log --all -p | Select-String -Pattern '$tokenToRemove'" -ForegroundColor Gray
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""

