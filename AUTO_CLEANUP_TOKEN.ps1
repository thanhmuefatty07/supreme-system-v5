# ============================================================================
# 🚨 AUTOMATED TOKEN CLEANUP FROM GIT HISTORY
# 
# This script automates the entire token cleanup process:
# 1. Checks prerequisites (Java, BFG)
# 2. Installs missing dependencies
# 3. Creates backup
# 4. Runs cleanup
# 5. Verifies cleanup
# 6. Provides force push instructions
# 
# Date: 2025-11-13
# ============================================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "🚨 AUTOMATED TOKEN CLEANUP FROM GIT HISTORY" -ForegroundColor Red
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
Write-Host ""

# Token to remove (from Git history)
$tokenToRemove = "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"
$replacement = "REVOKED_TOKEN_REMOVED_FROM_HISTORY"

# Step 1: Critical Warning
Write-Host "⚠️  CRITICAL WARNING" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host ""
Write-Host "This script will:" -ForegroundColor White
Write-Host "  1. Remove token from ALL Git history" -ForegroundColor White
Write-Host "  2. Rewrite ALL commit SHAs" -ForegroundColor White
Write-Host "  3. Require force push to remote" -ForegroundColor White
Write-Host ""
Write-Host "🔴 BEFORE PROCEEDING:" -ForegroundColor Red
Write-Host "  ✅ Revoke token at: https://github.com/settings/tokens" -ForegroundColor Yellow
Write-Host "  ✅ Backup repository (clone to another location)" -ForegroundColor Yellow
Write-Host "  ✅ Notify team (if shared repository)" -ForegroundColor Yellow
Write-Host ""

$confirm = Read-Host "Have you revoked the token and backed up the repository? (YES/NO)"
if ($confirm -ne "YES") {
    Write-Host ""
    Write-Host "❌ Operation cancelled. Please revoke token and backup first." -ForegroundColor Red
    Write-Host ""
    Write-Host "Steps:" -ForegroundColor Cyan
    Write-Host "  1. Go to: https://github.com/settings/tokens" -ForegroundColor White
    Write-Host "  2. Find and revoke token: $tokenToRemove" -ForegroundColor White
    Write-Host "  3. Clone backup: git clone https://github.com/thanhmuefatty07/supreme-system-v5.git backup-repo" -ForegroundColor White
    Write-Host "  4. Run this script again" -ForegroundColor White
    Write-Host ""
    exit 0
}

# Step 2: Check and Install Java
Write-Host ""
Write-Host "Step 1: Checking Java installation..." -ForegroundColor Blue
try {
    $javaVersion = java -version 2>&1 | Select-Object -First 1
    Write-Host "✅ Java found: $javaVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Java not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Attempting to install Java..." -ForegroundColor Yellow
    
    # Try Chocolatey
    try {
        Write-Host "Trying Chocolatey..." -ForegroundColor Cyan
        choco install openjdk -y
        Write-Host "✅ Java installed via Chocolatey" -ForegroundColor Green
    } catch {
        # Try Winget
        try {
            Write-Host "Trying Winget..." -ForegroundColor Cyan
            winget install Microsoft.OpenJDK.17
            Write-Host "✅ Java installed via Winget" -ForegroundColor Green
        } catch {
            Write-Host ""
            Write-Host "❌ Automatic installation failed!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Please install Java manually:" -ForegroundColor Yellow
            Write-Host "  1. Go to: https://www.java.com/download/" -ForegroundColor White
            Write-Host "  2. Download and install Java" -ForegroundColor White
            Write-Host "  3. Restart PowerShell" -ForegroundColor White
            Write-Host "  4. Run this script again" -ForegroundColor White
            Write-Host ""
            exit 1
        }
    }
}

# Step 3: Download BFG
Write-Host ""
Write-Host "Step 2: Checking BFG Repo-Cleaner..." -ForegroundColor Blue
$bfgPath = "bfg.jar"
if (-not (Test-Path $bfgPath)) {
    Write-Host "⚠️  BFG not found. Downloading..." -ForegroundColor Yellow
    try {
        $bfgUrl = "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar"
        Write-Host "Downloading from: $bfgUrl" -ForegroundColor Cyan
        Invoke-WebRequest -Uri $bfgUrl -OutFile $bfgPath
        Write-Host "✅ BFG downloaded successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Download failed: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please download BFG manually:" -ForegroundColor Yellow
        Write-Host "  1. Go to: https://rtyley.github.io/bfg-repo-cleaner/" -ForegroundColor White
        Write-Host "  2. Download: bfg.jar" -ForegroundColor White
        Write-Host "  3. Save to: $(Get-Location)\bfg.jar" -ForegroundColor White
        Write-Host "  4. Run this script again" -ForegroundColor White
        Write-Host ""
        exit 1
    }
} else {
    Write-Host "✅ BFG found: $bfgPath" -ForegroundColor Green
}

# Step 4: Create backup branch
Write-Host ""
Write-Host "Step 3: Creating backup branch..." -ForegroundColor Blue
$backupBranch = "backup-before-token-cleanup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
try {
    git branch $backupBranch 2>$null
    Write-Host "✅ Backup created: $backupBranch" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Backup branch creation failed (may already exist)" -ForegroundColor Yellow
}

# Step 5: Create tokens.txt
Write-Host ""
Write-Host "Step 4: Creating tokens.txt file..." -ForegroundColor Blue
$tokensFile = "tokens.txt"
"$tokenToRemove==>$replacement" | Out-File -FilePath $tokensFile -Encoding UTF8 -NoNewline
Write-Host "✅ Created: $tokensFile" -ForegroundColor Green
Write-Host "   Pattern: $tokenToRemove ==> $replacement" -ForegroundColor Gray

# Step 6: Final confirmation
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
Write-Host ""
Write-Host "⚠️  FINAL CONFIRMATION" -ForegroundColor Yellow
Write-Host ""
Write-Host "Ready to remove token from Git history:" -ForegroundColor White
Write-Host "  Token: $tokenToRemove" -ForegroundColor Gray
Write-Host ""
Write-Host "This will:" -ForegroundColor White
Write-Host "  • Remove token from ALL commits" -ForegroundColor White
Write-Host "  • Rewrite ALL commit SHAs" -ForegroundColor White
Write-Host "  • Require force push to remote" -ForegroundColor White
Write-Host ""

$finalConfirm = Read-Host "Type 'CLEANUP' to proceed with cleanup"
if ($finalConfirm -ne "CLEANUP") {
    Write-Host ""
    Write-Host "❌ Operation cancelled." -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

# Step 7: Run BFG
Write-Host ""
Write-Host "Step 5: Running BFG Repo-Cleaner..." -ForegroundColor Blue
Write-Host "This may take several minutes depending on repository size..." -ForegroundColor Yellow
Write-Host ""

try {
    java -jar bfg.jar --replace-text $tokensFile
    Write-Host ""
    Write-Host "✅ BFG completed successfully" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "❌ BFG failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check Java installation: java -version" -ForegroundColor White
    Write-Host "  2. Check BFG file exists: Test-Path bfg.jar" -ForegroundColor White
    Write-Host "  3. Check tokens.txt exists: Test-Path tokens.txt" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Step 8: Clean up Git references
Write-Host ""
Write-Host "Step 6: Cleaning up Git references..." -ForegroundColor Blue
try {
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive
    Write-Host "✅ Git cleanup completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Git cleanup warning: $_" -ForegroundColor Yellow
}

# Step 9: Verify cleanup
Write-Host ""
Write-Host "Step 7: Verifying token removal..." -ForegroundColor Blue

$tokenStillExists = git log --all -p | Select-String -Pattern [regex]::Escape($tokenToRemove) -Quiet
if ($tokenStillExists) {
    Write-Host "⚠️  WARNING: Token may still exist in history!" -ForegroundColor Yellow
    Write-Host "   Please verify manually:" -ForegroundColor Yellow
    Write-Host "   git log --all -p | Select-String -Pattern 'ghp_[A-Za-z0-9]{36}'" -ForegroundColor Gray
} else {
    Write-Host "✅ Token not found in Git history" -ForegroundColor Green
}

# Step 10: Instructions for force push
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
Write-Host "3. Verify remote URL is clean:" -ForegroundColor White
Write-Host "   git remote get-url origin" -ForegroundColor Gray
Write-Host "   # Expected: https://github.com/thanhmuefatty07/supreme-system-v5.git" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Final verification:" -ForegroundColor White
Write-Host "   git log --all -p | Select-String -Pattern 'ghp_[A-Za-z0-9]{36}'" -ForegroundColor Gray
Write-Host "   # Expected: No matches" -ForegroundColor Gray
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""

