# Emergency: Clean Gemini API Keys from Git History
# PowerShell version for Windows

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Red
Write-Host "  EMERGENCY: Cleaning Gemini API Keys from Git History" -ForegroundColor Red
Write-Host "=========================================================" -ForegroundColor Red
Write-Host ""

# Step 0: Verify keys are revoked
Write-Host "Step 0: Verify keys are revoked" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray
Write-Host "IMPORTANT: Have you revoked these keys in Google Cloud Console?" -ForegroundColor Yellow
Write-Host "   - AIzaSyB5v7LHHgdj7AMpi8Drngi7UsRhb4tLvcE" -ForegroundColor White
Write-Host "   - AIzaSyAgakXQVcSD5BadqMsNwxgZ86qs01natAI" -ForegroundColor White
Write-Host ""
$revoked = Read-Host "Have you revoked the keys? (yes/no)"

if ($revoked -ne "yes") {
    Write-Host "Please revoke keys first before proceeding!" -ForegroundColor Red
    Write-Host "Go to: https://console.cloud.google.com/apis/credentials" -ForegroundColor Yellow
    exit 1
}

Write-Host "Proceeding with cleanup..." -ForegroundColor Green
Write-Host ""

# Step 1: Backup
Write-Host "Step 1: Backup repository" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray
$backupName = "supreme-system-v5-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$backupPath = Join-Path ".." $backupName

if (Test-Path $backupPath) {
    Write-Host "Backup already exists: $backupPath" -ForegroundColor Yellow
} else {
    Write-Host "Creating backup..." -ForegroundColor White
    Copy-Item -Path "." -Destination $backupPath -Recurse -Exclude ".git"
    Write-Host "Backup created: $backupPath" -ForegroundColor Green
}
Write-Host ""

# Step 2: Check for Java/BFG
Write-Host "Step 2: Checking cleanup method" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray

$useBFG = $false
if (Get-Command java -ErrorAction SilentlyContinue) {
    Write-Host "Java found - Will try to use BFG Repo-Cleaner" -ForegroundColor Green
    $useBFG = $true
} else {
    Write-Host "Java not found - Will use git filter-branch" -ForegroundColor Yellow
}

Write-Host ""

# Step 3: Create replacement file
Write-Host "Step 3: Create replacement file" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray

$keysFile = "keys_to_remove.txt"
$keysContent = "AIzaSyB5v7LHHgdj7AMpi8Drngi7UsRhb4tLvcE==>***REMOVED***`nAIzaSyAgakXQVcSD5BadqMsNwxgZ86qs01natAI==>***REMOVED***"
$keysContent | Out-File -FilePath $keysFile -Encoding UTF8

Write-Host "Replacement file created: $keysFile" -ForegroundColor Green
Write-Host ""

# Step 4: Clean with BFG or git filter-branch
Write-Host "Step 4: Cleaning Git history" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray
Write-Host "This may take several minutes..." -ForegroundColor White
Write-Host ""

if ($useBFG) {
    $bfgPath = Join-Path $PWD "bfg-1.14.0.jar"
    if (-not (Test-Path $bfgPath)) {
        Write-Host "Downloading BFG Repo-Cleaner..." -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar" -OutFile $bfgPath
            Write-Host "BFG downloaded" -ForegroundColor Green
        } catch {
            Write-Host "Could not download BFG. Using git filter-branch instead." -ForegroundColor Yellow
            $useBFG = $false
        }
    }
    
    if ($useBFG -and (Test-Path $bfgPath)) {
        Write-Host "Using BFG Repo-Cleaner (recommended)..." -ForegroundColor Green
        java -jar $bfgPath --replace-text $keysFile .git 2>&1 | Out-Null
        Write-Host "BFG completed" -ForegroundColor Green
    }
}

if (-not $useBFG) {
    Write-Host "Using git filter-branch..." -ForegroundColor Yellow
    Write-Host "Removing file containing keys..." -ForegroundColor White
    
    # Remove the file that contains keys
    git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch RUN_OPTIMIZER.sh 2>$null || exit 0' --prune-empty --tag-name-filter cat -- --all 2>$null
    
    Write-Host "Git filter-branch completed" -ForegroundColor Green
}

Write-Host ""

# Step 5: Clean up Git
Write-Host "Step 5: Clean up Git" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray

Remove-Item -Path ".git\refs\original" -Recurse -Force -ErrorAction SilentlyContinue
git reflog expire --expire=now --all 2>$null
git gc --prune=now --aggressive 2>$null

Write-Host "Git cleaned" -ForegroundColor Green
Write-Host ""

# Step 6: Verify
Write-Host "Step 6: Verify keys removed" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray

$remaining1 = git log --all -S"AIzaSyB5v7LHHgdj7AMpi8Drngi7UsRhb4tLvcE" --oneline 2>$null
if (-not $remaining1) {
    Write-Host "Key 1: REMOVED" -ForegroundColor Green
} else {
    Write-Host "Key 1: Still found" -ForegroundColor Yellow
    Write-Host "   Commits: $remaining1" -ForegroundColor Gray
}

$remaining2 = git log --all -S"AIzaSyAgakXQVcSD5BadqMsNwxgZ86qs01natAI" --oneline 2>$null
if (-not $remaining2) {
    Write-Host "Key 2: REMOVED" -ForegroundColor Green
} else {
    Write-Host "Key 2: Still found" -ForegroundColor Yellow
    Write-Host "   Commits: $remaining2" -ForegroundColor Gray
}

Write-Host ""

# Step 7: Force push
Write-Host "Step 7: Force push to GitHub" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray
Write-Host "WARNING: This will REWRITE GitHub history!" -ForegroundColor Red
Write-Host ""
$ready = Read-Host "Ready to force push? (yes/no)"

if ($ready -eq "yes") {
    Write-Host "Pushing to GitHub..." -ForegroundColor White
    git push origin --force --all 2>&1
    git push origin --force --tags 2>&1
    Write-Host "Pushed to GitHub" -ForegroundColor Green
} else {
    Write-Host "Skipped push. Run manually:" -ForegroundColor Yellow
    Write-Host "   git push origin --force --all" -ForegroundColor Gray
    Write-Host "   git push origin --force --tags" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
Write-Host "CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Verify on GitHub that keys are gone" -ForegroundColor White
Write-Host "2. Generate new Gemini API keys" -ForegroundColor White
Write-Host "3. Add to .env (NOT committed to Git)" -ForegroundColor White
Write-Host "4. Install pre-commit hooks to prevent future leaks" -ForegroundColor White
Write-Host ""
