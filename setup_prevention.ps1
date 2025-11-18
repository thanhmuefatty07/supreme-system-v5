# Setup Prevention Measures for API Key Leaks
# PowerShell version

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
Write-Host "  Setting Up Prevention Measures" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
Write-Host ""

# Step 1: Install Pre-commit Hook
Write-Host "Step 1: Installing Pre-commit Hook" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray

$hookDir = ".git\hooks"
if (-not (Test-Path $hookDir)) {
    New-Item -ItemType Directory -Path $hookDir -Force | Out-Null
}

# Create PowerShell pre-commit hook
$psHookContent = @'
# Pre-commit hook to prevent API key leaks (PowerShell)

$stagedFiles = git diff --cached --name-only

# Check for Gemini API keys
$geminiPattern = "AIzaSy[A-Za-z0-9_-]{33}"
foreach ($file in $stagedFiles) {
    if ($file) {
        $content = git show ":$file" 2>$null
        if ($content -match $geminiPattern) {
            Write-Host "ERROR: Gemini API key detected in commit!" -ForegroundColor Red
            Write-Host "File: $file" -ForegroundColor Yellow
            Write-Host "Please remove API keys before committing." -ForegroundColor Yellow
            exit 1
        }
    }
}

# Check for generic API key patterns
$apiKeyPattern = 'api[_-]?key["'']?\s*[:=]\s*["''][a-zA-Z0-9]{20,}'
foreach ($file in $stagedFiles) {
    if ($file) {
        $content = git show ":$file" 2>$null
        if ($content -match $apiKeyPattern) {
            Write-Host "ERROR: API key pattern detected!" -ForegroundColor Red
            Write-Host "File: $file" -ForegroundColor Yellow
            exit 1
        }
    }
}

# Check for sensitive files
$sensitivePattern = "\.env$|\.key$|\.secret$|credentials"
$sensitiveFiles = $stagedFiles | Where-Object { $_ -match $sensitivePattern }
if ($sensitiveFiles) {
    Write-Host "ERROR: Sensitive file detected in commit!" -ForegroundColor Red
    Write-Host "Files:" -ForegroundColor Yellow
    $sensitiveFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    exit 1
}

exit 0
'@

$psHookPath = Join-Path $hookDir "pre-commit.ps1"
$psHookContent | Out-File -FilePath $psHookPath -Encoding UTF8

# Create batch wrapper for Git
$batchHook = @'
@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0pre-commit.ps1"
if %ERRORLEVEL% NEQ 0 exit %ERRORLEVEL%
'@

$batchHookPath = Join-Path $hookDir "pre-commit"
$batchHook | Out-File -FilePath $batchHookPath -Encoding ASCII

Write-Host "Pre-commit hook installed" -ForegroundColor Green
Write-Host ""

# Step 2: Update .gitignore
Write-Host "Step 2: Updating .gitignore" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray

$gitignoreAdditions = @"

# Gemini & AI API Keys
.gemini_api_key
gemini_credentials.json
ai_keys/
*.gemini

# Generic secrets
*.key
*.secret
*.pem
credentials/
secrets/
.env*
!.env.example

# Logs that might contain keys
*.log
logs/
debug.log
"@

if (Test-Path ".gitignore") {
    $currentContent = Get-Content ".gitignore" -Raw
    if ($currentContent -notmatch "Gemini & AI API Keys") {
        Add-Content -Path ".gitignore" -Value $gitignoreAdditions
        Write-Host ".gitignore updated" -ForegroundColor Green
    } else {
        Write-Host ".gitignore already has prevention rules" -ForegroundColor Green
    }
} else {
    $gitignoreAdditions | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Host ".gitignore created" -ForegroundColor Green
}

Write-Host ""

# Step 3: Create .env.example
Write-Host "Step 3: Creating .env.example" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray

$envExample = @'
# Gemini API Configuration
# Copy this to .env and add your actual key
# NEVER commit .env to Git!

GEMINI_API_KEY=your_gemini_api_key_here

# Other configurations
PROJECT_ENV=development
LOG_LEVEL=INFO

# Binance API (for trading)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_secret_here
BINANCE_TESTNET=true

# Bybit API (alternative exchange)
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_SECRET_KEY=your_bybit_secret_here
BYBIT_TESTNET=true
'@

if (-not (Test-Path ".env.example")) {
    $envExample | Out-File -FilePath ".env.example" -Encoding UTF8
    Write-Host ".env.example created" -ForegroundColor Green
} else {
    Write-Host ".env.example already exists" -ForegroundColor Green
}

Write-Host ""

# Step 4: Test hook
Write-Host "Step 4: Testing pre-commit hook" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Gray

$testFile = "test_api_key.txt"
"API_KEY=AIzaSyB5v7LHHgdj7AMpi8Drngi7UsRhb4tLvcE" | Out-File -FilePath $testFile -Encoding UTF8
git add $testFile 2>$null

Write-Host "Testing hook (should block commit)..." -ForegroundColor White
& $psHookPath 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Hook is working! (correctly blocked test commit)" -ForegroundColor Green
} else {
    Write-Host "Hook may not be working correctly" -ForegroundColor Yellow
}

git reset HEAD $testFile 2>$null
Remove-Item $testFile -Force -ErrorAction SilentlyContinue

Write-Host ""

Write-Host "=========================================================" -ForegroundColor Green
Write-Host "PREVENTION SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Protection measures installed:" -ForegroundColor Cyan
Write-Host "  Pre-commit hook (blocks API key commits)" -ForegroundColor White
Write-Host "  Enhanced .gitignore" -ForegroundColor White
Write-Host "  .env.example template" -ForegroundColor White
Write-Host ""
Write-Host "Next: Commit these changes" -ForegroundColor Yellow
Write-Host "  git add .gitignore .env.example" -ForegroundColor Gray
Write-Host "  git commit -m 'security: Add prevention measures for API keys'" -ForegroundColor Gray
Write-Host "  git push origin main" -ForegroundColor Gray
Write-Host ""
