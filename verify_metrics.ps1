# verify_metrics.ps1
# Script to verify actual test count and coverage metrics

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "   SUPREME SYSTEM V5 - METRICS VERIFICATION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the right directory
if (!(Test-Path "pytest.ini")) {
    Write-Host "[ERROR] pytest.ini not found. Are you in the repo root?" -ForegroundColor Red
    exit 1
}

# Step 1: Collect test count
Write-Host "[1/4] Collecting test count..." -ForegroundColor Yellow
try {
    $testCollection = pytest tests/ --collect-only -q 2>&1 | Out-String
    $testCountMatch = [regex]::Match($testCollection, "collected (\d+) item")
    if ($testCountMatch.Success) {
        $testCount = [int]$testCountMatch.Groups[1].Value
        Write-Host "[OK] Tests collected: $testCount" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Could not parse test count from pytest output" -ForegroundColor Yellow
        $testCount = 0
    }
} catch {
    Write-Host "[ERROR] Error collecting tests: $_" -ForegroundColor Red
    $testCount = 0
}

Write-Host ""

# Step 2: Run tests with coverage
Write-Host "[2/4] Running tests with coverage..." -ForegroundColor Yellow
Write-Host "This may take 2-5 minutes..." -ForegroundColor Gray
Write-Host ""

try {
    $startTime = Get-Date
    $coverageOutput = pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-report=json -q 2>&1 | Tee-Object -FilePath "test_results.txt" | Out-String
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Host "[OK] Tests completed in $([math]::Round($duration, 1)) seconds" -ForegroundColor Green
    Write-Host ""
    
    # Parse results
    $passedTests = ([regex]::Matches($coverageOutput, "PASSED")).Count
    $failedTests = ([regex]::Matches($coverageOutput, "FAILED")).Count
    $skippedTests = ([regex]::Matches($coverageOutput, "SKIPPED")).Count
    $errorTests = ([regex]::Matches($coverageOutput, "ERROR")).Count
    
    Write-Host "Test Results:" -ForegroundColor Cyan
    Write-Host "  [PASS]   $passedTests" -ForegroundColor Green
    Write-Host "  [FAIL]   $failedTests" -ForegroundColor $(if ($failedTests -eq 0) { "Green" } else { "Red" })
    Write-Host "  [SKIP]   $skippedTests" -ForegroundColor Yellow
    Write-Host "  [ERROR]  $errorTests" -ForegroundColor $(if ($errorTests -eq 0) { "Green" } else { "Red" })
    Write-Host ""
    
} catch {
    Write-Host "[ERROR] Error running tests: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Parse coverage from JSON
Write-Host "[3/4] Parsing coverage report..." -ForegroundColor Yellow
try {
    if (Test-Path "coverage.json") {
        $coverageJson = Get-Content "coverage.json" | ConvertFrom-Json
        $coveragePercent = [math]::Round($coverageJson.totals.percent_covered, 2)
        Write-Host "[OK] Total Coverage: $coveragePercent%" -ForegroundColor Green
    } else {
        $coverageLine = $coverageOutput | Select-String -Pattern "TOTAL.*?(\d+)%" | Select-Object -Last 1
        if ($coverageLine) {
            $coveragePercent = [regex]::Match($coverageLine, "(\d+)%").Groups[1].Value
            Write-Host "[OK] Total Coverage: $coveragePercent%" -ForegroundColor Green
        } else {
            Write-Host "[WARN] Could not parse coverage percentage" -ForegroundColor Yellow
            $coveragePercent = 0
        }
    }
} catch {
    Write-Host "[ERROR] Error parsing coverage: $_" -ForegroundColor Red
    $coveragePercent = 0
}

Write-Host ""

# Step 4: Compare with README claims
Write-Host "[4/4] Comparing with README claims..." -ForegroundColor Yellow
Write-Host ""

$readmeClaims = @{
    Tests = 474
    Coverage = 27
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "                  VERIFICATION RESULTS                          " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "README CLAIMS:" -ForegroundColor Yellow
Write-Host "  Tests:     $($readmeClaims.Tests)" -ForegroundColor White
Write-Host "  Coverage:  $($readmeClaims.Coverage)%" -ForegroundColor White
Write-Host ""

Write-Host "ACTUAL RESULTS:" -ForegroundColor Yellow
Write-Host "  Tests:     $testCount" -ForegroundColor White
Write-Host "  Coverage:  $coveragePercent%" -ForegroundColor White
if ($testCount -gt 0) {
    Write-Host "  Pass Rate: $([math]::Round(($passedTests / $testCount) * 100, 1))%" -ForegroundColor White
}
Write-Host ""

# Calculate differences
$testDiff = $testCount - $readmeClaims.Tests
$coverageDiff = $coveragePercent - $readmeClaims.Coverage

Write-Host "ANALYSIS:" -ForegroundColor Yellow

if ($testDiff -eq 0) {
    Write-Host "  [OK] Test count: EXACT MATCH" -ForegroundColor Green
    $testStatus = "MATCH"
} elseif ($testDiff -gt 0) {
    Write-Host "  [OK] Test count: +$testDiff tests (BETTER)" -ForegroundColor Green
    $testStatus = "BETTER"
} else {
    Write-Host "  [WARN] Test count: $testDiff tests (OVERCLAIMED)" -ForegroundColor Red
    $testStatus = "OVERCLAIMED"
}

if ([Math]::Abs($coverageDiff) -lt 1) {
    Write-Host "  [OK] Coverage: Match (within 1%)" -ForegroundColor Green
    $coverageStatus = "MATCH"
} elseif ($coverageDiff -gt 0) {
    Write-Host "  [OK] Coverage: +$([math]::Round($coverageDiff, 2))% (BETTER)" -ForegroundColor Green
    $coverageStatus = "BETTER"
} else {
    Write-Host "  [WARN] Coverage: $([math]::Round($coverageDiff, 2))% (OVERCLAIMED)" -ForegroundColor Red
    $coverageStatus = "OVERCLAIMED"
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Save results to JSON
$results = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    actual = @{
        tests = $testCount
        coverage = $coveragePercent
        passed = $passedTests
        failed = $failedTests
        skipped = $skippedTests
        errors = $errorTests
    }
    claimed = $readmeClaims
    differences = @{
        tests = $testDiff
        coverage = $coverageDiff
    }
    status = @{
        tests = $testStatus
        coverage = $coverageStatus
    }
}

$results | ConvertTo-Json -Depth 10 | Set-Content "verification_results.json"

Write-Host "Files Generated:" -ForegroundColor Cyan
Write-Host "  - test_results.txt" -ForegroundColor Gray
Write-Host "  - coverage.json" -ForegroundColor Gray
Write-Host "  - htmlcov/index.html" -ForegroundColor Gray
Write-Host "  - verification_results.json" -ForegroundColor Gray
Write-Host ""

if ($testStatus -eq "OVERCLAIMED" -or $coverageStatus -eq "OVERCLAIMED") {
    Write-Host "[ACTION REQUIRED] Update README with actual metrics" -ForegroundColor Red
    Write-Host "   Run: .\update_readme.ps1" -ForegroundColor Yellow
} else {
    Write-Host "[SUCCESS] Metrics match or exceed claims!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open htmlcov/index.html to view coverage details" -ForegroundColor White
Write-Host "  2. Run .\update_readme.ps1 to update README (if needed)" -ForegroundColor White
Write-Host "  3. Run .\create_issue.ps1 to create GitHub tracking issue" -ForegroundColor White
Write-Host ""



