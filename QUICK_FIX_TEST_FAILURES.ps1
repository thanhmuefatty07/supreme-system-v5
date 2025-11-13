# Quick Fix Script for Test Failures (PowerShell)
# Date: 2025-11-13

Write-Host ""
Write-Host "🔧 Quick Fix: Installing Missing Dependencies" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# Install missing dependencies
pip install numba psutil memory-profiler pytest-mock pytest-timeout

Write-Host ""
Write-Host "✅ Dependencies installed" -ForegroundColor Green
Write-Host ""
Write-Host "🔍 Running tests to identify failures..." -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# Run tests and collect failures
python -m pytest tests/ --tb=no -q 2>&1 | Select-String -Pattern "FAILED|ERROR" | Out-File -FilePath "test_failures.txt" -Encoding UTF8

$failedCount = (Get-Content "test_failures.txt" | Select-String -Pattern "FAILED").Count
$errorCount = (Get-Content "test_failures.txt" | Select-String -Pattern "ERROR").Count

Write-Host ""
Write-Host "Found: $failedCount failed tests, $errorCount errors" -ForegroundColor Yellow
Write-Host ""
Write-Host "📋 Failure summary saved to: test_failures.txt" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review test_failures.txt" -ForegroundColor White
Write-Host "  2. Fix import errors" -ForegroundColor White
Write-Host "  3. Fix path issues" -ForegroundColor White
Write-Host "  4. Re-run tests: pytest tests/ -v" -ForegroundColor White
Write-Host ""

