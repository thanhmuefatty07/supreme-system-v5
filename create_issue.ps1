# create_issue.ps1
# Create GitHub Issue for metrics tracking

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "   GITHUB ISSUE CREATION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if (!(Test-Path "verification_results.json")) {
    Write-Host "❌ Error: verification_results.json not found" -ForegroundColor Red
    exit 1
}

$results = Get-Content "verification_results.json" | ConvertFrom-Json

try {
    gh --version | Out-Null
    Write-Host "✅ GitHub CLI detected" -ForegroundColor Green
} catch {
    Write-Host "❌ GitHub CLI not found" -ForegroundColor Red
    Write-Host "   Install: winget install GitHub.cli" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

$issueTitle = "🔍 Metrics Verification - $(Get-Date -Format 'yyyy-MM-dd')"

$issueBody = @"
## 📊 Verification Summary

**Date:** $($results.timestamp)

### Metrics Comparison

| Metric | README | Actual | Status |
|--------|--------|--------|--------|
| Tests | $($results.claimed.tests) | $($results.actual.tests) | $(if ($results.status.tests -eq 'MATCH') { '✅' } elseif ($results.status.tests -eq 'BETTER') { '✅ Better' } else { '⚠️ Overclaimed' }) |
| Coverage | $($results.claimed.coverage)% | $($results.actual.coverage)% | $(if ($results.status.coverage -eq 'MATCH') { '✅' } elseif ($results.status.coverage -eq 'BETTER') { '✅ Better' } else { '⚠️ Overclaimed' }) |

### Test Results
- ✅ Passed: $($results.actual.passed)
- ❌ Failed: $($results.actual.failed)
- ⏭️ Skipped: $($results.actual.skipped)
- 💥 Errors: $($results.actual.errors)

### Action Items
- [ ] Review coverage report
- [ ] Update README if needed
- [ ] Plan to reach 80% coverage
- [ ] Close issue when done
"@

Write-Host "Creating issue..." -ForegroundColor Yellow
$issueUrl = gh issue create --title $issueTitle --body $issueBody --label "metrics,verification" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Issue created: $issueUrl" -ForegroundColor Green
    Start-Process $issueUrl
} else {
    Write-Host "❌ Failed to create issue" -ForegroundColor Red
}

Write-Host ""



