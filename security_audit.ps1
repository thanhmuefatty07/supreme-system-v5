# Comprehensive Security Audit Script
# Checks Git history for sensitive information

$ErrorActionPreference = "Continue"

# Colors
function Write-Red { param($text) Write-Host $text -ForegroundColor Red }
function Write-Yellow { param($text) Write-Host $text -ForegroundColor Yellow }
function Write-Green { param($text) Write-Host $text -ForegroundColor Green }
function Write-Cyan { param($text) Write-Host $text -ForegroundColor Cyan }

Write-Host ""
Write-Cyan "COMPREHENSIVE SECURITY AUDIT"
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

$ISSUES_FOUND = 0

# Function to search commits
function Search-Commits {
    param(
        [string]$Pattern,
        [string]$Description,
        [string]$Severity = "MEDIUM"
    )
    
    Write-Host "─────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "Searching: $Description" -ForegroundColor White
    Write-Host "─────────────────────────────────────────" -ForegroundColor Gray
    
    try {
        $results = git log --all --oneline --grep="$Pattern" 2>$null | Select-Object -First 10
        
        if ($results) {
            if ($Severity -eq "CRITICAL") {
                Write-Red "🚨 CRITICAL: FOUND!"
                $script:ISSUES_FOUND++
            } elseif ($Severity -eq "HIGH") {
                Write-Yellow "⚠️  HIGH: FOUND!"
                $script:ISSUES_FOUND++
            } else {
                Write-Yellow "⚠️  FOUND:"
            }
            $results | ForEach-Object { Write-Host $_ }
            Write-Host ""
            return $true
        } else {
            Write-Green "✅ Clean"
            Write-Host ""
            return $false
        }
    } catch {
        Write-Green "✅ Clean (no matches)"
        Write-Host ""
        return $false
    }
}

# Function to search commit content
function Search-Content {
    param(
        [string]$Pattern,
        [string]$Description,
        [string]$Severity = "MEDIUM"
    )
    
    Write-Host "─────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "Searching content: $Description" -ForegroundColor White
    Write-Host "─────────────────────────────────────────" -ForegroundColor Gray
    
    try {
        $results = git log --all -S"$Pattern" --oneline 2>$null | Select-Object -First 10
        
        if ($results) {
            if ($Severity -eq "CRITICAL") {
                Write-Red "🚨 CRITICAL: FOUND!"
                $script:ISSUES_FOUND++
            } elseif ($Severity -eq "HIGH") {
                Write-Yellow "⚠️  HIGH: FOUND!"
                $script:ISSUES_FOUND++
            } else {
                Write-Yellow "⚠️  FOUND:"
            }
            $results | ForEach-Object { Write-Host $_ }
            Write-Host ""
            return $true
        } else {
            Write-Green "✅ Clean"
            Write-Host ""
            return $false
        }
    } catch {
        Write-Green "✅ Clean (no matches)"
        Write-Host ""
        return $false
    }
}

Write-Cyan "PHASE 1: CRITICAL SECURITY ISSUES"
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Critical: API Keys & Secrets
Search-Content "api_key" "API Keys" "CRITICAL"
Search-Content "API_KEY" "API_KEY (uppercase)" "CRITICAL"
Search-Content "secret_key" "Secret Keys" "CRITICAL"
Search-Content "SECRET_KEY" "SECRET_KEY (uppercase)" "CRITICAL"
Search-Content "password" "Passwords" "CRITICAL"
Search-Content "PASSWORD" "PASSWORD (uppercase)" "CRITICAL"
Search-Content "token" "Auth Tokens" "CRITICAL"
Search-Content "TOKEN" "TOKEN (uppercase)" "CRITICAL"
Search-Content "sk-" "OpenAI/Stripe API Keys" "CRITICAL"
Search-Content "pk_" "Stripe Public Keys" "CRITICAL"

Write-Host ""
Write-Cyan "PHASE 2: PRICING & VALUE INFORMATION"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# High: Pricing/Value
Search-Commits "\$[0-9]" "Dollar amounts" "HIGH"
Search-Commits "value.*\$" "Value mentions with $" "HIGH"
Search-Commits "price" "Price mentions" "HIGH"
Search-Commits "worth" "Worth mentions" "HIGH"
Search-Commits "cost.*\$" "Cost with amounts" "HIGH"
Search-Commits "[0-9]+K.*value" "Value in thousands" "HIGH"
Search-Commits "sale.*[0-9]" "Sale prices" "HIGH"
Search-Commits "revenue" "Revenue mentions" "HIGH"
Search-Content "\$[0-9]" "Dollar amounts in content" "HIGH"

Write-Host ""
Write-Cyan "PHASE 3: PERSONAL INFORMATION"
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Personal info
Search-Commits "@gmail\.com" "Gmail addresses" "HIGH"
Search-Commits "@.*\.com" "Email addresses" "HIGH"
Search-Content "@gmail.com" "Gmail in content" "HIGH"
Search-Content "@.*\.com" "Email in content" "HIGH"
Search-Commits "phone.*[0-9]" "Phone numbers" "MEDIUM"
Search-Commits "address.*[0-9]" "Physical addresses" "MEDIUM"

Write-Host ""
Write-Cyan "PHASE 4: CREDENTIALS & DATABASE"
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Database & Credentials
Search-Content "DB_PASSWORD" "Database passwords" "CRITICAL"
Search-Content "mongodb://" "MongoDB URLs" "CRITICAL"
Search-Content "postgres://" "PostgreSQL URLs" "CRITICAL"
Search-Content "mysql://" "MySQL URLs" "CRITICAL"
Search-Content "redis://" "Redis URLs" "CRITICAL"

Write-Host ""
Write-Cyan "PHASE 5: CLOUD & SERVICE CREDENTIALS"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Cloud credentials
Search-Content "AWS_ACCESS" "AWS credentials" "CRITICAL"
Search-Content "AZURE_" "Azure credentials" "CRITICAL"
Search-Content "GCP_" "Google Cloud credentials" "CRITICAL"
Search-Content "HEROKU_" "Heroku credentials" "CRITICAL"

Write-Host ""
Write-Cyan "PHASE 6: TRADING & EXCHANGE"
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

# Trading specific
Search-Content "binance.*key" "Binance API keys" "CRITICAL"
Search-Content "BINANCE.*KEY" "BINANCE API keys" "CRITICAL"
Search-Content "coinbase.*key" "Coinbase API keys" "CRITICAL"
Search-Content "kraken.*key" "Kraken API keys" "CRITICAL"
Search-Content "exchange.*secret" "Exchange secrets" "CRITICAL"
Search-Content "bybit.*key" "Bybit API keys" "CRITICAL"

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Cyan "AUDIT SUMMARY"
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

if ($ISSUES_FOUND -eq 0) {
    Write-Green "✅ NO CRITICAL ISSUES FOUND!"
    Write-Host "Your repository appears clean and safe." -ForegroundColor White
} else {
    Write-Red "[WARNING] FOUND $ISSUES_FOUND POTENTIAL ISSUES!"
    Write-Host ""
    Write-Host "ACTION REQUIRED:" -ForegroundColor Yellow
    Write-Host "1. Review each finding above" -ForegroundColor White
    Write-Host "2. Determine if sensitive data is exposed" -ForegroundColor White
    Write-Host "3. Use git filter-branch or BFG to remove" -ForegroundColor White
    Write-Host "4. Force push cleaned history" -ForegroundColor White
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Cyan "DETAILED COMMIT SCAN"
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Recent commits (last 30):" -ForegroundColor White
git log --oneline -30

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Green "✅ AUDIT COMPLETE"
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

