# setup_week2.ps1 - PowerShell version for Windows

Write-Host "🚀 Supreme System V5 - Week 2 Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (Test-Path .env) {
    Write-Host "⚠️  .env already exists" -ForegroundColor Yellow
    $overwrite = Read-Host "Overwrite? (y/n)"
    if ($overwrite -ne 'y') {
        Write-Host "Setup cancelled" -ForegroundColor Red
        exit
    }
}

# Create .env file
@"
# Supreme System V5 - API Keys Configuration
# ⚠️  NEVER commit this file to Git!

# Gemini API Keys (6 keys for parallel processing)
# Get keys from: https://aistudio.google.com/apikey
GEMINI_API_KEY_1=YOUR_KEY_1_HERE
GEMINI_API_KEY_2=YOUR_KEY_2_HERE
GEMINI_API_KEY_3=YOUR_KEY_3_HERE
GEMINI_API_KEY_4=YOUR_KEY_4_HERE
GEMINI_API_KEY_5=YOUR_KEY_5_HERE
GEMINI_API_KEY_6=YOUR_KEY_6_HERE

# Optional: GitHub token for automated PR creation
# Get from: https://github.com/settings/tokens
GITHUB_TOKEN=YOUR_GITHUB_TOKEN_HERE
"@ | Out-File -FilePath .env -Encoding utf8

Write-Host "✅ .env file created" -ForegroundColor Green
Write-Host ""

# Prompt for keys
Write-Host "Enter your 6 Gemini API keys:" -ForegroundColor Cyan
Write-Host "(Get from: https://aistudio.google.com/apikey)" -ForegroundColor Gray
Write-Host ""

for ($i = 1; $i -le 6; $i++) {
    $key = Read-Host "Key $i"
    (Get-Content .env) -replace "YOUR_KEY_${i}_HERE", $key | Set-Content .env
}

Write-Host ""
Write-Host "✅ All keys configured in .env" -ForegroundColor Green
Write-Host ""

# Verify keys
Write-Host "Verifying keys..." -ForegroundColor Cyan
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
keys = [os.getenv(f'GEMINI_API_KEY_{i}') for i in range(1, 7)]
valid = [k for k in keys if k and len(k) > 30]
print(f'✅ Valid keys: {len(valid)}/6')
if len(valid) < 6:
    print('⚠️  Some keys are missing or invalid')
    exit(1)
"

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -q google-generativeai python-dotenv matplotlib pre-commit
Write-Host "✅ Dependencies installed" -ForegroundColor Green

# Setup pre-commit
Write-Host ""
Write-Host "Setting up pre-commit hooks..." -ForegroundColor Cyan
pre-commit install
pre-commit install --hook-type commit-msg
Write-Host "✅ Pre-commit hooks installed" -ForegroundColor Green

# Verify setup
Write-Host ""
Write-Host "🔍 Verifying setup..." -ForegroundColor Cyan
python -c "
import sys
sys.path.append('scripts')
from gemini_analyzer import GeminiAnalyzer

import os
keys = [os.getenv(f'GEMINI_API_KEY_{i}') for i in range(1, 7) if os.getenv(f'GEMINI_API_KEY_{i}')]
analyzer = GeminiAnalyzer(keys)
print('✅ GeminiAnalyzer initialized successfully')
print(f'✅ {len(keys)} API keys loaded')
"

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Run analysis: python scripts\gemini_analyzer.py" -ForegroundColor Gray
Write-Host "2. Review report in analysis_reports\" -ForegroundColor Gray
Write-Host "3. Generate tests: python scripts\test_generator.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Happy coding! 🚀" -ForegroundColor Cyan



