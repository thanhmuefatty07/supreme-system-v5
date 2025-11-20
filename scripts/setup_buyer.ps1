# setup_buyer.ps1 - Windows PowerShell version for buyer setup

Write-Host "🎉 Supreme System V5 - Buyer Setup" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists and has real keys
if (Test-Path .env) {
    Write-Host "⚠️  .env file already exists" -ForegroundColor Yellow

    # Check if it has real keys
    $envContent = Get-Content .env
    if ($envContent -match "AIza") {
        Write-Host "❌ .env contains real API keys!" -ForegroundColor Red
        Write-Host "   Please remove .env and use .env.example" -ForegroundColor Red
        Write-Host "   Or run: python scripts\prepare_for_sale.py" -ForegroundColor Red
        exit 1
    }

    Write-Host "✅ .env appears to be template (no real keys)" -ForegroundColor Green
} else {
    Write-Host "📄 Creating .env from template..." -ForegroundColor Cyan
    Copy-Item .env.example .env
    Write-Host "✅ .env created from .env.example" -ForegroundColor Green
}

Write-Host ""

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
pip install -q google-generativeai python-dotenv matplotlib pre-commit
Write-Host "✅ Dependencies installed" -ForegroundColor Green

# Setup pre-commit
Write-Host ""
Write-Host "🔧 Setting up pre-commit hooks..." -ForegroundColor Cyan
pre-commit install
pre-commit install --hook-type commit-msg
Write-Host "✅ Pre-commit hooks installed" -ForegroundColor Green

# Verify setup
Write-Host ""
Write-Host "🔍 Verifying setup..." -ForegroundColor Cyan

# Check if keys are configured (should be placeholders)
$keysConfigured = 0
for ($i = 1; $i -le 6; $i++) {
    $keyLine = Get-Content .env | Where-Object { $_ -match "GEMINI_API_KEY_$i" }
    if ($keyLine) {
        $key = ($keyLine -split '=')[1]
        if ($key -and $key -ne "YOUR_KEY_${i}_HERE") {
            $keysConfigured++
        }
    }
}

Write-Host "📊 Keys status: $keysConfigured/6 configured" -ForegroundColor White

if ($keysConfigured -eq 6) {
    Write-Host "✅ All keys configured - testing connection..." -ForegroundColor Green

    python -c "
import os
from dotenv import load_dotenv
load_dotenv()

keys = [os.getenv(f'GEMINI_API_KEY_{i}') for i in range(1, 7)]
valid = [k for k in keys if k and len(k) > 30 and k.startswith('AIza')]
print(f'✅ Valid keys: {len(valid)}/6')

if len(valid) == 6:
    # Test one key
    try:
        import google.generativeai as genai
        genai.configure(api_key=keys[0])
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content('Test connection')
        print('✅ Gemini API connection successful!')
    except Exception as e:
        print(f'⚠️  API test failed: {e}')
        print('   Keys may be invalid or network issue')
else:
    print('⚠️  Some keys appear invalid')
"
} else {
    Write-Host "⚠️  Keys not fully configured (using template values)" -ForegroundColor Yellow
    Write-Host "   Configure your keys in .env to enable full functionality" -ForegroundColor Gray
}

Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "✅ BUYER SETUP COMPLETE!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Configure your 6 Gemini API keys in .env" -ForegroundColor Gray
Write-Host "2. Run analysis: python scripts\gemini_analyzer.py" -ForegroundColor Gray
Write-Host "3. Generate tests: python scripts\test_generator.py" -ForegroundColor Gray
Write-Host ""
Write-Host "📚 Documentation: docs\week2\DEVELOPMENT_PLAN.md" -ForegroundColor Gray
Write-Host "🆘 Support: docs\BUYER_SETUP_GUIDE.md" -ForegroundColor Gray
Write-Host ""
Write-Host "Welcome to Supreme System V5! 🚀" -ForegroundColor Cyan



