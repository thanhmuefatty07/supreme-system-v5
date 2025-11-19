#!/bin/bash

# setup_buyer.sh - Automated buyer setup for Supreme System V5

echo "🎉 Supreme System V5 - Buyer Setup"
echo "=================================="
echo ""

# Check if .env exists and has real keys
if [ -f .env ]; then
    echo "⚠️  .env file already exists"

    # Check if it has real keys
    if grep -q "AIza" .env; then
        echo "❌ .env contains real API keys!"
        echo "   Please remove .env and use .env.example"
        echo "   Or run: python scripts/prepare_for_sale.py"
        exit 1
    fi

    echo "✅ .env appears to be template (no real keys)"
else
    echo "📄 Creating .env from template..."
    cp .env.example .env
    echo "✅ .env created from .env.example"
fi

echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q google-generativeai python-dotenv matplotlib pre-commit
echo "✅ Dependencies installed"

# Setup pre-commit
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg
echo "✅ Pre-commit hooks installed"

# Verify setup
echo ""
echo "🔍 Verifying setup..."

# Check if keys are configured (should be placeholders)
keys_configured=0
for i in {1..6}; do
    key=$(grep "GEMINI_API_KEY_$i" .env | cut -d'=' -f2)
    if [ "$key" != "YOUR_KEY_${i}_HERE" ] && [ -n "$key" ]; then
        keys_configured=$((keys_configured + 1))
    fi
done

echo "📊 Keys status: $keys_configured/6 configured"

if [ $keys_configured -eq 6 ]; then
    echo "✅ All keys configured - testing connection..."

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
    echo '⚠️  Some keys appear invalid'
"
else
    echo "⚠️  Keys not fully configured (using template values)"
    echo "   Configure your keys in .env to enable full functionality"
fi

echo ""
echo "=================================="
echo "✅ BUYER SETUP COMPLETE!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Configure your 6 Gemini API keys in .env"
echo "2. Run analysis: python scripts/gemini_analyzer.py"
echo "3. Generate tests: python scripts/test_generator.py"
echo ""
echo "📚 Documentation: docs/week2/DEVELOPMENT_PLAN.md"
echo "🆘 Support: docs/BUYER_SETUP_GUIDE.md"
echo ""
echo "Welcome to Supreme System V5! 🚀"



