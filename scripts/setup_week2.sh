#!/bin/bash

# setup_week2.sh - Setup Week 2 environment for Supreme System V5

echo "🚀 Supreme System V5 - Week 2 Setup"
echo "====================================="
echo ""

# Check if .env exists
if [ -f .env ]; then
    echo "⚠️  .env already exists"
    read -p "Overwrite? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled"
        exit 1
    fi
fi

# Create .env file
echo "Creating .env file..."
cat > .env << 'EOF'
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
EOF

echo "✅ .env file created"
echo ""

# Prompt for keys
echo "Enter your 6 Gemini API keys:"
echo "(Get from: https://aistudio.google.com/apikey)"
echo ""

for i in {1..6}; do
    read -p "Key $i: " key
    sed -i "s/YOUR_KEY_${i}_HERE/$key/" .env
done

echo ""
echo "✅ All keys configured in .env"
echo ""

# Verify keys
echo "Verifying keys..."
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

echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q google-generativeai python-dotenv matplotlib pre-commit
echo "✅ Dependencies installed"

# Setup pre-commit
echo "Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg
echo "✅ Pre-commit hooks installed"

# Verify setup
echo ""
echo "🔍 Verifying setup..."
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

echo ""
echo "====================================="
echo "✅ SETUP COMPLETE!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Run analysis: python scripts/gemini_analyzer.py"
echo "2. Review report in analysis_reports/"
echo "3. Generate tests: python scripts/test_generator.py"
echo ""
echo "Happy coding! 🚀"

