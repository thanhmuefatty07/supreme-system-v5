#!/bin/bash

# 🚀 Quick Start Script for Gemini Coverage Optimizer
# Supreme System V5 - AI-Powered Test Generation

set -e  # Exit on error

echo ""
echo "======================================================================"
echo "🤖 GEMINI AI COVERAGE OPTIMIZER - FREE TIER"
echo "Supreme System V5 - From 31% to 80%+ Coverage"
echo "======================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set Gemini API key
export GOOGLE_API_KEY="AIzaSyBH8mRSlNVKQoRi5uCrEJikTJlqhRhPA-g"
export GEMINI_API_KEY="AIzaSyBH8mRSlNVKQoRi5uCrEJikTJlqhRhPA-g"

# Configuration
export TARGET_COVERAGE="80"
export AI_PROVIDER="gemini"
export MAX_ITERATIONS="5"
export BATCH_SIZE="10"
export COVERAGE_CORE="sysmon"  # Python 3.12 fast mode

echo -e "${BLUE}Configuration:${NC}"
echo "  🎯 Target Coverage: ${TARGET_COVERAGE}%"
echo "  🤖 AI Provider: ${AI_PROVIDER} (FREE tier)"
echo "  🔄 Max Iterations: ${MAX_ITERATIONS}"
echo "  📦 Batch Size: ${BATCH_SIZE}"
echo ""

# Step 1: Install dependencies
echo -e "${YELLOW}Step 1: Installing dependencies...${NC}"
if ! python -c "import google.generativeai" 2>/dev/null; then
    echo "  Installing google-generativeai..."
    pip install -q google-generativeai
    echo -e "  ${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "  ${GREEN}✓ Dependencies already installed${NC}"
fi

# Step 2: Verify API key
echo -e "\n${YELLOW}Step 2: Verifying Gemini API key...${NC}"
python -c "
import google.generativeai as genai
import sys
try:
    genai.configure(api_key='AIzaSyBH8mRSlNVKQoRi5uCrEJikTJlqhRhPA-g')
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content('test')
    print('  ✓ Gemini API key verified and working!')
    sys.exit(0)
except Exception as e:
    print(f'  ✗ API key verification failed: {e}')
    sys.exit(1)
" && echo -e "${GREEN}✓ API key verified${NC}" || { echo -e "${RED}✗ API verification failed${NC}"; exit 1; }

# Step 3: Analyze initial coverage
echo -e "\n${YELLOW}Step 3: Analyzing initial coverage...${NC}"
if [ ! -f "coverage.xml" ]; then
    echo "  Running tests to generate coverage report..."
    pytest --cov=src --cov-report=xml -q || true
fi

INITIAL_COV=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); print(f'{float(tree.getroot().attrib[\"line-rate\"])*100:.1f}')" 2>/dev/null || echo "31.0")
echo -e "  ${BLUE}Initial Coverage: ${INITIAL_COV}%${NC}"

# Step 4: Run AI Coverage Optimizer
echo -e "\n${YELLOW}Step 4: Running AI Coverage Optimizer...${NC}"
echo "  🤖 Using Gemini 2.0 Flash (FREE tier)"
echo "  🎯 Target: ${TARGET_COVERAGE}%"
echo "  💰 Cost: $0.00 (FREE!)"
echo ""

python -m src.ai.gemini_coverage_optimizer \
    --source-dir src \
    --target-coverage ${TARGET_COVERAGE} \
    --provider gemini \
    --max-iterations ${MAX_ITERATIONS} \
    --verbose

OPTIMIZER_EXIT_CODE=$?

# Step 5: Final coverage measurement
echo -e "\n${YELLOW}Step 5: Measuring final coverage...${NC}"
pytest --cov=src --cov-report=xml --cov-report=term -q

FINAL_COV=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); print(f'{float(tree.getroot().attrib[\"line-rate\"])*100:.1f}')" 2>/dev/null || echo "0")
IMPROVEMENT=$(python -c "print(f'{${FINAL_COV} - ${INITIAL_COV}:.1f}')" 2>/dev/null || echo "0")

echo ""
echo "======================================================================"
echo -e "${GREEN}🎯 OPTIMIZATION COMPLETE!${NC}"
echo "======================================================================"
echo ""
echo -e "${BLUE}📊 Coverage Results:${NC}"
echo "  Initial Coverage:  ${INITIAL_COV}%"
echo "  Final Coverage:    ${FINAL_COV}%"
echo -e "  Improvement:       ${GREEN}+${IMPROVEMENT}%${NC}"
echo ""
echo -e "${BLUE}💰 Cost:${NC}"
echo -e "  Total Cost:        ${GREEN}\$0.00 (FREE!)${NC}"
echo "  Provider:          Gemini 2.0 Flash"
echo ""

if (( $(echo "${FINAL_COV} >= ${TARGET_COVERAGE}" | bc -l) )); then
    echo -e "${GREEN}✅ TARGET ACHIEVED: ${FINAL_COV}% >= ${TARGET_COVERAGE}%${NC}"
    echo -e "${GREEN}✅ Ready for production deployment!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Target not fully achieved: ${FINAL_COV}% < ${TARGET_COVERAGE}%${NC}"
    echo "  Run again with --max-iterations 10 for more improvement"
    exit 1
fi
