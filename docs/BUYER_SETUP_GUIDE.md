# Supreme System V5 - Buyer Setup Guide

## 🎉 Welcome!

You've acquired **Supreme System V5**, a production-ready algorithmic trading platform with AI-powered testing infrastructure.

**Key Feature:** The system can auto-generate comprehensive test suites using Gemini AI, ensuring your codebase maintains 80%+ coverage with minimal manual effort.

---

## 🔑 Step 1: Obtain Gemini API Keys (5 minutes)

### Why You Need Keys

The system uses **6 Gemini API keys** for:

- **Parallel test generation** (90 requests/minute)
- **Code analysis & optimization**
- **Documentation generation**
- **Quality assurance automation**

**Cost:** FREE tier (1,500 requests/day per key) is sufficient for normal development.

### How to Get Keys

1. **Go to Google AI Studio**
   - Visit: https://aistudio.google.com
   - Sign in with your Google account

2. **Create API Key**
   - Click "Get API Key"
   - Click "Create API Key"
   - Select "Create new project"
   - **Repeat 6 times** to get 6 keys

3. **Copy Keys**
   - You'll see: `AIza...` (39 characters)
   - **Important:** Save keys in secure location (password manager)

**Video Guide:** [How to Create Gemini API Keys](https://youtu.be/example)

---

## ⚙️ Step 2: Configure Keys (2 minutes)

### Setup Environment

```
# 1. Navigate to project directory
cd supreme-system-v5

# 2. Create .env file from template
cp .env.example .env

# 3. Edit .env with your favorite editor
nano .env  # or notepad .env (Windows)
```

### Enter Your Keys

Edit `.env` file:

```
# Supreme System V5 - Your API Keys
# ⚠️  NEVER commit this file to Git!

# Gemini API Keys (6 keys for parallel processing)
GEMINI_API_KEY_1=YOUR_KEY_1_HERE
GEMINI_API_KEY_2=YOUR_KEY_2_HERE
GEMINI_API_KEY_3=YOUR_KEY_3_HERE
GEMINI_API_KEY_4=YOUR_KEY_4_HERE
GEMINI_API_KEY_5=YOUR_KEY_5_HERE
GEMINI_API_KEY_6=YOUR_KEY_6_HERE

# Optional: GitHub token (for automated PR creation)
# Get from: https://github.com/settings/tokens
GITHUB_TOKEN=YOUR_GITHUB_TOKEN_HERE
```

**Replace `YOUR_KEY_X_HERE` with your actual keys.**

---

## 🚀 Step 3: Run Automated Setup (1 minute)

```
# Run setup script (automatically configures everything)
./scripts/setup_buyer.sh  # or setup_buyer.ps1 on Windows
```

**What the script does:**

- ✅ Verifies all 6 keys are valid
- ✅ Installs dependencies
- ✅ Sets up pre-commit hooks
- ✅ Initializes test infrastructure
- ✅ Runs verification analysis

**Expected output:**

```
✅ Setup complete!
✅ 6 API keys verified
✅ Dependencies installed
✅ Pre-commit hooks configured

Next steps:
1. Run analysis: python scripts/gemini_analyzer.py
2. Generate tests: python scripts/test_generator.py
3. Start developing!

Documentation: docs/week2/DEVELOPMENT_PLAN.md
```

---

## 📊 Step 4: Verify Everything Works

### Quick Test

```
# Run a quick test
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

keys = [os.getenv(f'GEMINI_API_KEY_{i}') for i in range(1, 7)]
print(f'✅ Loaded {len([k for k in keys if k])} keys')

# Test one key
import google.generativeai as genai
genai.configure(api_key=keys[0])
model = genai.GenerativeModel('gemini-2.0-flash-exp')
response = model.generate_content('Say hello')
print('✅ Gemini API working!')
"

# Expected: "✅ Loaded 6 keys" and "✅ Gemini API working!"
```

---

## 🎯 Step 5: Start Development

### Week 2 Quick Start

```
# 1. Run comprehensive analysis
python scripts/gemini_analyzer.py
# → Creates analysis_reports/[timestamp]/FULL_ANALYSIS_REPORT.md

# 2. Review analysis
open analysis_reports/[timestamp]/FULL_ANALYSIS_REPORT.md

# 3. Generate tests for high-impact files
python scripts/test_generator.py
# → Creates 10 comprehensive test files

# 4. Run tests
pytest tests/ -v --cov=src

# 5. Track progress
python scripts/coverage_tracker.py
```

**Expected timeline:** 7 days to reach 80% coverage

---

## 💡 Understanding the System

### Why 6 Keys?

**Parallel Processing Power:**

- 6 keys × 15 requests/minute = **90 RPM total**
- Batch generate 10 test files in **20 minutes** (vs 2 hours with 1 key)
- Scale linearly - more keys = faster

**Smart Rotation:**

- Scripts automatically distribute load
- No manual key management needed
- Fault tolerance (if 1 key fails, others continue)

### Use Cases

**1. Test Generation**

```
# Generate comprehensive tests for any module
python scripts/test_generator.py
```

**2. Code Analysis**

```
# Analyze entire codebase
python scripts/gemini_analyzer.py
```

**3. Quality Monitoring**

```
# Track coverage trends
python scripts/coverage_tracker.py
```

---

## 📚 Documentation

- **Master Development Plan:** `docs/week2/DEVELOPMENT_PLAN.md`
- **Commit Standards:** `docs/week2/COMMIT_STANDARDS.md`
- **Branching Strategy:** `docs/week2/BRANCHING_STRATEGY.md`
- **Full Setup Guide:** `README_WEEK2.md`

---

## 🔐 Security Best Practices

### **DO:**

- ✅ Keep `.env` file private (gitignore)
- ✅ Rotate keys every 30 days
- ✅ Monitor API usage daily
- ✅ Use separate keys per developer (if team)
- ✅ Set up usage alerts

### **DON'T:**

- ❌ Commit `.env` to Git
- ❌ Share keys publicly
- ❌ Use keys in public CI/CD (use GitHub Secrets)
- ❌ Ignore rate limit warnings

---

## 🎓 Value Proposition

**Why This System is Valuable:**

1. **Time Savings:** Generate 10 comprehensive test files in 20 minutes (vs 20 hours manual)
2. **Quality Assurance:** Maintain 80%+ coverage automatically
3. **Scalability:** Add new features with confidence
4. **AI-Powered:** Leverage Google's latest AI models
5. **Professional:** Enterprise-grade development practices

**Estimated Value:** $5,000 - $10,000 in saved development time

---

## 📞 Support

**If you need help:**

1. Review documentation in `docs/week2/`
2. Check logs: `gemini_analysis.log`
3. Open issue: GitHub Issues → "Buyer Support"
4. Email: [your-support-email]

---

## 🎉 Success!

You're now ready to develop with AI-powered testing infrastructure. Follow the Week 2 plan to achieve 80% coverage in 7 days.

**Happy coding!** 🚀

---

**Last Updated:** November 19, 2025
**Package Version:** v1.0.0

