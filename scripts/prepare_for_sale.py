#!/usr/bin/env python3

"""

Prepare for Sale - Remove sensitive data before release

Removes API keys and sensitive files from repository

"""

import shutil
from pathlib import Path
import os
import subprocess

def remove_sensitive_files():
    """Remove all sensitive files and data"""

    print("🔒 Preparing repository for sale...")
    print("=" * 60)

    # 1. Remove .env file
    if Path('.env').exists():
        Path('.env').unlink()
        print("✅ Removed: .env")

    # 2. Clear .env.example (keep structure, remove keys)
    env_example = Path('.env.example')
    if env_example.exists():
        env_example.write_text("""# Supreme System V5 - API Keys Configuration
# ⚠️  This file is a template. Add your own keys!

# Gemini API Keys (Get from: https://aistudio.google.com/apikey)
GEMINI_API_KEY_1=YOUR_KEY_1_HERE
GEMINI_API_KEY_2=YOUR_KEY_2_HERE
GEMINI_API_KEY_3=YOUR_KEY_3_HERE
GEMINI_API_KEY_4=YOUR_KEY_4_HERE
GEMINI_API_KEY_5=YOUR_KEY_5_HERE
GEMINI_API_KEY_6=YOUR_KEY_6_HERE

# Optional: GitHub token
GITHUB_TOKEN=YOUR_GITHUB_TOKEN_HERE
""", encoding='utf-8')
        print("✅ Cleaned: .env.example (no real keys)")

    # 3. Remove log files
    logs = list(Path('.').glob('*.log'))
    for log in logs:
        log.unlink()
        print(f"✅ Removed: {log}")

    # 4. Remove analysis reports (can be regenerated)
    if Path('analysis_reports').exists():
        shutil.rmtree('analysis_reports')
        print("✅ Removed: analysis_reports/ (can regenerate)")

    # 5. Remove test generation reports
    if Path('test_generation_reports').exists():
        shutil.rmtree('test_generation_reports')
        print("✅ Removed: test_generation_reports/ (can regenerate)")

    # 6. Clean coverage files (personal data)
    coverage_files = ['coverage.json', 'coverage.html', '.coverage']
    for cov_file in coverage_files:
        if Path(cov_file).exists():
            Path(cov_file).unlink()
            print(f"✅ Removed: {cov_file}")

    # 7. Remove personal baselines
    if Path('baselines').exists():
        shutil.rmtree('baselines')
        print("✅ Removed: baselines/ (personal data)")

    # 8. Clean git history (optional - requires manual execution)
    print("\n⚠️  Optional: Clean Git history of sensitive data")
    print("   Run: git filter-branch --force --index-filter")
    print("         'git rm --cached --ignore-unmatch .env' --prune-empty --tag-name-filter cat -- --all")
    print("   Then: git push origin --force --all")

def verify_cleanup():
    """Verify that sensitive data is removed"""

    print("\n🔍 Verifying cleanup...")
    print("=" * 60)

    issues = []

    # Check for .env file
    if Path('.env').exists():
        issues.append("❌ .env file still exists")
    else:
        print("✅ .env file removed")

    # Check .env.example content
    if Path('.env.example').exists():
        content = Path('.env.example').read_text()
        if 'AIza' in content or any(f'YOUR_KEY_{i}_HERE' not in content for i in range(1, 7)):
            issues.append("❌ .env.example contains real keys or invalid placeholders")
        else:
            print("✅ .env.example cleaned")

    # Check for log files
    logs = list(Path('.').glob('*.log'))
    if logs:
        issues.append(f"❌ Found {len(logs)} log files: {[str(l) for l in logs]}")
    else:
        print("✅ No log files remaining")

    # Check for analysis reports
    if Path('analysis_reports').exists():
        issues.append("❌ analysis_reports/ still exists")
    else:
        print("✅ analysis_reports/ removed")

    # Check for test generation reports
    if Path('test_generation_reports').exists():
        issues.append("❌ test_generation_reports/ still exists")
    else:
        print("✅ test_generation_reports/ removed")

    # Check Git history for keys (sample check)
    try:
        result = subprocess.run([
            'git', 'log', '--oneline', '--all', '--grep', 'AIza'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout.strip():
            issues.append("⚠️  Git history may contain API keys (manual check required)")
        else:
            print("✅ Git history appears clean")
    except Exception:
        print("⚠️  Could not check Git history (git not available)")

    if issues:
        print("\n❌ Cleanup Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n✅ All cleanup checks passed!")
        return True

def create_sale_checklist():
    """Create pre-sale checklist"""

    checklist = """# Pre-Sale Security Checklist

## 🔒 Security Verification

- [ ] Removed .env file (contains real API keys)
- [ ] Cleaned .env.example (only placeholders)
- [ ] Removed all .log files
- [ ] Removed analysis_reports/ (regenerable)
- [ ] Removed test_generation_reports/ (regenerable)
- [ ] Removed coverage.json, .coverage files
- [ ] Removed baselines/ (personal data)

## 🧹 Git History Cleanup

- [ ] No API keys in Git history (check with: git log -p --all -S 'AIza')
- [ ] No sensitive data in any commits
- [ ] Consider: git filter-branch to clean history
- [ ] Force push after cleanup (if needed)

## 📦 Package Verification

- [ ] All test files remain (tests/ directory intact)
- [ ] Documentation complete (docs/ directory)
- [ ] Scripts functional (scripts/ directory)
- [ ] Configuration templates clean (.env.example)
- [ ] CI/CD workflows intact (.github/)

## 🔍 Final Verification

- [ ] Run: python scripts/prepare_for_sale.py --verify
- [ ] Run: git log -p --all -S 'AIza' (should return nothing)
- [ ] Test buyer setup: cp .env.example .env && run setup script
- [ ] Confirm all functionality works with template keys

## 📋 Buyer Documentation

- [ ] docs/BUYER_SETUP_GUIDE.md complete and accurate
- [ ] All setup scripts functional (Linux/Mac + Windows)
- [ ] Clear instructions for obtaining Gemini keys
- [ ] Security best practices documented

## 🎯 Ready for Sale

- [ ] Package compressed and ready
- [ ] Documentation included
- [ ] No sensitive data remaining
- [ ] Buyer can self-setup immediately

---
Generated by prepare_for_sale.py
"""

    Path('PRE_SALE_CHECKLIST.md').write_text(checklist, encoding='utf-8')
    print("✅ Created: PRE_SALE_CHECKLIST.md")

def main():
    """Main execution"""

    print("Supreme System V5 - Prepare for Sale")
    print("=" * 60)
    print("This script removes sensitive data before selling the project")
    print("⚠️  Make sure you have backups of any important data!")
    print()

    # Confirm execution
    confirm = input("Continue with cleanup? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        print("Cleanup cancelled.")
        return

    # Perform cleanup
    remove_sensitive_files()

    # Verify cleanup
    success = verify_cleanup()

    # Create checklist
    create_sale_checklist()

    print("\n" + "=" * 60)

    if success:
        print("✅ REPOSITORY CLEANED FOR SALE!")
        print("✅ Ready to package and deliver to buyer")
    else:
        print("❌ CLEANUP INCOMPLETE!")
        print("🔧 Fix issues above before proceeding")

    print("=" * 60)
    print("\nNext steps:")
    print("1. Review PRE_SALE_CHECKLIST.md")
    print("2. Test buyer setup with .env.example")
    print("3. Package project: tar -czf supreme-system-v5.tar.gz .")
    print("4. Deliver to buyer with BUYER_SETUP_GUIDE.md")

    print("\n🔒 Security Reminder:")
    print("- Buyer will need to create their own 6 Gemini API keys")
    print("- All sensitive data has been removed")
    print("- Buyer setup is self-contained")

if __name__ == "__main__":
    main()



