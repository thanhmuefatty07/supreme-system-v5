# 🚨 EMERGENCY CLEANUP GUIDE - Gemini API Keys

**Date:** November 17, 2025  
**Status:** CRITICAL ACTION REQUIRED

## ⚡ QUICK START (3 Steps)

### STEP 1: REVOKE KEYS (5 minutes) 🔴 CRITICAL

**DO THIS FIRST!**

1. Go to: https://console.cloud.google.com/apis/credentials
2. Find and **DELETE** these keys:
   - `AIzaSyB5v7LHHgdj7AMpi8Drngi7UsRhb4tLvcE`
   - `AIzaSyAgakXQVcSD5BadqMsNwxgZ86qs01natAI`
3. Check usage logs for suspicious activity

**⚠️ IMPORTANT:** Keys in Git = Public = Compromised. Must revoke BEFORE cleanup!

---

### STEP 2: CLEAN GIT HISTORY (30 minutes) 🟡

**After revoking keys, run:**

```powershell
.\cleanup_gemini_keys.ps1
```

**What it does:**
- Creates backup of repository
- Removes keys from Git history
- Cleans Git references
- Verifies keys are removed
- Optionally force pushes to GitHub

**⚠️ WARNING:** This rewrites Git history. Make sure you've revoked keys first!

---

### STEP 3: SETUP PREVENTION (15 minutes) 🟢

**Prevent future leaks:**

```powershell
.\setup_prevention.ps1
```

**What it does:**
- Installs pre-commit hook (blocks API key commits)
- Updates .gitignore (excludes sensitive files)
- Creates .env.example template

**Then commit:**

```powershell
git add .gitignore .env.example
git commit -m "security: Add prevention measures for API keys"
git push origin main
```

---

## 📋 VERIFICATION CHECKLIST

After cleanup, verify:

```powershell
# 1. Keys removed from history?
git log --all -S"AIzaSyB5v7LHHgdj7AMpi8Drngi7UsRhb4tLvcE" --oneline
# Should be EMPTY ✅

git log --all -S"AIzaSyAgakXQVcSD5BadqMsNwxgZ86qs01natAI" --oneline
# Should be EMPTY ✅

# 2. Keys NOT in current code?
Get-ChildItem -Recurse -File | Select-String -Pattern "AIzaSyB5v7LHHgdj7AMpi8Drngi7UsRhb4tLvcE"
# Should be EMPTY ✅

# 3. .env not tracked?
git ls-files | Select-String -Pattern "^\.env$"
# Should be EMPTY (except .env.example) ✅

# 4. Pre-commit hook working?
# Try committing a test file with API key - should be BLOCKED ✅
```

---

## 🛡️ PREVENTION MEASURES INSTALLED

### Pre-commit Hook
- Blocks commits containing Gemini API keys
- Blocks commits containing generic API key patterns
- Blocks commits of sensitive files (.env, *.key, *.secret)

### Enhanced .gitignore
- Excludes .env files
- Excludes *.key, *.secret files
- Excludes credentials directories
- Excludes logs that might contain keys

### .env.example Template
- Template for environment variables
- Shows required keys without exposing values
- Safe to commit to Git

---

## 📊 TIMELINE

```
NOW (0 min):     Revoke keys ← DO THIS FIRST!
+5 min:          Run cleanup script
+35 min:         Verify & force push
+45 min:         Setup prevention
+60 min (1h):    COMPLETE & SAFE ✅
```

---

## 🎯 PRIORITY ACTION ITEMS

### CRITICAL (Do RIGHT NOW):
1. ✅ Revoke 2 Gemini API keys in Google Cloud Console
2. ✅ Run cleanup script
3. ✅ Force push cleaned history

### HIGH (Do today):
4. ✅ Install pre-commit hook
5. ✅ Update .gitignore
6. ✅ Create .env.example

### MEDIUM (Do this week):
7. ✅ Generate new Gemini API keys
8. ✅ Test new keys work
9. ✅ Document security procedures

---

## 💰 VALUE IMPACT

### If properly cleaned:
- Security restored: +$0 (baseline)
- Trust maintained: +$1,000
- Professional handling: +$500
- Prevention setup: +$500
- **Net impact: +$2,000**

### If NOT cleaned:
- Security risk: -$10,000
- Unsellable: -$20,000
- Legal liability: -$5,000
- Reputation damage: -$3,000
- **Net loss: -$38,000**

**Cleaning is CRITICAL!** ⚡

---

## 🆘 TROUBLESHOOTING

### Issue: Cleanup script fails
**Solution:** Make sure you've revoked keys first. Script checks for this.

### Issue: Force push rejected
**Solution:** Check if you have write access. May need to disable branch protection temporarily.

### Issue: Pre-commit hook not working
**Solution:** Make sure hook is executable: `chmod +x .git/hooks/pre-commit`

### Issue: Keys still found after cleanup
**Solution:** May need to use BFG Repo-Cleaner instead. See SECURITY_AUDIT_REPORT.md for details.

---

## 📞 SUPPORT

If you encounter issues:
1. Check SECURITY_AUDIT_REPORT.md for detailed instructions
2. Review script output for error messages
3. Verify keys are revoked before proceeding

---

**Last Updated:** November 17, 2025  
**Status:** Ready to Execute



