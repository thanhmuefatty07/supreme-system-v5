# Security Key Rotation Guide
**Date:** November 16, 2025  
**Status:** CRITICAL - Immediate Action Required

## 🚨 Security Alert

The `.env` file was previously tracked in git history. Even though it has been removed from tracking, **it still exists in git history** and may contain exposed secrets.

## ⚠️ Immediate Actions Required

### Step 1: Remove .env from Git History

**Option A: Using BFG Repo-Cleaner (Recommended)**

```bash
# 1. Download BFG Repo-Cleaner
# https://rtyley.github.io/bfg-repo-cleaner/

# 2. Clone a fresh copy of your repo
git clone --mirror https://github.com/thanhmuefatty07/supreme-system-v5.git temp-repo.git

# 3. Remove .env from history
java -jar bfg.jar --delete-files .env temp-repo.git

# 4. Clean up
cd temp-repo.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. Force push (WARNING: This rewrites history!)
git push --force

# 6. Delete temp repo
cd ..
rm -rf temp-repo.git
```

**Option B: Using git filter-branch (Alternative)**

```bash
# Remove .env from entire history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push
git push origin --force --all
git push origin --force --tags
```

**⚠️ WARNING:** Both methods rewrite git history. All collaborators must re-clone the repository.

### Step 2: Rotate All Exposed Keys

Based on codebase analysis, rotate these keys immediately:

#### Binance API Keys
1. **Login to Binance Account**
2. Go to API Management
3. **Revoke/Delete** existing API keys
4. Create new API keys
5. Update `.env` file locally with new keys
6. **Never commit** `.env` to git

#### Alpha Vantage API Key
1. **Login to Alpha Vantage**
2. Go to API Keys section
3. **Regenerate** API key
4. Update `.env` file locally

#### GitHub Token (if used)
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. **Revoke** any exposed tokens
3. Generate new token if needed
4. Update `.env` file locally

#### Other Secrets
- Database passwords
- JWT secrets
- Any other API keys in `.env`

### Step 3: Verify .env is Ignored

```bash
# Check if .env is tracked
git ls-files | grep "^\.env$"

# Should return nothing. If it returns .env, run:
git rm --cached .env

# Verify .gitignore
cat .gitignore | grep "^\.env$"

# Should show: .env
```

### Step 4: Update All Collaborators

**Important:** After removing `.env` from history:

1. **Notify all collaborators** to:
   - Re-clone the repository
   - Create `.env` from `.env.example`
   - Add their own API keys

2. **Update CI/CD pipelines** to use secrets management:
   - GitHub Secrets
   - Environment variables
   - Never hardcode secrets

## 📋 Checklist

- [ ] Remove `.env` from git history (BFG or filter-branch)
- [ ] Rotate Binance API keys
- [ ] Rotate Alpha Vantage API key
- [ ] Rotate GitHub token (if used)
- [ ] Rotate database passwords
- [ ] Rotate JWT secrets
- [ ] Verify `.env` is in `.gitignore`
- [ ] Verify `.env` is NOT tracked by git
- [ ] Update `.env.example` with all required variables
- [ ] Notify all collaborators
- [ ] Update CI/CD to use secrets management
- [ ] Document new key rotation process

## 🔒 Best Practices Going Forward

### 1. Never Commit Secrets
- ✅ Use `.env.example` as template
- ✅ Add `.env` to `.gitignore`
- ✅ Use environment variables in production
- ❌ Never commit `.env` files
- ❌ Never hardcode secrets in code

### 2. Use Secrets Management
- **Development:** `.env` file (gitignored)
- **CI/CD:** GitHub Secrets / GitLab CI Variables
- **Production:** AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault

### 3. Regular Rotation
- Rotate API keys every 90 days
- Rotate database passwords every 180 days
- Rotate JWT secrets every 365 days
- Document rotation dates

### 4. Monitoring
- Set up alerts for unauthorized API usage
- Monitor for suspicious activity
- Review access logs regularly

## 📝 Example .env Structure

```bash
# API Keys (NEVER COMMIT THESE!)
BINANCE_API_KEY=your_new_key_here
BINANCE_SECRET_KEY=your_new_secret_here
ALPHA_VANTAGE_KEY=your_new_key_here

# Database (NEVER COMMIT THESE!)
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Security (NEVER COMMIT THESE!)
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Feature Flags (Safe to commit defaults)
ENABLE_AI_OPTIMIZER=true
LOG_LEVEL=INFO
```

## 🆘 Emergency Response

If keys are compromised:

1. **Immediately revoke** all exposed keys
2. **Rotate** all secrets within 1 hour
3. **Review** access logs for unauthorized usage
4. **Notify** affected services/users
5. **Document** incident and response

## 📞 Support

For questions or issues:
- Review: `scripts/cleanup_git_secrets.sh`
- Contact: thanhmuefatty07@gmail.com

---

**Last Updated:** November 16, 2025  
**Next Review:** December 16, 2025

