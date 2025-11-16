# API Keys Rotation Checklist
**Date:** November 16, 2025  
**Status:** CRITICAL - Rotate Immediately

## 🚨 Exposed Keys Found in Git History

The following API keys/secrets were potentially exposed in git history and **MUST be rotated immediately**:

### 1. Binance API Keys ⚠️ CRITICAL
**Service:** Binance Exchange  
**Risk Level:** HIGH - Trading account access

**Action Required:**
1. Login to Binance Account
2. Go to API Management: https://www.binance.com/en/my/settings/api-management
3. **DELETE** existing API keys immediately
4. Create new API keys with minimal permissions:
   - Enable: Spot & Margin Trading (if needed)
   - **Disable:** Futures, Withdrawals
   - IP Restriction: Add your server IPs
5. Update `.env` file locally with new keys
6. **Never commit** `.env` to git

**Files Using Binance Keys:**
- `src/data/binance_client.py`
- `src/config/config.py`
- `src/utils/secrets_manager.py`
- `src/di/container.py`

### 2. Alpha Vantage API Key
**Service:** Alpha Vantage (Market Data)  
**Risk Level:** MEDIUM - API quota abuse

**Action Required:**
1. Login to Alpha Vantage: https://www.alphavantage.co/support/#api-key
2. **Regenerate** API key
3. Update `.env` file locally
4. Monitor API usage for suspicious activity

**Files Using Alpha Vantage:**
- `src/data/` (if integrated)

### 3. GitHub Token (if used)
**Service:** GitHub API  
**Risk Level:** HIGH - Repository access

**Action Required:**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. **Revoke** any exposed tokens
3. Generate new token with minimal permissions:
   - Only select required scopes
   - Set expiration date
4. Update `.env` file locally
5. Use GitHub Secrets for CI/CD instead

**Files Using GitHub Token:**
- CI/CD workflows (if any)
- Scripts that interact with GitHub API

### 4. OpenAI API Key (if used)
**Service:** OpenAI  
**Risk Level:** HIGH - Billing abuse

**Action Required:**
1. Login to OpenAI: https://platform.openai.com/api-keys
2. **Revoke** exposed keys
3. Create new API key
4. Set usage limits
5. Update `.env` file locally

**Files Using OpenAI:**
- `src/ai/coverage_optimizer.py`
- `src/ai/__main__.py`

### 5. Database Credentials
**Service:** PostgreSQL/Redis  
**Risk Level:** CRITICAL - Data breach

**Action Required:**
1. **Change database passwords** immediately
2. Rotate Redis passwords
3. Update connection strings in `.env`
4. Verify no unauthorized access occurred
5. Review database access logs

**Files Using Database:**
- `src/data/data_storage.py`
- `src/config/settings.py`

### 6. JWT Secret
**Service:** Application Authentication  
**Risk Level:** HIGH - Session hijacking

**Action Required:**
1. Generate new JWT secret:
   ```python
   import secrets
   print(secrets.token_urlsafe(32))
   ```
2. Update `.env` file
3. **Force logout** all existing sessions
4. Invalidate all existing tokens

**Files Using JWT:**
- `src/security/zero_trust.py`

### 7. Encryption Keys
**Service:** Application Encryption  
**Risk Level:** CRITICAL - Data decryption

**Action Required:**
1. Generate new encryption keys
2. Re-encrypt all encrypted data
3. Update `.env` file
4. Rotate keys in secrets manager

**Files Using Encryption:**
- `src/security/quantum_crypto.py`
- `src/utils/secrets_manager.py`

## 📋 Rotation Priority

| Priority | Service | Timeframe | Impact if Compromised |
|----------|---------|-----------|----------------------|
| **P0** | Binance API | **Immediate** | Financial loss, account takeover |
| **P0** | Database | **Immediate** | Data breach, compliance violation |
| **P1** | GitHub Token | **Within 1 hour** | Repository compromise |
| **P1** | OpenAI API | **Within 1 hour** | Billing abuse |
| **P2** | JWT Secret | **Within 24 hours** | Session hijacking |
| **P2** | Encryption Keys | **Within 24 hours** | Data decryption |
| **P3** | Alpha Vantage | **Within 48 hours** | API quota abuse |

## ✅ Verification Steps

After rotation, verify:

1. **Binance:**
   ```bash
   # Test new API key
   python -c "from src.data.binance_client import BinanceClient; client = BinanceClient(); print(client.test_connection())"
   ```

2. **Database:**
   ```bash
   # Test connection
   python -c "from src.data.data_storage import DataStorage; storage = DataStorage(); print('Connected')"
   ```

3. **GitHub:**
   ```bash
   # Test token
   curl -H "Authorization: token YOUR_NEW_TOKEN" https://api.github.com/user
   ```

4. **All Services:**
   - Run test suite: `pytest tests/`
   - Check logs for authentication errors
   - Monitor for suspicious activity

## 🔒 Post-Rotation Security

1. **Enable 2FA** on all accounts
2. **Set API rate limits** where possible
3. **Enable IP whitelisting** for API keys
4. **Monitor usage** for anomalies
5. **Set up alerts** for unauthorized access
6. **Review access logs** regularly

## 📝 Rotation Log

| Date | Service | Old Key (Last 4) | New Key (Last 4) | Rotated By | Status |
|------|---------|------------------|------------------|------------|--------|
| 2025-11-16 | Binance API | [PENDING] | [PENDING] | [PENDING] | ⏳ |
| 2025-11-16 | Database | [PENDING] | [PENDING] | [PENDING] | ⏳ |
| 2025-11-16 | GitHub Token | [PENDING] | [PENDING] | [PENDING] | ⏳ |

## 🆘 Emergency Contacts

- **Binance Support:** https://www.binance.com/en/support
- **GitHub Support:** https://support.github.com
- **Database Admin:** [Contact Info]
- **Security Team:** [Contact Info]

---

**⚠️ IMPORTANT:** Complete all rotations within 24 hours. Document all changes in this file.

**Last Updated:** November 16, 2025

