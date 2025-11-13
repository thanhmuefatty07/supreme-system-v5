# 🚀 Option 1: BFG Repo-Cleaner - Hướng Dẫn Chi Tiết

**Date:** 2025-11-13  
**Method:** BFG Repo-Cleaner (Easiest & Fastest)  
**Estimated Time:** 10-15 phút

---

## ✅ **ĐÃ CHUẨN BỊ SẴN**

- ✅ Script tự động: `scripts/cleanup-token-bfg.ps1`
- ✅ Documentation: `CRITICAL_TOKEN_REMOVAL.md`
- ✅ Token đã được xác định: `YOUR_EXPOSED_TOKEN_HERE`

---

## 📋 **YÊU CẦU HỆ THỐNG**

### **1. Java Runtime Environment (JRE)**

**Kiểm tra:**
```powershell
java -version
```

**Nếu chưa có, cài đặt:**

**Option A: Chocolatey (Recommended)**
```powershell
# Install Chocolatey nếu chưa có
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Java
choco install openjdk -y
```

**Option B: Manual Download**
1. Go to: https://www.java.com/download/
2. Download Java for Windows
3. Install và restart PowerShell

**Option C: Winget (Windows 11)**
```powershell
winget install Microsoft.OpenJDK.17
```

---

### **2. BFG Repo-Cleaner**

**Script sẽ tự động download, hoặc download manual:**

```powershell
# Download BFG
Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar" -OutFile "bfg.jar"
```

**Hoặc download từ:** https://rtyley.github.io/bfg-repo-cleaner/

---

## 🚀 **CÁCH THỰC HIỆN**

### **Method 1: Sử dụng Script Tự Động (Recommended)**

```powershell
# Chạy script tự động
.\scripts\cleanup-token-bfg.ps1
```

**Script sẽ:**
1. ✅ Kiểm tra Java installation
2. ✅ Download BFG nếu chưa có
3. ✅ Tạo backup branch
4. ✅ Tạo tokens.txt file
5. ✅ Chạy BFG cleanup
6. ✅ Clean up Git references
7. ✅ Verify token removal
8. ✅ Hướng dẫn force push

---

### **Method 2: Manual Commands**

**Nếu bạn muốn chạy từng bước:**

```powershell
# Step 1: Tạo backup branch
git branch backup-before-token-cleanup-$(Get-Date -Format 'yyyyMMdd-HHmmss')

# Step 2: Tạo tokens.txt
$token = "YOUR_EXPOSED_TOKEN_HERE"
"$token==>REVOKED_TOKEN_REMOVED_FROM_HISTORY" | Out-File -FilePath "tokens.txt" -Encoding UTF8 -NoNewline

# Step 3: Download BFG (nếu chưa có)
Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar" -OutFile "bfg.jar"

# Step 4: Chạy BFG
java -jar bfg.jar --replace-text tokens.txt

# Step 5: Clean up Git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Step 6: Verify
git log --all -p | Select-String -Pattern "YOUR_EXPOSED_TOKEN_HERE"
# Should return: No matches

# Step 7: Force push (WARNING: Rewrites remote history)
git push origin --force --all
git push origin --force --tags
```

---

## ⚠️ **QUAN TRỌNG TRƯỚC KHI CHẠY**

### **🔴 CHECKLIST BẮT BUỘC:**

- [ ] ✅ **Đã revoke token** tại https://github.com/settings/tokens
- [ ] ✅ **Đã backup repository** (clone về máy khác)
- [ ] ✅ **Đã thông báo team** (nếu shared repo)
- [ ] ✅ **Đã kiểm tra Java** (`java -version`)
- [ ] ✅ **Đã đọc warnings** về force push

---

## 📊 **QUY TRÌNH CHI TIẾT**

### **Phase 1: Preparation (2 phút)**

1. **Revoke token:**
   - Go to: https://github.com/settings/tokens
   - Find token: `YOUR_EXPOSED_TOKEN_HERE`
   - Click "Revoke"

2. **Install Java** (nếu chưa có):
   ```powershell
   choco install openjdk -y
   # Hoặc download từ java.com
   ```

3. **Verify setup:**
   ```powershell
   java -version
   git --version
   ```

---

### **Phase 2: Cleanup (5-10 phút)**

**Chạy script:**
```powershell
.\scripts\cleanup-token-bfg.ps1
```

**Hoặc manual:**
```powershell
# Tạo tokens.txt
"YOUR_EXPOSED_TOKEN_HERE==>REVOKED_TOKEN_REMOVED" | Out-File tokens.txt -NoNewline

# Download BFG
Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar" -OutFile "bfg.jar"

# Chạy cleanup
java -jar bfg.jar --replace-text tokens.txt

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

### **Phase 3: Verification (1 phút)**

```powershell
# Kiểm tra token đã được remove
git log --all -p | Select-String -Pattern "YOUR_EXPOSED_TOKEN_HERE"
# Expected: No matches

# Kiểm tra commit history
git log --oneline -10
```

---

### **Phase 4: Push to Remote (2 phút)**

**⚠️ WARNING: Force push sẽ rewrite remote history!**

```powershell
# Force push tất cả branches
git push origin --force --all

# Force push tags
git push origin --force --tags
```

**Sau khi push:**
```powershell
# Update remote URL (remove token)
git remote set-url origin https://github.com/thanhmuefatty07/supreme-system-v5.git

# Verify remote URL
git remote get-url origin
```

---

## 🔍 **VERIFICATION**

### **Check 1: Token không còn trong history**
```powershell
git log --all -p | Select-String -Pattern "YOUR_EXPOSED_TOKEN_HERE"
```
**Expected:** No matches ✅

### **Check 2: File đã được clean**
```powershell
git show HEAD:GITHUB_TOKEN_SECURITY.md | Select-String -Pattern "ghp_"
```
**Expected:** No matches ✅

### **Check 3: Remote URL không chứa token**
```powershell
git remote get-url origin
```
**Expected:** `https://github.com/thanhmuefatty07/supreme-system-v5.git` ✅

---

## 🛡️ **SAFETY MEASURES**

### **Backup Before Cleanup:**
```powershell
# Clone backup
git clone https://github.com/thanhmuefatty07/supreme-system-v5.git backup-repo

# Hoặc tạo backup branch
git branch backup-before-cleanup-$(Get-Date -Format 'yyyyMMdd-HHmmss')
```

### **Test on Local First:**
```powershell
# Test trên local branch trước
git checkout -b test-cleanup
# Chạy cleanup trên test branch
# Verify kết quả
# Nếu OK, merge vào main
```

---

## 📝 **TROUBLESHOOTING**

### **Issue 1: Java not found**
```powershell
# Install Java
choco install openjdk -y
# Hoặc download từ java.com
```

### **Issue 2: BFG download failed**
```powershell
# Download manual từ browser
# Save as: bfg.jar trong project root
```

### **Issue 3: Force push rejected**
```powershell
# Check branch protection rules
# May need to disable temporarily
# Or use: git push origin --force --all --no-verify
```

### **Issue 4: Token still found**
```powershell
# Check if token exists in other files
git log --all --name-only | Select-String -Pattern "token"

# Re-run BFG with more specific pattern
java -jar bfg.jar --replace-text tokens.txt --no-blob-protection
```

---

## ✅ **SUCCESS CRITERIA**

Sau khi hoàn thành, bạn sẽ có:

- ✅ Token đã được remove khỏi Git history
- ✅ Tất cả commit SHAs đã được rewrite
- ✅ Remote repository đã được update
- ✅ Remote URL không chứa token
- ✅ Verification passed (no token found)

---

## 🎯 **NEXT STEPS AFTER CLEANUP**

1. ✅ **Update remote URL** (remove token)
2. ✅ **Verify cleanup** (check history)
3. ✅ **Review access logs** (check unauthorized access)
4. ✅ **Update documentation** (mark as completed)
5. ✅ **Notify team** (if shared repo)

---

## 📚 **REFERENCES**

- BFG Repo-Cleaner: https://rtyley.github.io/bfg-repo-cleaner/
- Git Filter-Branch: https://git-scm.com/docs/git-filter-branch
- GitHub Token Security: `CRITICAL_TOKEN_REMOVAL.md`

---

**Last Updated:** 2025-11-13  
**Status:** ✅ **READY TO EXECUTE**  
**Script:** `scripts/cleanup-token-bfg.ps1`

