# ⚡ Option 1 Quick Start - BFG Cleanup

**Thời gian:** 10 phút  
**Độ khó:** Dễ  
**Yêu cầu:** Java + BFG

---

## 🚀 **3 BƯỚC NHANH**

### **Bước 1: Cài Java (2 phút)**

```powershell
# Option A: Chocolatey
choco install openjdk -y

# Option B: Winget (Windows 11)
winget install Microsoft.OpenJDK.17

# Verify
java -version
```

---

### **Bước 2: Chạy Script (5 phút)**

```powershell
# Chạy script tự động
.\scripts\cleanup-token-bfg.ps1
```

**Script sẽ tự động:**
- ✅ Download BFG
- ✅ Tạo backup
- ✅ Clean token từ history
- ✅ Verify cleanup

---

### **Bước 3: Force Push (2 phút)**

```powershell
# Push cleaned history
git push origin --force --all
git push origin --force --tags

# Update remote URL
git remote set-url origin https://github.com/thanhmuefatty07/supreme-system-v5.git
```

---

## ✅ **XONG!**

Token đã được remove khỏi Git history!

---

**Chi tiết:** Xem `OPTION1_BFG_CLEANUP_GUIDE.md`

