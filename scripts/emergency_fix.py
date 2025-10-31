#!/usr/bin/env python3
"""
Emergency Fix Script - Immediate Resolution for All Issues
Execute black + ruff fixes in one go
"""

import subprocess
import sys
import os

def execute_emergency_fix():
    """Execute immediate fix for all formatting and linting issues"""
    
    print("🚨 EMERGENCY FIX PROTOCOL INITIATED")
    print("Target: Fix ALL 21 files + remaining lint issues")
    print("=" * 60)
    
    # Step 1: Black formatting (nuclear)
    print("\n🔥 Step 1: Nuclear Black Formatting...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "black", ".", 
            "--line-length", "88", 
            "--target-version", "py311"
        ], check=True, capture_output=True, text=True)
        print("✅ Black formatting completed")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Black failed:", e.stderr)
        return False
    except Exception as e:
        print("❌ Black error:", str(e))
        return False
    
    # Step 2: Ruff auto-fix
    print("\n🛠️ Step 2: Ruff Auto-Fix...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "ruff", "check", ".", 
            "--fix", "--unsafe-fixes"
        ], capture_output=True, text=True)
        print("✅ Ruff auto-fix completed")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print("⚠️ Ruff fix warning:", str(e))
    
    # Step 3: Final validation
    print("\n🔍 Step 3: Final Validation...")
    try:
        # Check black status
        black_result = subprocess.run([
            sys.executable, "-m", "black", "--check", "."
        ], capture_output=True, text=True)
        
        if black_result.returncode == 0:
            print("✅ Black: All files properly formatted")
        else:
            print("⚠️ Black: Some files may need manual attention")
        
        # Check ruff status
        ruff_result = subprocess.run([
            sys.executable, "-m", "ruff", "check", "."
        ], capture_output=True, text=True)
        
        if ruff_result.returncode == 0:
            print("✅ Ruff: All issues resolved")
        else:
            print("⚠️ Ruff: Some issues remain:")
            print(ruff_result.stdout)
        
        # Overall success check
        success = black_result.returncode == 0 and ruff_result.returncode == 0
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 EMERGENCY FIX SUCCESSFUL!")
            print("✅ All formatting and linting issues resolved")
            print("🚀 System is now CI-ready")
        else:
            print("⚠️ EMERGENCY FIX PARTIALLY SUCCESSFUL")
            print("Some issues may require manual intervention")
        
        return success
        
    except Exception as e:
        print("❌ Validation error:", str(e))
        return False

def main():
    """Main emergency fix execution"""
    try:
        success = execute_emergency_fix()
        
        print("\n🎯 NEXT STEPS:")
        if success:
            print("1. git add .")
            print("2. git commit -m 'Emergency fix: Black + Ruff formatting'")
            print("3. git push")
            print("4. Monitor CI for green status")
            sys.exit(0)
        else:
            print("1. Review remaining issues")
            print("2. Manual intervention may be required")
            print("3. Run nuclear_cleanup.py for detailed analysis")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Emergency fix interrupted")
        sys.exit(1)
    except Exception as e:
        print("\n💥 Emergency fix failed:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
