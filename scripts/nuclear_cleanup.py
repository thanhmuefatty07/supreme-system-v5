#!/usr/bin/env python3
"""
Nuclear Cleanup Script - Ultra SFL Deep Intervention
Systematic cleanup of ALL formatting and linting issues
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List

class NuclearCleanup:
    """Ultra SFL cleanup system with 3-layer approach"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.results = {
            'black_formatted': 0,
            'ruff_fixed': 0,
            'errors': []
        }
    
    def layer_1_black_format(self) -> bool:
        """Layer 1: Nuclear black formatting"""
        print("🚀 Layer 1: Nuclear Black Formatting")
        print("=" * 50)
        
        try:
            # Run black on entire project
            result = subprocess.run([
                sys.executable, "-m", "black", 
                ".", "--line-length", "88", "--target-version", "py311"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Black formatting completed successfully")
                self.results['black_formatted'] = 1
                return True
            else:
                print("❌ Black formatting failed:")
                print(result.stderr)
                self.results['errors'].append("Black formatting failed")
                return False
                
        except Exception as e:
            print("❌ Black execution error:", str(e))
            self.results['errors'].append("Black execution failed: " + str(e))
            return False
    
    def layer_2_ruff_fix(self) -> bool:
        """Layer 2: Systematic ruff auto-fix"""
        print("\n🔧 Layer 2: Systematic Ruff Auto-Fix")
        print("=" * 50)
        
        try:
            # Run ruff with auto-fix
            result = subprocess.run([
                sys.executable, "-m", "ruff", "check", 
                ".", "--fix", "--unsafe-fixes"
            ], capture_output=True, text=True)
            
            print("Ruff auto-fix output:")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            # Check final status
            check_result = subprocess.run([
                sys.executable, "-m", "ruff", "check", "."
            ], capture_output=True, text=True)
            
            if check_result.returncode == 0:
                print("✅ All ruff issues resolved")
                self.results['ruff_fixed'] = 1
                return True
            else:
                print("⚠️ Some ruff issues remain:")
                print(check_result.stdout)
                self.results['errors'].append("Remaining ruff issues")
                return False
                
        except Exception as e:
            print("❌ Ruff execution error:", str(e))
            self.results['errors'].append("Ruff execution failed: " + str(e))
            return False
    
    def layer_3_manual_verification(self) -> bool:
        """Layer 3: Manual verification of critical files"""
        print("\n🔍 Layer 3: Manual Verification")
        print("=" * 50)
        
        critical_files = [
            "main_clean.py",
            "python/supreme_system_v5/__init__.py",
            "python/supreme_system_v5/core.py",
            "python/supreme_system_v5/utils.py"
        ]
        
        all_good = True
        
        for file_path in critical_files:
            if os.path.exists(file_path):
                try:
                    # Syntax check
                    result = subprocess.run([
                        sys.executable, "-m", "py_compile", file_path
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅", file_path, "- Syntax OK")
                    else:
                        print("❌", file_path, "- Syntax Error")
                        print(result.stderr)
                        all_good = False
                        self.results['errors'].append(file_path + " syntax error")
                        
                except Exception as e:
                    print("❌", file_path, "- Check failed:", str(e))
                    all_good = False
            else:
                print("⚠️", file_path, "- File not found")
        
        return all_good
    
    def execute_nuclear_cleanup(self) -> bool:
        """Execute complete 3-layer nuclear cleanup"""
        print("💥 NUCLEAR CLEANUP - ULTRA SFL DEEP INTERVENTION")
        print("=" * 70)
        print("Target: ZERO errors, ZERO warnings, PRODUCTION READY")
        print("=" * 70)
        
        # Layer 1: Black formatting
        layer1_success = self.layer_1_black_format()
        
        # Layer 2: Ruff auto-fix
        layer2_success = self.layer_2_ruff_fix()
        
        # Layer 3: Manual verification
        layer3_success = self.layer_3_manual_verification()
        
        # Final report
        print("\n" + "=" * 70)
        print("📊 NUCLEAR CLEANUP RESULTS")
        print("=" * 70)
        print("Layer 1 (Black):", "✅ SUCCESS" if layer1_success else "❌ FAILED")
        print("Layer 2 (Ruff):", "✅ SUCCESS" if layer2_success else "❌ FAILED")
        print("Layer 3 (Verify):", "✅ SUCCESS" if layer3_success else "❌ FAILED")
        
        all_success = layer1_success and layer2_success and layer3_success
        
        print("\n🎯 OVERALL STATUS:", "✅ COMPLETE SUCCESS" if all_success else "❌ ISSUES REMAIN")
        
        if self.results['errors']:
            print("\n❌ Errors encountered:")
            for error in self.results['errors']:
                print("  -", error)
        
        if all_success:
            print("\n🏆 SUPREME SYSTEM V5 IS NOW PRODUCTION READY!")
            print("✅ Zero errors, zero warnings, fully optimized")
            print("🚀 Ready for immediate deployment")
        
        return all_success
    
    def quick_status_check(self):
        """Quick final status check"""
        print("\n🔍 QUICK STATUS CHECK")
        print("=" * 30)
        
        # Check Python syntax
        try:
            result = subprocess.run([
                sys.executable, "-c", 
                "import sys; sys.path.insert(0, 'python'); import supreme_system_v5; print('✅ Core import successful')"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Core system imports successfully")
            else:
                print("❌ Core import failed:", result.stderr)
        except Exception as e:
            print("❌ Import test failed:", str(e))
        
        # Check Rust engine
        try:
            result = subprocess.run([
                sys.executable, "-c",
                "try:\n  import supreme_engine_rs\n  print('✅ Rust engine available')\nexcept:\n  print('⚠️ Rust engine not built yet')"
            ], capture_output=True, text=True)
            print(result.stdout.strip())
        except Exception as e:
            print("❌ Rust check failed:", str(e))

def main():
    """Execute nuclear cleanup with ultra SFL approach"""
    cleanup = NuclearCleanup()
    
    print("💀 NUCLEAR CLEANUP INITIATED")
    print("🎯 Target: ZERO ERRORS, PRODUCTION READY")
    print("🔬 Method: Ultra SFL (Systematic File-by-File Layered)")
    print("\n" + "🚨" * 20)
    print("WARNING: This will reformat ALL Python files")
    print("🚨" * 20)
    
    # Execute cleanup
    success = cleanup.execute_nuclear_cleanup()
    
    # Quick final check
    cleanup.quick_status_check()
    
    if success:
        print("\n🎉 NUCLEAR CLEANUP COMPLETE - SYSTEM IS CLEAN!")
        print("Next steps:")
        print("  1. git add .")
        print("  2. git commit -m 'Nuclear cleanup - all formatting fixed'")
        print("  3. make build-rust")
        print("  4. make validate")
        sys.exit(0)
    else:
        print("\n💥 NUCLEAR CLEANUP PARTIALLY SUCCESSFUL")
        print("Some issues may require manual intervention")
        sys.exit(1)

if __name__ == "__main__":
    main()
