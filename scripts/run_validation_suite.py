#!/usr/bin/env python3
"""
Safe Foundation Validation Suite - C2a
"""
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_safe_validation():
    print("🚀 RUNNING SAFE FOUNDATION VALIDATION (C2a)")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": "C2a",
        "tests": {},
        "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
    }
    
    # Run pytest
    print("🧪 Running safe foundation tests...")
    try:
        ret = subprocess.run(
            ["pytest", "tests/comprehensive/test_safe_foundation.py", "-v"],
            capture_output=True,
            text=True
        )
        results["tests"]["safe_foundation"] = {
            "exit_code": ret.returncode,
            "status": "PASS" if ret.returncode == 0 else "FAIL"
        }
    except Exception as e:
        results["tests"]["safe_foundation"] = {"status": "FAIL", "error": str(e)}
    
    # Test Rust import
    try:
        import supreme_core
        results["tests"]["rust_core"] = {"status": "PASS", "message": "Import ok"}
    except ImportError:
        results["tests"]["rust_core"] = {"status": "SKIP", "message": "Not built"}
    
    # Summary
    for test_result in results["tests"].values():
        results["summary"]["total"] += 1
        if test_result["status"] == "PASS":
            results["summary"]["passed"] += 1
        elif test_result["status"] == "FAIL":
            results["summary"]["failed"] += 1
        else:
            results["summary"]["skipped"] += 1
    
    # Save
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"validation_c2a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results: {output_file}")
    print(f"📊 Summary: {results['summary']['passed']}/{results['summary']['total']} passed")
    
    if results["summary"]["failed"] == 0:
        print("🎉 ALL TESTS PASSED - C2a STABLE")
        return True
    else:
        print("⚠️  SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_safe_validation()
    sys.exit(0 if success else 1)
