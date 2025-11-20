#!/usr/bin/env python3
"""
Wrapper script to run Smart Coverage Hunter with better control
"""

import subprocess
import signal
import sys
import time

def signal_handler(sig, frame):
    print('\n⏹️ Received interrupt signal. Stopping smart hunter...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("🚀 Starting Smart Coverage Hunter...")
    print("Press Ctrl+C to stop gracefully")
    print("-" * 50)

    try:
        # Run the smart hunter
        result = subprocess.run([
            sys.executable, "scripts/smart_coverage_hunter.py"
        ], check=False)

        if result.returncode == 0:
            print("✅ Smart Hunter completed successfully!")
        else:
            print(f"⚠️ Smart Hunter finished with code: {result.returncode}")

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"❌ Error running smart hunter: {e}")

    print("\n📊 Checking final results...")
    # Could add coverage check here
