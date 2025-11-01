#!/usr/bin/env python3

"""
Performance optimization script for Supreme System V5

Sets process priorities and system optimizations
"""

import os
import sys
import subprocess
import psutil
from pathlib import Path

class PerformanceOptimizer:
    def __init__(self):
        self.trading_priority = -20  # Highest priority
        self.dashboard_priority = 19  # Lowest priority

    def setup_trading_process(self):
        """Configure trading process for maximum performance"""
        try:
            # Set process priority
            os.nice(self.trading_priority)
            print(f"✅ Set trading process priority to {self.trading_priority}")

            # Set CPU affinity (cores 0, 1, 2 for trading)
            trading_cores = [0, 1, 2]
            psutil.Process().cpu_affinity(trading_cores)
            print(f"✅ Set trading CPU affinity to cores {trading_cores}")

            # Lock memory to prevent swap
            if self._can_mlock():
                import mlock
                mlock.mlockall(mlock.MCL_CURRENT | mlock.MCL_FUTURE)
                print("✅ Memory locked to prevent swap delays")

            # Set I/O priority to real-time
            subprocess.run(['ionice', '-c', '1', '-n', '4', '-p', str(os.getpid())])
            print("✅ Set real-time I/O priority")

        except Exception as e:
            print(f"⚠️ Trading optimization warning: {e}")

    def setup_dashboard_process(self):
        """Configure dashboard process for background operation"""
        try:
            # Set low priority
            os.nice(self.dashboard_priority)
            print(f"✅ Set dashboard process priority to {self.dashboard_priority}")

            # Set CPU affinity (core 3 only for dashboard)
            dashboard_cores = [3]
            psutil.Process().cpu_affinity(dashboard_cores)
            print(f"✅ Set dashboard CPU affinity to core {dashboard_cores}")

            # Set I/O priority to idle
            subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())])
            print("✅ Set idle I/O priority")

        except Exception as e:
            print(f"⚠️ Dashboard optimization warning: {e}")

    def _can_mlock(self):
        """Check if process can lock memory"""
        try:
            import mlock
            return os.getuid() == 0 or self._has_capability('CAP_IPC_LOCK')
        except ImportError:
            return False

    def _has_capability(self, capability):
        """Check if process has specific capability"""
        try:
            result = subprocess.run(['getcap', '/proc/self/exe'],
                                  capture_output=True, text=True)
            return capability in result.stdout
        except:
            return False

def main():
    optimizer = PerformanceOptimizer()

    # Determine process type from environment
    process_type = os.getenv('PRIORITY', 'unknown')

    if process_type == 'trading':
        print("🎯 Optimizing for trading performance...")
        optimizer.setup_trading_process()
    elif process_type == 'dashboard':
        print("📊 Optimizing for dashboard background operation...")
        optimizer.setup_dashboard_process()
    else:
        print("❓ Unknown process type. Set PRIORITY environment variable.")
        sys.exit(1)

    print("✅ Performance optimization completed")

if __name__ == "__main__":
    main()
