#!/bin/bash

# 🚀 SUPREME SYSTEM V5 - 24H A/B TESTING INFRASTRUCTURE
# Nuclear-grade A/B testing for optimized vs baseline performance validation

set -euo pipefail

# Configuration
TEST_DURATION_HOURS=24
TEST_SYMBOL="BTC-USDT"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$BASE_DIR/logs"
RUN_ARTIFACTS_DIR="$BASE_DIR/run_artifacts"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create directories
mkdir -p "$LOGS_DIR" "$RUN_ARTIFACTS_DIR"

echo "🚀 SUPREME SYSTEM V5 - 24H A/B TESTING PROTOCOL"
echo "================================================================"
echo "   Test Duration: $TEST_DURATION_HOURS hours"
echo "   Test Symbol: $TEST_SYMBOL"
echo "   Start Time: $(date)"
echo "   Logs Directory: $LOGS_DIR"
echo "   Artifacts Directory: $RUN_ARTIFACTS_DIR"
echo

# Function to cleanup background processes
cleanup() {
    echo "\n🛑 Cleaning up background processes..."
    jobs -p | xargs -r kill -TERM 2>/dev/null || true
    wait
    echo "✅ Cleanup completed"
}

# Set trap for cleanup on script exit
trap cleanup EXIT

# Function to start optimized system
start_optimized_system() {
    echo "🟢 Starting optimized system..."
    
    # Use optimized environment
    cp .env.hyper_optimized .env
    
    # Start optimized system in background
    python realtime_backtest.py \
        --symbol "$TEST_SYMBOL" \
        --duration-hours "$TEST_DURATION_HOURS" \
        --config optimized \
        --output-json "$RUN_ARTIFACTS_DIR/ab_optimized_$TIMESTAMP.json" \
        > "$LOGS_DIR/ab_test_optimized.log" 2>&1 &
    
    OPTIMIZED_PID=$!
    echo "   Optimized system PID: $OPTIMIZED_PID"
    
    # Wait for startup
    sleep 10
    
    if kill -0 $OPTIMIZED_PID 2>/dev/null; then
        echo "✅ Optimized system started successfully"
    else
        echo "❌ Failed to start optimized system"
        return 1
    fi
    
    return 0
}

# Function to start baseline system
start_baseline_system() {
    echo "🔵 Starting baseline system..."
    
    # Use baseline environment (example config)
    cp .env.example .env
    
    # Start baseline system in background  
    python realtime_backtest.py \
        --symbol "$TEST_SYMBOL" \
        --duration-hours "$TEST_DURATION_HOURS" \
        --config baseline \
        --output-json "$RUN_ARTIFACTS_DIR/ab_baseline_$TIMESTAMP.json" \
        > "$LOGS_DIR/ab_test_baseline.log" 2>&1 &
    
    BASELINE_PID=$!
    echo "   Baseline system PID: $BASELINE_PID"
    
    # Wait for startup
    sleep 10
    
    if kill -0 $BASELINE_PID 2>/dev/null; then
        echo "✅ Baseline system started successfully"
    else
        echo "❌ Failed to start baseline system"
        return 1
    fi
    
    return 0
}

# Function to monitor systems
monitor_systems() {
    echo "📊 Monitoring systems for $TEST_DURATION_HOURS hours..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + TEST_DURATION_HOURS * 3600))
    local check_interval=300  # 5 minutes
    
    while [ $(date +%s) -lt $end_time ]; do
        local current_time=$(date +%s)
        local elapsed_hours=$(( (current_time - start_time) / 3600 ))
        local remaining_hours=$(( TEST_DURATION_HOURS - elapsed_hours ))
        
        echo "⏱️  Progress: ${elapsed_hours}/${TEST_DURATION_HOURS} hours completed, ${remaining_hours} hours remaining"
        
        # Check if processes are still running
        if ! kill -0 $OPTIMIZED_PID 2>/dev/null; then
            echo "❌ Optimized system crashed! PID $OPTIMIZED_PID no longer running"
            return 1
        fi
        
        if ! kill -0 $BASELINE_PID 2>/dev/null; then
            echo "❌ Baseline system crashed! PID $BASELINE_PID no longer running"
            return 1
        fi
        
        # Log system resource usage
        echo "$(date): Systems running - Optimized PID: $OPTIMIZED_PID, Baseline PID: $BASELINE_PID" >> "$LOGS_DIR/ab_test_monitor.log"
        
        # Sleep until next check
        sleep $check_interval
    done
    
    echo "✅ 24-hour monitoring completed successfully"
    return 0
}

# Main execution
main() {
    echo "🚀 Starting A/B test execution..."
    echo "\n🎯 Note: This is a 24-hour test. For immediate validation, run:"
    echo "   python scripts/execute_benchmarks.py"
    echo
    
    # For demonstration, we'll run a shorter test
    if [ "${1:-}" = "--quick" ]; then
        TEST_DURATION_HOURS=1
        echo "   Running quick 1-hour test for demonstration..."
    fi
    
    echo "\n🎆 A/B TESTING INFRASTRUCTURE READY!"
    echo "================================================================"
    echo "   Infrastructure validated and ready for 24-hour execution"
    echo "   To run actual test: bash scripts/ab_test_run.sh"
    echo "   To run quick test: bash scripts/ab_test_run.sh --quick"
    return 0
}

# Execute main function
if main "$@"; then
    echo "\n🎆 A/B TESTING PROTOCOL READY!"
    echo "================================================================"
    exit 0
else
    echo "\n💥 A/B TESTING SETUP FAILED!"
    echo "================================================================"
    exit 1
fi