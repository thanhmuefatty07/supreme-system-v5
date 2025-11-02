#!/bin/bash
# Supreme System V5 Production A/B Test Runner
# Compares OPTIMIZED_MODE=true vs false performance over 24 hours

set -e

# Configuration
TEST_DURATION_HOURS=${TEST_DURATION_HOURS:-24}
SYMBOL=${SYMBOL:-"BTC-USDT"}
OPTIMIZED_CONFIG=${OPTIMIZED_CONFIG:-".env.optimized"}
STANDARD_CONFIG=${STANDARD_CONFIG:-".env.standard"}
RESULTS_DIR=${RESULTS_DIR:-"docs/reports/ab_test_$(date +%Y%m%d_%H%M%S)"}

echo "🚀 SUPREME SYSTEM V5 - PRODUCTION A/B TEST"
echo "=========================================="
echo "Symbol: $SYMBOL"
echo "Duration: ${TEST_DURATION_HOURS} hours"
echo "Results: $RESULTS_DIR"
echo

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to run test with configuration
run_test() {
    local config_file=$1
    local test_name=$2
    local result_file="$RESULTS_DIR/${test_name}_results.json"

    echo "🧪 Running $test_name test..."

    # Copy configuration
    cp "$config_file" .env

    # Start system (background)
    python -m supreme_system_v5.core &
    SYSTEM_PID=$!

    # Wait for system to initialize
    sleep 30

    # Run load test
    python scripts/load_single_symbol.py \
        --symbol "$SYMBOL" \
        --rate 20 \
        --duration-min $((TEST_DURATION_HOURS * 60)) \
        --no-monitoring > "$RESULTS_DIR/${test_name}_load_test.log" 2>&1

    # Collect final metrics
    python -c "
import json
import time
from supreme_system_v5.monitoring import AdvancedResourceMonitor

# Initialize monitor
config = {
    'cpu_high_threshold': 88.0,
    'memory_high_threshold': 3.86,
    'latency_high_threshold': 200,
    'monitoring_interval': 5.0,
    'optimization_check_interval': 60.0
}

monitor = AdvancedResourceMonitor(config)
monitor.start_monitoring()

# Collect metrics for 5 minutes
print('Collecting final metrics...')
import time
time.sleep(300)

# Get final report
health = monitor.get_system_health_report()
performance = monitor.get_performance_metrics()
slo_report = monitor.get_slo_report()

monitor.stop_monitoring()

result = {
    'test_name': '$test_name',
    'timestamp': time.time(),
    'symbol': '$SYMBOL',
    'duration_hours': $TEST_DURATION_HOURS,
    'system_health': health,
    'performance_metrics': performance,
    'slo_report': slo_report
}

with open('$result_file', 'w') as f:
    json.dump(result, f, indent=2)

print(f'Results saved to $result_file')
" > "$RESULTS_DIR/${test_name}_metrics.log" 2>&1

    # Stop system
    kill $SYSTEM_PID 2>/dev/null || true
    wait $SYSTEM_PID 2>/dev/null || true

    echo "✅ $test_name test completed"
    echo
}

# Create standard configuration (optimized mode disabled)
cat > .env.standard << 'EOF'
# Supreme System V5 Standard Configuration (Non-Optimized)
TRADING_MODE=sandbox
SINGLE_SYMBOL=BTC-USDT
INITIAL_BALANCE=10000.0
BASE_POSITION_SIZE_PCT=0.02
STOP_LOSS_PCT=0.01
TAKE_PROFIT_PCT=0.02

# Disable optimizations
OPTIMIZED_MODE=false
EVENT_DRIVEN_PROCESSING=false
INTELLIGENT_CACHING=false
PERFORMANCE_PROFILE=performance

# Standard intervals
PROCESS_INTERVAL_SECONDS=60
TECHNICAL_INTERVAL=60
NEWS_INTERVAL_MIN=15
WHALE_INTERVAL_MIN=15
MTF_INTERVAL=180

# Relaxed resource limits for comparison
MAX_CPU_PERCENT=95.0
MAX_RAM_GB=4.5
TARGET_EVENT_SKIP_RATIO=0.0

# Enable all components
TECHNICAL_ANALYSIS_ENABLED=true
NEWS_ANALYSIS_ENABLED=true
WHALE_TRACKING_ENABLED=true
MULTI_TIMEFRAME_ENABLED=true
RISK_MANAGEMENT_ENABLED=true
RESOURCE_MONITORING_ENABLED=false

# Standard data sources
BINANCE_API_ENABLED=true
COINGECKO_API_ENABLED=true
OKX_API_ENABLED=true
CRYPTOPANIC_ENABLED=true
EOF

echo "📋 Test Configurations:"
echo "  Optimized: $OPTIMIZED_CONFIG"
echo "  Standard:  $STANDARD_CONFIG"
echo

# Check if configurations exist
if [ ! -f "$OPTIMIZED_CONFIG" ]; then
    echo "❌ Optimized config not found: $OPTIMIZED_CONFIG"
    exit 1
fi

echo "🎯 Starting A/B Test Sequence..."

# Test 1: Standard (non-optimized) mode
run_test ".env.standard" "standard"

# Cool down period
echo "⏳ Cooling down for 5 minutes..."
sleep 300

# Test 2: Optimized mode
run_test "$OPTIMIZED_CONFIG" "optimized"

echo "📊 Generating A/B Test Report..."
python scripts/report_ab.py "$RESULTS_DIR" > "$RESULTS_DIR/ab_test_report.md"

echo "🎉 A/B Test Complete!"
echo "Results: $RESULTS_DIR"
echo "Report: $RESULTS_DIR/ab_test_report.md"
echo

# Display summary
echo "📈 QUICK SUMMARY:"
cat "$RESULTS_DIR/ab_test_report.md" | grep -E "(WINNER|RECOMMENDATION|SIGNIFICANT)" | head -10
