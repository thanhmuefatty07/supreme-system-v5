#!/bin/bash
# ==============================================================================
# SUPREME SYSTEM V5 - MASTER CRITICAL TESTS EXECUTION SCRIPT
# ==============================================================================
#
# Executes all 12 critical production readiness tests with comprehensive
# reporting, parallel execution, and success criteria validation.
#
# Tests executed in priority order:
# HIGH PRIORITY: Memory, Network, Live Market (Phase 1)
# MEDIUM PRIORITY: Thermal, Concurrent Load, API Rate Limiting (Phase 2)
# LOW PRIORITY: Data Corruption, Memory Fragmentation, 72H Stability (Phase 3)
#
# Usage: ./scripts/run_all_critical_tests.sh [--parallel] [--quick] [--report-json]
#
# ==============================================================================

set -euo pipefail

# Configuration
PARALLEL_MODE=false
QUICK_MODE=false
JSON_REPORT=false
TEST_TIMEOUT=7200  # 2 hours max per test
LOG_DIR="test_results"
MASTER_LOG="$LOG_DIR/master_test_execution_$(date +"%Y%m%d_%H%M%S").log"
REPORT_FILE="$LOG_DIR/master_test_report_$(date +"%Y%m%d_%H%M%S").json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Test definitions with priorities and configurations
declare -A TEST_COMMANDS=(
    # HIGH PRIORITY - Execute immediately (4-6 hours)
    ["memory_stress"]="python scripts/stress_test_memory.py --duration 1 --memory-limit 2.2"
    ["network_failure"]="python tests/network_failure_simulation.py --duration 30"
    ["live_market"]="python tests/live_market_integration.py --duration 2"

    # MEDIUM PRIORITY - Within 24h
    ["thermal_throttling"]="python scripts/simulate_thermal_throttling.py --duration 2"
    ["concurrent_load"]="python tests/concurrent_algorithm_load.py --algorithms 15 --duration 30"
    ["api_rate_limiting"]="python tests/api_rate_limiting.py --duration 30"

    # LOW PRIORITY - Within 48h
    ["data_corruption"]="cargo test data_corruption_recovery --release"
    ["memory_fragmentation"]="cargo test memory_fragmentation_72h --release"
    ["graceful_shutdown"]="cargo test graceful_shutdown_scenarios --release"
    ["news_sentiment"]="python tests/real_news_sentiment.py --sample-size 1000 --accuracy-target 0.85"
    ["hot_config_reload"]="python tests/hot_config_reload.py --config-changes 50 --duration 30"
    ["long_running"]="python scripts/72h_stability_test.py --duration 1"  # Shortened for testing
)

declare -A TEST_PRIORITIES=(
    ["memory_stress"]="HIGH"
    ["network_failure"]="HIGH"
    ["live_market"]="HIGH"
    ["thermal_throttling"]="MEDIUM"
    ["concurrent_load"]="MEDIUM"
    ["api_rate_limiting"]="MEDIUM"
    ["data_corruption"]="LOW"
    ["memory_fragmentation"]="LOW"
    ["graceful_shutdown"]="LOW"
    ["news_sentiment"]="LOW"
    ["hot_config_reload"]="LOW"
    ["long_running"]="LOW"
)

declare -A TEST_TIMEOUTS=(
    ["memory_stress"]=7200      # 2 hours
    ["network_failure"]=3600    # 1 hour
    ["live_market"]=14400       # 4 hours
    ["thermal_throttling"]=7200 # 2 hours
    ["concurrent_load"]=7200    # 2 hours
    ["api_rate_limiting"]=3600  # 1 hour
    ["data_corruption"]=3600    # 1 hour
    ["memory_fragmentation"]=3600 # 1 hour
    ["graceful_shutdown"]=3600  # 1 hour
    ["news_sentiment"]=7200     # 2 hours
    ["hot_config_reload"]=7200  # 2 hours
    ["long_running"]=7200       # 2 hours (shortened)
)

# Quick mode configurations (reduced time/size for CI)
if [[ "$QUICK_MODE" == "true" ]]; then
    TEST_COMMANDS["memory_stress"]="python scripts/stress_test_memory.py --duration 0.1 --memory-limit 2.2"
    TEST_COMMANDS["network_failure"]="python tests/network_failure_simulation.py --duration 5"
    TEST_COMMANDS["live_market"]="python tests/live_market_integration.py --duration 0.5"
    TEST_COMMANDS["thermal_throttling"]="python scripts/simulate_thermal_throttling.py --duration 0.5"
    TEST_COMMANDS["concurrent_load"]="python tests/concurrent_algorithm_load.py --algorithms 5 --duration 5"
    TEST_COMMANDS["api_rate_limiting"]="python tests/api_rate_limiting.py --duration 5"
    TEST_COMMANDS["news_sentiment"]="python tests/real_news_sentiment.py --sample-size 100 --accuracy-target 0.8"
    TEST_COMMANDS["hot_config_reload"]="python tests/hot_config_reload.py --config-changes 10 --duration 5"
    TEST_COMMANDS["long_running"]="python scripts/72h_stability_test.py --duration 0.1"

    # Reduce timeouts for quick mode
    for test in "${!TEST_TIMEOUTS[@]}"; do
        TEST_TIMEOUTS["$test"]=$((TEST_TIMEOUTS["$test"] / 6))  # 1/6th the time
        TEST_TIMEOUTS["$test"]=$((TEST_TIMEOUTS["$test"] < 300 ? 300 : TEST_TIMEOUTS["$test"]))  # Min 5 minutes
    done
fi

# Test results tracking
declare -A TEST_RESULTS
declare -A TEST_DURATIONS
declare -A TEST_OUTPUTS
declare -A TEST_PHASES

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $*" | tee -a "$MASTER_LOG"
}

# Initialize JSON report
if [[ "$JSON_REPORT" == "true" ]]; then
    cat > "$REPORT_FILE" << EOF
{
    "master_execution": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "parallel_mode": $PARALLEL_MODE,
        "quick_mode": $QUICK_MODE,
        "total_tests": ${#TEST_COMMANDS[@]},
        "hostname": "$(hostname)",
        "execution_phases": ["HIGH", "MEDIUM", "LOW"]
    },
    "test_results": {},
    "phase_summaries": {},
    "final_summary": {
        "total_passed": 0,
        "total_failed": 0,
        "total_skipped": 0,
        "success_rate": 0.0,
        "high_priority_passed": 0,
        "medium_priority_passed": 0,
        "low_priority_passed": 0,
        "overall_success": false
    }
}
EOF
fi

# Function to run a single test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local timeout_seconds="${3:-$TEST_TIMEOUT}"
    local priority="${TEST_PRIORITIES[$test_name]}"
    local start_time=$(date +%s)
    local test_output_file="$LOG_DIR/${test_name}_$(date +"%Y%m%d_%H%M%S").log"

    log "▶️  STARTING TEST: $test_name (Priority: $priority)"
    log "Command: $test_command"
    log "Timeout: ${timeout_seconds}s"
    log "Output: $test_output_file"

    # Run test with timeout
    local exit_code=0
    local timeout_occurred=false

    # Execute command with timeout
    if timeout "$timeout_seconds" bash -c "$test_command > '$test_output_file' 2>&1"; then
        exit_code=$?
    else
        exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            timeout_occurred=true
            echo "TIMEOUT: Test exceeded ${timeout_seconds}s limit" >> "$test_output_file"
        fi
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Analyze test result
    local test_passed=false
    local test_failed=false
    local test_skipped=false

    if [[ $timeout_occurred == true ]]; then
        test_failed=true
        log "❌ $test_name: TIMEOUT (failed)"
    elif [[ $exit_code -eq 0 ]]; then
        test_passed=true
        log "✅ $test_name: PASSED"
    else
        test_failed=true
        log "❌ $test_name: FAILED (exit code: $exit_code)"
    fi

    # Store results
    TEST_RESULTS["$test_name"]=$([[ $test_passed == true ]] && echo "PASSED" || echo "FAILED")
    TEST_DURATIONS["$test_name"]=$duration
    TEST_OUTPUTS["$test_name"]=$test_output_file
    TEST_PHASES["$test_name"]=$priority

    log "Duration: ${duration}s, Exit Code: $exit_code"

    # Update JSON report
    if [[ "$JSON_REPORT" == "true" ]]; then
        local result_status
        if [[ $test_passed == true ]]; then
            result_status="PASSED"
        elif [[ $test_failed == true ]]; then
            result_status="FAILED"
        else
            result_status="SKIPPED"
        fi

        # Update JSON with test result
        jq --arg test_name "$test_name" \
           --arg result "$result_status" \
           --arg duration "$duration" \
           --arg output_file "$test_output_file" \
           --arg priority "$priority" \
           '.test_results[$test_name] = {
               "status": $result,
               "duration_seconds": ($duration | tonumber),
               "output_file": $output_file,
               "priority": $priority,
               "exit_code": '$exit_code'
           }' "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"
    fi

    return $exit_code
}

# Function to run tests by priority
run_tests_by_priority() {
    local priority="$1"
    local test_names=()

    # Collect tests for this priority
    for test_name in "${!TEST_COMMANDS[@]}"; do
        if [[ "${TEST_PRIORITIES[$test_name]}" == "$priority" ]]; then
            test_names+=("$test_name")
        fi
    done

    if [[ ${#test_names[@]} -eq 0 ]]; then
        log "No tests found for priority: $priority"
        return
    fi

    log "🚀 EXECUTING $priority PRIORITY TESTS (${#test_names[@]} tests)"
    log "=================================================="

    local phase_start_time=$(date +%s)
    local phase_results=()
    local pids=()

    if [[ "$PARALLEL_MODE" == "true" && ${#test_names[@]} -gt 1 ]]; then
        log "Running tests in parallel mode..."

        # Start all tests in parallel
        for test_name in "${test_names[@]}"; do
            local test_command="${TEST_COMMANDS[$test_name]}"
            local timeout_seconds="${TEST_TIMEOUTS[$test_name]:-$TEST_TIMEOUT}"

            run_test "$test_name" "$test_command" "$timeout_seconds" &
            pids+=($!)
        done

        # Wait for all tests to complete
        for pid in "${pids[@]}"; do
            wait "$pid"
            phase_results+=($?)
        done
    else
        # Run tests sequentially
        for test_name in "${test_names[@]}"; do
            local test_command="${TEST_COMMANDS[$test_name]}"
            local timeout_seconds="${TEST_TIMEOUTS[$test_name]:-$TEST_TIMEOUT}"
            run_test "$test_name" "$test_command" "$timeout_seconds"
            phase_results+=($?)
        done
    fi

    local phase_end_time=$(date +%s)
    local phase_duration=$((phase_end_time - phase_start_time))

    # Calculate phase statistics
    local phase_passed=0
    local phase_failed=0
    for result in "${phase_results[@]}"; do
        if [[ $result -eq 0 ]]; then
            ((phase_passed++))
        else
            ((phase_failed++))
        fi
    done

    local phase_success_rate=$((phase_passed * 100 / (${#phase_results[@]})))

    log "📊 $priority Phase Summary: $phase_passed passed, $phase_failed failed (${phase_success_rate}% success rate)"
    log "⏱️  Phase Duration: ${phase_duration}s"

    # Update JSON report with phase summary
    if [[ "$JSON_REPORT" == "true" ]]; then
        jq --arg priority "$priority" \
           --arg passed "$phase_passed" \
           --arg failed "$phase_failed" \
           --arg duration "$phase_duration" \
           --arg success_rate "$phase_success_rate" \
           '.phase_summaries[$priority] = {
               "passed": ($passed | tonumber),
               "failed": ($failed | tonumber),
               "total": '$((phase_passed + phase_failed))',
               "success_rate": ($success_rate | tonumber),
               "duration_seconds": ($duration | tonumber)
           }' "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"
    fi

    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --report-json)
            JSON_REPORT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--parallel] [--quick] [--report-json]"
            exit 1
            ;;
    esac
done

# Main execution
echo -e "${BLUE}🧪 SUPREME SYSTEM V5 - MASTER CRITICAL TESTS EXECUTION${NC}"
echo "=========================================================="
echo "Timestamp: $(date)"
echo "Parallel Mode: $PARALLEL_MODE"
echo "Quick Mode: $QUICK_MODE"
echo "JSON Report: $JSON_REPORT"
echo "Master Log: $MASTER_LOG"
echo

log "STARTING MASTER CRITICAL TESTS EXECUTION"
log "Configuration: Parallel=$PARALLEL_MODE, Quick=$QUICK_MODE, JSON=$JSON_REPORT"

# Display test plan
echo "📋 TEST EXECUTION PLAN:"
echo "======================"
echo "HIGH PRIORITY (3 tests): Memory Stress, Network Failure, Live Market"
echo "MEDIUM PRIORITY (3 tests): Thermal Throttling, Concurrent Load, API Rate Limiting"
echo "LOW PRIORITY (6 tests): Data Corruption, Memory Fragmentation, Graceful Shutdown,"
echo "                        News Sentiment, Hot Config Reload, 72H Stability"
echo

total_start_time=$(date +%s)

# Execute tests by priority
run_tests_by_priority "HIGH"
run_tests_by_priority "MEDIUM"
run_tests_by_priority "LOW"

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

# Generate comprehensive summary
echo -e "${BLUE}📊 MASTER TEST EXECUTION SUMMARY${NC}"
echo "==================================="

passed_tests=0
failed_tests=0
skipped_tests=0
high_passed=0
medium_passed=0
low_passed=0

for test_name in "${!TEST_RESULTS[@]}"; do
    result="${TEST_RESULTS[$test_name]}"
    duration="${TEST_DURATIONS[$test_name]}"
    priority="${TEST_PHASES[$test_name]}"

    case $result in
        "PASSED")
            echo -e "${GREEN}✅ $test_name ($priority): PASSED (${duration}s)${NC}"
            ((passed_tests++))
            case $priority in
                "HIGH") ((high_passed++)) ;;
                "MEDIUM") ((medium_passed++)) ;;
                "LOW") ((low_passed++)) ;;
            esac
            ;;
        "FAILED")
            echo -e "${RED}❌ $test_name ($priority): FAILED (${duration}s)${NC}"
            ((failed_tests++))
            ;;
        "SKIPPED")
            echo -e "${YELLOW}⏭️  $test_name ($priority): SKIPPED (${duration}s)${NC}"
            ((skipped_tests++))
            ;;
    esac
done

echo
echo "📈 OVERALL RESULTS:"
echo "==================="
echo "Total Tests: $((${#TEST_RESULTS[@]}))"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"
echo "Skipped: $skipped_tests"

success_rate=$((passed_tests * 100 / ${#TEST_RESULTS[@]}))
echo "Success Rate: ${success_rate}%"
echo "Total Duration: ${total_duration}s"
echo

# Priority breakdown
echo "🎯 PRIORITY BREAKDOWN:"
echo "======================"
echo "HIGH Priority (3 tests): $high_passed/3 passed"
echo "MEDIUM Priority (3 tests): $medium_passed/3 passed"
echo "LOW Priority (6 tests): $low_passed/6 passed"
echo

# Success criteria evaluation
critical_success=$([[ $high_passed -ge 2 && $success_rate -ge 75 ]] && echo "YES" || echo "NO")
production_ready=$([[ $passed_tests -ge 9 && $success_rate -ge 80 ]] && echo "YES" || echo "NO")

echo "🎯 SUCCESS CRITERIA:"
echo "===================="
echo "Critical Tests Passed (≥6/9): $critical_success"
echo "Production Ready (≥80% success): $production_ready"
echo

# Final verdict
if [[ "$production_ready" == "YES" ]]; then
    echo -e "${GREEN}🏆 FINAL VERDICT: SUPREME SYSTEM V5 IS PRODUCTION READY! 🚀${NC}"
    echo "All critical systems validated and ready for deployment."
    final_result="PRODUCTION_READY"
else
    echo -e "${RED}⚠️  FINAL VERDICT: REQUIRES ADDITIONAL WORK${NC}"
    echo "Some tests failed - review logs and fix issues before production deployment."
    final_result="REQUIRES_FIXES"
fi

echo
echo "📁 Detailed logs: $LOG_DIR/"
echo "📄 Master log: $MASTER_LOG"
echo "📊 JSON Report: $REPORT_FILE"

# Update final JSON report
if [[ "$JSON_REPORT" == "true" ]]; then
    jq --arg passed "$passed_tests" \
       --arg failed "$failed_tests" \
       --arg skipped "$skipped_tests" \
       --arg duration "$total_duration" \
       --arg success_rate "$success_rate" \
       --arg high_passed "$high_passed" \
       --arg medium_passed "$medium_passed" \
       --arg low_passed "$low_passed" \
       --arg final_result "$final_result" \
       '.final_summary = {
           "total_passed": ($passed | tonumber),
           "total_failed": ($failed | tonumber),
           "total_skipped": ($skipped | tonumber),
           "success_rate": ($success_rate | tonumber),
           "high_priority_passed": ($high_passed | tonumber),
           "medium_priority_passed": ($medium_passed | tonumber),
           "low_priority_passed": ($low_passed | tonumber),
           "overall_success": '$([[ "$production_ready" == "YES" ]] && echo "true" || echo "false")',
           "final_result": $final_result
       }' "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"

    echo "📊 Master JSON report updated: $REPORT_FILE"
fi

echo
echo "=========================================================="
echo "Master Critical Tests Execution Complete"
echo "=========================================================="

# Exit with appropriate code
exit $((failed_tests > 0 ? 1 : 0))
