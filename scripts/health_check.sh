#!/bin/bash

# Supreme System V5 - Automated Health Check Script
# Phase 4: 24H Monitoring & Optimization

set -euo pipefail

# Configuration
HEALTH_LIVE_ENDPOINT="${HEALTH_LIVE_ENDPOINT:-http://localhost:8000/health/live}"
HEALTH_READY_ENDPOINT="${HEALTH_READY_ENDPOINT:-http://localhost:8000/health/ready}"
METRICS_ENDPOINT="${METRICS_ENDPOINT:-http://localhost:9090/metrics}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
MAX_FAILURES="${MAX_FAILURES:-3}"
LOG_FILE="${LOG_FILE:-logs/health_check.log}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Health check function
check_endpoint() {
    local endpoint=$1
    local name=$2
    
    if curl -f -s --max-time 5 "$endpoint" > /dev/null 2>&1; then
        return 0
    else
        log_error "$name check failed: $endpoint"
        return 1
    fi
}

# Perform health checks
perform_health_check() {
    local failures=0
    local overall_status="HEALTHY"
    
    # Check liveness
    if ! check_endpoint "$HEALTH_LIVE_ENDPOINT" "Liveness"; then
        failures=$((failures + 1))
        overall_status="UNHEALTHY"
    fi
    
    # Check readiness
    if ! check_endpoint "$HEALTH_READY_ENDPOINT" "Readiness"; then
        failures=$((failures + 1))
        if [ "$overall_status" != "UNHEALTHY" ]; then
            overall_status="DEGRADED"
        fi
    fi
    
    # Check metrics endpoint
    if ! check_endpoint "$METRICS_ENDPOINT" "Metrics"; then
        log_warning "Metrics endpoint unavailable"
    fi
    
    # Log result
    if [ $failures -eq 0 ]; then
        log_success "All health checks passed"
        return 0
    else
        log_error "Health check failed: $overall_status ($failures failures)"
        return 1
    fi
}

# Main monitoring loop
main() {
    log "Starting health check monitoring..."
    log "Endpoints:"
    log "  Live: $HEALTH_LIVE_ENDPOINT"
    log "  Ready: $HEALTH_READY_ENDPOINT"
    log "  Metrics: $METRICS_ENDPOINT"
    log "Check interval: ${CHECK_INTERVAL}s"
    log "Max failures: $MAX_FAILURES"
    log ""
    
    local consecutive_failures=0
    
    while true; do
        if perform_health_check; then
            consecutive_failures=0
        else
            consecutive_failures=$((consecutive_failures + 1))
            
            if [ $consecutive_failures -ge $MAX_FAILURES ]; then
                log_error "CRITICAL: Service unhealthy after $consecutive_failures consecutive failures"
                log_error "Alerting operations team..."
                
                # Send alert (implement based on your alerting system)
                # Example: curl -X POST "$ALERT_WEBHOOK_URL" -d "Service unhealthy"
                
                # Exit with error code
                exit 1
            fi
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# Run main function
main "$@"




