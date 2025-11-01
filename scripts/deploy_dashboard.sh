#!/bin/bash

# Supreme System V5 - Dashboard Deployment Script
# Automated deployment with performance validation

set -euo pipefail

# Configuration
COMPOSE_FILE="docker-compose.performance.yml"
TRADING_CONTAINER="supreme-system-v5_supreme-trading_1"
DASHBOARD_CONTAINER="supreme-system-v5_supreme-dashboard_1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking deployment requirements..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is required but not installed"
        exit 1
    fi

    # Check system resources
    total_memory=$(free -m | awk 'NR==2{print $2}')
    if [ "$total_memory" -lt 4000 ]; then
        log_warning "System has less than 4GB RAM (${total_memory}MB detected)"
        log_warning "Performance may be impacted"
    fi

    log_success "Requirements check passed"
}

validate_configuration() {
    log_info "Validating configuration..."

    # Check if compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    # Validate compose file syntax
    if ! docker-compose -f "$COMPOSE_FILE" config --quiet; then
        log_error "Invalid Docker Compose configuration"
        exit 1
    fi

    # Check environment variables
    if [ ! -f ".env" ]; then
        log_warning ".env file not found, using defaults"
        cp .env.example .env
    fi

    log_success "Configuration validation passed"
}

measure_baseline_performance() {
    log_info "Measuring baseline trading performance..."

    # Start only trading core
    docker-compose -f "$COMPOSE_FILE" up -d supreme-trading postgres-master

    # Wait for trading core to stabilize
    sleep 60

    # Run baseline performance test
    if [ -f "scripts/validate_performance.py" ]; then
        python3 scripts/validate_performance.py --baseline
        log_success "Baseline performance measured"
    else
        log_warning "Performance validation script not found"
    fi
}

deploy_dashboard() {
    log_info "Deploying dashboard with resource isolation..."

    # Deploy dashboard services
    docker-compose -f "$COMPOSE_FILE" up -d supreme-dashboard postgres-replica

    # Wait for services to start
    log_info "Waiting for services to stabilize..."
    sleep 45

    # Check container health
    check_container_health
}

check_container_health() {
    log_info "Checking container health..."

    # Check trading container
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "supreme-trading.*Up"; then
        log_success "Trading container is running"
    else
        log_error "Trading container is not running properly"
        docker logs --tail 20 "$TRADING_CONTAINER"
        exit 1
    fi

    # Check dashboard container
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "supreme-dashboard.*Up"; then
        log_success "Dashboard container is running"
    else
        log_error "Dashboard container is not running properly"
        docker logs --tail 20 "$DASHBOARD_CONTAINER"
        exit 1
    fi

    # Test dashboard endpoint
    max_retries=10
    retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        if curl -s http://localhost:5000/health > /dev/null; then
            log_success "Dashboard is responding"
            break
        else
            log_info "Waiting for dashboard to respond... (attempt $((retry_count + 1))/$max_retries)"
            sleep 10
            retry_count=$((retry_count + 1))
        fi
    done

    if [ $retry_count -eq $max_retries ]; then
        log_error "Dashboard failed to respond after $max_retries attempts"
        exit 1
    fi
}

validate_performance_impact() {
    log_info "Validating performance impact of dashboard..."

    # Run performance validation
    if [ -f "scripts/validate_performance.py" ]; then
        python3 scripts/validate_performance.py --with-dashboard

        # Check if performance is acceptable
        if python3 scripts/validate_performance.py --check-acceptable; then
            log_success "Performance impact is within acceptable limits"
        else
            log_error "Dashboard performance impact exceeds acceptable limits"
            log_error "Consider optimizing dashboard or increasing system resources"
            exit 1
        fi
    else
        log_warning "Performance validation script not found, skipping validation"
    fi
}

setup_monitoring() {
    log_info "Setting up performance monitoring..."

    # Create logs directory
    mkdir -p logs

    # Start performance monitoring in background
    if [ -f "scripts/monitor_performance.py" ]; then
        nohup python3 scripts/monitor_performance.py > logs/performance_monitor.log 2>&1 &
        echo $! > logs/monitor.pid
        log_success "Performance monitoring started (PID: $(cat logs/monitor.pid))"
    else
        log_warning "Performance monitoring script not found"
    fi
}

print_deployment_summary() {
    log_info "Deployment Summary:"
    echo "==================="

    # Container status
    echo "📦 Container Status:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep supreme-

    # Resource usage
    echo ""
    echo "💾 Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep supreme-

    # System status
    echo ""
    echo "🖥️  System Status:"
    echo "Memory: $(free -h | awk 'NR==2{printf "%.1f/%.1fGB (%.0f%%)", $3/1024, $2/1024, $3*100/$2}')"
    echo "CPU: $(uptime | awk -F'load average:' '{print $2}' | awk '{print "Load average:" $1 $2 $3}')"

    # Dashboard access
    echo ""
    echo "🌐 Dashboard Access:"
    echo "URL: http://localhost:5000"
    echo "Status: $(curl -s http://localhost:5000/health | jq -r '.status' 2>/dev/null || echo 'Unknown')"

    # Next steps
    echo ""
    echo "📋 Next Steps:"
    echo "1. Open http://localhost:5000 to access the dashboard"
    echo "2. Monitor performance with: tail -f logs/performance_monitor.log"
    echo "3. Check trading performance: docker logs -f $TRADING_CONTAINER"

    log_success "Dashboard deployment completed successfully!"
}

cleanup_on_failure() {
    log_error "Deployment failed, cleaning up..."

    # Stop containers
    docker-compose -f "$COMPOSE_FILE" down

    # Stop monitoring if running
    if [ -f logs/monitor.pid ]; then
        kill "$(cat logs/monitor.pid)" 2>/dev/null || true
        rm logs/monitor.pid
    fi

    exit 1
}

main() {
    echo "🚀 Supreme System V5 - Dashboard Deployment"
    echo "==========================================="

    # Set up error handling
    trap cleanup_on_failure ERR

    # Deployment steps
    check_requirements
    validate_configuration
    measure_baseline_performance
    deploy_dashboard
    validate_performance_impact
    setup_monitoring
    print_deployment_summary

    log_success "🎉 Deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --quick        Skip performance validation (faster deployment)"
        echo "  --no-monitor   Skip performance monitoring setup"
        exit 0
        ;;
    --quick)
        log_info "Quick deployment mode enabled"
        measure_baseline_performance() { log_info "Skipping baseline measurement"; }
        validate_performance_impact() { log_info "Skipping performance validation"; }
        ;;
    --no-monitor)
        log_info "Monitoring disabled"
        setup_monitoring() { log_info "Skipping monitoring setup"; }
        ;;
esac

# Run main deployment
main "$@"
