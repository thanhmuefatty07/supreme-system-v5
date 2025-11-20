#!/bin/bash

# Supreme System V5 - Production Deployment Script
# Version: 5.0.0
# Date: November 11, 2025

set -euo pipefail

# Configuration
DEPLOY_ENV=${DEPLOY_ENV:-production}
DOCKER_IMAGE="supreme-system-v5:${DEPLOY_ENV}"
CONTAINER_NAME="supreme-system-v5-${DEPLOY_ENV}"
BACKUP_DIR="/opt/supreme-system/backups"
LOG_DIR="/opt/supreme-system/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Performing pre-deployment checks..."

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if required environment variables are set
    required_vars=("BINANCE_API_KEY" "BINANCE_API_SECRET" "SYSTEM_SECRET_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done

    # Check if required directories exist
    directories=("$BACKUP_DIR" "$LOG_DIR")
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_warn "Directory $dir does not exist, creating..."
            sudo mkdir -p "$dir"
            sudo chown $(whoami):$(whoami) "$dir"
        fi
    done

    # Check disk space (require at least 5GB free)
    available_space=$(df / | tail -1 | awk '{print $4}')
    if (( available_space < 5242880 )); then  # 5GB in KB
        log_error "Insufficient disk space. Require at least 5GB free."
        exit 1
    fi

    log_success "Pre-deployment checks completed"
}

# Create backup
create_backup() {
    log_info "Creating backup of current deployment..."

    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_file="$BACKUP_DIR/supreme_system_backup_$timestamp.tar.gz"

    # Stop current container if running
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        log_info "Stopping current container..."
        docker stop "$CONTAINER_NAME" || true
    fi

    # Backup configuration and data
    if [[ -d "/opt/supreme-system/config" ]]; then
        tar -czf "$backup_file" \
            -C /opt/supreme-system \
            config/ data/ logs/ 2>/dev/null || true
        log_success "Backup created: $backup_file"
    else
        log_warn "No existing configuration to backup"
    fi
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."

    # Build with build args for security
    DOCKER_BUILDKIT=1 docker build \
        --target production \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --build-arg DEPLOY_ENV="$DEPLOY_ENV" \
        -t "$DOCKER_IMAGE" \
        .

    if [[ $? -eq 0 ]]; then
        log_success "Docker image built successfully"
    else
        log_error "Docker image build failed"
        exit 1
    fi
}

# Run pre-deployment tests
run_pre_deployment_tests() {
    log_info "Running pre-deployment tests..."

    # Test Docker image
    log_info "Testing Docker image..."
    if docker run --rm --entrypoint python "$DOCKER_IMAGE" -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from config.settings import validate_configuration
    validate_configuration()
    print('Configuration validation passed')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    sys.exit(1)
"; then
        log_success "Docker image tests passed"
    else
        log_error "Docker image tests failed"
        exit 1
    fi
}

# Deploy application
deploy_application() {
    log_info "Deploying application..."

    # Remove existing container if it exists
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    # Run new container
    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        --env-file .env \
        -e DEPLOY_ENV="$DEPLOY_ENV" \
        -v "$LOG_DIR:/app/logs" \
        -v "/opt/supreme-system/data:/app/data" \
        -v "/opt/supreme-system/config:/app/config" \
        -p 8501:8501 \
        -p 8001:8001 \
        --memory="2g" \
        --cpus="1.0" \
        --security-opt no-new-privileges \
        --cap-drop ALL \
        --cap-add NET_BIND_SERVICE \
        --read-only \
        -v /tmp \
        "$DOCKER_IMAGE"

    if [[ $? -eq 0 ]]; then
        log_success "Application deployed successfully"
    else
        log_error "Application deployment failed"
        exit 1
    fi
}

# Health check
perform_health_check() {
    log_info "Performing health checks..."

    max_attempts=30
    attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts"

        # Check if container is running
        if ! docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
            log_error "Container is not running"
            exit 1
        fi

        # Check health endpoint
        if curl -f -s http://localhost:8001/health > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi

        sleep 10
        ((attempt++))
    done

    log_error "Health check failed after $max_attempts attempts"
    exit 1
}

# Post-deployment validation
post_deployment_validation() {
    log_info "Performing post-deployment validation..."

    # Check logs for errors
    error_count=$(docker logs "$CONTAINER_NAME" 2>&1 | grep -i error | wc -l)
    if [[ $error_count -gt 0 ]]; then
        log_warn "Found $error_count error messages in logs during startup"
        docker logs "$CONTAINER_NAME" | grep -i error | tail -10
    fi

    # Validate configuration loading
    if docker exec "$CONTAINER_NAME" python -c "
import sys
sys.path.insert(0, '/app/src')
from config.settings import get_settings
settings = get_settings()
print(f'Environment: {settings.system.environment}')
print(f'Trading symbols: {len(settings.trading.symbols)}')
print('Configuration validation successful')
" 2>/dev/null; then
        log_success "Configuration validation passed"
    else
        log_error "Configuration validation failed"
        exit 1
    fi

    # Check resource usage
    container_stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep "$CONTAINER_NAME")
    log_info "Container resource usage: $container_stats"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."

    # Check if Prometheus is configured
    if [[ -f "monitoring/prometheus.yml" ]]; then
        log_info "Prometheus configuration found"
        # Additional monitoring setup can be added here
    fi

    # Setup log rotation
    if [[ -f "/etc/logrotate.d/supreme-system" ]]; then
        log_info "Log rotation already configured"
    else
        sudo tee /etc/logrotate.d/supreme-system > /dev/null << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(whoami)
    postrotate
        docker exec $CONTAINER_NAME kill -USR1 1 2>/dev/null || true
    endscript
}
EOF
        log_success "Log rotation configured"
    fi
}

# Cleanup old deployments
cleanup_old_deployments() {
    log_info "Cleaning up old deployments..."

    # Remove old images (keep last 3)
    docker images supreme-system-v5 --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}" | \
        tail -n +2 | sort -k3 -r | tail -n +4 | awk '{print $2}' | \
        xargs -r docker rmi 2>/dev/null || true

    # Remove old containers (keep last 2)
    docker ps -a --filter "name=supreme-system-v5" --filter "status=exited" \
        --format "{{.Names}}\t{{.CreatedAt}}" | sort -k2 | head -n -2 | \
        awk '{print $1}' | xargs -r docker rm 2>/dev/null || true

    log_success "Cleanup completed"
}

# Rollback function
rollback() {
    log_error "Deployment failed, initiating rollback..."

    # Stop failed container
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    # Find latest backup
    latest_backup=$(ls -t "$BACKUP_DIR"/supreme_system_backup_*.tar.gz 2>/dev/null | head -1)

    if [[ -n "$latest_backup" && -f "$latest_backup" ]]; then
        log_info "Restoring from backup: $latest_backup"

        # Restore configuration and data
        sudo tar -xzf "$latest_backup" -C /opt/supreme-system 2>/dev/null || true

        # Restart previous version (assuming it exists)
        if docker images | grep -q "supreme-system-v5.*latest"; then
            docker run -d \
                --name "${CONTAINER_NAME}-rollback" \
                --env-file .env \
                -v "$LOG_DIR:/app/logs" \
                -v "/opt/supreme-system/data:/app/data" \
                supreme-system-v5:latest
            log_success "Rollback completed"
        else
            log_error "No previous version available for rollback"
        fi
    else
        log_error "No backup available for rollback"
    fi

    exit 1
}

# Main deployment function
main() {
    log_info "Starting Supreme System V5 production deployment"
    log_info "Environment: $DEPLOY_ENV"
    log_info "Image: $DOCKER_IMAGE"
    log_info "Container: $CONTAINER_NAME"

    # Trap errors for rollback
    trap rollback ERR

    # Execute deployment steps
    pre_deployment_checks
    create_backup
    build_image
    run_pre_deployment_tests
    deploy_application
    perform_health_check
    post_deployment_validation
    setup_monitoring
    cleanup_old_deployments

    log_success "🎉 Supreme System V5 deployment completed successfully!"
    log_info "Application is running at:"
    log_info "  - Dashboard: http://localhost:8501"
    log_info "  - Metrics: http://localhost:8001"
    log_info "  - Health Check: http://localhost:8001/health"

    # Display monitoring information
    echo
    log_info "Monitoring Information:"
    echo "  - Logs: $LOG_DIR"
    echo "  - Backups: $BACKUP_DIR"
    echo "  - Container: $CONTAINER_NAME"
    echo
    log_info "Useful commands:"
    echo "  - View logs: docker logs $CONTAINER_NAME"
    echo "  - Restart: docker restart $CONTAINER_NAME"
    echo "  - Stop: docker stop $CONTAINER_NAME"
    echo "  - Update: ./scripts/deploy_production.sh"
}

# Run main function
main "$@"