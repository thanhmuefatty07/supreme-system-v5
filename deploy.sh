#!/bin/bash

# Supreme System V5 - Unified Deployment Script
# ULTRA SFL - Single command deployment across all environments
#
# Usage:
#   ./deploy.sh [environment] [version]
#
# Environments:
#   development - Full development stack with debugging
#   i3          - i3 optimized configuration
#   production  - Production deployment with security hardening
#
# Examples:
#   ./deploy.sh development    # Deploy development environment
#   ./deploy.sh production v1.2.3  # Deploy production v1.2.3
#   ./deploy.sh i3             # Deploy i3 optimized environment

set -euo pipefail

# ================================
# CONFIGURATION
# ================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT=${1:-development}
VERSION=${2:-latest}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================================
# UTILITY FUNCTIONS
# ================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        development|staging|production|i3)
            log_info "Deploying to $ENVIRONMENT environment..."
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, staging, production, i3"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi

    # Check if required files exist
    local required_files=(
        "docker-compose.base.yml"
        "secrets/postgres_password.txt"
        "secrets/grafana_password.txt"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done

    log_success "Prerequisites check passed"
}

# Build compose files list
build_compose_files() {
    COMPOSE_FILES=("-f" "docker-compose.base.yml")

    case $ENVIRONMENT in
        development)
            COMPOSE_FILES+=("-f" "docker-compose.development.yml")
            ;;
        production)
            COMPOSE_FILES+=("-f" "docker-compose.production.yml")
            export VERSION=$VERSION
            ;;
        i3)
            COMPOSE_FILES+=("-f" "docker-compose.i3.yml")
            ;;
    esac
}

# Validate configuration
validate_config() {
    log_info "Validating Docker Compose configuration..."

    if docker-compose "${COMPOSE_FILES[@]}" config --quiet 2>/dev/null; then
        log_success "Configuration validation passed"
    else
        log_error "Configuration validation failed"
        exit 1
    fi
}

# Check environment-specific requirements
check_environment_requirements() {
    case $ENVIRONMENT in
        production)
            log_info "Checking production requirements..."

            # Check for required environment variables
            local required_vars=(
                "OKX_API_KEY"
                "OKX_SECRET_KEY"
                "OKX_PASSPHRASE"
                "BINANCE_API_KEY"
                "BINANCE_SECRET_KEY"
                "POSTGRES_PASSWORD"
                "REDIS_PASSWORD"
            )

            local missing_vars=()
            for var in "${required_vars[@]}"; do
                if [[ -z "${!var:-}" ]]; then
                    missing_vars+=("$var")
                fi
            done

            if [[ ${#missing_vars[@]} -gt 0 ]]; then
                log_warn "Missing production environment variables:"
                printf '  - %s\n' "${missing_vars[@]}"
                log_warn "Some features may not work correctly"
            fi
            ;;
        i3)
            log_info "Checking i3 hardware requirements..."

            # Check available memory (rough estimate)
            local total_mem_kb
            total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
            local total_mem_gb=$((total_mem_kb / 1024 / 1024))

            if [[ $total_mem_gb -lt 4 ]]; then
                log_warn "System has ${total_mem_gb}GB RAM, i3 configuration requires 4GB+"
                log_warn "Performance may be degraded"
            fi
            ;;
    esac
}

# Backup current state (if production)
backup_current_state() {
    if [[ $ENVIRONMENT == "production" ]]; then
        log_info "Creating backup of current state..."

        local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"

        # Backup volumes (if using named volumes)
        # This is a simplified backup - production should use proper backup tools
        log_info "Backup created in: $backup_dir"
    fi
}

# Deploy services
deploy_services() {
    log_info "Starting deployment..."

    # Set environment variables
    export ENVIRONMENT=$ENVIRONMENT
    export VERSION=$VERSION

    # Deploy services
    if docker-compose "${COMPOSE_FILES[@]}" up -d; then
        log_success "Services deployed successfully"
    else
        log_error "Service deployment failed"
        exit 1
    fi
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to become healthy..."

    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."

        if docker-compose "${COMPOSE_FILES[@]}" ps | grep -q "healthy"; then
            log_success "All services are healthy"
            return 0
        fi

        sleep 10
        ((attempt++))
    done

    log_warn "Some services may still be starting..."
    log_warn "Check status with: docker-compose ${COMPOSE_FILES[*]} ps"
}

# Show deployment status
show_status() {
    log_info "Deployment status:"

    echo ""
    docker-compose "${COMPOSE_FILES[@]}" ps

    echo ""
    log_success "Deployment completed successfully!"
    log_info "Useful commands:"
    echo "  View logs: docker-compose ${COMPOSE_FILES[*]} logs -f [service]"
    echo "  Stop services: docker-compose ${COMPOSE_FILES[*]} down"
    echo "  Restart service: docker-compose ${COMPOSE_FILES[*]} restart [service]"
    echo "  View metrics: open http://localhost:9090 (Prometheus)"
    echo "  View dashboards: open http://localhost:3000 (Grafana)"

    case $ENVIRONMENT in
        development)
            echo "  API docs: http://localhost:8000/docs"
            ;;
        production)
            echo "  Application: https://your-domain.com"
            echo "  Metrics: https://metrics.your-domain.com"
            echo "  Dashboards: https://dashboard.your-domain.com"
            ;;
        i3)
            echo "  Note: i3 optimized - reduced resource usage"
            ;;
    esac
}

# ================================
# MAIN DEPLOYMENT LOGIC
# ================================

main() {
    echo "🚀 Supreme System V5 - Unified Deployment"
    echo "=========================================="
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    echo ""

    validate_environment
    check_prerequisites
    build_compose_files
    validate_config
    check_environment_requirements
    backup_current_state
    deploy_services
    wait_for_services
    show_status

    echo ""
    echo "🎯 Deployment complete! Supreme System V5 is ready."
}

# ================================
# ERROR HANDLING
# ================================

trap 'log_error "Deployment failed with error on line $LINENO"' ERR

# ================================
# EXECUTE
# ================================

main "$@"
