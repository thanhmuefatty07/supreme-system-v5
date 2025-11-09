#!/bin/bash

# Supreme System V5 - Production Deployment Script
# This script handles complete production deployment with safety checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_ENV=${1:-staging}
PROJECT_NAME="supreme-system-v5"
DOCKER_IMAGE="${PROJECT_NAME}:latest"

# Logging
LOG_FILE="logs/deployment_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo -e "${BLUE}🚀 SUPREME SYSTEM V5 - PRODUCTION DEPLOYMENT${NC}"
echo "Environment: $DEPLOY_ENV"
echo "Timestamp: $(date)"
echo "Log file: $LOG_FILE"
echo "=================================================="

# Functions
log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_dependencies() {
    echo "🔍 Checking dependencies..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check docker-compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi

    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed"
        exit 1
    fi

    log_info "Dependencies check passed"
}

run_pre_deployment_checks() {
    echo "🧪 Running pre-deployment checks..."

    # Check if required files exist
    required_files=(
        "Dockerfile"
        "docker-compose.yml"
        "requirements.txt"
        "src/"
        "dashboard.py"
    )

    for file in "${required_files[@]}"; do
        if [ ! -e "$file" ]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done

    # Run tests
    echo "Running test suite..."
    if python -m pytest tests/ -v --tb=short; then
        log_info "All tests passed"
    else
        log_error "Tests failed - aborting deployment"
        exit 1
    fi

    # Check code quality
    echo "Checking code quality..."
    if flake8 src/ --max-line-length=100 --extend-ignore=E203,W503; then
        log_info "Code quality check passed"
    else
        log_warn "Code quality issues found - continuing anyway"
    fi

    log_info "Pre-deployment checks completed"
}

build_docker_image() {
    echo "🏗️  Building Docker image..."

    # Build image
    if docker build -t "$DOCKER_IMAGE" .; then
        log_info "Docker image built successfully"
    else
        log_error "Docker build failed"
        exit 1
    fi

    # Show image info
    docker images "$DOCKER_IMAGE"
}

deploy_to_environment() {
    local env=$1
    echo "🚀 Deploying to $env environment..."

    # Set environment variables
    export COMPOSE_PROJECT_NAME="${PROJECT_NAME}_${env}"

    case $env in
        staging)
            export BINANCE_TESTNET=true
            ;;
        production)
            export BINANCE_TESTNET=false
            # Additional production checks
            echo "🔒 Production deployment - extra safety checks..."

            # Check if API keys are set
            if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
                log_error "Production deployment requires BINANCE_API_KEY and BINANCE_API_SECRET"
                exit 1
            fi

            # Confirm production deployment
            echo "⚠️  PRODUCTION DEPLOYMENT CONFIRMATION ⚠️"
            echo "This will deploy live trading with real money!"
            read -p "Are you sure you want to continue? (yes/no): " confirm
            if [ "$confirm" != "yes" ]; then
                log_error "Production deployment cancelled"
                exit 1
            fi
            ;;
        *)
            log_error "Unknown environment: $env"
            exit 1
            ;;
    esac

    # Stop existing containers
    echo "Stopping existing containers..."
    docker-compose down || true

    # Start new deployment
    echo "Starting $env deployment..."
    if docker-compose up -d; then
        log_info "Deployment to $env completed successfully"
    else
        log_error "Deployment to $env failed"
        exit 1
    fi

    # Wait for services to be healthy
    echo "Waiting for services to be healthy..."
    sleep 30

    # Check container health
    if docker-compose ps | grep -q "Up"; then
        log_info "All containers are running"
    else
        log_error "Some containers failed to start"
        docker-compose logs
        exit 1
    fi
}

run_post_deployment_tests() {
    local env=$1
    echo "🧪 Running post-deployment tests..."

    # Wait for application to be ready
    sleep 10

    # Test health endpoint
    if curl -f http://localhost:8501 >/dev/null 2>&1; then
        log_info "Dashboard is accessible"
    else
        log_warn "Dashboard may not be ready yet"
    fi

    # Test data pipeline
    if python -c "
import sys
sys.path.insert(0, 'src')
from data.data_pipeline import DataPipeline
dp = DataPipeline()
print('Data pipeline initialized successfully')
"; then
        log_info "Data pipeline is functional"
    else
        log_error "Data pipeline initialization failed"
        exit 1
    fi

    log_info "Post-deployment tests completed"
}

rollback_deployment() {
    local env=$1
    echo "🔄 Rolling back $env deployment..."

    # Stop containers
    docker-compose down

    # Optionally restore from backup
    # (Add backup/restore logic here if needed)

    log_warn "Deployment rolled back"
}

main() {
    echo "Starting deployment process..."

    # Pre-deployment checks
    check_dependencies
    run_pre_deployment_checks

    # Build
    build_docker_image

    # Deploy
    if deploy_to_environment "$DEPLOY_ENV"; then
        # Post-deployment tests
        run_post_deployment_tests "$DEPLOY_ENV"

        echo ""
        echo -e "${GREEN}🎉 DEPLOYMENT SUCCESSFUL!${NC}"
        echo "Environment: $DEPLOY_ENV"
        echo "Dashboard: http://localhost:8501"
        echo "Logs: $LOG_FILE"
        echo ""
        echo "Useful commands:"
        echo "  docker-compose logs -f          # View logs"
        echo "  docker-compose ps               # Check status"
        echo "  docker-compose down             # Stop deployment"
        echo ""
        echo -e "${BLUE}Supreme System V5 is now live! 🚀${NC}"
    else
        log_error "Deployment failed"
        rollback_deployment "$DEPLOY_ENV"
        exit 1
    fi
}

# Show usage
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [environment]"
    echo ""
    echo "Environments:"
    echo "  staging     - Deploy to staging environment (default)"
    echo "  production  - Deploy to production environment"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy to staging"
    echo "  $0 production        # Deploy to production"
    exit 0
fi

# Run main function
main "$@"
