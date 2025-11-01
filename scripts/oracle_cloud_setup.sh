#!/bin/bash

# Supreme System V5 - Oracle Cloud Always Free Setup
# ARM64 Ubuntu 22.04 - VM.Standard.A1.Flex (4 OCPUs, 16GB RAM)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Configuration
ORACLE_IP="${ORACLE_IP:-your_oracle_ip}"
ORACLE_USER="${ORACLE_USER:-ubuntu}"
PROJECT_DIR="/opt/supreme-system-v5"

setup_oracle_cloud() {
    log_info "Setting up Oracle Cloud Always Free VM..."

    # Update system
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y curl wget git htop iotop sysstat

    # Install Docker and Docker Compose
    log_info "Installing Docker ecosystem..."
    sudo apt install -y docker.io docker-compose-plugin
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker ubuntu

    # Install Python 3.11 and pip
    log_info "Installing Python 3.11..."
    sudo apt install -y python3.11 python3.11-dev python3-pip

    # Install system monitoring tools
    log_info "Installing monitoring tools..."
    sudo apt install -y prometheus prometheus-node-exporter grafana

    # Configure firewall
    log_info "Configuring firewall..."
    sudo ufw allow ssh
    sudo ufw allow 80
    sudo ufw allow 443
    sudo ufw allow 3000  # Grafana
    sudo ufw allow 8000  # Trading API
    sudo ufw allow 8080  # WebSocket
    sudo ufw allow 9090  # Prometheus
    echo "y" | sudo ufw enable

    # Create project directory
    sudo mkdir -p $PROJECT_DIR
    sudo chown ubuntu:ubuntu $PROJECT_DIR

    log_success "Oracle Cloud base setup completed"
}

optimize_oracle_performance() {
    log_info "Optimizing Oracle ARM64 performance..."

    # ARM64 specific optimizations
    cat << EOF | sudo tee -a /etc/sysctl.conf
# ARM64 Performance Optimizations
vm.swappiness = 1
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
vm.vfs_cache_pressure = 50

# Network optimizations for ARM
net.core.rmem_max = 33554432
net.core.wmem_max = 33554432
net.core.somaxconn = 65535
net.ipv4.tcp_rmem = 4096 87380 33554432
net.ipv4.tcp_wmem = 4096 65536 33554432
net.ipv4.tcp_max_syn_backlog = 65535

# ARM memory management
vm.min_free_kbytes = 262144
vm.page-cluster = 0
EOF

    sudo sysctl -p

    # CPU frequency scaling
    echo 'performance' | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor

    # Disable CPU frequency scaling service
    sudo systemctl disable ondemand
    sudo systemctl stop ondemand

    log_success "Oracle ARM64 performance optimized"
}

setup_monitoring_stack() {
    log_info "Setting up monitoring stack..."

    # Configure Prometheus
    sudo mkdir -p /etc/prometheus
    cat << EOF | sudo tee /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'supreme-trading'
    static_configs:
      - targets: ['localhost:8001']
EOF

    # Configure Grafana
    sudo mkdir -p /etc/grafana/provisioning/datasources
    cat << EOF | sudo tee /etc/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
EOF

    # Start services
    sudo systemctl enable prometheus
    sudo systemctl enable prometheus-node-exporter
    sudo systemctl enable grafana-server
    sudo systemctl start prometheus
    sudo systemctl start prometheus-node-exporter
    sudo systemctl start grafana-server

    log_success "Monitoring stack configured"
}

setup_ai_environment() {
    log_info "Setting up AI/ML environment..."

    # Install PyTorch for ARM64
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Install ML libraries
    pip3 install \
        transformers \
        tensorflow-cpu \
        scikit-learn \
        pandas \
        numpy \
        matplotlib \
        seaborn \
        jupyterlab

    # Install GPU acceleration (if available)
    pip3 install \
        onnxruntime \
        openvino \
        tvm

    # Create AI workspace
    mkdir -p $PROJECT_DIR/ai/models
    mkdir -p $PROJECT_DIR/ai/data
    mkdir -p $PROJECT_DIR/ai/notebooks

    log_success "AI environment configured"
}

deploy_supreme_system() {
    log_info "Deploying Supreme System V5..."

    cd $PROJECT_DIR

    # Clone repository
    if [ ! -d ".git" ]; then
        git clone https://github.com/thanhmuefatty07/supreme-system-v5.git .
    fi

    # Create environment file
    cat << EOF > .env.oracle
# Oracle Cloud Configuration
ENVIRONMENT=oracle
ORACLE_IP=$ORACLE_IP

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=supreme_trading
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=supreme_password_oracle

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Trading Configuration
TRADING_MODE=production
ENABLE_AI=true
ENABLE_DASHBOARD=true
MAX_MEMORY_GB=14
CPU_CORES=4

# AI Configuration
AI_MODEL_PATH=/opt/supreme-system-v5/ai/models
AI_DATA_PATH=/opt/supreme-system-v5/ai/data
AI_BATCH_SIZE=16
AI_LEARNING_RATE=0.001

# Monitoring
GRAFANA_ADMIN_PASSWORD=supreme_admin_2024
PROMETHEUS_RETENTION=30d
EOF

    # Setup database
    docker run -d \
        --name postgres-oracle \
        -e POSTGRES_DB=supreme_trading \
        -e POSTGRES_USER=trading_user \
        -e POSTGRES_PASSWORD=supreme_password_oracle \
        -p 5432:5432 \
        -v postgres_oracle_data:/var/lib/postgresql/data \
        postgres:15-alpine

    # Setup Redis
    docker run -d \
        --name redis-oracle \
        -p 6379:6379 \
        -v redis_oracle_data:/data \
        redis:7-alpine redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru

    # Wait for services
    sleep 30

    # Deploy Supreme System
    docker-compose -f docker-compose.oracle.yml up -d

    log_success "Supreme System V5 deployed on Oracle Cloud"
}

verify_deployment() {
    log_info "Verifying deployment..."

    # Check services
    services=("postgres-oracle" "redis-oracle" "supreme-trading" "supreme-dashboard")
    for service in "${services[@]}"; do
        if docker ps | grep -q "$service"; then
            log_success "$service: RUNNING"
        else
            log_error "$service: NOT RUNNING"
        fi
    done

    # Check endpoints
    endpoints=(
        "http://localhost:8001/health:Trading API"
        "http://localhost:5000/health:Dashboard"
        "http://localhost:3000:Grafana"
        "http://localhost:9090:Prometheus"
    )

    for endpoint in "${endpoints[@]}"; do
        url=$(echo $endpoint | cut -d: -f1)
        name=$(echo $endpoint | cut -d: -f2)
        if curl -s --max-time 10 "$url" > /dev/null; then
            log_success "$name: ACCESSIBLE"
        else
            log_warning "$name: NOT ACCESSIBLE"
        fi
    done

    # Performance check
    log_info "Performance verification..."
    memory_usage=$(free -h | awk 'NR==2{printf "%.1fGB used / %.1fGB total", $3/1024, $2/1024}')
    cpu_usage=$(uptime | awk -F'load average:' '{print $2}' | awk '{print "Load:" $1 $2 $3}')

    log_info "Memory: $memory_usage"
    log_info "CPU: $cpu_usage"

    log_success "Deployment verification completed"
}

main() {
    echo "🚀 Supreme System V5 - Oracle Cloud Deployment"
    echo "============================================="
    echo "Target: ARM64 Ubuntu 22.04, 4 OCPUs, 16GB RAM"
    echo "Cost: $0 (Always Free Tier)"
    echo ""

    # Run setup phases
    setup_oracle_cloud
    optimize_oracle_performance
    setup_monitoring_stack
    setup_ai_environment
    deploy_supreme_system
    verify_deployment

    echo ""
    log_success "🎉 Oracle Cloud deployment completed!"
    echo ""
    echo "Access Points:"
    echo "📊 Dashboard: http://$ORACLE_IP:5000"
    echo "📈 Grafana: http://$ORACLE_IP:3000 (admin/supreme_admin_2024)"
    echo "🔍 Prometheus: http://$ORACLE_IP:9090"
    echo "🤖 Trading API: http://$ORACLE_IP:8001"
    echo ""
    echo "Next: Configure local i3-4GB system for hybrid deployment"
}

# Run main function
main "$@"
