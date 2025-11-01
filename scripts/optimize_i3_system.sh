#!/bin/bash

# Supreme System V5 - i3-4GB Local System Optimization
# Intel i3-8xxx + 4GB RAM + GPU Support

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

# Detect hardware
detect_hardware() {
    log_info "Detecting i3-4GB hardware configuration..."

    # CPU detection
    CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
    CPU_CORES=$(nproc)
    CPU_THREADS=$(lscpu | grep "CPU(s)" | head -1 | awk '{print $2}')

    log_info "CPU: $CPU_MODEL"
    log_info "Cores: $CPU_CORES, Threads: $CPU_THREADS"

    # Memory detection
    TOTAL_MEMORY=$(free -g | awk 'NR==2{print $2}')
    AVAILABLE_MEMORY=$(free -g | awk 'NR==2{print $7}')

    log_info "Total RAM: ${TOTAL_MEMORY}GB"
    log_info "Available RAM: ${AVAILABLE_MEMORY}GB"

    if [ "$TOTAL_MEMORY" -lt 4 ]; then
        log_warning "System has less than 4GB RAM. Performance may be limited."
    fi

    # GPU detection
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        GPU_NAME=$(echo $GPU_INFO | cut -d, -f1)
        GPU_MEMORY=$(echo $GPU_INFO | cut -d, -f2)
        log_info "GPU: NVIDIA $GPU_NAME (${GPU_MEMORY}MB)"
        HAS_NVIDIA_GPU=true
    elif lspci | grep -i vga | grep -i intel &> /dev/null; then
        log_info "GPU: Intel integrated graphics"
        HAS_INTEL_GPU=true
    else
        log_info "GPU: None detected (CPU-only mode)"
        HAS_GPU=false
    fi
}

optimize_memory() {
    log_info "Optimizing memory management for 4GB RAM..."

    # Disable swap to prevent memory thrashing
    sudo swapoff -a
    sudo sed -i '/swap/ s/^/#/' /etc/fstab

    # Optimize kernel memory management
    cat << EOF | sudo tee -a /etc/sysctl.conf
# i3-4GB Memory Optimizations
vm.swappiness = 1
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2
vm.vfs_cache_pressure = 50
vm.min_free_kbytes = 32768

# Network memory (reduced for low RAM)
net.core.rmem_max = 1048576
net.core.wmem_max = 1048576
net.core.rmem_default = 262144
net.core.wmem_default = 262144
EOF

    sudo sysctl -p

    # Limit systemd journal size
    sudo sed -i 's/#SystemMaxUse=/SystemMaxUse=50M/' /etc/systemd/journald.conf
    sudo systemctl restart systemd-journald

    log_success "Memory optimization completed"
}

optimize_cpu() {
    log_info "Optimizing CPU scheduling for i3..."

    # Set CPU governor to performance
    if [ -d "/sys/devices/system/cpu/cpufreq" ]; then
        echo 'performance' | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
        log_success "CPU governor set to performance mode"
    fi

    # Disable CPU frequency scaling service
    sudo systemctl disable ondemand 2>/dev/null || true
    sudo systemctl stop ondemand 2>/dev/null || true

    # Optimize I/O scheduler
    for device in /sys/block/*; do
        if [ -f "$device/queue/scheduler" ]; then
            echo 'deadline' | sudo tee "$device/queue/scheduler" > /dev/null
        fi
    done

    log_success "CPU and I/O optimization completed"
}

optimize_services() {
    log_info "Optimizing system services for minimal memory usage..."

    # Disable unnecessary services
    services_to_disable=(
        "bluetooth.service"
        "cups.service"
        "avahi-daemon.service"
        "ModemManager.service"
        "wpa_supplicant.service"
    )

    for service in "${services_to_disable[@]}"; do
        if sudo systemctl is-active --quiet "$service" 2>/dev/null; then
            sudo systemctl disable "$service" 2>/dev/null || true
            sudo systemctl stop "$service" 2>/dev/null || true
        fi
    done

    # Optimize systemd
    sudo mkdir -p /etc/systemd/system.conf.d
    cat << EOF | sudo tee /etc/systemd/system.conf.d/optimization.conf
[Manager]
DefaultLimitNOFILE=1024
DefaultTasksMax=512
EOF

    log_success "System services optimized"
}

setup_docker_limits() {
    log_info "Setting up Docker resource limits for i3-4GB..."

    # Create Docker daemon configuration
    sudo mkdir -p /etc/docker
    cat << EOF | sudo tee /etc/docker/daemon.json
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "max-concurrent-downloads": 1,
    "max-concurrent-uploads": 1,
    "registry-mirrors": ["https://mirror.gcr.io"]
}
EOF

    sudo systemctl restart docker

    # Create Docker Compose override for i3
    mkdir -p docker-compose
    cat << EOF > docker-compose/docker-compose.i3.yml
version: '3.8'

services:
  # Override resource limits for i3-4GB
  supreme-trading:
    deploy:
      resources:
        limits:
          memory: 1.5G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.5'

  supreme-dashboard:
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.5'
        reservations:
          memory: 128M
          cpus: '0.25'

  postgres-master:
    environment:
      POSTGRES_SHARED_BUFFERS: 256MB
      POSTGRES_EFFECTIVE_CACHE_SIZE: 1GB
      POSTGRES_MAINTENANCE_WORK_MEM: 64MB

  redis:
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
EOF

    log_success "Docker resource limits configured"
}

setup_gpu_acceleration() {
    log_info "Setting up GPU acceleration..."

    if [ "$HAS_NVIDIA_GPU" = true ]; then
        # Install NVIDIA drivers and CUDA (if not present)
        if ! command -v nvidia-smi &> /dev/null; then
            log_warning "NVIDIA drivers not detected. Installing..."
            # Note: This requires manual intervention for proprietary drivers
            log_info "Please install NVIDIA drivers manually for CUDA support"
        else
            log_success "NVIDIA GPU detected, installing CUDA acceleration"
            pip3 install cupy-cuda11x numba[cuda]
        fi

    elif [ "$HAS_INTEL_GPU" = true ]; then
        log_info "Setting up Intel GPU acceleration..."
        pip3 install intel-extension-for-pytorch
        pip3 install onednn-cpu  # Intel oneDNN

    else
        log_info "No GPU detected, optimizing for CPU-only mode..."
        pip3 install numba
        export NUMBA_CPU_NAME=$(lscpu | grep "Model name" | cut -d: -f2 | xargs | tr ' ' '_')
    fi
}

optimize_python() {
    log_info "Optimizing Python environment..."

    # Install optimized Python packages
    pip3 install --upgrade pip

    # Install core dependencies with optimizations
    pip3 install \
        numpy \
        pandas \
        psutil \
        redis \
        aiohttp \
        websockets \
        setproctitle \
        memory-profiler

    # Install ML libraries (lightweight versions)
    pip3 install \
        scikit-learn \
        lightgbm \
        xgboost \
        catboost

    # Set Python memory optimizations
    export PYTHONMALLOC=malloc
    export PYTHONOPTIMIZE=1

    log_success "Python environment optimized"
}

create_performance_profile() {
    log_info "Creating performance profile..."

    # Create systemd service for performance monitoring
    cat << EOF | sudo tee /etc/systemd/system/supreme-performance.service
[Unit]
Description=Supreme System V5 Performance Monitor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/supreme-system-v5/scripts/monitor_performance.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload

    # Create performance monitoring script
    cat << 'EOF' > /usr/local/bin/supreme-performance-check
#!/bin/bash
echo "=== Supreme System V5 Performance Check ==="
echo "Memory: $(free -h | awk 'NR==2{print $3"/"$2" ("$3*100/$2"%)"}')"
echo "CPU Load: $(uptime | awk -F'load average:' '{print $2}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"
echo "Top processes:"
ps aux --sort=-%mem | head -6 | awk 'NR>1{printf "  %-15s %5.1f%% MEM %5.1f%% CPU\n", $11, $4, $3}'
EOF

    sudo chmod +x /usr/local/bin/supreme-performance-check

    log_success "Performance monitoring profile created"
}

deploy_local_system() {
    log_info "Deploying Supreme System V5 locally..."

    # Create project directory
    sudo mkdir -p /opt/supreme-system-v5
    sudo chown $USER:$USER /opt/supreme-system-v5

    # Clone repository
    if [ ! -d "/opt/supreme-system-v5/.git" ]; then
        git clone https://github.com/thanhmuefatty07/supreme-system-v5.git /opt/supreme-system-v5
    fi

    cd /opt/supreme-system-v5

    # Create local environment file
    cat << EOF > .env.local
# Local i3-4GB Configuration
ENVIRONMENT=local
ORACLE_IP=${ORACLE_IP:-localhost}

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=supreme_trading
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=supreme_password_local

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Trading Configuration
TRADING_MODE=production
ENABLE_AI=false  # Disable AI locally to save RAM
ENABLE_DASHBOARD=true
MAX_MEMORY_GB=2.5
CPU_CORES=4

# Performance Limits
TRADING_MEMORY_LIMIT=1.5G
DASHBOARD_MEMORY_LIMIT=256M
REDIS_MEMORY_LIMIT=256M
POSTGRES_MEMORY_LIMIT=512M

# GPU Configuration
GPU_ACCELERATION=${HAS_NVIDIA_GPU:-false}
INTEL_GPU=${HAS_INTEL_GPU:-false}
EOF

    # Start with i3-specific compose file
    docker-compose -f docker-compose.yml -f docker-compose/docker-compose.i3.yml up -d

    log_success "Supreme System V5 deployed locally"
}

verify_local_deployment() {
    log_info "Verifying local deployment..."

    # Check memory usage
    memory_usage=$(free -h | awk 'NR==2{printf "%.1fG/%.1fG", $3/1024, $2/1024}')
    log_info "Memory usage: $memory_usage"

    # Check Docker containers
    running_containers=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep supreme | wc -l)
    if [ "$running_containers" -ge 3 ]; then
        log_success "All containers running"
    else
        log_warning "Some containers may not be running"
    fi

    # Performance check
    supreme-performance-check

    # Test endpoints
    sleep 10
    if curl -s http://localhost:5000/health > /dev/null; then
        log_success "Dashboard accessible"
    else
        log_warning "Dashboard not accessible"
    fi

    log_success "Local deployment verification completed"
}

main() {
    echo "⚡ Supreme System V5 - i3-4GB Local Optimization"
    echo "==============================================="
    echo "Target: Intel i3-8xxx + 4GB RAM + GPU"
    echo "Goal: S+ Performance Grade with $0 investment"
    echo ""

    # Initialize GPU detection
    HAS_NVIDIA_GPU=false
    HAS_INTEL_GPU=false
    HAS_GPU=false

    # Run optimization phases
    detect_hardware
    optimize_memory
    optimize_cpu
    optimize_services
    setup_docker_limits
    setup_gpu_acceleration
    optimize_python
    create_performance_profile
    deploy_local_system
    verify_local_deployment

    echo ""
    log_success "🎉 i3-4GB optimization completed!"
    echo ""
    echo "Performance Commands:"
    echo "📊 Check performance: supreme-performance-check"
    echo "📈 Monitor system: htop"
    echo "💾 Memory usage: free -h"
    echo ""
    echo "Access Points:"
    echo "📊 Dashboard: http://localhost:5000"
    echo "🔍 System Monitor: htop"
    echo ""
    echo "Next Steps:"
    echo "1. Connect to Oracle Cloud instance"
    echo "2. Run oracle_cloud_setup.sh on Oracle VM"
    echo "3. Configure hybrid local↔cloud synchronization"
}

# Run main function
main "$@"
