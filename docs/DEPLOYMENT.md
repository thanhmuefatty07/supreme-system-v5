# Supreme System V5 - Deployment Guide

## Prerequisites

### System Requirements
- **OS**: Linux/Windows/macOS
- **Python**: 3.8 - 3.11
- **Memory**: 8GB+ RAM recommended
- **Storage**: 50GB+ for historical data
- **Network**: Stable internet connection

### Dependencies
```bash
pip install -e .[dev,dashboard,security,performance]
```

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/thanhmuefatty07/supreme-system-v5.git
cd supreme-system-v5
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -e .
pip install -e .[dev,dashboard,security,performance]
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### 5. Run Pre-deployment Checks
```bash
# Run tests
pytest tests/ --cov=src --cov-report=term-missing

# Run security checks
bandit -r src/
safety check

# Run type checking
mypy src/
```

## Development Deployment

### Local Development Setup
```bash
# Install pre-commit hooks
pre-commit install

# Run development server
streamlit run src/monitoring/dashboard.py --server.port 8501

# Run CLI commands
python -m src.cli --help
```

### Docker Development
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -e .[dev,dashboard,security,performance]

EXPOSE 8501
CMD ["streamlit", "run", "src/monitoring/dashboard.py", "--server.port", "8501"]
```

```bash
docker build -t supreme-system-dev .
docker run -p 8501:8501 supreme-system-dev
```

## Production Deployment

### 1. Server Preparation

#### Ubuntu/Debian Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install system dependencies
sudo apt install build-essential -y
```

#### Security Hardening
```bash
# Create dedicated user
sudo useradd -m -s /bin/bash supreme-trader

# Set proper permissions
sudo chown -R supreme-trader:supreme-trader /opt/supreme-system
sudo chmod 700 /opt/supreme-system
```

### 2. Application Deployment

#### Manual Deployment
```bash
# Create deployment directory
sudo mkdir -p /opt/supreme-system
cd /opt/supreme-system

# Clone repository
sudo -u supreme-trader git clone https://github.com/thanhmuefatty07/supreme-system-v5.git .

# Create virtual environment
sudo -u supreme-trader python3 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install -e .
pip install gunicorn uvicorn

# Configure environment
cp .env.example .env
# Edit .env with production settings
```

#### Docker Production Deployment
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Copy application
COPY . /app
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Install dependencies
RUN pip install --user -e .
RUN pip install --user gunicorn uvicorn

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "src.cli", "live", "start"]
```

### 3. Configuration Management

#### Environment Variables
```bash
# Production environment file
cat > /opt/supreme-system/.env << EOF
BINANCE_API_KEY=your_production_api_key
BINANCE_API_SECRET=your_production_secret
BINANCE_TESTNET=false
INITIAL_CAPITAL=100000
LOG_LEVEL=WARNING
ENABLE_DASHBOARD=true
DASHBOARD_PORT=8501
EOF
```

#### Configuration Validation
```bash
# Test configuration
python -c "from src.config.config import get_config; print('Config loaded successfully')"
```

### 4. Service Management

#### Systemd Service
```bash
# Create systemd service
cat > /etc/systemd/system/supreme-system.service << EOF
[Unit]
Description=Supreme System V5 Trading Engine
After=network.target

[Service]
Type=simple
User=supreme-trader
WorkingDirectory=/opt/supreme-system
ExecStart=/opt/supreme-system/venv/bin/python -m src.cli live start
Restart=always
RestartSec=10
Environment=PATH=/opt/supreme-system/venv/bin

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable supreme-system
sudo systemctl start supreme-system
sudo systemctl status supreme-system
```

#### Process Monitoring
```bash
# Monitor service
sudo journalctl -u supreme-system -f

# Check service status
sudo systemctl status supreme-system

# Restart service
sudo systemctl restart supreme-system
```

### 5. Monitoring Setup

#### Dashboard Deployment
```bash
# Install dashboard as service
cat > /etc/systemd/system/supreme-dashboard.service << EOF
[Unit]
Description=Supreme System Dashboard
After=network.target

[Service]
Type=simple
User=supreme-trader
WorkingDirectory=/opt/supreme-system
ExecStart=/opt/supreme-system/venv/bin/streamlit run src/monitoring/dashboard.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable supreme-dashboard
sudo systemctl start supreme-dashboard
```

#### Log Management
```bash
# Configure log rotation
cat > /etc/logrotate.d/supreme-system << EOF
/opt/supreme-system/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 supreme-trader supreme-trader
    postrotate
        systemctl reload supreme-system
    endscript
}
EOF
```

### 6. Backup and Recovery

#### Automated Backups
```bash
# Create backup script
cat > /opt/supreme-system/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/supreme-system/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database/data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz /opt/supreme-system/data/

# Backup configuration (without secrets)
tar -czf $BACKUP_DIR/config_$DATE.tar.gz --exclude='*.key' --exclude='*.secret' /opt/supreme-system/config/

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /opt/supreme-system/backup.sh
```

#### Cron Job for Backups
```bash
# Add to crontab
echo "0 2 * * * /opt/supreme-system/backup.sh" | crontab -
```

### 7. Security Configuration

#### API Key Security
```bash
# Use encrypted storage for API keys
python -c "
from cryptography.fernet import Fernet
key = Fernet.generate_key()
print('Encryption key:', key.decode())
"

# Store encrypted keys in environment
echo "API_KEY_ENCRYPTION_KEY=your_generated_key" >> /opt/supreme-system/.env
```

#### Network Security
```bash
# Configure firewall
sudo ufw allow 8501/tcp  # Dashboard
sudo ufw allow 22/tcp    # SSH
sudo ufw --force enable

# Use fail2ban for SSH protection
sudo apt install fail2ban -y
```

### 8. Performance Optimization

#### Memory Tuning
```bash
# Increase system limits
echo "supreme-trader soft nofile 65536" >> /etc/security/limits.conf
echo "supreme-trader hard nofile 65536" >> /etc/security/limits.conf
```

#### CPU Affinity (Optional)
```bash
# Pin process to specific CPU cores
taskset -c 0-3 python -m src.cli live start
```

### 9. Scaling Considerations

#### Horizontal Scaling
```bash
# Use multiple instances with load balancer
# Configure Redis for shared state
pip install redis

# Update configuration for Redis
echo "REDIS_HOST=localhost" >> /opt/supreme-system/.env
echo "REDIS_PORT=6379" >> /opt/supreme-system/.env
```

#### Database Scaling
```bash
# Use PostgreSQL for larger deployments
sudo apt install postgresql postgresql-contrib -y

# Configure connection
echo "DATABASE_URL=postgresql://user:password@localhost/supreme_db" >> /opt/supreme-system/.env
```

### 10. Monitoring and Alerting

#### Prometheus Setup
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'supreme-system'
    static_configs:
      - targets: ['localhost:8000']
```

#### Grafana Dashboard
```bash
# Install Grafana
sudo apt install grafana -y
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Reinstall dependencies
pip install -e . --force-reinstall
```

#### Permission Issues
```bash
# Fix ownership
sudo chown -R supreme-trader:supreme-trader /opt/supreme-system
```

#### Memory Issues
```bash
# Check memory usage
ps aux --sort=-%mem | head

# Adjust configuration
echo "MAX_MEMORY_USAGE=0.8" >> /opt/supreme-system/.env
```

#### Network Issues
```bash
# Test connectivity
curl -f https://api.binance.com/api/v3/ping

# Check DNS
nslookup api.binance.com
```

## Rollback Procedures

### Emergency Stop
```bash
# Immediate shutdown
sudo systemctl stop supreme-system

# Check positions manually
python -c "from src.trading.live_trading_engine import LiveTradingEngine; engine = LiveTradingEngine(); print(engine.get_status())"
```

### Version Rollback
```bash
# Rollback to previous version
cd /opt/supreme-system
git checkout v4.9.0
pip install -e .

# Restart services
sudo systemctl restart supreme-system
```

## Maintenance

### Regular Tasks
- **Daily**: Check logs and performance metrics
- **Weekly**: Run full test suite and security scans
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Review and update risk parameters

### Health Checks
```bash
# Automated health check script
cat > /opt/supreme-system/health_check.py << 'EOF'
#!/usr/bin/env python3
from src.config.config import get_config
from src.data.binance_client import BinanceClient

def health_check():
    try:
        # Test configuration
        config = get_config()
        print("✓ Configuration loaded")

        # Test API connectivity
        client = BinanceClient(
            api_key=config.get('binance.api_key'),
            api_secret=config.get('binance.api_secret'),
            testnet=config.get('binance.testnet', True)
        )

        if client.test_connection():
            print("✓ API connection successful")
        else:
            print("✗ API connection failed")
            return False

        print("✓ All health checks passed")
        return True

    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

if __name__ == "__main__":
    health_check()
EOF
```

This deployment guide provides comprehensive instructions for setting up Supreme System V5 in both development and production environments, with security, monitoring, and maintenance considerations.
