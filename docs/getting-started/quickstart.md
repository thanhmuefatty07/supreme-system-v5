# Quick Start Guide

Get Supreme System V5 running in under 15 minutes.

## Prerequisites

- Docker & Docker Compose
- 4GB RAM minimum
- Linux/Mac (Windows WSL2 supported)

## Step 1: Access Repository

Contact [thanhmuefatty07@gmail.com](mailto:thanhmuefatty07@gmail.com?subject=Supreme%20System%20V5%20-%20Evaluation%20License%20Request) for evaluation access token.

## Step 2: Clone & Setup

```bash
git clone https://github.com/thanhmuefatty07/supreme-system-v5.git
cd supreme-system-v5

# Copy environment template
cp .env.example .env

# Edit configuration (use demo keys provided)
nano .env
```

## Step 3: Start Demo Environment

```bash
docker-compose -f docker-compose.demo.yml up -d
```

Wait 2-3 minutes for all services to initialize.

## Step 4: Access Interfaces

- **Dashboard**: http://localhost:8501
- **Grafana Metrics**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Step 5: Run Demo Backtest

In the dashboard:

1. Navigate to "Backtest" tab
2. Select pre-loaded demo data (BTC/USDT, 365 days)
3. Click "Run Backtest"
4. View results in <30 seconds

## Expected Results

- Sharpe Ratio: ~1.8
- Max Drawdown: <18%
- Total trades: ~120
- Win rate: ~58%

## Next Steps

- [Configuration Guide](configuration.md)
- [Architecture Overview](../architecture/overview.md)
- [Production Deployment](docker.md)

