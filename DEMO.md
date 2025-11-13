# 🎬 Supreme System V5 - Demo Guide

## What's Included

This demo environment includes:

- ✅ **365 days** of historical market data (5 symbols)
- ✅ **Pre-configured** backtest scenarios
- ✅ **Live metrics** dashboard (Grafana)
- ✅ **Paper trading** simulation mode
- ✅ **Zero configuration** - works out of the box

---

## Quick Start (< 15 minutes)

### Step 1: Start Environment

```bash
docker-compose -f docker-compose.demo.yml up -d
```

Wait for startup (check status):

```bash
docker-compose -f docker-compose.demo.yml ps
```

**Expected output:**
```
NAME                  STATUS
supreme-system-demo  Up (healthy)
prometheus-demo      Up
grafana-demo         Up
```

### Step 2: Access Dashboard

Open browser: **http://localhost:8501**

### Step 3: Run Demo Backtest

1. Click "Backtest" tab
2. Select symbol: **BTC/USDT**
3. Date range: **Last 365 days** (pre-loaded)
4. Strategy: **Neuromorphic Trend Following**
5. Click "Run Backtest"

**Expected results** (< 30 seconds):

- Sharpe Ratio: ~1.8
- Max Drawdown: <18%
- Total trades: ~120
- Win rate: ~58%

### Step 4: View Metrics

**Grafana**: http://localhost:3000 (admin/admin)

Pre-loaded dashboards:

- **Trading Metrics** - Latency, throughput, positions, PnL
- **System Performance** - Resource usage, errors
- **Risk Analytics** - Drawdown, exposure

**Prometheus**: http://localhost:9090

---

## Demo Features

### 1. Backtest Engine

- Historical data: 365 days, 1-minute candles
- Symbols: BTC, ETH, SOL, BNB, XRP
- Realistic slippage & fees
- Risk management simulation

### 2. Paper Trading (Optional)

Enable in `.env`:

```bash
ENABLE_PAPER_TRADING=true
PAPER_CAPITAL=10000
```

### 3. Live Metrics

- Latency monitoring (simulated <10μs)
- Throughput tracking
- Position management
- PnL calculation

---

## Limitations (Demo Mode)

- ❌ No real exchange connections
- ❌ No live data (historical only)
- ❌ Limited to demo symbols
- ❌ 30-day evaluation period

**For production use**: [Contact Sales](mailto:thanhmuefatty07@gmail.com?subject=Supreme%20System%20V5%20-%20Commercial%20License%20Inquiry)

---

## Troubleshooting

### Dashboard not loading?

```bash
docker-compose -f docker-compose.demo.yml logs supreme-system
```

**Common issues:**
- Port 8501 already in use → Change port in `docker-compose.demo.yml`
- Service not healthy → Check logs for errors
- Missing data → Run `python scripts/generate_demo_data.py`

### Metrics missing?

Check Prometheus: http://localhost:9090/targets

**Verify:**
- Prometheus is scraping metrics
- Targets are UP
- Metrics are being collected

### Services won't start?

```bash
# Check Docker resources
docker system df

# Restart services
docker-compose -f docker-compose.demo.yml restart

# View all logs
docker-compose -f docker-compose.demo.yml logs
```

---

## Next Steps

Impressed with the demo? Let's talk:

- 📞 [Schedule Demo Call](mailto:thanhmuefatty07@gmail.com?subject=Supreme%20System%20V5%20-%20Demo%20Request)
- 💼 [Commercial Licensing](mailto:thanhmuefatty07@gmail.com?subject=Supreme%20System%20V5%20-%20Commercial%20License%20Inquiry)
- 📚 [Full Documentation](docs/)

---

## Demo Data Generation

If you need to regenerate demo data:

```bash
# Generate default data (365 days, 5 symbols)
python scripts/generate_demo_data.py

# Custom generation
python scripts/generate_demo_data.py \
  --symbols BTC/USDT,ETH/USDT \
  --days 180 \
  --timeframe 1h \
  --output-dir data/demo
```

**Options:**
- `--symbols`: Comma-separated list (default: BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT)
- `--days`: Number of days (default: 365)
- `--timeframe`: Data timeframe - 1min, 5min, 15min, 1h, 4h, 1d (default: 1min)
- `--output-dir`: Output directory (default: data/demo)

---

**Built with** ❤️ **by Supreme System V5 Team**

