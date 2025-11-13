# Configuration Guide

Configure Supreme System V5 for your environment.

## Environment Variables

Create `.env` file:

```bash
# Trading Configuration
MODE=demo
SYMBOLS=BTC/USDT,ETH/USDT
DATA_DAYS=365

# API Keys (for production)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# Monitoring
LOG_LEVEL=INFO
```

## Demo Mode

Demo mode includes:
- Pre-loaded historical data
- Paper trading simulation
- Limited to demo symbols

## Production Mode

Production mode requires:
- Valid API keys
- Commercial license
- Proper risk management configuration

