# 🔄 Bybit Integration & Migration Guide

**Supreme System V5 - Switching from Binance to Bybit**

---

## 📋 Overview

This guide explains how to:
1. **Setup Bybit API credentials**
2. **Switch from Binance to Bybit**
3. **Use both exchanges simultaneously**
4. **Migrate existing configurations**

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install pybit>=5.7.0
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Get Bybit API Keys

1. **Create Bybit Account**: https://www.bybit.com/
2. **Go to API Management**: https://www.bybit.com/app/user/api-management
3. **Create API Key**:
   - Enable: **Spot Trading** (for spot trading)
   - Enable: **Derivatives Trading** (for futures, if needed)
   - **Disable**: Withdrawals (for security)
   - Set **IP Whitelist** (recommended)
4. **Save API Key and Secret** (secret only shown once!)

### Step 3: Configure Environment

Add to your `.env` file:

```bash
# Exchange Selection
API_PRIMARY_EXCHANGE=bybit

# Bybit API Credentials
BYBIT_API_KEY=your_api_key_here
BYBIT_SECRET_KEY=your_secret_key_here
BYBIT_TESTNET=true  # Set to false for production
```

### Step 4: Test Connection

```python
from src.data.bybit_client import AsyncBybitClient

async def test():
    client = AsyncBybitClient()
    await client.initialize_session()
    
    if await client.test_connection():
        print("✅ Bybit connection successful!")
    else:
        print("❌ Connection failed")
    
    await client.close_session()

# Run test
import asyncio
asyncio.run(test())
```

---

## 🔄 Migration from Binance

### Option A: Complete Switch (Recommended)

**Replace Binance with Bybit:**

1. **Update `.env`**:
   ```bash
   # Change primary exchange
   API_PRIMARY_EXCHANGE=bybit
   
   # Add Bybit credentials
   BYBIT_API_KEY=your_bybit_key
   BYBIT_SECRET_KEY=your_bybit_secret
   BYBIT_TESTNET=true
   
   # Keep Binance keys for reference (optional)
   # BINANCE_API_KEY=...
   # BINANCE_SECRET_KEY=...
   ```

2. **Update Code** (if using direct client):
   ```python
   # Old (Binance)
   from src.data.binance_client import AsyncBinanceClient
   client = AsyncBinanceClient()
   
   # New (Bybit)
   from src.data.bybit_client import AsyncBybitClient
   client = AsyncBybitClient()
   ```

3. **Test Everything**:
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Test data download
   python -c "
   from src.data.bybit_client import BybitClient
   client = BybitClient()
   df = client.get_historical_klines('BTCUSDT', '1h', '2024-01-01', '2024-01-02')
   print(f'Downloaded {len(df)} records')
   "
   ```

### Option B: Multi-Exchange Support

**Use both exchanges simultaneously:**

```python
from src.data.binance_client import AsyncBinanceClient
from src.data.bybit_client import AsyncBybitClient

# Initialize both
binance = AsyncBinanceClient()
bybit = AsyncBybitClient()

# Use based on symbol or strategy
if symbol in ['BTCUSDT', 'ETHUSDT']:
    data = await bybit.get_historical_klines(symbol, '1h', start_date, end_date)
else:
    data = await binance.get_historical_klines(symbol, '1h', start_date, end_date)
```

---

## 📊 Key Differences: Binance vs Bybit

### API Endpoints

| Feature | Binance | Bybit |
|---------|---------|-------|
| **Base URL** | `https://api.binance.com` | `https://api.bybit.com` |
| **Testnet** | `https://testnet.binance.vision` | `https://api-testnet.bybit.com` |
| **Klines Limit** | 1000 per request | 200 per request |
| **Rate Limits** | 1200 requests/min | 120 requests/10s |

### Symbol Format

- **Binance**: `BTCUSDT`, `ETHUSDT` (no separator)
- **Bybit**: `BTCUSDT`, `ETHUSDT` (same format)

### Interval Format

Both use similar intervals, but Bybit uses numeric codes:
- `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d` → Same
- Bybit also supports: `1`, `3`, `5`, `15`, `30`, `60`, `120`, `240`, `360`, `720`, `D`, `W`, `M`

### Data Structure

**Binance Klines:**
```python
[timestamp, open, high, low, close, volume, close_time, quote_volume, trades, ...]
```

**Bybit Klines:**
```python
[timestamp, open, high, low, close, volume, turnover]
```

Both are automatically converted to pandas DataFrame with same structure.

---

## 🔧 Configuration Options

### Environment Variables

```bash
# Exchange Selection
API_PRIMARY_EXCHANGE=bybit  # or 'binance'

# Bybit Configuration
BYBIT_API_KEY=your_key
BYBIT_SECRET_KEY=your_secret
BYBIT_TESTNET=true

# Binance Configuration (if using both)
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
BINANCE_TESTNET=true
```

### Programmatic Configuration

```python
from src.data.bybit_client import AsyncBybitClient

# Direct initialization
client = AsyncBybitClient(
    api_key='your_key',
    api_secret='your_secret',
    testnet=True
)

# Using secrets manager
from src.utils.secrets_manager import setup_bybit_credentials

setup_bybit_credentials(
    api_key='your_key',
    api_secret='your_secret',
    testnet=True
)

# Then use without explicit credentials
client = AsyncBybitClient()  # Auto-loads from secrets manager
```

---

## 📝 Code Examples

### Download Historical Data

```python
from src.data.bybit_client import AsyncBybitClient
import asyncio

async def download_data():
    client = AsyncBybitClient()
    await client.initialize_session()
    
    # Download BTCUSDT 1-hour candles
    df = await client.get_historical_klines(
        symbol='BTCUSDT',
        interval='1h',
        start_date='2024-01-01',
        end_date='2024-01-31',
        limit=200
    )
    
    print(f"Downloaded {len(df)} records")
    print(df.head())
    
    await client.close_session()

asyncio.run(download_data())
```

### Get Symbol Information

```python
from src.data.bybit_client import AsyncBybitClient

async def get_info():
    client = AsyncBybitClient()
    await client.initialize_session()
    
    info = await client.get_symbol_info('BTCUSDT')
    print(info)
    
    await client.close_session()

asyncio.run(get_info())
```

### Synchronous Usage

```python
from src.data.bybit_client import BybitClient

# Synchronous wrapper
client = BybitClient()

# Test connection
if client.test_connection():
    print("✅ Connected!")

# Get historical data
df = client.get_historical_klines(
    symbol='BTCUSDT',
    interval='1h',
    start_date='2024-01-01',
    end_date='2024-01-02'
)

print(df)
```

---

## ⚠️ Important Notes

### Rate Limits

- **Bybit**: 120 requests per 10 seconds
- **Binance**: 1200 requests per minute
- **Recommendation**: Use rate limiting delays (default: 0.05s)

### Testnet

- **Bybit Testnet**: https://testnet.bybit.com/
- **Testnet Keys**: Separate from production keys
- **Always test with testnet first!**

### Security

1. **Never commit API keys** to git
2. **Use IP whitelisting** on Bybit
3. **Disable withdrawals** on API keys
4. **Rotate keys regularly** (every 90 days)
5. **Use secrets manager** for production

### Error Handling

```python
try:
    df = await client.get_historical_klines(...)
    if df is None:
        print("Failed to download data")
except Exception as e:
    print(f"Error: {e}")
```

---

## 🐛 Troubleshooting

### Import Error: `pybit` not found

```bash
pip install pybit>=5.7.0
```

### Connection Failed

1. Check API keys are correct
2. Verify testnet setting matches your keys
3. Check IP whitelist (if enabled)
4. Verify API key permissions

### Rate Limit Errors

- Increase `rate_limit_delay` in client initialization
- Reduce request frequency
- Use connection pooling

### Data Format Issues

The client automatically converts Bybit format to pandas DataFrame. If you see issues:
- Check symbol format (should be uppercase, no separator)
- Verify interval format
- Check date format (YYYY-MM-DD)

---

## 📚 Additional Resources

- **Bybit API Docs**: https://bybit-exchange.github.io/docs/v5/
- **Pybit Library**: https://github.com/bybit-exchange/pybit
- **Bybit Testnet**: https://testnet.bybit.com/

---

## ✅ Migration Checklist

- [ ] Install `pybit` library
- [ ] Create Bybit API keys
- [ ] Add credentials to `.env`
- [ ] Test connection
- [ ] Update `API_PRIMARY_EXCHANGE=bybit`
- [ ] Test data download
- [ ] Update any hardcoded Binance references
- [ ] Run full test suite
- [ ] Verify trading strategies work
- [ ] Update documentation

---

**Last Updated**: 2025-01-16  
**Version**: 1.0

