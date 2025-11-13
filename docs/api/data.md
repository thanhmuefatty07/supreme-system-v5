# Data API

API for accessing market data and historical information.

## Endpoints

### Get Market Data

```python
GET /api/data/market?symbol=BTC/USDT&timeframe=1h
```

### Get Historical Data

```python
GET /api/data/historical?symbol=BTC/USDT&days=365
```

### Get Indicators

```python
GET /api/data/indicators?symbol=BTC/USDT&indicator=RSI
```

