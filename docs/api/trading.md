# Trading API

API for executing trades and managing positions.

## Endpoints

### Place Order

```python
POST /api/trading/order
{
    "symbol": "BTC/USDT",
    "side": "buy",
    "quantity": 0.1,
    "order_type": "market"
}
```

### Get Positions

```python
GET /api/trading/positions
```

### Get Balance

```python
GET /api/trading/balance
```

