# Risk Management

Comprehensive risk management for institutional-grade trading.

## Features

- Circuit breakers
- Position limits
- Drawdown control
- Multi-layer validation

## Configuration

Configure risk parameters in `config/risk_config.py`:

```python
MAX_POSITION_SIZE = 0.1  # 10% of portfolio
MAX_DRAWDOWN = 0.20      # 20% max drawdown
CIRCUIT_BREAKER_THRESHOLD = 0.05  # 5% loss triggers halt
```

