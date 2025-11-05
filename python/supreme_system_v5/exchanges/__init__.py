"""
Exchange connectors for Supreme System V5
Real-time market data and order execution
Agent Mode: Complete exchange ecosystem with MEXC, Binance, OKX
"""

from typing import Dict, List, Any

# Exchange connector imports with error handling
try:
    from .okx_connector import OKXConnector
    OKX_AVAILABLE = True
except ImportError:
    OKX_AVAILABLE = False

try:
    from .binance_connector import BinanceConnector
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

try:
    from .mexc_connector import MEXCConnector, create_mexc_connector, DEFAULT_MEXC_CONFIG
    MEXC_AVAILABLE = True
except ImportError:
    MEXC_AVAILABLE = False

# Base exchange interface
try:
    from .base import BaseExchange
    BaseExchangeConnector = BaseExchange
except ImportError:
    # Fallback - create basic base class
    class BaseExchangeConnector:
        def __init__(self, config):
            self.config = config
            pass
    BaseExchange = BaseExchangeConnector

# Configuration class
class ExchangeConfig:
    """Exchange configuration container"""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)

# Exchange factory function
def create_exchange_connector(exchange_name: str, config: Dict[str, Any]):
    """Factory function to create exchange connectors"""
    exchange_name = exchange_name.lower()
    
    if exchange_name == 'mexc' and MEXC_AVAILABLE:
        return MEXCConnector(config)
    elif exchange_name == 'binance' and BINANCE_AVAILABLE:
        return BinanceConnector(config)
    elif exchange_name == 'okx' and OKX_AVAILABLE:
        return OKXConnector(config)
    else:
        raise ValueError(f"Exchange '{exchange_name}' not available or not supported")

# Get available exchanges
def get_available_exchanges() -> List[str]:
    """Get list of available exchange connectors"""
    exchanges = []
    
    if MEXC_AVAILABLE:
        exchanges.append('mexc')
    if BINANCE_AVAILABLE:
        exchanges.append('binance')
    if OKX_AVAILABLE:
        exchanges.append('okx')
        
    return exchanges

# Exchange availability status
EXCHANGE_STATUS = {
    'mexc': MEXC_AVAILABLE,
    'binance': BINANCE_AVAILABLE,
    'okx': OKX_AVAILABLE
}

__all__ = [
    "BaseExchangeConnector",
    "BaseExchange", 
    "ExchangeConfig",
    "OKXConnector",
    "BinanceConnector", 
    "MEXCConnector",
    "create_mexc_connector",
    "DEFAULT_MEXC_CONFIG",
    "create_exchange_connector",
    "get_available_exchanges",
    "OKX_AVAILABLE",
    "BINANCE_AVAILABLE",
    "MEXC_AVAILABLE",
    "EXCHANGE_STATUS",
]