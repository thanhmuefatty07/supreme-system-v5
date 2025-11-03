# Strategy interface normalization for Supreme System V5
# This module wraps any existing strategy to provide a stable
# interface expected by the backtest engine: add_price_data(...) and generate_signal(...)

from typing import Any, Dict, Optional

class StrategyInterfaceAdapter:
    """
    Adapter to normalize strategy interfaces.
    Ensures the engine can always call:
      - add_price_data(symbol, price, volume=None, ts=None)
      - generate_signal(data: Dict[str, Any]) -> Optional[Dict[str, Any]]
    """
    def __init__(self, strategy: Any):
        self._s = strategy

    # --- add_price_data normalization ---
    def add_price_data(self, symbol: str, price: float, volume: Optional[float] = None, ts: Optional[float] = None) -> None:
        """Call underlying method with flexible signature."""
        if hasattr(self._s, 'add_price_data'):
            fn = getattr(self._s, 'add_price_data')
            try:
                # Try full signature
                fn(symbol, price, volume, ts)
            except TypeError:
                # Fallbacks for shorter signatures
                try:
                    fn(symbol, price, volume)
                except TypeError:
                    try:
                        fn(symbol, price)
                    except TypeError:
                        # Last resort: call with a single dict payload
                        fn({'symbol': symbol, 'price': price, 'volume': volume, 'timestamp': ts})
        elif hasattr(self._s, 'update'):
            # Some strategies use update(...) naming
            getattr(self._s, 'update')(symbol, price, volume, ts)
        else:
            # No method defined; ignore to avoid breaking the loop
            pass

    # --- generate_signal normalization ---
    def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a signal dict with fields: action, confidence, size (optional)."""
        # Preferred name
        if hasattr(self._s, 'generate_signal'):
            return getattr(self._s, 'generate_signal')(data)
        # Common alternates
        for alt in ('analyze', 'analyze_signal', 'signal', 'create_signal'):
            if hasattr(self._s, alt):
                return getattr(self._s, alt)(data)
        # If not available, return HOLD to keep engine stable
        return {'action': 'HOLD', 'confidence': 0.0, 'size': 0.0}
