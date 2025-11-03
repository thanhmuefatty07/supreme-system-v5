# Portfolio state helpers to ensure compatibility with engine calls
from dataclasses import dataclass

@dataclass
class PortfolioState:
    total_balance: float = 0.0
    positions_value: float = 0.0

    @property
    def total_value(self) -> float:
        """Compatibility alias used by some modules.
        total_value = cash + mark-to-market positions.
        """
        return float(self.total_balance) + float(self.positions_value)
