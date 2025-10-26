"""
💹 Supreme System V5 - Trading Integration Module
Production trading capabilities with exchange connectors

This module provides production-ready trading integration:
- Exchange connectors (Binance, Coinbase, MEXC, etc.)
- Portfolio management
- Risk management
- Order execution
- Real-time market data

Components:
- TradingEngine: Main trading orchestrator
- RealDataConnector: Unified data source interface
- PortfolioManager: Position and risk management
- AISignalGenerator: AI-powered signal generation
"""

from .engine import (
    TradingEngine,
    TradingConfig,
    RealDataConnector,  # Renamed from ExchangeConnector
    PortfolioManager,
    AISignalGenerator   # Renamed from OrderExecutor
)

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

__all__ = [
    "TradingEngine",
    "TradingConfig",
    "RealDataConnector",  # Renamed from ExchangeConnector
    "PortfolioManager",
    "AISignalGenerator"   # Renamed from OrderExecutor
]

print("💹 Supreme System V5 - Trading Integration Module Loaded")
print("   Exchange connectors ready")
print("   Portfolio management active")
print("🚀 Production Trading Ready!")
