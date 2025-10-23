"""
🌐 Supreme System V5 - REST API Module
RESTful API interface for remote control and monitoring

This module provides HTTP/REST API access to Supreme System V5:
- System control endpoints
- Real-time performance metrics
- Trading status and portfolio
- AI component monitoring
- WebSocket real-time streams

Components:
- APIServer: FastAPI-based REST server
- WebSocketHandler: Real-time data streaming
- AuthMiddleware: Security and authentication
- MetricsCollector: Performance monitoring
"""

from .server import (
    APIServer,
    APIConfig,
    start_api_server,
    demo_api_server
)

__version__ = "5.0.0"
__author__ = "Supreme System V5 Team"

__all__ = [
    "APIServer",
    "APIConfig",
    "start_api_server",
    "demo_api_server"
]

# API specifications
API_SPECS = {
    "framework": "FastAPI",
    "async_support": True,
    "websocket_support": True,
    "auth_methods": ["API_KEY", "JWT"],
    "default_port": 8000
}

print("🌐 Supreme System V5 - REST API Module Loaded")
print(f"   Framework: {API_SPECS['framework']}")
print(f"   Port: {API_SPECS['default_port']}")
print("🚀 Production API Ready!")
