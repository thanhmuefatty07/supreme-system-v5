#!/usr/bin/env python3
"""
Supreme System V5 - Real-time Demo

Demonstrates real-time WebSocket functionality.
Shows how to connect to Binance streams and process live data.
"""

import sys
import time
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.realtime_client import BinanceWebSocketClient


def price_callback(data):
    """Handle price ticker updates."""
    if 's' in data and 'c' in data:
        symbol = data['s']
        price = float(data['c'])
        volume = float(data.get('v', 0))
        print(".4f"


def trade_callback(data):
    """Handle trade updates."""
    if 's' in data and 'p' in data and 'q' in data:
        symbol = data['s']
        price = float(data['p'])
        quantity = float(data['q'])
        is_buyer_maker = data.get('m', False)
        trade_type = "SELL" if is_buyer_maker else "BUY"
        print(".4f"


def kline_callback(data):
    """Handle kline/candlestick updates."""
    if 'k' in data:
        kline = data['k']
        symbol = data['s']
        interval = kline['i']
        close_price = float(kline['c'])
        volume = float(kline['v'])
        is_closed = kline['x']  # True if kline is closed

        status = "CLOSED" if is_closed else "OPEN"
        print(".4f"


def depth_callback(data):
    """Handle order book depth updates."""
    if 'bids' in data and 'asks' in data:
        symbol = data['s']
        best_bid = float(data['bids'][0][0]) if data['bids'] else 0
        best_ask = float(data['asks'][0][0]) if data['asks'] else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0

        print(".4f"


def demo_price_streams():
    """Demo price ticker streams."""
    print("\n💰 PRICE STREAMS DEMO")
    print("=" * 50)

    client = BinanceWebSocketClient()

    # Subscribe to multiple price streams
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
    for symbol in symbols:
        client.subscribe_price_stream(symbol, price_callback)

    print(f"Subscribed to price streams for: {', '.join(symbols)}")

    return client


def demo_trade_streams():
    """Demo trade streams."""
    print("\n📈 TRADE STREAMS DEMO")
    print("=" * 50)

    client = BinanceWebSocketClient()

    # Subscribe to trade streams
    symbols = ['BTCUSDT', 'ETHUSDT']
    for symbol in symbols:
        client.subscribe_trade_stream(symbol, trade_callback)

    print(f"Subscribed to trade streams for: {', '.join(symbols)}")

    return client


def demo_kline_streams():
    """Demo kline/candlestick streams."""
    print("\n📊 KLINE STREAMS DEMO")
    print("=" * 50)

    client = BinanceWebSocketClient()

    # Subscribe to kline streams
    client.subscribe_kline_stream('BTCUSDT', '1m', kline_callback)
    client.subscribe_kline_stream('ETHUSDT', '1m', kline_callback)

    print("Subscribed to 1-minute kline streams for BTCUSDT and ETHUSDT")

    return client


def demo_depth_streams():
    """Demo order book depth streams."""
    print("\n📚 DEPTH STREAMS DEMO")
    print("=" * 50)

    client = BinanceWebSocketClient()

    # Subscribe to depth streams
    client.subscribe_depth_stream('BTCUSDT', 5, depth_callback)

    print("Subscribed to order book depth stream for BTCUSDT (top 5 levels)")

    return client


def demo_multi_stream():
    """Demo multiple stream types simultaneously."""
    print("\n🎯 MULTI-STREAM DEMO")
    print("=" * 50)

    client = BinanceWebSocketClient()

    # Add multiple types of streams
    client.subscribe_price_stream('BTCUSDT', price_callback)
    client.subscribe_trade_stream('BTCUSDT', trade_callback)
    client.subscribe_kline_stream('BTCUSDT', '1m', kline_callback)
    client.subscribe_depth_stream('BTCUSDT', 10, depth_callback)

    print("Subscribed to ALL stream types for BTCUSDT")
    print("- Price ticker updates")
    print("- Individual trades")
    print("- 1-minute klines")
    print("- Order book depth (top 10)")

    return client


def demo_metrics():
    """Demo metrics and monitoring."""
    print("\n📊 METRICS DEMO")
    print("=" * 50)

    client = BinanceWebSocketClient()

    def metrics_callback():
        """Print metrics periodically."""
        metrics = client.get_metrics()
        print("
📈 Current Metrics:"        print(f"   Messages Received: {metrics['messages_received']}")
        print(f"   Messages Processed: {metrics['messages_processed']}")
        print(f"   Active Streams: {metrics['active_streams']}")
        print(f"   Connection Healthy: {metrics['connection_healthy']}")
        if 'messages_per_second' in metrics:
            print(".2f"
    # Add price stream for some activity
    client.subscribe_price_stream('BTCUSDT', price_callback)

    # Print metrics every 10 seconds
    import threading
    def periodic_metrics():
        while client.is_running:
            time.sleep(10)
            metrics_callback()

    metrics_thread = threading.Thread(target=periodic_metrics, daemon=True)
    metrics_thread.start()

    return client


def main():
    """Run real-time demo."""
    print("🚀 SUPREME SYSTEM V5 - REAL-TIME WEBSOCKET DEMO")
    print("=" * 60)
    print("This demo shows real-time data streaming capabilities.")
    print("Note: Requires internet connection to Binance WebSocket API.")
    print("Press Ctrl+C to stop the demo.")
    print("=" * 60)

    # Demo menu
    demos = {
        '1': ('Price Streams', demo_price_streams),
        '2': ('Trade Streams', demo_trade_streams),
        '3': ('Kline Streams', demo_kline_streams),
        '4': ('Depth Streams', demo_depth_streams),
        '5': ('Multi-Stream', demo_multi_stream),
        '6': ('Metrics Demo', demo_metrics)
    }

    while True:
        print("\nSelect demo:")
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print("  0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == '0':
            break

        if choice in demos:
            demo_name, demo_func = demos[choice]

            print(f"\n▶️ Starting {demo_name}...")
            print("Press Ctrl+C to stop this demo and return to menu")

            client = demo_func()

            def signal_handler(sig, frame):
                print("
⏹️ Stopping demo..."                client.stop()

            signal.signal(signal.SIGINT, signal_handler)

            try:
                client.start()

                # Wait for demo to run
                while client.is_running:
                    time.sleep(1)

            except KeyboardInterrupt:
                client.stop()
                print("Demo stopped by user")

            print(f"✅ {demo_name} completed")

        else:
            print("❌ Invalid choice")

    print("\n🎉 Demo session ended!")
    print("💡 Key Takeaways:")
    print("   • WebSocket client handles multiple stream types")
    print("   • Automatic reconnection on connection loss")
    print("   • Intelligent data buffering and callbacks")
    print("   • Real-time metrics and monitoring")
    print("   • Production-ready for live trading systems")


if __name__ == "__main__":
    main()
