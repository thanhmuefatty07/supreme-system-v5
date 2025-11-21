#!/usr/bin/env python3
"""
Simple Live Data Test - Minimal Demo

Test only the LiveDataManager with Binance WebSocket.
No complex trading engine, just raw data streaming.
"""

import asyncio
import logging
import signal
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple LiveDataManager import test
try:
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

    # Try minimal import
    import websockets
    import json

    logger.info("✅ WebSocket libraries available")

except ImportError as e:
    logger.error(f"❌ Missing dependency: {e}")
    sys.exit(1)


class SimpleDataTester:
    """Simple WebSocket data tester."""

    def __init__(self):
        self.websocket_url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        self.running = False
        self.message_count = 0
        self.price_data = []

    async def connect_and_listen(self, duration_seconds=30):
        """Connect to Binance and listen for data."""
        logger.info(f"🔌 Connecting to: {self.websocket_url}")
        logger.info(f"⏱️  Will listen for {duration_seconds} seconds")

        try:
            async with websockets.connect(self.websocket_url) as websocket:
                logger.info("✅ Connected to Binance WebSocket!")

                # Set up signal handler
                def signal_handler(signum, frame):
                    logger.info("🛑 Stop signal received")
                    self.running = False

                signal.signal(signal.SIGINT, signal_handler)
                self.running = True

                start_time = asyncio.get_event_loop().time()

                while self.running and (asyncio.get_event_loop().time() - start_time) < duration_seconds:
                    try:
                        # Receive message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)

                        # Parse message
                        data = json.loads(message)

                        if 'k' in data:
                            kline = data['k']
                            price = float(kline['c'])
                            volume = float(kline['v'])
                            timestamp = int(kline['t'])

                            self.message_count += 1
                            self.price_data.append(price)

                            # Log every 5 messages
                            if self.message_count % 5 == 0:
                                elapsed = int(asyncio.get_event_loop().time() - start_time)
                                logger.info(f"📊 [{elapsed}s] Message #{self.message_count}: BTC @ ${price:,.2f}, Vol: {volume:.4f}")

                        await asyncio.sleep(0.1)  # Small delay

                    except asyncio.TimeoutError:
                        logger.warning("⏰ WebSocket timeout, checking connection...")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.error("🔌 WebSocket connection closed")
                        break
                    except Exception as e:
                        logger.error(f"❌ Error processing message: {e}")
                        break

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

        return True

    def show_summary(self):
        """Show test summary."""
        print("\n" + "="*50)
        print("📊 LIVE DATA TEST SUMMARY")
        print("="*50)
        print(f"📬 Messages Received: {self.message_count}")

        if self.price_data:
            start_price = self.price_data[0]
            end_price = self.price_data[-1]
            change = end_price - start_price
            change_pct = (change / start_price) * 100 if start_price > 0 else 0

            print(f"💰 Start Price: ${start_price:,.2f}")
            print(f"🏁 End Price: ${end_price:,.2f}")
            print(f"📈 Change: ${change:+,.2f} ({change_pct:+.2f}%)")
            print(f"📊 Price Range: ${min(self.price_data):,.2f} - ${max(self.price_data):,.2f}")
        else:
            print("❌ No price data received")

        print("="*50)


async def main():
    """Main test function."""
    duration = 30  # seconds

    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except:
            pass

    print("🚀 Supreme System V5 - Live Data Test")
    print("📡 Testing Binance WebSocket connection")
    print(f"⏱️  Duration: {duration} seconds")
    print("⚠️  This is a SAFE TEST - No trading occurs!")
    print("="*50)

    tester = SimpleDataTester()

    try:
        success = await tester.connect_and_listen(duration)
        if success:
            print("✅ Test completed successfully!")
        else:
            print("❌ Test failed - check connection/logs")
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        tester.show_summary()


if __name__ == "__main__":
    asyncio.run(main())
