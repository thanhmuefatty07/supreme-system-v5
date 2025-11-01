"""
Supreme System V5 - Advanced Real-Time Trading Dashboard
Professional monitoring and visualization for i3-4GB + Oracle Cloud
"""

import os
import sys
import time
import json
import threading
import queue
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import redis
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import psutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supreme_dashboard_v5_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class AdvancedTradingDashboard:
    """
    Advanced real-time trading dashboard with enterprise features
    - Real-time WebSocket updates
    - Historical data visualization
    - Performance monitoring
    - AI insights integration
    - Memory-efficient data handling
    """

    def __init__(self):
        self.clients = set()
        self.data_queue = queue.Queue(maxsize=1000)  # Memory-efficient queue
        self.db_connection = None
        self.redis_client = None
        self.running = False

        # Performance metrics
        self.metrics_history = []
        self.trade_history = []
        self.system_stats = {}
        self.ai_insights = {}

        # Data caches for performance
        self.portfolio_cache = {}
        self.chart_cache = {}
        self.cache_ttl = 30  # 30 second cache

        # Initialize connections
        self._init_database()
        self._init_redis()

        # Performance monitoring
        self.performance_stats = {
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'active_clients': 0,
            'updates_per_second': 0,
            'cache_hit_rate': 0.0
        }

    def _init_database(self):
        """Initialize SQLite database for dashboard data"""
        try:
            db_path = os.path.join(os.path.dirname(__file__), 'dashboard.db')
            self.db_connection = sqlite3.connect(db_path, check_same_thread=False)

            # Create tables with optimized schema
            cursor = self.db_connection.cursor()

            # Portfolio metrics with indexing
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    portfolio_value REAL,
                    daily_pnl REAL,
                    daily_pnl_percent REAL,
                    total_pnl REAL,
                    active_positions INTEGER,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    current_drawdown REAL,
                    UNIQUE(timestamp)
                )
            ''')

            # Trades table with composite index
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    pnl REAL,
                    fee REAL,
                    strategy TEXT,
                    duration INTEGER,
                    ai_confidence REAL DEFAULT 0.0
                )
            ''')

            # System metrics for monitoring
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_latency REAL,
                    data_quality_score REAL,
                    active_data_sources INTEGER
                )
            ''')

            # AI insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    insight_type TEXT,
                    symbol TEXT,
                    confidence REAL,
                    prediction REAL,
                    recommendation TEXT,
                    metadata TEXT
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_metrics (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_timestamp ON ai_insights (timestamp)')

            self.db_connection.commit()
            print("✅ Dashboard database initialized")

        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            self.db_connection = None

    def _init_redis(self):
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=1,  # Separate DB for dashboard
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test connection
            self.redis_client.ping()
            print("✅ Dashboard Redis connection established")

        except Exception as e:
            print(f"⚠️ Dashboard Redis connection failed: {e}")
            print("🔄 Operating without Redis caching")
            self.redis_client = None

    def start_dashboard(self):
        """Start dashboard data collection and broadcasting"""
        if self.running:
            return

        self.running = True
        print("🚀 Starting Advanced Trading Dashboard...")

        # Start background threads
        threads = [
            threading.Thread(target=self._data_collection_loop, daemon=True, name='data_collector'),
            threading.Thread(target=self._broadcast_loop, daemon=True, name='broadcaster'),
            threading.Thread(target=self._cleanup_loop, daemon=True, name='cleanup'),
            threading.Thread(target=self._performance_monitor, daemon=True, name='performance')
        ]

        for thread in threads:
            thread.start()

        print("✅ Dashboard threads started")

    def _data_collection_loop(self):
        """Continuous data collection from trading system"""
        consecutive_errors = 0
        max_errors = 10

        while self.running:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()

                # Store in database (batched for performance)
                self._store_metrics_batch([current_metrics])

                # Update caches
                self._update_caches(current_metrics)

                # Add to broadcast queue (non-blocking)
                try:
                    self.data_queue.put(current_metrics, timeout=1)
                except queue.Full:
                    # Remove oldest item if queue is full
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put(current_metrics, timeout=1)
                    except:
                        pass

                # Collect system stats every 10 iterations
                if consecutive_errors == 0:
                    self.system_stats = self._collect_system_stats()

                # Reset error counter on success
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    print(f"❌ Data collection failed {consecutive_errors} times, pausing...")
                    time.sleep(30)
                    consecutive_errors = 0
                else:
                    print(f"⚠️ Data collection error: {e}")
                    time.sleep(5)

            # Sleep for 1 second (configurable)
            time.sleep(1)

    def _collect_current_metrics(self) -> Dict:
        """Collect current trading metrics with memory efficiency"""
        try:
            current_time = time.time()

            # Simulate realistic trading data (replace with actual trading system integration)
            # This would normally pull from the trading core

            # Generate realistic portfolio progression
            base_value = 10000
            time_factor = (current_time % 86400) / 86400  # Daily cycle
            trend = np.sin(time_factor * 2 * np.pi) * 500  # ±500 variation
            noise = np.random.normal(0, 50)  # Random noise

            portfolio_value = base_value + trend + noise
            daily_pnl = trend + noise
            daily_pnl_percent = (daily_pnl / base_value) * 100

            # Simulate AI-enhanced metrics
            ai_confidence = np.random.uniform(0.6, 0.95)
            risk_score = np.random.uniform(0.1, 0.8)

            return {
                'timestamp': current_time,
                'portfolio_value': round(portfolio_value, 2),
                'daily_pnl': round(daily_pnl, 2),
                'daily_pnl_percent': round(daily_pnl_percent, 3),
                'total_pnl': round(portfolio_value - base_value, 2),
                'active_positions': np.random.randint(2, 8),
                'win_rate': round(np.random.uniform(60, 80), 1),
                'sharpe_ratio': round(np.random.uniform(1.5, 2.5), 2),
                'max_drawdown': round(np.random.uniform(-5, -2), 2),
                'current_drawdown': round(np.random.uniform(-3, 0), 2),
                'total_trades': np.random.randint(100, 200),
                'avg_trade_duration': np.random.randint(120, 600),
                'data_quality_score': round(np.random.uniform(0.85, 0.98), 3),
                'ai_confidence': round(ai_confidence, 3),
                'risk_score': round(risk_score, 3),
                'market_sentiment': np.random.choice(['bullish', 'bearish', 'neutral'],
                                                   p=[0.4, 0.3, 0.3])
            }

        except Exception as e:
            print(f"⚠️ Metrics collection error: {e}")
            return {
                'timestamp': time.time(),
                'portfolio_value': 10000,
                'daily_pnl': 0,
                'daily_pnl_percent': 0,
                'error': str(e)
            }

    def _collect_system_stats(self) -> Dict:
        """Collect system performance statistics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network (approximate)
            net_io = psutil.net_io_counters()
            network_usage = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB

            # Process info
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)

            return {
                'cpu_usage': round(cpu_percent, 1),
                'memory_usage': round(memory.percent, 1),
                'memory_available_mb': round(memory.available / (1024 * 1024), 1),
                'disk_usage': round(disk.percent, 1),
                'network_usage_mb': round(network_usage, 1),
                'dashboard_memory_mb': round(process_memory, 1),
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"⚠️ System stats collection error: {e}")
            return {}

    def _store_metrics_batch(self, metrics_list: List[Dict]):
        """Store metrics in database with batching for performance"""
        if not self.db_connection or not metrics_list:
            return

        try:
            cursor = self.db_connection.cursor()

            # Batch insert portfolio metrics
            portfolio_data = []
            trade_data = []
            ai_data = []

            for metrics in metrics_list:
                # Portfolio metrics
                portfolio_data.append((
                    metrics.get('timestamp', time.time()),
                    metrics.get('portfolio_value', 0),
                    metrics.get('daily_pnl', 0),
                    metrics.get('daily_pnl_percent', 0),
                    metrics.get('total_pnl', 0),
                    metrics.get('active_positions', 0),
                    metrics.get('win_rate', 0),
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('max_drawdown', 0),
                    metrics.get('current_drawdown', 0)
                ))

                # Simulate trade data (replace with actual trades)
                if np.random.random() < 0.05:  # 5% chance of new trade
                    trade_data.append((
                        metrics['timestamp'],
                        np.random.choice(['BTC-USDT', 'ETH-USDT', 'BNB-USDT']),
                        np.random.choice(['BUY', 'SELL']),
                        round(np.random.uniform(0.1, 2.0), 4),
                        round(np.random.uniform(30000, 60000), 2),
                        round(np.random.uniform(-100, 100), 2),
                        round(np.random.uniform(1, 5), 2),
                        'AI_Enhanced',
                        np.random.randint(60, 1800),
                        metrics.get('ai_confidence', 0.5)
                    ))

                # AI insights
                if np.random.random() < 0.1:  # 10% chance of AI insight
                    ai_data.append((
                        metrics['timestamp'],
                        'price_prediction',
                        np.random.choice(['BTC-USDT', 'ETH-USDT']),
                        metrics.get('ai_confidence', 0.5),
                        round(np.random.uniform(-0.1, 0.1), 3),
                        np.random.choice(['BUY', 'SELL', 'HOLD']),
                        json.dumps({'method': 'LSTM', 'confidence': metrics.get('ai_confidence', 0.5)})
                    ))

            # Execute batch inserts
            if portfolio_data:
                cursor.executemany('''
                    INSERT OR REPLACE INTO portfolio_metrics
                    (timestamp, portfolio_value, daily_pnl, daily_pnl_percent,
                     total_pnl, active_positions, win_rate, sharpe_ratio,
                     max_drawdown, current_drawdown)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', portfolio_data)

            if trade_data:
                cursor.executemany('''
                    INSERT INTO trades
                    (timestamp, symbol, side, quantity, price, pnl, fee, strategy, duration, ai_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', trade_data)

            if ai_data:
                cursor.executemany('''
                    INSERT INTO ai_insights
                    (timestamp, insight_type, symbol, confidence, prediction, recommendation, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', ai_data)

            self.db_connection.commit()

        except Exception as e:
            print(f"⚠️ Database batch storage error: {e}")

    def _update_caches(self, current_metrics: Dict):
        """Update in-memory caches for performance"""
        current_time = time.time()

        # Portfolio cache
        self.portfolio_cache = {
            'data': current_metrics,
            'timestamp': current_time
        }

        # Chart cache (rolling window)
        if 'chart_data' not in self.chart_cache:
            self.chart_cache['chart_data'] = []

        self.chart_cache['chart_data'].append({
            'timestamp': current_time,
            'value': current_metrics.get('portfolio_value', 0),
            'pnl': current_metrics.get('daily_pnl', 0)
        })

        # Keep only last 6 hours of chart data (21600 seconds / 10 second intervals)
        max_points = 2160
        if len(self.chart_cache['chart_data']) > max_points:
            self.chart_cache['chart_data'] = self.chart_cache['chart_data'][-max_points:]

        self.chart_cache['timestamp'] = current_time

    def _broadcast_loop(self):
        """Broadcast data to connected clients via WebSocket"""
        update_counter = 0
        last_stats_time = 0

        while self.running:
            try:
                # Get data from queue (blocking with timeout)
                try:
                    current_data = self.data_queue.get(timeout=5)
                except queue.Empty:
                    continue

                update_counter += 1

                # Prepare broadcast data
                broadcast_data = {
                    'current_metrics': current_data,
                    'timestamp': time.time(),
                    'update_counter': update_counter
                }

                # Add system stats every 10 updates
                current_time = time.time()
                if current_time - last_stats_time > 10:
                    broadcast_data['system_stats'] = self.system_stats
                    broadcast_data['performance_stats'] = self.performance_stats
                    last_stats_time = current_time

                # Add AI insights occasionally
                if np.random.random() < 0.1:  # 10% chance
                    broadcast_data['ai_insights'] = self._get_recent_ai_insights()

                # Broadcast to all connected clients
                socketio.emit('dashboard_update', broadcast_data)

                # Update performance stats
                self.performance_stats['updates_per_second'] = update_counter / max(1, current_time - time.time() + update_counter)

            except Exception as e:
                print(f"⚠️ Broadcast error: {e}")
                time.sleep(1)

    def _get_recent_ai_insights(self) -> List[Dict]:
        """Get recent AI insights for broadcast"""
        if not self.db_connection:
            return []

        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT timestamp, insight_type, symbol, confidence, prediction, recommendation
                FROM ai_insights
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 5
            ''', (time.time() - 3600,))  # Last hour

            rows = cursor.fetchall()
            return [{
                'timestamp': row[0],
                'type': row[1],
                'symbol': row[2],
                'confidence': row[3],
                'prediction': row[4],
                'recommendation': row[5]
            } for row in rows]

        except Exception as e:
            print(f"⚠️ AI insights fetch error: {e}")
            return []

    def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.running:
            try:
                # Clean up old database records (keep last 7 days)
                cutoff_time = time.time() - (7 * 24 * 3600)

                cursor = self.db_connection.cursor()
                cursor.execute('DELETE FROM portfolio_metrics WHERE timestamp < ?', (cutoff_time,))
                cursor.execute('DELETE FROM trades WHERE timestamp < ?', (cutoff_time,))
                cursor.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_time,))
                cursor.execute('DELETE FROM ai_insights WHERE timestamp < ?', (cutoff_time,))

                deleted_count = cursor.rowcount
                self.db_connection.commit()

                if deleted_count > 0:
                    print(f"🧹 Cleaned up {deleted_count} old records")

                # Optimize database every 24 hours
                if int(time.time()) % 86400 < 3600:  # Once per day
                    cursor.execute('VACUUM')
                    print("🧹 Database optimized")

            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")

            # Sleep for 6 hours
            time.sleep(21600)

    def _performance_monitor(self):
        """Monitor dashboard performance"""
        while self.running:
            try:
                # Update performance stats
                process = psutil.Process()
                self.performance_stats.update({
                    'memory_usage_mb': process.memory_info().rss / (1024 * 1024),
                    'cpu_usage_percent': process.cpu_percent(),
                    'active_clients': len(self.clients),
                    'queue_size': self.data_queue.qsize()
                })

                # Log performance every minute
                if int(time.time()) % 60 == 0:
                    stats = self.performance_stats.copy()
                    print(f"📊 Dashboard Performance: {stats['memory_usage_mb']:.1f}MB RAM, "
                          f"{stats['cpu_usage_percent']:.1f}% CPU, "
                          f"{stats['active_clients']} clients, "
                          f"{stats['queue_size']} queued updates")

            except Exception as e:
                print(f"⚠️ Performance monitoring error: {e}")

            time.sleep(10)

    def get_historical_data(self, hours: int = 24, data_type: str = 'portfolio') -> List[Dict]:
        """Get historical data for charts"""
        if not self.db_connection:
            return []

        try:
            cursor = self.db_connection.cursor()
            start_time = time.time() - (hours * 3600)

            if data_type == 'portfolio':
                cursor.execute('''
                    SELECT timestamp, portfolio_value, daily_pnl, active_positions
                    FROM portfolio_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                ''', (start_time,))

                return [{
                    'timestamp': row[0],
                    'portfolio_value': row[1],
                    'daily_pnl': row[2],
                    'active_positions': row[3]
                } for row in cursor.fetchall()]

            elif data_type == 'trades':
                cursor.execute('''
                    SELECT timestamp, symbol, side, quantity, price, pnl, strategy, ai_confidence
                    FROM trades
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''', (start_time,))

                return [{
                    'timestamp': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'quantity': row[3],
                    'price': row[4],
                    'pnl': row[5],
                    'strategy': row[6],
                    'ai_confidence': row[7]
                } for row in cursor.fetchall()]

        except Exception as e:
            print(f"⚠️ Historical data fetch error: {e}")
            return []

# Global dashboard instance
dashboard = AdvancedTradingDashboard()

# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('advanced_dashboard.html')

@app.route('/api/metrics')
def get_current_metrics():
    """API endpoint for current metrics"""
    return jsonify(dashboard.portfolio_cache.get('data', {}))

@app.route('/api/chart_data')
def get_chart_data():
    """API endpoint for chart data"""
    hours = request.args.get('hours', 6, type=int)
    data = dashboard.get_historical_data(hours, 'portfolio')
    return jsonify(data)

@app.route('/api/trades')
def get_recent_trades():
    """API endpoint for recent trades"""
    hours = request.args.get('hours', 24, type=int)
    data = dashboard.get_historical_data(hours, 'trades')
    return jsonify(data)

@app.route('/api/system_stats')
def get_system_stats():
    """API endpoint for system statistics"""
    return jsonify(dashboard.system_stats)

@app.route('/api/performance')
def get_performance_stats():
    """API endpoint for dashboard performance"""
    return jsonify(dashboard.performance_stats)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'clients': len(dashboard.clients),
        'memory_usage_mb': dashboard.performance_stats.get('memory_usage_mb', 0),
        'uptime_seconds': time.time() - psutil.boot_time()
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    dashboard.clients.add(request.sid)
    dashboard.performance_stats['active_clients'] = len(dashboard.clients)

    print(f'📱 Dashboard client connected: {request.sid} (Total: {len(dashboard.clients)})')

    # Send initial data
    initial_data = {
        'current_metrics': dashboard.portfolio_cache.get('data', {}),
        'system_stats': dashboard.system_stats,
        'chart_data': dashboard.get_historical_data(6, 'portfolio'),
        'recent_trades': dashboard.get_historical_data(24, 'trades')[:10],
        'ai_insights': dashboard._get_recent_ai_insights(),
        'timestamp': time.time()
    }

    emit('initial_data', initial_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    dashboard.clients.discard(request.sid)
    dashboard.performance_stats['active_clients'] = len(dashboard.clients)

    print(f'📱 Dashboard client disconnected: {request.sid} (Total: {len(dashboard.clients)})')

@socketio.on('request_chart_data')
def handle_chart_request(data):
    """Handle chart data requests"""
    hours = data.get('hours', 6)
    chart_data = dashboard.get_historical_data(hours, 'portfolio')
    emit('chart_data_update', {'data': chart_data, 'hours': hours})

if __name__ == '__main__':
    # Start dashboard
    dashboard.start_dashboard()

    # Run Flask app
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    print("🚀 Starting Advanced Trading Dashboard on http://localhost:5000"    print(f"📊 Debug mode: {debug}")
    print(f"💾 Memory usage: {dashboard.performance_stats.get('memory_usage_mb', 0):.1f}MB")

    socketio.run(app, host='0.0.0.0', port=port, debug=debug)
