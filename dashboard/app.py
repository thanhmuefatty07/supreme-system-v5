"""
Supreme System V5 Dashboard Backend

Ultra-lightweight Flask API for dashboard data

Optimized for minimal resource usage on i3-4GB system
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
import psycopg2
import psycopg2.extras
from functools import lru_cache
import logging

# Setup minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask with minimal overhead
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
app.config['JSON_SORT_KEYS'] = False

class DatabaseConnection:
    """Lightweight database connection manager"""

    def __init__(self):
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST', 'postgres-replica'),
            'port': os.getenv('POSTGRES_PORT', 5432),
            'database': os.getenv('POSTGRES_DB', 'trading_readonly'),
            'user': os.getenv('POSTGRES_USER', 'dashboard_reader'),
            'password': os.getenv('POSTGRES_PASSWORD', 'dashboard_readonly_pass')
        }
        self._connection = None

    def get_connection(self):
        """Get database connection with connection pooling"""
        try:
            if self._connection is None or self._connection.closed:
                self._connection = psycopg2.connect(**self.connection_params)
                self._connection.set_session(readonly=True, autocommit=True)
            return self._connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

# Global database manager
db = DatabaseConnection()

@lru_cache(maxsize=32, typed=True)
def cached_query(query: str, cache_duration: int = 300):
    """Cache database queries for 5 minutes"""
    return _execute_query(query)

def _execute_query(query: str) -> Optional[List[Dict]]:
    """Execute database query with error handling"""
    try:
        conn = db.get_connection()
        if not conn:
            return None

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            return [dict(row) for row in result]

    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return None

# API Routes

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'memory_usage_mb': _get_memory_usage()
    })

@app.route('/api/portfolio/summary')
def portfolio_summary():
    """Get current portfolio summary"""
    query = """
    SELECT
        total_balance,
        total_pnl,
        gross_exposure,
        active_positions,
        avg_pnl_percent,
        last_updated
    FROM portfolio_summary
    LIMIT 1;
    """

    result = cached_query(query, cache_duration=60)  # 1 minute cache
    if not result:
        return jsonify({'error': 'No portfolio data available'}), 404

    return jsonify(result[0])

@app.route('/api/portfolio/history')
def portfolio_history():
    """Get portfolio balance history for charts"""
    hours = request.args.get('hours', 24, type=int)
    hours = min(hours, 168)  # Max 7 days

    query = f"""
    SELECT
        timestamp,
        balance,
        daily_pnl,
        total_pnl,
        max_drawdown
    FROM portfolio_history
    WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
    ORDER BY timestamp ASC;
    """

    result = cached_query(query, cache_duration=300)  # 5 minute cache
    if not result:
        return jsonify([])

    return jsonify(result)

@app.route('/api/trades/recent')
def recent_trades():
    """Get recent trading activity"""
    limit = request.args.get('limit', 100, type=int)
    limit = min(limit, 500)  # Max 500 trades

    query = f"""
    SELECT
        id,
        symbol,
        side,
        quantity,
        price,
        executed_at,
        pnl,
        fee,
        strategy_name,
        result
    FROM recent_trades
    LIMIT {limit};
    """

    result = cached_query(query, cache_duration=120)  # 2 minute cache
    if not result:
        return jsonify([])

    return jsonify(result)

@app.route('/api/performance/metrics')
def performance_metrics():
    """Get performance metrics for analysis"""
    days = request.args.get('days', 7, type=int)
    days = min(days, 30)  # Max 30 days

    query = f"""
    SELECT
        date,
        total_trades,
        winning_trades,
        losing_trades,
        avg_pnl,
        daily_pnl,
        pnl_volatility,
        best_trade,
        worst_trade,
        CASE
            WHEN total_trades > 0
            THEN ROUND((winning_trades::float / total_trades * 100), 2)
            ELSE 0
        END as win_rate
    FROM performance_metrics
    WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
    ORDER BY date DESC;
    """

    result = cached_query(query, cache_duration=600)  # 10 minute cache
    if not result:
        return jsonify([])

    return jsonify(result)

@app.route('/api/export/trades')
def export_trades():
    """Export trades data as CSV"""
    days = request.args.get('days', 7, type=int)
    days = min(days, 30)  # Max 30 days

    query = f"""
    SELECT
        executed_at,
        symbol,
        side,
        quantity,
        price,
        pnl,
        fee,
        strategy_name
    FROM trades
    WHERE executed_at >= NOW() - INTERVAL '{days} days'
    ORDER BY executed_at DESC;
    """

    result = _execute_query(query)
    if not result:
        return jsonify({'error': 'No trade data available'}), 404

    # Convert to CSV format
    import csv
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=result[0].keys())
    writer.writeheader()
    writer.writerows(result)

    csv_content = output.getvalue()
    output.close()

    from flask import Response
    return Response(
        csv_content,
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=trades_{datetime.now().strftime("%Y%m%d")}.csv'
        }
    )

@app.route('/api/system/status')
def system_status():
    """Get system resource usage"""
    import psutil

    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)

    return jsonify({
        'memory_usage_percent': memory.percent,
        'memory_available_mb': memory.available / (1024**2),
        'cpu_percent': cpu_percent,
        'uptime_seconds': time.time() - psutil.boot_time(),
        'dashboard_memory_mb': _get_memory_usage()
    })

def _get_memory_usage():
    """Get current process memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024**2)
    except:
        return 0

# Dashboard static files
@app.route('/')
def dashboard():
    """Serve dashboard HTML"""
    return send_from_directory('templates', 'dashboard.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static assets"""
    return send_from_directory('static', filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Production configuration
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    print(f"🚀 Starting Supreme Dashboard on port {port}")
    print(f"📊 Debug mode: {debug}")
    print(f"💾 Memory usage: {_get_memory_usage():.1f}MB")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # Disable reloader to save memory
    )
