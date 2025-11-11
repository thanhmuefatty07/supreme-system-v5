#!/usr/bin/env python3
"""
Supreme System V5 - Real-time Monitoring Dashboard

Interactive dashboard for monitoring trading system performance,
real-time data streams, and system health.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import threading
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.data_pipeline import DataPipeline
from data.realtime_client import BinanceWebSocketClient
from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy


class MonitoringDashboard:
    """Real-time monitoring dashboard for Supreme System V5."""

    def __init__(self):
        self.pipeline = DataPipeline()
        self.ws_client = None
        self.strategies = self._initialize_strategies()

        # Dashboard state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.last_update = datetime.now()

        # Data buffers for charts
        self.price_data = {}
        self.signal_data = []
        self.system_metrics = {}

    def _initialize_strategies(self):
        """Initialize trading strategies for monitoring."""
        return {
            'Moving Average': MovingAverageStrategy(),
            'Mean Reversion': MeanReversionStrategy(),
            'Momentum': MomentumStrategy(),
            'Breakout': BreakoutStrategy()
        }

    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.ws_client = BinanceWebSocketClient()

        # Subscribe to key streams
        self.ws_client.subscribe_price_stream('BTCUSDT')
        self.ws_client.subscribe_price_stream('ETHUSDT')
        self.ws_client.subscribe_trade_stream('BTCUSDT')
        self.ws_client.subscribe_kline_stream('BTCUSDT', '1m')

        # Start WebSocket client
        self.ws_client.start()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        st.success("✅ Real-time monitoring started!")

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.ws_client:
            self.ws_client.stop()

        st.info("⏹️ Monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._update_data()
                self.last_update = datetime.now()
                time.sleep(1)  # Update every second
            except Exception as e:
                st.error(f"Monitoring error: {e}")
                time.sleep(5)

    def _update_data(self):
        """Update dashboard data."""
        if not self.ws_client:
            return

        # Update price data
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            price_data = self.ws_client.get_stream_data(f"{symbol.lower()}@ticker")
            if price_data:
                latest = price_data[-1]
                self.price_data[symbol] = {
                    'price': float(latest.get('c', 0)),
                    'volume': float(latest.get('v', 0)),
                    'change': float(latest.get('P', 0)),  # 24h change %
                    'timestamp': datetime.now()
                }

        # Update system metrics
        self.system_metrics = {
            'pipeline_status': self.pipeline.get_pipeline_status(),
            'ws_metrics': self.ws_client.get_metrics() if self.ws_client else {},
            'data_info': self.pipeline.storage.get_data_info()
        }

        # Generate trading signals for visualization
        self._generate_signals()

    def _generate_signals(self):
        """Generate trading signals for visualization."""
        # Get recent BTC data for signal generation
        btc_data = self.pipeline.get_data('BTCUSDT', '1h', end_date=datetime.now().strftime('%Y-%m-%d'))

        if btc_data is None or btc_data.empty:
            return

        # Generate signals from all strategies
        signals = []
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(btc_data)
                signals.append({
                    'timestamp': datetime.now(),
                    'strategy': strategy_name,
                    'signal': signal,
                    'price': btc_data['close'].iloc[-1]
                })
            except Exception as e:
                signals.append({
                    'timestamp': datetime.now(),
                    'strategy': strategy_name,
                    'signal': 0,
                    'price': btc_data['close'].iloc[-1],
                    'error': str(e)
                })

        # Keep last 100 signals
        self.signal_data.extend(signals)
        if len(self.signal_data) > 100:
            self.signal_data = self.signal_data[-100:]


def create_price_chart(data_dict):
    """Create real-time price chart."""
    if not data_dict:
        return px.line(title="No Price Data Available")

    symbols = list(data_dict.keys())
    prices = [data['price'] for data in data_dict.values()]

    fig = go.Figure()

    for symbol, data in data_dict.items():
        fig.add_trace(go.Scatter(
            x=[data['timestamp']],
            y=[data['price']],
            mode='lines+markers',
            name=symbol,
            line=dict(width=2)
        ))

    fig.update_layout(
        title="Real-time Cryptocurrency Prices",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        height=400
    )

    return fig


def create_signal_chart(signals_data):
    """Create trading signals visualization."""
    if not signals_data:
        return px.scatter(title="No Signal Data Available")

    df = pd.DataFrame(signals_data)

    # Create subplot for each strategy
    strategies = df['strategy'].unique()
    fig = make_subplots(
        rows=len(strategies),
        cols=1,
        subplot_titles=strategies,
        shared_xaxes=True
    )

    colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'blue'}

    for i, strategy in enumerate(strategies, 1):
        strategy_data = df[df['strategy'] == strategy]

        for signal_val, signal_name in [(-1, 'SELL'), (0, 'HOLD'), (1, 'BUY')]:
            signal_points = strategy_data[strategy_data['signal'] == signal_val]
            if not signal_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=signal_points['timestamp'],
                        y=signal_points['price'],
                        mode='markers',
                        name=f"{strategy} {signal_name}",
                        marker=dict(
                            color=colors.get(signal_name, 'gray'),
                            size=8,
                            symbol='triangle-up' if signal_name == 'BUY' else 'triangle-down' if signal_name == 'SELL' else 'circle'
                        )
                    ),
                    row=i, col=1
                )

    fig.update_layout(height=600, title_text="Trading Signals by Strategy")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Price (USDT)")

    return fig


def create_system_health_gauge(metrics):
    """Create system health gauge chart."""
    if not metrics:
        return go.Figure()

    # Calculate overall health score
    pipeline_health = 1.0 if metrics.get('pipeline_status', {}).get('components', {}).get('data_storage', {}).get('data_info', {}).get('total_files', 0) > 0 else 0.0
    ws_health = 1.0 if metrics.get('ws_metrics', {}).get('connection_healthy', False) else 0.0

    overall_health = (pipeline_health + ws_health) / 2 * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_health,
        title={'text': "System Health"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ]
        }
    ))

    return fig


def create_metrics_cards(metrics):
    """Create metrics info cards."""
    if not metrics:
        return []

    ws_metrics = metrics.get('ws_metrics', {})
    data_info = metrics.get('data_info', {})

    cards = [
        {
            'title': 'WebSocket Status',
            'value': '🟢 Connected' if ws_metrics.get('connection_healthy') else '🔴 Disconnected',
            'delta': f"{ws_metrics.get('messages_received', 0)} msgs"
        },
        {
            'title': 'Active Streams',
            'value': ws_metrics.get('active_streams', 0),
            'delta': f"{ws_metrics.get('messages_per_second', 0):.1f} msg/s"
        },
        {
            'title': 'Data Files',
            'value': data_info.get('total_files', 0),
            'delta': f"{data_info.get('total_size_mb', 0):.1f} MB"
        },
        {
            'title': 'Strategies',
            'value': 4,
            'delta': 'All Active'
        }
    ]

    return cards


def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Supreme System V5 - Monitoring Dashboard",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Supreme System V5 - Real-time Monitoring Dashboard")
    st.markdown("---")

    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = MonitoringDashboard()

    dashboard = st.session_state.dashboard

    # Control panel
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Start Monitoring", type="primary"):
            dashboard.start_monitoring()

    with col2:
        if st.button("⏹️ Stop Monitoring"):
            dashboard.stop_monitoring()

    with col3:
        if st.button("🔄 Refresh Data"):
            dashboard._update_data()
            st.success("Data refreshed!")

    # System status
    st.subheader("📊 System Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        health_gauge = create_system_health_gauge(dashboard.system_metrics)
        st.plotly_chart(health_gauge, use_container_width=True)

    with status_col2:
        st.metric(
            "Last Update",
            dashboard.last_update.strftime("%H:%M:%S"),
            f"{(datetime.now() - dashboard.last_update).seconds}s ago"
        )

        monitoring_status = "🟢 Active" if dashboard.is_monitoring else "🔴 Inactive"
        st.metric("Monitoring Status", monitoring_status)

    # Metrics cards
    st.subheader("📈 Key Metrics")

    metrics_cards = create_metrics_cards(dashboard.system_metrics)
    cols = st.columns(len(metrics_cards))

    for i, card in enumerate(metrics_cards):
        with cols[i]:
            st.metric(card['title'], card['value'], card['delta'])

    # Real-time price chart
    st.subheader("💰 Real-time Prices")
    price_chart = create_price_chart(dashboard.price_data)
    st.plotly_chart(price_chart, use_container_width=True)

    # Trading signals
    st.subheader("🎯 Trading Signals")
    signal_chart = create_signal_chart(dashboard.signal_data)
    st.plotly_chart(signal_chart, use_container_width=True)

    # Data pipeline status
    st.subheader("🔧 Data Pipeline Status")

    pipeline_col1, pipeline_col2 = st.columns(2)

    with pipeline_col1:
        st.markdown("**Components Status:**")
        pipeline_status = dashboard.system_metrics.get('pipeline_status', {}).get('components', {})
        for component, status in pipeline_status.items():
            healthy = status.get('healthy', False) if isinstance(status, dict) else True
            icon = "✅" if healthy else "❌"
            st.write(f"{icon} {component.replace('_', ' ').title()}")

    with pipeline_col2:
        st.markdown("**Data Summary:**")
        data_info = dashboard.system_metrics.get('data_info', {})
        st.write(f"📁 Total Files: {data_info.get('total_files', 0)}")
        st.write(f"💾 Total Size: {data_info.get('total_size_mb', 0):.1f} MB")
        st.write(f"🪙 Symbols: {data_info.get('total_symbols', 0)}")
        st.write(f"⏰ Intervals: {data_info.get('total_intervals', 0)}")

    # Raw data viewer
    with st.expander("🔍 Raw Data Viewer"):
        st.subheader("Recent Price Data")
        if dashboard.price_data:
            price_df = pd.DataFrame.from_dict(dashboard.price_data, orient='index')
            st.dataframe(price_df)

        st.subheader("Recent Signals")
        if dashboard.signal_data:
            signals_df = pd.DataFrame(dashboard.signal_data[-20:])  # Last 20 signals
            st.dataframe(signals_df)

    # Auto-refresh
    if dashboard.is_monitoring:
        time.sleep(2)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
