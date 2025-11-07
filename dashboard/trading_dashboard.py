import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

class TradingDashboard:
    """
    Real-time trading dashboard với interactive visualizations
    """
    def __init__(self, portfolio_manager, agents):
        self.portfolio_manager = portfolio_manager
        self.agents = agents
        self.setup_page()
    
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Supreme System V5 - Trading Dashboard",
            page_icon="🚀",
            layout="wide"
        )
    
    def display_header(self):
        """Display dashboard header với real-time metrics"""
        st.title("🚀 Supreme System V5 - Live Trading Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        metrics = self.portfolio_manager.calculate_performance_metrics()
        col1.metric("Portfolio Value", f"${metrics.get('current_portfolio_value', 0):,.2f}")
        col2.metric("Total Return", f"{metrics.get('total_return_percent', 0):.2f}%")
        col3.metric("Active Positions", metrics.get('active_positions', 0))
        col4.metric("Total Trades", metrics.get('total_trades', 0))
    
    def display_performance_chart(self):
        """Interactive performance chart"""
        if len(self.portfolio_manager.performance_history) > 1:
            df = pd.DataFrame(self.portfolio_manager.performance_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['total_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00ff88', width=3)
            ))
            fig.update_layout(
                title="Portfolio Performance Over Time",
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_trade_history(self):
        """Display recent trade history"""
        if self.portfolio_manager.order_executor.order_history:
            st.subheader("📋 Recent Trade History")
            trades_df = pd.DataFrame(self.portfolio_manager.order_executor.order_history)
            st.dataframe(trades_df.tail(10), use_container_width=True)
    
    def display_agent_performance(self):
        """Display agent performance comparison"""
        st.subheader("🤖 Agent Performance")
        agent_data = []
        for agent in self.agents:
            agent_data.append({
                "Agent ID": agent.agent_id,
                "Strategy": agent.__class__.__name__,
                "Current Position": getattr(agent, 'position', 0)
            })
        if agent_data:
            st.dataframe(pd.DataFrame(agent_data), use_container_width=True)
    
    def run(self):
        """Run the dashboard"""
        self.display_header()
        col1, col2 = st.columns([2, 1])
        with col1:
            self.display_performance_chart()
            self.display_trade_history()
        with col2:
            self.display_agent_performance()
            st.subheader("⚡ Quick Actions")
            if st.button("🔄 Refresh Data"):
                st.rerun()
