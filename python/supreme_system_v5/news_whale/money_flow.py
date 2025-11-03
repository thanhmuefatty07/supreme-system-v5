"""
Money Flow Aggregator for Supreme System V5.
Exchange flow analysis and money movement tracking.
"""

from typing import Dict, List, Optional, Any, NamedTuple
import time
from collections import defaultdict
from .whale_tracking import WhaleTransaction

class ExchangeFlow:
    """Exchange flow data structure."""

    def __init__(self, exchange_from: str, exchange_to: str, btc_amount: float,
                 usd_value: float, timestamp: float):
        self.exchange_from = exchange_from
        self.exchange_to = exchange_to
        self.btc_amount = btc_amount
        self.usd_value = usd_value
        self.timestamp = timestamp
        self.flow_type = 'transfer'  # Could be 'arbitrage', 'rebalancing', etc.

class MoneyFlowAggregator:
    """
    Money flow analysis across exchanges.

    Tracks BTC movements between exchanges to identify:
    - Arbitrage opportunities
    - Large rebalancing moves
    - Exchange-specific accumulation/distribution
    - Inter-exchange whale activities
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize money flow aggregator.

        Args:
            config: Configuration parameters
        """
        self.config = config

        # Flow tracking
        self.exchange_balances = defaultdict(float)  # BTC balance per exchange
        self.recent_flows = []  # Recent exchange flows
        self.flow_history = defaultdict(list)  # Historical flows per exchange pair

        # Analysis windows
        self.analysis_window_hours = config.get('analysis_window_hours', 24)
        self.max_history_size = config.get('max_history_size', 1000)

        # Thresholds
        self.large_flow_threshold_btc = config.get('large_flow_threshold_btc', 50)
        self.arbitrage_threshold_pct = config.get('arbitrage_threshold_pct', 0.5)  # 0.5%

    def add_whale_transaction(self, tx: WhaleTransaction):
        """
        Process whale transaction for flow analysis.

        Args:
            tx: Whale transaction to analyze
        """
        # For transfers between exchanges, update balances
        if tx.transaction_type == 'transfer':
            # This would require additional API data in real implementation
            # For now, we'll track based on transaction patterns
            pass

    def add_exchange_flow(self, flow: ExchangeFlow):
        """
        Add exchange flow data.

        Args:
            flow: Exchange flow to add
        """
        # Update balances
        self.exchange_balances[flow.exchange_from] -= flow.btc_amount
        self.exchange_balances[flow.exchange_to] += flow.btc_amount

        # Store flow
        self.recent_flows.append(flow)

        # Maintain history size
        if len(self.recent_flows) > self.max_history_size:
            self.recent_flows.pop(0)

        # Track flow history
        flow_key = f"{flow.exchange_from}->{flow.exchange_to}"
        self.flow_history[flow_key].append(flow)

        # Clean old history (keep only recent flows)
        cutoff_time = time.time() - (self.analysis_window_hours * 3600)
        for flows in self.flow_history.values():
            flows[:] = [f for f in flows if f.timestamp > cutoff_time]

    def get_exchange_balances(self) -> Dict[str, float]:
        """Get current BTC balances per exchange."""
        return dict(self.exchange_balances)

    def get_large_flows(self, hours: int = 24) -> List[ExchangeFlow]:
        """
        Get large exchange flows within time window.

        Args:
            hours: Hours to look back

        Returns:
            List of large flows
        """
        cutoff_time = time.time() - (hours * 3600)

        large_flows = []
        for flow in self.recent_flows:
            if (flow.timestamp > cutoff_time and
                flow.btc_amount >= self.large_flow_threshold_btc):
                large_flows.append(flow)

        return large_flows

    def analyze_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """
        Analyze potential arbitrage opportunities based on flows.

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        # This would analyze price differences between exchanges
        # and correlate with flow patterns in a real implementation

        # Mock analysis for demonstration
        exchanges = list(self.exchange_balances.keys())
        if len(exchanges) >= 2:
            # Simulate potential arbitrage detection
            for i, ex1 in enumerate(exchanges):
                for ex2 in exchanges[i+1:]:
                    # Check for significant flow imbalance
                    flow_key_1to2 = f"{ex1}->{ex2}"
                    flow_key_2to1 = f"{ex2}->{ex1}"

                    flow_1to2 = len(self.flow_history[flow_key_1to2])
                    flow_2to1 = len(self.flow_history[flow_key_2to1])

                    if abs(flow_1to2 - flow_2to1) > 5:  # Significant imbalance
                        opportunities.append({
                            'exchange_pair': f"{ex1}-{ex2}",
                            'flow_imbalance': flow_1to2 - flow_2to1,
                            'potential_arbitrage': True
                        })

        return opportunities

    def get_flow_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of money flows.

        Args:
            hours: Hours to analyze

        Returns:
            Flow summary statistics
        """
        cutoff_time = time.time() - (hours * 3600)

        # Filter recent flows
        recent_flows = [f for f in self.recent_flows if f.timestamp > cutoff_time]

        if not recent_flows:
            return {'error': 'No flow data available'}

        total_volume = sum(f.btc_amount for f in recent_flows)
        total_value = sum(f.usd_value for f in recent_flows)

        # Flow directions
        inflows = defaultdict(float)
        outflows = defaultdict(float)

        for flow in recent_flows:
            outflows[flow.exchange_from] += flow.btc_amount
            inflows[flow.exchange_to] += flow.btc_amount

        return {
            'total_flows': len(recent_flows),
            'total_btc_volume': total_volume,
            'total_usd_value': total_value,
            'time_window_hours': hours,
            'exchange_inflows': dict(inflows),
            'exchange_outflows': dict(outflows),
            'net_positions': {
                exchange: inflows[exchange] - outflows[exchange]
                for exchange in set(list(inflows.keys()) + list(outflows.keys()))
            }
        }

# Demo function for testing
def demo_money_flow():
    """Demonstrate money flow analysis capabilities."""
    print("💰 SUPREME SYSTEM V5 - Money Flow Analysis Demo")
    print("=" * 60)

    # Initialize aggregator
    config = {
        'analysis_window_hours': 24,
        'max_history_size': 100,
        'large_flow_threshold_btc': 25
    }

    flow_aggregator = MoneyFlowAggregator(config)

    # Simulate some exchange flows
    exchanges = ['Binance', 'Coinbase', 'Kraken', 'Gemini']

    print("📊 Simulating exchange flows...")

    for i in range(10):
        # Generate random flow
        from_ex = random.choice(exchanges)
        to_ex = random.choice([ex for ex in exchanges if ex != from_ex])
        btc_amount = random.uniform(10, 100)
        price = random.uniform(45000, 55000)
        usd_value = btc_amount * price

        flow = ExchangeFlow(
            exchange_from=from_ex,
            exchange_to=to_ex,
            btc_amount=btc_amount,
            usd_value=usd_value,
            timestamp=time.time()
        )

        flow_aggregator.add_exchange_flow(flow)
        time.sleep(0.01)  # Small delay for timestamps

    # Analyze flows
    summary = flow_aggregator.get_flow_summary()
    large_flows = flow_aggregator.get_large_flows()
    arbitrage_ops = flow_aggregator.analyze_arbitrage_opportunities()

    print("\n💹 FLOW SUMMARY (24h):")
    print(f"   Total Flows: {summary['total_flows']}")
    print(".2f")
    print(".2f")

    print("\n🏦 EXCHANGE NET POSITIONS:")
    for exchange, position in summary['net_positions'].items():
        direction = "📈 INFLOW" if position > 0 else "📉 OUTFLOW" if position < 0 else "➡️ BALANCED"
        print(".2f")

    print(f"\n🚨 LARGE FLOWS (>25 BTC): {len(large_flows)} detected")
    for flow in large_flows[:3]:  # Show first 3
        print(".2f")

    if arbitrage_ops:
        print(f"\n🎯 ARBITRAGE OPPORTUNITIES: {len(arbitrage_ops)} detected")
        for op in arbitrage_ops[:2]:
            print(f"   {op['exchange_pair']}: Flow imbalance {op['flow_imbalance']}")
    else:
        print("\n🎯 ARBITRAGE OPPORTUNITIES: None detected")

    print("\n✅ Money Flow Analysis Demo Complete")
    print("   System ready for inter-exchange flow monitoring!")

if __name__ == "__main__":
    import random
    demo_money_flow()
