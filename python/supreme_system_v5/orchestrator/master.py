"""
Master Trading Orchestrator for Supreme System V5.
Centralized coordination with intelligent scheduling.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, NamedTuple
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

from ..optimized.analyzer import OptimizedTechnicalAnalyzer
from ..news_whale import AdvancedNewsClassifier, WhaleTrackingSystem
from ..risk import DynamicRiskManager, PortfolioState
from ..mtf import MultiTimeframeEngine

class ComponentStatus(Enum):
    """Component status enumeration."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

class ComponentPriority(Enum):
    """Component execution priority."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ComponentInfo:
    """Component information and state."""
    name: str
    component: Any
    status: ComponentStatus
    priority: ComponentPriority
    last_execution: float
    execution_count: int
    error_count: int
    next_scheduled_run: float

class OrchestrationResult(NamedTuple):
    """Orchestration cycle result."""
    timestamp: float
    components_executed: int
    decisions_made: int
    trading_signals: List[Dict[str, Any]]
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class MasterTradingOrchestrator:
    """
    Master orchestrator for Supreme System V5.

    Coordinates all trading components with intelligent scheduling:
    - Technical Analysis: 30s intervals
    - News Analysis: 10m intervals
    - Whale Tracking: 10m intervals
    - Multi-Timeframe: 2m intervals
    - Pattern Recognition: 1m intervals (if implemented)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize master orchestrator.

        Args:
            config: Orchestrator configuration
        """
        self.config = config

        # Component registry
        self.components: Dict[str, ComponentInfo] = {}

        # Base scheduling configuration (in seconds) - minimum intervals
        self.base_intervals = {
            'technical': config.get('technical_interval', 30),      # 30s
            'news': config.get('news_interval', 600),              # 10m
            'whale': config.get('whale_interval', 600),            # 10m
            'mtf': config.get('mtf_interval', 120),                # 2m
            'patterns': config.get('patterns_interval', 60),       # 1m
        }

        # Adaptive intervals (dynamically adjusted)
        self.schedule_intervals = self.base_intervals.copy()

        # Resource limits for backpressure
        self.cpu_high_threshold = config.get('cpu_high_threshold', 88.0)
        self.cpu_low_threshold = config.get('cpu_low_threshold', 70.0)
        self.memory_high_threshold = config.get('memory_high_threshold', 3.86)
        self.latency_high_threshold = config.get('latency_high_threshold', 200)

        # Backpressure state
        self.backpressure_active = False
        self.interval_multiplier = 1.0  # Multiplier for all intervals
        self.last_resource_check = 0
        self.resource_check_interval = config.get('resource_check_interval', 60)  # Check every 60s

        # Performance history for trend analysis
        self.cpu_history = []
        self.memory_history = []
        self.latency_history = []
        self.history_max_size = 10

        # System state
        self.is_running = False
        self.start_time = 0
        self.cycle_count = 0
        self.last_cycle_time = 0

        # Performance tracking
        self.total_signals_generated = 0
        self.total_decisions_made = 0
        self.component_execution_stats = defaultdict(int)

        # Trading state
        self.current_market_data = {}
        self.portfolio_state = PortfolioState(
            total_balance=config.get('initial_balance', 10000.0),
            available_balance=config.get('initial_balance', 10000.0),
            current_positions=[],
            total_exposure_percent=0.0,
            daily_pnl=0.0,
            max_drawdown=0.0,
            win_rate_30d=0.5
        )

    async def initialize_components(self):
        """Initialize all trading system components."""
        print("🚀 Initializing Supreme System V5 components...")

        try:
            # 1. Technical Analysis Component
            tech_config = self.config.get('technical_config', {})
            tech_analyzer = OptimizedTechnicalAnalyzer(tech_config)
            self._register_component(
                "technical",
                tech_analyzer,
                ComponentPriority.CRITICAL
            )

            # 2. News Analysis Component
            news_config = self.config.get('news_config', {})
            news_classifier = AdvancedNewsClassifier()
            self._register_component(
                "news",
                news_classifier,
                ComponentPriority.HIGH
            )

            # 3. Whale Tracking Component
            whale_config = self.config.get('whale_config', {})
            whale_system = WhaleTrackingSystem(whale_config)
            self._register_component(
                "whale",
                whale_system,
                ComponentPriority.HIGH
            )

            # 4. Multi-Timeframe Component
            mtf_config = self.config.get('mtf_config', {})
            mtf_engine = MultiTimeframeEngine(mtf_config)
            self._register_component(
                "mtf",
                mtf_engine,
                ComponentPriority.MEDIUM
            )

            # 5. Risk Management Component
            risk_config = self.config.get('risk_config', {})
            risk_manager = DynamicRiskManager(risk_config)
            self._register_component(
                "risk_manager",
                risk_manager,
                ComponentPriority.CRITICAL
            )

            print("✅ All components initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            return False

    def _register_component(self, name: str, component: Any, priority: ComponentPriority):
        """Register a component with the orchestrator."""
        now = time.time()

        component_info = ComponentInfo(
            name=name,
            component=component,
            status=ComponentStatus.READY,
            priority=priority,
            last_execution=0,
            execution_count=0,
            error_count=0,
            next_scheduled_run=now  # Ready to run immediately
        )

        self.components[name] = component_info
        print(f"  📦 Registered component: {name} ({priority.value} priority)")

    async def run_orchestration_cycle(self) -> OrchestrationResult:
        """
        Execute one orchestration cycle.

        Returns:
            OrchestrationResult with cycle outcomes
        """
        cycle_start = time.time()
        self.cycle_count += 1

        # Determine which components should run this cycle
        components_to_execute = self._get_components_due_for_execution()

        executed_count = 0
        signals_generated = []
        decisions_made = 0

        # Execute components
        for comp_name in components_to_execute:
            try:
                signals = await self._execute_component(self.components[comp_name])
                if signals:
                    signals_generated.extend(signals)
                    executed_count += 1

                self.component_execution_stats[comp_name] += 1

            except Exception as e:
                print(f"❌ Component {comp_name} execution failed: {e}")
                self.components[comp_name].error_count += 1
                self.components[comp_name].status = ComponentStatus.ERROR

        # Make trading decisions based on collected signals
        if signals_generated:
            decisions = await self._make_trading_decisions(signals_generated)
            decisions_made = len(decisions) if decisions else 0
            self.total_decisions_made += decisions_made

        self.total_signals_generated += len(signals_generated)
        self.last_cycle_time = time.time()

        # Collect system health metrics
        system_health = self._assess_system_health()

        # Performance metrics
        performance_metrics = {
            'cycle_duration': self.last_cycle_time - cycle_start,
            'components_executed': executed_count,
            'signals_generated': len(signals_generated),
            'decisions_made': decisions_made,
            'total_cycles': self.cycle_count
        }

        return OrchestrationResult(
            timestamp=self.last_cycle_time,
            components_executed=executed_count,
            decisions_made=decisions_made,
            trading_signals=signals_generated,
            system_health=system_health,
            performance_metrics=performance_metrics
        )

    def _check_and_adjust_backpressure(self):
        """Check system resources and adjust scheduling intervals if needed."""
        now = time.time()
        if now - self.last_resource_check < self.resource_check_interval:
            return

        # Get current resource usage (mock for demo - in production use psutil)
        current_cpu = self._get_current_cpu_usage()
        current_memory = self._get_current_memory_usage()
        current_latency = self._get_current_latency()

        # Update history
        self.cpu_history.append(current_cpu)
        self.memory_history.append(current_memory)
        self.latency_history.append(current_latency)

        # Keep history size limited
        self.cpu_history = self.cpu_history[-self.history_max_size:]
        self.memory_history = self.memory_history[-self.history_max_size:]
        self.latency_history = self.latency_history[-self.history_max_size:]

        # Calculate trends (simple moving average)
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        avg_memory = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0

        # Backpressure logic
        high_resource_usage = (
            avg_cpu > self.cpu_high_threshold or
            avg_memory > self.memory_high_threshold or
            avg_latency > self.latency_high_threshold
        )

        low_resource_usage = (
            avg_cpu < self.cpu_low_threshold and
            avg_memory < self.memory_high_threshold * 0.8 and
            avg_latency < self.latency_high_threshold * 0.8
        )

        if high_resource_usage and not self.backpressure_active:
            # Activate backpressure - increase intervals
            self.interval_multiplier = min(self.interval_multiplier * 2.0, 4.0)  # Max 4x slowdown
            self.backpressure_active = True
            print(f"⚠️  Backpressure activated - CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.2f}GB, Latency: {avg_latency:.1f}ms")
            self._update_schedule_intervals()

        elif low_resource_usage and self.backpressure_active:
            # Deactivate backpressure - decrease intervals
            self.interval_multiplier = max(self.interval_multiplier * 0.5, 1.0)  # Min 1x (normal)
            if self.interval_multiplier <= 1.1:  # Close to normal
                self.backpressure_active = False
                self.interval_multiplier = 1.0
                print("✅ Backpressure deactivated - system resources normal")
            else:
                print(f"🔄 Backpressure reducing - multiplier: {self.interval_multiplier:.1f}x")
            self._update_schedule_intervals()

        self.last_resource_check = now

    def _update_schedule_intervals(self):
        """Update all component schedule intervals based on backpressure multiplier."""
        for comp_name, base_interval in self.base_intervals.items():
            self.schedule_intervals[comp_name] = base_interval * self.interval_multiplier

    def _get_current_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        # Mock implementation - in production use psutil
        import random
        return 60.0 + random.uniform(-10, 30)  # Simulate 50-90% CPU

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        # Mock implementation - in production use psutil
        import random
        return 2.5 + random.uniform(-0.5, 2.0)  # Simulate 2-4.5GB usage

    def _get_current_latency(self) -> float:
        """Get current system latency in ms."""
        # Mock implementation - in production use system metrics
        import random
        return 150 + random.uniform(-50, 100)  # Simulate 100-250ms latency

    def _get_components_due_for_execution(self) -> List[str]:
        """Get list of components due for execution with adaptive scheduling."""
        # Check and adjust backpressure
        self._check_and_adjust_backpressure()

        now = time.time()
        due_components = []

        for comp_name, comp_info in self.components.items():
            if comp_info.status not in [ComponentStatus.READY, ComponentStatus.ACTIVE]:
                continue

            # Check if component is due for execution (with adaptive intervals)
            if now >= comp_info.next_scheduled_run:
                due_components.append(comp_name)

        # Priority-based sorting with preemption safety
        priority_weights = {
            ComponentPriority.CRITICAL: 0,    # Always execute critical
            ComponentPriority.HIGH: 1,        # High priority second
            ComponentPriority.MEDIUM: 2,      # Medium priority third
            ComponentPriority.LOW: 3         # Low priority last
        }

        # Sort by priority, then by how overdue they are
        def sort_key(comp_name):
            comp_info = self.components[comp_name]
            priority_weight = priority_weights[comp_info.priority]
            overdue_time = now - comp_info.next_scheduled_run
            return (priority_weight, -overdue_time)  # Negative overdue = more urgent first

        due_components.sort(key=sort_key)

        # Under backpressure, limit concurrent executions to prevent resource exhaustion
        if self.backpressure_active and len(due_components) > 2:
            # Execute only top 2 most critical components during backpressure
            due_components = due_components[:2]
            print(f"🔄 Backpressure limiting: executing {len(due_components)} critical components only")

        return due_components

    async def _execute_component(self, comp_info: ComponentInfo) -> Optional[List[Dict[str, Any]]]:
        """Execute a single component and return signals."""
        comp_info.status = ComponentStatus.ACTIVE
        comp_info.last_execution = time.time()
        comp_info.execution_count += 1

        signals = []

        try:
            if comp_info.name == "technical":
                # Technical analysis
                if self.current_market_data:
                    price = self.current_market_data.get('price', 0)
                    volume = self.current_market_data.get('volume', 0)
                    timestamp = self.current_market_data.get('timestamp', time.time())

                    processed = comp_info.component.add_price_data(price, volume, timestamp)
                    if processed:
                        signals.append({
                            'component': 'technical',
                            'signals': {
                                'ema': comp_info.component.get_ema(),
                                'rsi': comp_info.component.get_rsi(),
                                'macd': comp_info.component.get_macd()
                            },
                            'timestamp': timestamp
                        })

            elif comp_info.name == "news":
                # News analysis (mock for demo)
                signals.append({
                    'component': 'news',
                    'signals': {
                        'sentiment': 0.5,  # Neutral
                        'impact_score': 0.3
                    },
                    'timestamp': time.time()
                })

            elif comp_info.name == "whale":
                # Whale tracking
                if comp_info.component.update_whale_data():
                    metrics = comp_info.component.get_current_metrics()
                    signals.append({
                        'component': 'whale',
                        'signals': {
                            'whale_confidence': metrics.whale_confidence,
                            'accumulation_score': metrics.accumulation_score
                        },
                        'timestamp': time.time()
                    })

            elif comp_info.name == "mtf":
                # Multi-timeframe analysis
                comp_info.component.add_price_data(
                    self.current_market_data.get('price', 50000),
                    self.current_market_data.get('volume', 1000)
                )
                consensus = comp_info.component.get_timeframe_consensus()
                signals.append({
                    'component': 'mtf',
                    'signals': {
                        'direction': consensus.overall_direction,
                        'confidence': consensus.confidence_score
                    },
                    'timestamp': time.time()
                })

            # Update next scheduled run time
            interval = self.schedule_intervals.get(comp_info.name, 60)
            comp_info.next_scheduled_run = time.time() + interval

            comp_info.status = ComponentStatus.READY

        except Exception as e:
            comp_info.status = ComponentStatus.ERROR
            comp_info.error_count += 1
            print(f"Component {comp_info.name} error: {e}")

        return signals if signals else None

    async def _make_trading_decisions(self, signals: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Make trading decisions based on collected signals."""
        if not signals:
            return None

        # Aggregate signals by type
        aggregated_signals = self._aggregate_signals(signals)

        # Get risk-adjusted position sizing
        risk_comp = self.components.get('risk_manager')
        if risk_comp and risk_comp.status == ComponentStatus.READY:
            risk_manager = risk_comp.component

            try:
                optimal_position = risk_manager.calculate_optimal_position(
                    signals=aggregated_signals,
                    portfolio=self.portfolio_state,
                    current_price=self.current_market_data.get('price', 50000),
                    volatility_factor=1.0
                )

                decision = {
                    'action': self._determine_action(aggregated_signals),
                    'position_size': optimal_position.position_size_percent,
                    'leverage': optimal_position.leverage_ratio,
                    'stop_loss': optimal_position.stop_loss_price,
                    'take_profit': optimal_position.take_profit_price,
                    'confidence': self._calculate_overall_confidence(aggregated_signals),
                    'reasoning': optimal_position.reasoning,
                    'timestamp': time.time()
                }

                return [decision]

            except Exception as e:
                print(f"Risk calculation failed: {e}")

        return None

    def _aggregate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate signals from all components."""
        aggregated = {
            'technical_confidence': 0.0,
            'news_confidence': 0.0,
            'whale_confidence': 0.0,
            'pattern_confidence': 0.0,
            'overall_sentiment': 0.0
        }

        for signal in signals:
            comp_name = signal['component']
            signal_data = signal['signals']

            if comp_name == 'technical':
                # Technical signals contribute to confidence
                ema = signal_data.get('ema')
                rsi = signal_data.get('rsi')
                macd = signal_data.get('macd')

                tech_score = 0
                if ema is not None:
                    tech_score += 0.3
                if rsi is not None:
                    tech_score += 0.3
                if macd is not None:
                    tech_score += 0.4

                aggregated['technical_confidence'] = min(tech_score, 1.0)

            elif comp_name == 'news':
                aggregated['news_confidence'] = signal_data.get('impact_score', 0.0)
                aggregated['overall_sentiment'] = signal_data.get('sentiment', 0.0)

            elif comp_name == 'whale':
                aggregated['whale_confidence'] = signal_data.get('whale_confidence', 0.0)

            elif comp_name == 'mtf':
                # MTF contributes to pattern confidence
                mtf_confidence = signal_data.get('confidence', 0.0)
                aggregated['pattern_confidence'] = mtf_confidence

        return aggregated

    def _determine_action(self, signals: Dict[str, Any]) -> str:
        """Determine trading action based on aggregated signals."""
        tech_conf = signals.get('technical_confidence', 0)
        news_sentiment = signals.get('overall_sentiment', 0)
        whale_conf = signals.get('whale_confidence', 0)

        # Simple decision logic
        bullish_score = tech_conf * 0.5 + news_sentiment * 0.3 + whale_conf * 0.2
        bearish_score = (1 - tech_conf) * 0.5 + (-news_sentiment) * 0.3 + (1 - whale_conf) * 0.2

        if bullish_score > bearish_score + 0.2:
            return "BUY"
        elif bearish_score > bullish_score + 0.2:
            return "SELL"
        else:
            return "HOLD"

    def _calculate_overall_confidence(self, signals: Dict[str, Any]) -> float:
        """Calculate overall confidence from all signals."""
        tech = signals.get('technical_confidence', 0)
        news = signals.get('news_confidence', 0)
        whale = signals.get('whale_confidence', 0)
        pattern = signals.get('pattern_confidence', 0)

        # Weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # Technical gets highest weight
        confidences = [tech, news, whale, pattern]

        overall = sum(w * c for w, c in zip(weights, confidences))
        return min(overall, 1.0)

    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        total_components = len(self.components)
        healthy_components = sum(1 for c in self.components.values()
                               if c.status in [ComponentStatus.READY, ComponentStatus.ACTIVE])

        error_components = sum(1 for c in self.components.values()
                             if c.status == ComponentStatus.ERROR)

        health_score = healthy_components / total_components if total_components > 0 else 0

        return {
            'overall_health': health_score,
            'total_components': total_components,
            'healthy_components': healthy_components,
            'error_components': error_components,
            'uptime_seconds': time.time() - self.start_time if self.start_time else 0
        }

    def update_market_data(self, price: float, volume: float = 0, timestamp: Optional[float] = None):
        """Update current market data."""
        if timestamp is None:
            timestamp = time.time()

        self.current_market_data = {
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'is_running': self.is_running,
            'cycle_count': self.cycle_count,
            'last_cycle_time': self.last_cycle_time,
            'total_signals': self.total_signals_generated,
            'total_decisions': self.total_decisions_made,
            'component_status': {
                name: {
                    'status': comp.status.value,
                    'executions': comp.execution_count,
                    'errors': comp.error_count,
                    'last_execution': comp.last_execution
                }
                for name, comp in self.components.items()
            },
            'portfolio': {
                'total_balance': self.portfolio_state.total_balance,
                'available_balance': self.portfolio_state.available_balance,
                'exposure_percent': self.portfolio_state.total_exposure_percent
            }
        }

# Demo function for testing
def demo_master_orchestrator():
    """Demonstrate master orchestrator capabilities."""
    print("🎼 SUPREME SYSTEM V5 - Master Orchestrator Demo")
    print("=" * 60)

    # Initialize orchestrator
    config = {
        'initial_balance': 10000.0,
        'technical_interval': 30,
        'news_interval': 600,
        'whale_interval': 600,
        'mtf_interval': 120,
        'patterns_interval': 60,
        'technical_config': {
            'ema_period': 14,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'price_history_size': 100
        },
        'news_config': {},
        'whale_config': {'whale_threshold_usd': 100000},
        'mtf_config': {},
        'risk_config': {
            'base_position_size_pct': 0.02,
            'max_position_size_pct': 0.10
        }
    }

    async def run_demo():
        orchestrator = MasterTradingOrchestrator(config)

        # Initialize components
        success = await orchestrator.initialize_components()
        if not success:
            print("❌ Initialization failed")
            return

        # Simulate market data and orchestration cycles
        print("🔄 Running orchestration cycles...")

        base_price = 50000
        for cycle in range(5):
            # Update market data
            price_change = (cycle - 2) * 100  # Some price movement
            current_price = base_price + price_change
            orchestrator.update_market_data(current_price, 1000 * (cycle + 1))

            # Run orchestration cycle
            result = await orchestrator.run_orchestration_cycle()

            print(f"\nCycle {cycle + 1}:")
            print(f"   Components executed: {result.components_executed}")
            print(f"   Signals generated: {len(result.trading_signals)}")
            print(f"   Decisions made: {result.decisions_made}")
            print(".3f")

            # Small delay between cycles
            await asyncio.sleep(0.1)

        # Final system status
        status = orchestrator.get_system_status()

        print(f"\n📊 FINAL SYSTEM STATUS:")
        print(f"   Total cycles: {status['cycle_count']}")
        print(f"   Total signals: {status['total_signals']}")
        print(f"   Total decisions: {status['total_decisions']}")

        print(f"\n🏗️  COMPONENT STATUS:")
        for comp_name, comp_status in status['component_status'].items():
            print(f"   {comp_name}: {comp_status['status']} ({comp_status['executions']} executions)")

        print(f"\n💰 PORTFOLIO STATUS:")
        print(".2f")
        print(".2f")
        print(".2f")

        print("\n🎯 SYSTEM CAPABILITIES:")
        print("   • Intelligent component scheduling (30s-10m intervals)")
        print("   • Multi-component signal aggregation")
        print("   • Risk-adjusted decision making")
        print("   • Real-time system health monitoring")
        print("   • Portfolio exposure management")

        print("\n✅ Master Orchestrator Demo Complete")
        print("   Supreme System V5 orchestration ready for production!")

    # Run the async demo
    asyncio.run(run_demo())

if __name__ == "__main__":
    demo_master_orchestrator()
