import pytest
from src.monitoring.metrics_collector import MetricsCollector



def test_metrics_initialization():
    mc = MetricsCollector()
    mc.initialize(10000)
    m = mc.get_metrics()
    assert m.total_pnl == 0.0
    assert m.max_drawdown == 0.0



def test_pnl_tracking():
    mc = MetricsCollector()
    mc.initialize(1000)

    mc.record_trade(100) # Balance 1100, Peak 1100
    mc.record_trade(-50) # Balance 1050, Peak 1100, DD (1100-1050)/1100 = 4.5%

    m = mc.get_metrics()
    assert m.total_pnl == 50.0
    assert m.win_rate == 0.5 # 1 win, 1 loss
    assert m.total_trades == 2
    assert m.max_drawdown > 0.04 # Approx 4.5%



def test_latency_stats():
    mc = MetricsCollector()
    for i in range(100):
        mc.record_latency(10.0) # Constant 10ms

    m = mc.get_metrics()
    assert m.avg_latency_ms == 10.0
    assert m.p95_latency_ms == 10.0