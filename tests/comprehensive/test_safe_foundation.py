#!/usr/bin/env python3
import pytest
import time
from src.algorithms.safe_enhanced_orchestrator import SafeEnhancedOrchestrator, SafeAlgorithmConfig

def dummy_fast(data):
    return {"result": "success", "processed": len(data.get("values", []))}

def dummy_slow(data):
    time.sleep(0.1)
    return {"result": "slow_success"}

def dummy_memory_intensive(data):
    large = [i for i in range(100000)]
    return {"result": "memory", "size": len(large)}

class TestSafeFoundation:
    @pytest.fixture
    def orchestrator(self):
        return SafeEnhancedOrchestrator(total_memory_mb=50)
    
    @pytest.fixture
    def sample_data(self):
        return {"values": [1, 2, 3, 4, 5], "metadata": {"test": True}}
    
    def test_basic_execution(self, orchestrator, sample_data):
        orchestrator.register_algorithm(SafeAlgorithmConfig(
            name="fast",
            function=dummy_fast,
            memory_estimate_mb=10,
            timeout_seconds=5
        ))
        
        results = orchestrator.execute_safely(sample_data, ["fast"])
        
        assert "fast" in results
        assert "result" in results["fast"]
        assert results["fast"]["result"]["result"] == "success"
    
    def test_memory_constraint(self, orchestrator, sample_data):
        orchestrator.register_algorithm(SafeAlgorithmConfig(
            name="memory_hog",
            function=dummy_memory_intensive,
            memory_estimate_mb=100,
            timeout_seconds=5
        ))
        
        results = orchestrator.execute_safely(sample_data, ["memory_hog"])
        
        assert "memory_hog" in results
        assert "error" in results["memory_hog"]
    
    def test_multiple_algorithms(self, orchestrator, sample_data):
        for i in range(3):
            orchestrator.register_algorithm(SafeAlgorithmConfig(
                name=f"algo_{i}",
                function=dummy_fast,
                memory_estimate_mb=5,
                timeout_seconds=5
            ))
        
        results = orchestrator.execute_safely(sample_data, ["algo_0", "algo_1", "algo_2"])
        
        assert len(results) == 3
        for i in range(3):
            assert f"algo_{i}" in results
    
    def test_orchestrator_status(self, orchestrator):
        status = orchestrator.get_status()
        
        assert "total_algorithms" in status
        assert "has_rust_support" in status
        assert "total_memory_mb" in status
        assert status["total_memory_mb"] == 50

def test_rust_import_optional():
    try:
        import supreme_core
        mm = supreme_core.SafeMemoryManager()
        stats = mm.get_stats()
        assert "total_budget" in stats
        assert "current_usage" in stats
        print("✅ Rust core available")
    except ImportError:
        pytest.skip("Rust core not built")
