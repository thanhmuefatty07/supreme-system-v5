#!/usr/bin/env python3
"""
Parity Test Suite for Supreme System V5 Optimized Indicators
Validates mathematical equivalence between optimized and reference implementations
"""

import pytest
import sys
import os
import time
import random
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from supreme_system_v5.strategies import ScalpingStrategy

class TestParityIndicators:
    """Test suite for validating parity between optimized and reference indicators"""
    
    TOLERANCE = 1e-6  # Maximum allowed tolerance for parity validation
    
    @pytest.fixture
    def strategy_config(self):
        """Standard strategy configuration for testing"""
        return {
            'symbol': 'BTC-USDT',
            'position_size_pct': 0.02,
            'stop_loss_pct': 0.01,
            'take_profit_pct': 0.02,
            'ema_period': 14,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'price_history_size': 100,
            'event_config': {
                'min_price_change_pct': 0.0005,
                'min_volume_multiplier': 2.0,
                'max_time_gap_seconds': 30
            }
        }
    
    @pytest.fixture
    def historical_data(self):
        """Generate realistic historical price data for testing"""
        data = []
        base_price = 50000.0
        current_time = time.time()
        
        for i in range(100):
            # Generate trending price with realistic noise
            trend = 200 * (i / 100)  # Upward trend
            noise = random.uniform(-50, 50)  # Price noise
            price = base_price + trend + noise
            
            data.append({
                'price': round(price, 2),
                'volume': random.uniform(100, 1000),
                'timestamp': current_time + i
            })
        
        return data
    
    def test_ema_parity(self, strategy_config, historical_data):
        """Test EMA parity between optimized and reference implementations"""
        strategy = ScalpingStrategy(strategy_config)
        
        # Run parity validation
        results = strategy.validate_parity_with_reference(historical_data, self.TOLERANCE)
        
        # Assert parity passed
        assert results['ema_parity'], f"EMA parity failed. Violations: {results['parity_violations']}"
        
        # Check no violations for EMA specifically
        ema_violations = [v for v in results['parity_violations'] if 'EMA' in v['indicator']]
        assert len(ema_violations) == 0, f"EMA violations found: {ema_violations}"
        
        print(f"✅ EMA parity test PASSED - {results['total_points']} data points validated")
    
    def test_rsi_parity(self, strategy_config, historical_data):
        """Test RSI parity between optimized and reference implementations"""
        strategy = ScalpingStrategy(strategy_config)
        
        # Run parity validation
        results = strategy.validate_parity_with_reference(historical_data, self.TOLERANCE)
        
        # Assert parity passed
        assert results['rsi_parity'], f"RSI parity failed. Violations: {results['parity_violations']}"
        
        # Check no violations for RSI specifically
        rsi_violations = [v for v in results['parity_violations'] if 'RSI' in v['indicator']]
        assert len(rsi_violations) == 0, f"RSI violations found: {rsi_violations}"
        
        print(f"✅ RSI parity test PASSED - {results['total_points']} data points validated")
    
    def test_macd_parity(self, strategy_config, historical_data):
        """Test MACD parity between optimized and reference implementations"""
        strategy = ScalpingStrategy(strategy_config)
        
        # Run parity validation
        results = strategy.validate_parity_with_reference(historical_data, self.TOLERANCE)
        
        # Assert parity passed
        assert results['macd_parity'], f"MACD parity failed. Violations: {results['parity_violations']}"
        
        # Check no violations for MACD specifically
        macd_violations = [v for v in results['parity_violations'] if 'MACD' in v['indicator']]
        assert len(macd_violations) == 0, f"MACD violations found: {macd_violations}"
        
        print(f"✅ MACD parity test PASSED - {results['total_points']} data points validated")
    
    def test_all_indicators_parity(self, strategy_config, historical_data):
        """Test overall parity for all indicators combined"""
        strategy = ScalpingStrategy(strategy_config)
        
        # Run comprehensive parity validation
        results = strategy.validate_parity_with_reference(historical_data, self.TOLERANCE)
        
        # Assert all indicators passed parity
        assert results['parity_passed'], f"Overall parity failed. Violations: {results['parity_violations']}"
        assert results['ema_parity'], "EMA parity failed"
        assert results['rsi_parity'], "RSI parity failed" 
        assert results['macd_parity'], "MACD parity failed"
        
        print(f"✅ ALL INDICATORS parity test PASSED - {results['total_points']} data points validated")
        print(f"   - EMA: ✅ PASS")
        print(f"   - RSI: ✅ PASS")
        print(f"   - MACD: ✅ PASS")
        print(f"   - Total violations: {len(results['parity_violations'])}")
    
    def test_performance_vs_reference(self, strategy_config, historical_data):
        """Test that optimized implementation is faster than reference"""
        strategy = ScalpingStrategy(strategy_config)
        
        # Time optimized implementation
        start_time = time.perf_counter()
        for data_point in historical_data:
            strategy.add_price_data(
                data_point['price'],
                data_point['volume'], 
                data_point['timestamp']
            )
        optimized_time = time.perf_counter() - start_time
        
        # Time reference implementation (rough estimate)
        from supreme_system_v5.strategies import ReferenceTechnicalIndicators
        ref_indicators = ReferenceTechnicalIndicators(strategy_config)
        
        start_time = time.perf_counter()
        for data_point in historical_data:
            ref_indicators.add_price_data(
                data_point['price'],
                data_point['volume'],
                data_point['timestamp']
            )
        reference_time = time.perf_counter() - start_time
        
        # Calculate speedup
        speedup = reference_time / optimized_time if optimized_time > 0 else float('inf')
        
        print(f"📊 Performance Comparison:")
        print(f"   - Optimized time: {optimized_time:.6f}s")
        print(f"   - Reference time: {reference_time:.6f}s")
        print(f"   - Speedup: {speedup:.2f}x")
        
        # Assert optimized is faster (or at least not significantly slower)
        assert speedup >= 0.8, f"Optimized implementation slower than expected. Speedup: {speedup:.2f}x"
        
        print(f"✅ Performance test PASSED - {speedup:.2f}x speedup achieved")
    
    def test_tolerance_edge_cases(self, strategy_config):
        """Test edge cases around tolerance boundaries"""
        # Create minimal dataset that might trigger edge cases
        edge_case_data = [
            {'price': 50000.0, 'volume': 100, 'timestamp': time.time()},
            {'price': 50000.0, 'volume': 100, 'timestamp': time.time() + 1},  # No change
            {'price': 50100.0, 'volume': 100, 'timestamp': time.time() + 2},  # Small increase
            {'price': 49900.0, 'volume': 100, 'timestamp': time.time() + 3},  # Small decrease
            {'price': 50000.0, 'volume': 100, 'timestamp': time.time() + 4},  # Back to original
        ]
        
        # Extend with more data points
        for i in range(5, 50):
            edge_case_data.append({
                'price': 50000.0 + (i % 10 - 5) * 10,  # Oscillating
                'volume': 100 + i,
                'timestamp': time.time() + i
            })
        
        strategy = ScalpingStrategy(strategy_config)
        results = strategy.validate_parity_with_reference(edge_case_data, self.TOLERANCE)
        
        assert results['parity_passed'], f"Edge case parity failed: {results['parity_violations']}"
        
        print(f"✅ Edge cases parity test PASSED - {len(edge_case_data)} edge cases validated")


if __name__ == "__main__":
    # Run tests directly if script is executed
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
