#!/usr/bin/env python3
"""
Lazy Import Optimizer for Memory Efficiency - Option B Implementation
Reduces memory footprint through strategic lazy loading
"""

import sys
import os
import gc
import importlib
import functools
from typing import Dict, Any, Optional, Callable
import logging
from contextlib import contextmanager

# Ultra-minimal logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class LazyImportManager:
    """Manages lazy imports to reduce memory footprint"""
    
    def __init__(self):
        self._lazy_modules = {}
        self._loaded_modules = {}
        self._import_stats = {'total_imports': 0, 'lazy_imports': 0, 'memory_saved_mb': 0}
        
        # Set LEAN mode environment
        self._activate_lean_mode()
        
        logger.error("Lazy Import Manager initialized - LEAN mode activated")
    
    def _activate_lean_mode(self):
        """Activate LEAN mode for memory optimization"""
        lean_env_vars = {
            'LEAN_MODE': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONUNBUFFERED': '1',
            'PYTHONHASHSEED': '1',
            'OMP_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1'
        }
        
        for key, value in lean_env_vars.items():
            os.environ[key] = value
        
        # Configure pandas/numpy for minimal memory
        try:
            import pandas as pd
            pd.set_option('compute.use_numba', False)
            pd.set_option('mode.chained_assignment', None)  # Disable warnings
        except ImportError:
            pass
            
    def register_lazy_import(self, module_name: str, package: Optional[str] = None,
                           condition: Optional[Callable] = None):
        """Register a module for lazy loading"""
        self._lazy_modules[module_name] = {
            'package': package,
            'condition': condition,
            'loaded': False
        }
        
    def lazy_import(self, module_name: str, package: Optional[str] = None):
        """Import module only when accessed"""
        if module_name in self._loaded_modules:
            return self._loaded_modules[module_name]
        
        try:
            # Import only when needed
            if package:
                module = importlib.import_module(module_name, package)
            else:
                module = importlib.import_module(module_name)
            
            self._loaded_modules[module_name] = module
            self._import_stats['lazy_imports'] += 1
            
            return module
            
        except ImportError as e:
            logger.error(f"Lazy import failed for {module_name}: {e}")
            return None
    
    @contextmanager
    def temporary_import(self, module_name: str):
        """Temporarily import module and clean up after use"""
        module = self.lazy_import(module_name)
        try:
            yield module
        finally:
            # Clean up if possible
            if module_name in self._loaded_modules:
                del self._loaded_modules[module_name]
            if module_name in sys.modules:
                del sys.modules[module_name]
            gc.collect()
    
    def get_import_stats(self) -> Dict[str, Any]:
        """Get lazy import statistics"""
        return {
            'total_registered': len(self._lazy_modules),
            'currently_loaded': len(self._loaded_modules),
            'lazy_imports_used': self._import_stats['lazy_imports'],
            'memory_efficiency': 1.0 - (len(self._loaded_modules) / max(1, len(self._lazy_modules)))
        }


# Global lazy import manager
lazy_manager = LazyImportManager()

# Register common heavy imports for lazy loading
lazy_manager.register_lazy_import('numpy')
lazy_manager.register_lazy_import('pandas') 
lazy_manager.register_lazy_import('scipy')
lazy_manager.register_lazy_import('matplotlib')
lazy_manager.register_lazy_import('plotly')
lazy_manager.register_lazy_import('sklearn')


# Lazy import decorators and functions
def with_numpy(func):
    """Decorator for functions that need numpy"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        np = lazy_manager.lazy_import('numpy')
        if np is None:
            raise ImportError("numpy not available")
        return func(np, *args, **kwargs)
    return wrapper


def with_pandas(func):
    """Decorator for functions that need pandas"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pd = lazy_manager.lazy_import('pandas')
        if pd is None:
            raise ImportError("pandas not available")
        return func(pd, *args, **kwargs)
    return wrapper


def get_numpy():
    """Get numpy with lazy loading"""
    return lazy_manager.lazy_import('numpy')


def get_pandas():
    """Get pandas with lazy loading"""
    return lazy_manager.lazy_import('pandas')


@with_numpy
def calculate_technical_indicators_lazy(np, prices: list, period: int = 14) -> Dict[str, list]:
    """Calculate technical indicators with lazy numpy import"""
    prices_array = np.array(prices, dtype=np.float32)  # Use float32 for memory efficiency
    
    # EMA calculation (memory efficient)
    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(prices_array, dtype=np.float32)
    ema[0] = prices_array[0]
    
    for i in range(1, len(prices_array)):
        ema[i] = alpha * prices_array[i] + (1 - alpha) * ema[i-1]
    
    # RSI calculation (simplified)
    deltas = np.diff(prices_array)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Simple moving averages for RSI
    if len(gains) >= period:
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi_values = 100 - (100 / (1 + rs))
        
        # Pad RSI
        rsi = np.full(len(prices_array), 50.0, dtype=np.float32)
        rsi[period:] = rsi_values
    else:
        rsi = np.full(len(prices_array), 50.0, dtype=np.float32)
    
    return {
        'ema': ema.tolist(),
        'rsi': rsi.tolist()
    }


@with_pandas
def process_market_data_lazy(pd, data: list) -> Dict[str, Any]:
    """Process market data with lazy pandas import"""
    # Use minimal pandas processing
    df = pd.DataFrame(data, dtype='float32')  # Memory efficient dtype
    
    # Basic processing only
    summary = {
        'count': len(df),
        'mean_price': float(df.iloc[:, 3].mean()) if len(df) > 0 else 0.0,  # Close price
        'price_std': float(df.iloc[:, 3].std()) if len(df) > 1 else 0.0,
        'min_price': float(df.iloc[:, 3].min()) if len(df) > 0 else 0.0,
        'max_price': float(df.iloc[:, 3].max()) if len(df) > 0 else 0.0
    }
    
    # Clean up immediately
    del df
    
    return summary


class MemoryEfficientTradingEngine:
    """Trading engine with lazy imports and memory optimization"""
    
    def __init__(self):
        self.lean_mode = os.getenv('LEAN_MODE', '0') == '1'
        self._processing_stats = {'operations': 0, 'memory_cleanups': 0}
        
        logger.error(f"Memory Efficient Trading Engine initialized (LEAN mode: {self.lean_mode})")
    
    def process_market_data(self, ohlcv_data: list) -> Dict[str, Any]:
        """Process market data with memory efficiency"""
        self._processing_stats['operations'] += 1
        
        # Convert to numpy array only when needed
        with lazy_manager.temporary_import('numpy') as np:
            if np is None:
                # Fallback to pure Python
                return self._process_data_pure_python(ohlcv_data)
            
            data_array = np.array(ohlcv_data, dtype=np.float32)
            
            # Basic statistics
            close_prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, 0]
            
            stats = {
                'data_points': len(data_array),
                'price_mean': float(np.mean(close_prices)),
                'price_std': float(np.std(close_prices)),
                'price_min': float(np.min(close_prices)),
                'price_max': float(np.max(close_prices))
            }
            
            # Immediate cleanup
            del data_array, close_prices
            
        # Force garbage collection periodically
        if self._processing_stats['operations'] % 10 == 0:
            gc.collect()
            self._processing_stats['memory_cleanups'] += 1
        
        return stats
    
    def _process_data_pure_python(self, data: list) -> Dict[str, Any]:
        """Pure Python fallback processing (ultra memory efficient)"""
        if not data:
            return {'data_points': 0, 'price_mean': 0, 'price_std': 0, 'price_min': 0, 'price_max': 0}
        
        # Extract close prices (assume index 3)
        prices = [row[3] if len(row) > 3 else row[0] for row in data]
        
        count = len(prices)
        mean_price = sum(prices) / count if count > 0 else 0
        
        if count > 1:
            variance = sum((p - mean_price) ** 2 for p in prices) / (count - 1)
            price_std = variance ** 0.5
        else:
            price_std = 0
        
        return {
            'data_points': count,
            'price_mean': mean_price,
            'price_std': price_std,
            'price_min': min(prices) if prices else 0,
            'price_max': max(prices) if prices else 0
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing and memory optimization statistics"""
        import_stats = lazy_manager.get_import_stats()
        
        return {
            'processing_operations': self._processing_stats['operations'],
            'memory_cleanups': self._processing_stats['memory_cleanups'],
            'lazy_import_efficiency': import_stats['memory_efficiency'],
            'modules_loaded': import_stats['currently_loaded'],
            'lean_mode_active': self.lean_mode
        }


def optimize_imports_for_memory():
    """Optimize module imports for reduced memory footprint"""
    
    # Remove unnecessary modules from sys.modules
    modules_to_remove = []
    for module_name in sys.modules:
        # Keep essential modules
        if any(essential in module_name for essential in 
               ['supreme_system_v5', '__main__', 'builtins', 'sys', 'os']):
            continue
            
        # Remove heavy optional modules
        if any(heavy in module_name for heavy in 
               ['matplotlib', 'plotly', 'sklearn', 'tensorflow', 'torch']):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # Force garbage collection
    gc.collect()
    
    logger.error(f"Optimized imports: removed {len(modules_to_remove)} heavy modules")
    return len(modules_to_remove)


def setup_memory_efficient_environment():
    """Setup environment for maximum memory efficiency"""
    
    # Python optimization settings
    sys.dont_write_bytecode = True
    
    # Garbage collection optimization
    gc.set_threshold(200, 5, 5)  # More aggressive GC
    
    # Clear type cache
    if hasattr(sys, '_clear_type_cache'):
        sys._clear_type_cache()
    
    # Memory optimization environment variables
    memory_env = {
        'MALLOC_TRIM_THRESHOLD_': '50000',
        'MALLOC_MMAP_THRESHOLD_': '50000', 
        'PYTHONMALLOC': 'malloc',
        'PYTHONASYNCIODEBUG': '0'
    }
    
    os.environ.update(memory_env)
    
    logger.error("Memory-efficient environment configured")


# Lazy import functions for heavy modules
def get_numpy_lazy():
    """Get numpy with lazy import and memory optimization"""
    np = lazy_manager.lazy_import('numpy')
    if np is not None:
        # Configure numpy for minimal memory
        try:
            np.seterr(all='ignore')  # Reduce error checking overhead
        except:
            pass
    return np


def get_pandas_lazy():
    """Get pandas with lazy import and memory optimization"""
    pd = lazy_manager.lazy_import('pandas')
    if pd is not None:
        # Configure pandas for minimal memory
        try:
            pd.set_option('mode.chained_assignment', None)
            pd.set_option('display.max_rows', 10)  # Reduce display memory
            pd.set_option('display.max_columns', 5)
        except:
            pass
    return pd


def get_scipy_lazy():
    """Get scipy with lazy import"""
    return lazy_manager.lazy_import('scipy')


# Memory-efficient calculation functions
@contextmanager
def memory_efficient_calculation(max_mb: float = 5.0):
    """Context manager for memory-efficient calculations"""
    import psutil
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_used = end_memory - start_memory
        
        if memory_used > max_mb:
            logger.error(f"Memory usage warning: {memory_used:.1f}MB > {max_mb}MB limit")
        
        # Force cleanup
        gc.collect()


def calculate_indicators_memory_optimized(prices: list, volume: list = None) -> Dict[str, list]:
    """Calculate technical indicators with memory optimization"""
    
    with memory_efficient_calculation(2.0):  # 2MB limit for calculations
        np = get_numpy_lazy()
        if np is None:
            return {'error': 'numpy_not_available'}
        
        # Convert to float32 arrays (50% memory vs float64)
        price_array = np.array(prices, dtype=np.float32)
        
        # EMA 14
        ema_14 = _calculate_ema_optimized(np, price_array, 14)
        
        # RSI 14
        rsi_14 = _calculate_rsi_optimized(np, price_array, 14)
        
        # MACD
        macd_line, signal_line = _calculate_macd_optimized(np, price_array)
        
        # Convert back to lists and cleanup
        result = {
            'ema_14': ema_14.tolist(),
            'rsi_14': rsi_14.tolist(),
            'macd_line': macd_line.tolist(),
            'macd_signal': signal_line.tolist()
        }
        
        # Immediate cleanup
        del price_array, ema_14, rsi_14, macd_line, signal_line
        
        return result


def _calculate_ema_optimized(np, prices: np.ndarray, period: int) -> np.ndarray:
    """Memory-optimized EMA calculation"""
    alpha = np.float32(2.0 / (period + 1))
    ema = np.zeros_like(prices, dtype=np.float32)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema


def _calculate_rsi_optimized(np, prices: np.ndarray, period: int) -> np.ndarray:
    """Memory-optimized RSI calculation"""
    deltas = np.diff(prices).astype(np.float32)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Simple moving averages (memory efficient)
    if len(gains) >= period:
        avg_gains = np.convolve(gains, np.ones(period, dtype=np.float32)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period, dtype=np.float32)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi_values = 100 - (100 / (1 + rs))
        
        # Pad to match price length
        rsi = np.full(len(prices), 50.0, dtype=np.float32)
        rsi[period:] = rsi_values
    else:
        rsi = np.full(len(prices), 50.0, dtype=np.float32)
    
    return rsi


def _calculate_macd_optimized(np, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Memory-optimized MACD calculation"""
    ema_12 = _calculate_ema_optimized(np, prices, 12)
    ema_26 = _calculate_ema_optimized(np, prices, 26)
    
    macd_line = ema_12 - ema_26
    signal_line = _calculate_ema_optimized(np, macd_line, 9)
    
    # Cleanup intermediate arrays
    del ema_12, ema_26
    
    return macd_line, signal_line


class LEANModeTrader:
    """Ultra-lean trading engine for memory-constrained environments"""
    
    def __init__(self, memory_budget_mb: float = 100.0):
        self.memory_budget_mb = memory_budget_mb
        self.trading_engine = MemoryEfficientTradingEngine()
        
        # Setup memory monitoring
        try:
            import psutil
            self.process = psutil.Process()
            self.memory_monitoring = True
        except ImportError:
            self.memory_monitoring = False
            logger.error("psutil not available - memory monitoring disabled")
        
        # Initialize LEAN mode
        setup_memory_efficient_environment()
        optimize_imports_for_memory()
        
        logger.error(f"LEAN Mode Trader initialized ({memory_budget_mb}MB budget)")
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        if self.memory_monitoring:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def check_memory_budget(self) -> bool:
        """Check if within memory budget"""
        if not self.memory_monitoring:
            return True
        
        current_mb = self.get_current_memory_mb()
        return current_mb <= self.memory_budget_mb
    
    def execute_lean_trading_cycle(self, market_data: list) -> Dict[str, Any]:
        """Execute trading cycle in LEAN mode"""
        start_memory = self.get_current_memory_mb()
        
        try:
            # Process data with minimal memory footprint
            data_summary = self.trading_engine.process_market_data(market_data)
            
            # Calculate indicators with lazy imports
            if len(market_data) > 0:
                prices = [row[3] if len(row) > 3 else row[0] for row in market_data[-50:]]  # Last 50 only
                indicators = calculate_indicators_memory_optimized(prices)
                
                # Combine results
                result = {
                    'data_summary': data_summary,
                    'indicators': indicators,
                    'memory_usage_mb': self.get_current_memory_mb(),
                    'memory_increase_mb': self.get_current_memory_mb() - start_memory,
                    'budget_compliance': self.check_memory_budget(),
                    'processing_stats': self.trading_engine.get_processing_stats()
                }
            else:
                result = {
                    'error': 'no_market_data',
                    'memory_usage_mb': self.get_current_memory_mb()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"LEAN trading cycle error: {e}")
            return {
                'error': 'processing_failed',
                'message': str(e),
                'memory_usage_mb': self.get_current_memory_mb()
            }
        
        finally:
            # Always cleanup
            gc.collect()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        import_stats = lazy_manager.get_import_stats()
        processing_stats = self.trading_engine.get_processing_stats()
        
        return {
            'memory_budget_mb': self.memory_budget_mb,
            'current_memory_mb': self.get_current_memory_mb(),
            'budget_compliance': self.check_memory_budget(),
            'lean_mode_active': self.lean_mode,
            'import_optimization': import_stats,
            'processing_optimization': processing_stats,
            'memory_efficiency': {
                'lazy_imports_used': import_stats['lazy_imports_used'],
                'memory_cleanups': processing_stats['memory_cleanups'],
                'modules_loaded': import_stats['currently_loaded']
            }
        }


# Global LEAN mode trader instance
lean_trader = LEANModeTrader(memory_budget_mb=100.0)


def execute_option_b_optimization(market_data: list) -> Dict[str, Any]:
    """Execute Option B trading-first optimization"""
    logger.error("🎯 Executing Option B - Trading-First Optimization")
    
    # Execute LEAN trading cycle
    trading_result = lean_trader.execute_lean_trading_cycle(market_data)
    
    # Get optimization report
    optimization_report = lean_trader.get_optimization_report()
    
    # Combine results
    result = {
        'option_b_status': 'executed',
        'trading_result': trading_result,
        'optimization_report': optimization_report,
        'memory_efficiency': {
            'budget_compliance': optimization_report['budget_compliance'],
            'current_memory_mb': optimization_report['current_memory_mb'],
            'lean_mode': optimization_report['lean_mode_active']
        },
        'next_steps': [
            'Execute enhanced backtest with optimized signals',
            'Validate 68.9% win rate recovery',
            'Confirm 2.47 Sharpe ratio achievement', 
            'Proceed to Phase 3 Paper Trading'
        ]
    }
    
    logger.error("✅ Option B optimization executed successfully")
    return result


if __name__ == "__main__":
    print("🚀 Lazy Import Optimizer - Option B Implementation")
    print("Memory optimization: Lazy loading + LEAN mode")
    print("Target: Reduce memory footprint while maintaining functionality")
    
    # Test with minimal data
    test_data = [[i, i+1, i-1, i+0.5, 1000] for i in range(100, 110)]
    result = execute_option_b_optimization(test_data)
    
    print(f"Memory usage: {result['optimization_report']['current_memory_mb']:.1f}MB")
    print(f"Budget compliance: {result['memory_efficiency']['budget_compliance']}")
    print(f"LEAN mode: {result['memory_efficiency']['lean_mode']}")
    print("✅ Lazy import optimization ready")