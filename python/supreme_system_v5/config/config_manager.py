"""
Ultra-Optimized Configuration Manager for Supreme System V5.
Intelligent config management with auto-tuning, validation, and performance profiling.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

class ConfigValidator:
    """Advanced configuration validation with intelligent error correction."""

    def __init__(self):
        self.validation_rules = {
            'cpu_limits': {'min': 10, 'max': 100, 'default': 88},
            'memory_limits': {'min': 1.0, 'max': 8.0, 'default': 3.86},
            'indicator_periods': {'min': 2, 'max': 200, 'default': 14},
            'event_thresholds': {'min': 0.0001, 'max': 0.01, 'default': 0.001},
            'cache_ttl': {'min': 0.1, 'max': 10.0, 'default': 1.0},
            'processing_intervals': {'min': 1, 'max': 300, 'default': 30}
        }

    def validate_and_correct(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate configuration and auto-correct invalid values.

        Args:
            config: Raw configuration dictionary

        Returns:
            Tuple of (corrected_config, warning_messages)
        """
        corrected_config = config.copy()
        warnings = []

        # CPU limits validation
        cpu_limit = config.get('MAX_CPU_PERCENT', self.validation_rules['cpu_limits']['default'])
        if not isinstance(cpu_limit, (int, float)) or cpu_limit < self.validation_rules['cpu_limits']['min']:
            corrected_config['MAX_CPU_PERCENT'] = self.validation_rules['cpu_limits']['default']
            warnings.append(f"CPU limit {cpu_limit} invalid, corrected to {self.validation_rules['cpu_limits']['default']}%")
        elif cpu_limit > self.validation_rules['cpu_limits']['max']:
            corrected_config['MAX_CPU_PERCENT'] = self.validation_rules['cpu_limits']['max']
            warnings.append(f"CPU limit {cpu_limit}% too high, capped to {self.validation_rules['cpu_limits']['max']}%")

        # Memory limits validation
        mem_limit = config.get('MAX_RAM_GB', self.validation_rules['memory_limits']['default'])
        if not isinstance(mem_limit, (int, float)) or mem_limit < self.validation_rules['memory_limits']['min']:
            corrected_config['MAX_RAM_GB'] = self.validation_rules['memory_limits']['default']
            warnings.append(f"Memory limit {mem_limit}GB invalid, corrected to {self.validation_rules['memory_limits']['default']}GB")
        elif mem_limit > self.validation_rules['memory_limits']['max']:
            corrected_config['MAX_RAM_GB'] = self.validation_rules['memory_limits']['max']
            warnings.append(f"Memory limit {mem_limit}GB too high, capped to {self.validation_rules['memory_limits']['max']}GB")

        # Indicator periods validation
        for indicator in ['ema_period', 'rsi_period', 'macd_fast', 'macd_slow', 'macd_signal']:
            period = config.get(indicator.upper(), self.validation_rules['indicator_periods']['default'])
            if not isinstance(period, int) or period < self.validation_rules['indicator_periods']['min']:
                corrected_config[indicator.upper()] = self.validation_rules['indicator_periods']['default']
                warnings.append(f"{indicator} {period} invalid, corrected to {self.validation_rules['indicator_periods']['default']}")

        # Event thresholds validation
        price_change_pct = config.get('MIN_PRICE_CHANGE_PCT', self.validation_rules['event_thresholds']['default'])
        if not isinstance(price_change_pct, float) or price_change_pct < self.validation_rules['event_thresholds']['min']:
            corrected_config['MIN_PRICE_CHANGE_PCT'] = self.validation_rules['event_thresholds']['default']
            warnings.append(f"Price change threshold {price_change_pct} too low, corrected to {self.validation_rules['event_thresholds']['default']}")

        return corrected_config, warnings

class PerformanceProfiler:
    """Configuration performance profiler with auto-tuning recommendations."""

    def __init__(self):
        self.performance_history = []
        self.recommendations_cache = {}
        self.last_analysis_time = 0

    def analyze_performance_impact(self, config: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze configuration performance impact and generate tuning recommendations.

        Args:
            config: Current configuration
            metrics: Performance metrics

        Returns:
            Performance analysis with recommendations
        """
        analysis = {
            'cpu_efficiency': self._analyze_cpu_efficiency(config, metrics),
            'memory_efficiency': self._analyze_memory_efficiency(config, metrics),
            'processing_efficiency': self._analyze_processing_efficiency(config, metrics),
            'recommendations': self._generate_tuning_recommendations(config, metrics),
            'optimization_score': self._calculate_optimization_score(config, metrics)
        }

        # Cache analysis for future reference
        self.performance_history.append({
            'timestamp': time.time(),
            'config': config.copy(),
            'metrics': metrics.copy(),
            'analysis': analysis.copy()
        })

        # Keep history bounded
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]

        return analysis

    def _analyze_cpu_efficiency(self, config: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CPU efficiency based on configuration and metrics."""
        cpu_limit = config.get('MAX_CPU_PERCENT', 88)
        avg_cpu = metrics.get('avg_cpu_percent', 0)
        skip_ratio = metrics.get('skip_ratio', 0)

        efficiency_score = 1.0 - (avg_cpu / cpu_limit)  # Closer to 1.0 is better

        return {
            'efficiency_score': efficiency_score,
            'avg_cpu_usage': avg_cpu,
            'cpu_limit': cpu_limit,
            'event_filtering_impact': skip_ratio * 0.7,  # Estimated CPU savings from filtering
            'status': 'optimal' if efficiency_score > 0.8 else 'warning' if efficiency_score > 0.6 else 'critical'
        }

    def _analyze_memory_efficiency(self, config: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory efficiency."""
        mem_limit = config.get('MAX_RAM_GB', 3.86)
        memory_usage = metrics.get('avg_memory_gb', 0)
        buffer_size = config.get('price_history_size', 100)

        efficiency_score = 1.0 - (memory_usage / mem_limit)
        memory_per_buffer_entry = memory_usage / max(buffer_size, 1)  # Rough estimate

        return {
            'efficiency_score': efficiency_score,
            'memory_usage_gb': memory_usage,
            'memory_limit_gb': mem_limit,
            'memory_per_buffer_entry_kb': memory_per_buffer_entry * 1024,
            'fixed_memory_allocation': buffer_size <= 200,  # Optimal for i3
            'status': 'optimal' if efficiency_score > 0.7 else 'warning' if efficiency_score > 0.5 else 'critical'
        }

    def _analyze_processing_efficiency(self, config: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze processing efficiency."""
        latency = metrics.get('avg_indicator_latency_ms', 0)
        throughput = metrics.get('events_per_second', 0)
        cache_hit_ratio = metrics.get('cache_hit_ratio', 0)

        # Composite efficiency score
        latency_score = max(0, 1.0 - (latency / 200))  # 200ms target
        throughput_score = min(1.0, throughput / 100)   # 100 events/sec target
        cache_score = cache_hit_ratio

        overall_efficiency = (latency_score + throughput_score + cache_score) / 3

        return {
            'overall_efficiency': overall_efficiency,
            'latency_score': latency_score,
            'throughput_score': throughput_score,
            'cache_efficiency': cache_score,
            'bottleneck': 'latency' if latency > 100 else 'throughput' if throughput < 50 else 'memory' if cache_hit_ratio < 0.5 else 'none'
        }

    def _generate_tuning_recommendations(self, config: Dict[str, Any], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent tuning recommendations."""
        recommendations = []

        # CPU tuning recommendations
        cpu_usage = metrics.get('avg_cpu_percent', 0)
        if cpu_usage > 85:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'high',
                'action': 'increase_event_filtering',
                'parameter': 'MIN_PRICE_CHANGE_PCT',
                'current_value': config.get('MIN_PRICE_CHANGE_PCT', 0.001),
                'recommended_value': 0.002,
                'expected_impact': '15-25% CPU reduction',
                'reason': f'CPU usage at {cpu_usage:.1f}%, needs aggressive filtering'
            })

        # Memory tuning recommendations
        memory_usage = metrics.get('avg_memory_gb', 0)
        if memory_usage > 3.5:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'action': 'reduce_buffer_size',
                'parameter': 'price_history_size',
                'current_value': config.get('price_history_size', 100),
                'recommended_value': 50,
                'expected_impact': '30-50% memory reduction',
                'reason': f'Memory usage at {memory_usage:.2f}GB, approaching limit'
            })

        # Latency tuning recommendations
        latency = metrics.get('avg_indicator_latency_ms', 0)
        if latency > 50:
            recommendations.append({
                'type': 'latency_optimization',
                'priority': 'medium',
                'action': 'enable_aggressive_caching',
                'parameter': 'cache_enabled',
                'current_value': config.get('cache_enabled', True),
                'recommended_value': True,
                'expected_impact': '40-60% latency reduction',
                'reason': f'Indicator latency at {latency:.1f}ms, needs caching optimization'
            })

        # Cache tuning recommendations
        cache_hit_ratio = metrics.get('cache_hit_ratio', 0)
        if cache_hit_ratio < 0.7:
            recommendations.append({
                'type': 'cache_optimization',
                'priority': 'medium',
                'action': 'increase_cache_ttl',
                'parameter': 'cache_ttl_seconds',
                'current_value': config.get('cache_ttl_seconds', 1.0),
                'recommended_value': 2.0,
                'expected_impact': '20-30% cache hit ratio improvement',
                'reason': f'Cache hit ratio at {cache_hit_ratio:.2f}, can be improved'
            })

        return recommendations

    def _calculate_optimization_score(self, config: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """Calculate overall optimization score (0-100)."""
        cpu_score = self._analyze_cpu_efficiency(config, metrics)['efficiency_score']
        memory_score = self._analyze_memory_efficiency(config, metrics)['efficiency_score']
        processing_score = self._analyze_processing_efficiency(config, metrics)['overall_efficiency']

        # Weighted average
        weights = [0.4, 0.4, 0.2]  # CPU and Memory most important
        overall_score = sum(w * s for w, s in zip(weights, [cpu_score, memory_score, processing_score]))

        return min(100, max(0, overall_score * 100))

class ConfigManager:
    """
    Ultra-Optimized Configuration Manager with auto-tuning capabilities.

    Features:
    - Intelligent config validation and auto-correction
    - Performance profiling and tuning recommendations
    - Dynamic config optimization based on system metrics
    - Memory-efficient config caching
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to configuration file (.env format)
        """
        self.config_file = config_file or '.env'
        self.validator = ConfigValidator()
        self.profiler = PerformanceProfiler()
        self._config_cache = {}
        self._last_load_time = 0
        self._cache_ttl = 30  # 30 seconds cache

    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load and validate configuration with intelligent caching.

        Args:
            force_reload: Force reload from disk

        Returns:
            Validated and optimized configuration
        """
        current_time = time.time()

        # Check cache validity
        if not force_reload and current_time - self._last_load_time < self._cache_ttl:
            if self._config_cache:
                return self._config_cache.copy()

        # Load from environment file
        load_dotenv(self.config_file)

        # Build configuration dictionary
        config = {
            # Core optimization flags
            'OPTIMIZED_MODE': self._parse_bool_env('OPTIMIZED_MODE', True),
            'EVENT_DRIVEN_PROCESSING': self._parse_bool_env('EVENT_DRIVEN_PROCESSING', True),
            'INTELLIGENT_CACHING': self._parse_bool_env('INTELLIGENT_CACHING', True),
            'PERFORMANCE_PROFILE': os.getenv('PERFORMANCE_PROFILE', 'normal'),

            # Single symbol focus
            'SINGLE_SYMBOL': os.getenv('SINGLE_SYMBOL', 'BTC-USDT'),

            # Scheduling intervals
            'PROCESS_INTERVAL_SECONDS': self._parse_int_env('PROCESS_INTERVAL_SECONDS', 30),
            'TECHNICAL_INTERVAL': self._parse_int_env('TECHNICAL_INTERVAL', 30),
            'NEWS_INTERVAL_MIN': self._parse_int_env('NEWS_INTERVAL_MIN', 10),
            'WHALE_INTERVAL_MIN': self._parse_int_env('WHALE_INTERVAL_MIN', 10),
            'MTF_INTERVAL': self._parse_int_env('MTF_INTERVAL', 120),

            # Resource limits (i3-4GB optimization)
            'MAX_CPU_PERCENT': self._parse_float_env('MAX_CPU_PERCENT', 88.0),
            'MAX_RAM_GB': self._parse_float_env('MAX_RAM_GB', 3.86),
            'TARGET_EVENT_SKIP_RATIO': self._parse_float_env('TARGET_EVENT_SKIP_RATIO', 0.7),

            # Component enables
            'TECHNICAL_ANALYSIS_ENABLED': self._parse_bool_env('TECHNICAL_ANALYSIS_ENABLED', True),
            'NEWS_ANALYSIS_ENABLED': self._parse_bool_env('NEWS_ANALYSIS_ENABLED', True),
            'WHALE_TRACKING_ENABLED': self._parse_bool_env('WHALE_TRACKING_ENABLED', True),
            'MULTI_TIMEFRAME_ENABLED': self._parse_bool_env('MULTI_TIMEFRAME_ENABLED', True),
            'RISK_MANAGEMENT_ENABLED': self._parse_bool_env('RISK_MANAGEMENT_ENABLED', True),
            'RESOURCE_MONITORING_ENABLED': self._parse_bool_env('RESOURCE_MONITORING_ENABLED', True),

            # Advanced optimization parameters
            'ema_period': self._parse_int_env('EMA_PERIOD', 14),
            'rsi_period': self._parse_int_env('RSI_PERIOD', 14),
            'macd_fast': self._parse_int_env('MACD_FAST', 12),
            'macd_slow': self._parse_int_env('MACD_SLOW', 26),
            'macd_signal': self._parse_int_env('MACD_SIGNAL', 9),
            'price_history_size': self._parse_int_env('PRICE_HISTORY_SIZE', 100),
            'cache_enabled': self._parse_bool_env('CACHE_ENABLED', True),
            'cache_ttl_seconds': self._parse_float_env('CACHE_TTL_SECONDS', 1.0),
            'min_price_change_pct': self._parse_float_env('MIN_PRICE_CHANGE_PCT', 0.001),
            'min_volume_multiplier': self._parse_float_env('MIN_VOLUME_MULTIPLIER', 3.0),
            'max_time_gap_seconds': self._parse_int_env('MAX_TIME_GAP_SECONDS', 60),

            # Risk management parameters
            'base_position_size_pct': self._parse_float_env('BASE_POSITION_SIZE_PCT', 0.02),
            'max_position_size_pct': self._parse_float_env('MAX_POSITION_SIZE_PCT', 0.10),
            'base_leverage': self._parse_float_env('BASE_LEVERAGE', 5.0),
            'max_leverage': self._parse_float_env('MAX_LEVERAGE', 50.0),
            'max_portfolio_exposure': self._parse_float_env('MAX_PORTFOLIO_EXPOSURE', 0.50),
            'high_confidence_threshold': self._parse_float_env('HIGH_CONFIDENCE_THRESHOLD', 0.75),
            'medium_confidence_threshold': self._parse_float_env('MEDIUM_CONFIDENCE_THRESHOLD', 0.60),
            'low_confidence_threshold': self._parse_float_env('LOW_CONFIDENCE_THRESHOLD', 0.45),
        }

        # Validate and auto-correct configuration
        validated_config, warnings = self.validator.validate_and_correct(config)

        # Log warnings
        for warning in warnings:
            print(f"⚠️  Config Warning: {warning}")

        # Cache validated configuration
        self._config_cache = validated_config
        self._last_load_time = current_time

        return validated_config.copy()

    def optimize_config(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-optimize configuration based on performance metrics.

        Args:
            current_metrics: Current system performance metrics

        Returns:
            Optimized configuration recommendations
        """
        current_config = self.load_config()
        analysis = self.profiler.analyze_performance_impact(current_config, current_metrics)

        # Generate optimized config
        optimized_config = current_config.copy()

        # Apply recommendations
        for recommendation in analysis['recommendations']:
            if recommendation['priority'] == 'high':
                param = recommendation['parameter']
                new_value = recommendation['recommended_value']
                optimized_config[param] = new_value
                print(f"🔧 Auto-tuning: {param} = {new_value} ({recommendation['expected_impact']})")

        return optimized_config

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'config_validation_warnings': [],  # Would be populated during load
            'performance_history_length': len(self.profiler.performance_history),
            'optimization_score_trend': self._calculate_score_trend(),
            'recommended_actions': self._get_pending_recommendations()
        }

    def _calculate_score_trend(self) -> List[float]:
        """Calculate optimization score trend."""
        return [entry['analysis']['optimization_score'] for entry in self.profiler.performance_history[-10:]]

    def _get_pending_recommendations(self) -> List[Dict[str, Any]]:
        """Get pending optimization recommendations."""
        if not self.profiler.performance_history:
            return []

        latest_analysis = self.profiler.performance_history[-1]['analysis']
        return latest_analysis['recommendations']

    def _parse_bool_env(self, key: str, default: bool = False) -> bool:
        """Parse boolean environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')

    def _parse_int_env(self, key: str, default: int = 0) -> int:
        """Parse integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    def _parse_float_env(self, key: str, default: float = 0.0) -> float:
        """Parse float environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    def save_optimized_config(self, config: Dict[str, Any], filename: str = None):
        """
        Save optimized configuration to file.

        Args:
            config: Configuration to save
            filename: Output filename
        """
        if filename is None:
            filename = f".env.optimized.{int(time.time())}"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Supreme System V5 - Auto-Optimized Configuration\n")
            f.write(f"# Generated: {time.ctime()}\n\n")

            for key, value in config.items():
                if isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, float):
                    value = ".3f"
                f.write(f"{key}={value}\n")

        print(f"💾 Optimized config saved to {filename}")

    def create_performance_snapshot(self) -> Dict[str, Any]:
        """Create performance snapshot for analysis."""
        return {
            'timestamp': time.time(),
            'config': self.load_config(),
            'performance_history': self.profiler.performance_history[-5:],  # Last 5 entries
            'optimization_score': self.profiler.performance_history[-1]['analysis']['optimization_score'] if self.profiler.performance_history else 0
        }
