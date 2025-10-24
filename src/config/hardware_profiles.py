"""
⚡ Supreme System V5 - Hardware Profiles for Optimal Performance
Hardware-specific optimizations for different processor generations

Optimizations for:
- i3 8th generation (4C/4T, 65W TDP)
- i5 8th generation (4C/4T to 6C/6T)  
- Ultra-low resource configurations
"""

import psutil
import platform
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

class ProcessorType(Enum):
    I3_8TH_GEN = "i3_8th_gen"
    I5_8TH_GEN = "i5_8th_gen" 
    I7_8TH_GEN = "i7_8th_gen"
    UNKNOWN = "unknown"

class MemoryProfile(Enum):
    LOW_4GB = "4gb"
    MEDIUM_8GB = "8gb"
    HIGH_16GB = "16gb"

@dataclass
class HardwareProfile:
    """Hardware-specific configuration profile"""
    processor_type: ProcessorType
    memory_profile: MemoryProfile
    core_count: int
    base_frequency_ghz: float
    cache_mb: int
    tdp_watts: int
    
    # Performance targets
    max_cpu_usage_pct: float
    max_memory_usage_gb: float
    target_latency_ms: float
    
    # Optimizations
    enable_vectorization: bool
    use_single_thread: bool
    aggressive_gc: bool
    compress_data: bool
    reduced_history: bool
    
    # Trading specific
    max_symbols: int
    update_frequency_ms: int
    websocket_buffer_size: int
    metrics_retention_hours: int

# Hardware profiles configuration
HARDWARE_PROFILES = {
    ProcessorType.I3_8TH_GEN: {
        MemoryProfile.LOW_4GB: HardwareProfile(
            processor_type=ProcessorType.I3_8TH_GEN,
            memory_profile=MemoryProfile.LOW_4GB,
            core_count=4,
            base_frequency_ghz=3.6,
            cache_mb=6,
            tdp_watts=65,
            
            # Aggressive optimization for i3 + 4GB
            max_cpu_usage_pct=80.0,  # Higher CPU usage acceptable
            max_memory_usage_gb=3.0,  # Conservative memory usage
            target_latency_ms=100.0,  # Relaxed latency target
            
            # Optimizations enabled
            enable_vectorization=True,
            use_single_thread=True,   # Avoid thread contention
            aggressive_gc=True,       # Frequent garbage collection
            compress_data=True,       # Compress historical data
            reduced_history=True,     # Only 15 days history
            
            # Trading limitations
            max_symbols=5,            # Max 5 trading pairs
            update_frequency_ms=2000, # 2-second updates
            websocket_buffer_size=500,# Small WebSocket buffer  
            metrics_retention_hours=12 # 12 hours metrics only
        ),
        
        MemoryProfile.MEDIUM_8GB: HardwareProfile(
            processor_type=ProcessorType.I3_8TH_GEN,
            memory_profile=MemoryProfile.MEDIUM_8GB,
            core_count=4,
            base_frequency_ghz=3.6,
            cache_mb=6,
            tdp_watts=65,
            
            # Balanced optimization for i3 + 8GB
            max_cpu_usage_pct=75.0,
            max_memory_usage_gb=6.0,  # Can use more RAM
            target_latency_ms=75.0,   # Better latency target
            
            # Moderate optimizations
            enable_vectorization=True,
            use_single_thread=True,
            aggressive_gc=True,
            compress_data=True,
            reduced_history=False,    # Can handle 30 days history
            
            # More trading capacity
            max_symbols=10,           # 10 trading pairs
            update_frequency_ms=1000, # 1-second updates
            websocket_buffer_size=1000,
            metrics_retention_hours=24 # 24 hours metrics
        )
    },
    
    ProcessorType.I5_8TH_GEN: {
        MemoryProfile.LOW_4GB: HardwareProfile(
            processor_type=ProcessorType.I5_8TH_GEN,
            memory_profile=MemoryProfile.LOW_4GB,
            core_count=6,
            base_frequency_ghz=2.8,
            cache_mb=9,
            tdp_watts=65,
            
            # i5 can handle more with 4GB
            max_cpu_usage_pct=70.0,
            max_memory_usage_gb=3.2,
            target_latency_ms=50.0,   # Better latency
            
            # Reduced optimizations needed
            enable_vectorization=True,
            use_single_thread=False,  # Can use multi-threading
            aggressive_gc=True,
            compress_data=True,
            reduced_history=True,
            
            # Better trading capacity
            max_symbols=10,
            update_frequency_ms=500,  # 500ms updates
            websocket_buffer_size=1000,
            metrics_retention_hours=24
        ),
        
        MemoryProfile.MEDIUM_8GB: HardwareProfile(
            processor_type=ProcessorType.I5_8TH_GEN,
            memory_profile=MemoryProfile.MEDIUM_8GB,
            core_count=6,
            base_frequency_ghz=2.8,
            cache_mb=9,
            tdp_watts=65,
            
            # Standard configuration
            max_cpu_usage_pct=70.0,
            max_memory_usage_gb=6.5,
            target_latency_ms=25.0,   # Production target
            
            # Minimal optimizations
            enable_vectorization=True,
            use_single_thread=False,
            aggressive_gc=False,      # Normal GC
            compress_data=False,      # No compression needed
            reduced_history=False,
            
            # Full capacity
            max_symbols=20,
            update_frequency_ms=100,  # 100ms updates
            websocket_buffer_size=2000,
            metrics_retention_hours=72 # 3 days metrics
        )
    }
}

class HardwareDetector:
    """Automatic hardware detection and profile selection"""
    
    def __init__(self):
        self.system_info = self._detect_system()
        
    def _detect_system(self) -> Dict[str, Any]:
        """Detect current system specifications"""
        cpu_info = platform.processor()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        
        return {
            "processor": cpu_info,
            "memory_gb": round(memory_gb),
            "cpu_count": cpu_count,
            "cpu_freq_ghz": cpu_freq.max / 1000 if cpu_freq else 0,
            "platform": platform.system()
        }
    
    def detect_processor_type(self) -> ProcessorType:
        """Detect processor type from system info"""
        cpu_info = self.system_info["processor"].lower()
        
        if "i3" in cpu_info and any(gen in cpu_info for gen in ["8100", "8300", "8350"]):
            return ProcessorType.I3_8TH_GEN
        elif "i5" in cpu_info and any(gen in cpu_info for gen in ["8400", "8500", "8600"]):
            return ProcessorType.I5_8TH_GEN
        elif "i7" in cpu_info and any(gen in cpu_info for gen in ["8700", "8750"]):
            return ProcessorType.I7_8TH_GEN
        else:
            return ProcessorType.UNKNOWN
    
    def detect_memory_profile(self) -> MemoryProfile:
        """Detect memory profile from system specs"""
        memory_gb = self.system_info["memory_gb"]
        
        if memory_gb <= 5:
            return MemoryProfile.LOW_4GB
        elif memory_gb <= 10:
            return MemoryProfile.MEDIUM_8GB
        else:
            return MemoryProfile.HIGH_16GB
    
    def get_optimal_profile(self) -> HardwareProfile:
        """Get optimal hardware profile for current system"""
        processor_type = self.detect_processor_type()
        memory_profile = self.detect_memory_profile()
        
        # Fallback to i3 8th gen profile if unknown
        if processor_type == ProcessorType.UNKNOWN:
            processor_type = ProcessorType.I3_8TH_GEN
            
        # Get profile or fallback to most conservative
        if processor_type in HARDWARE_PROFILES:
            profiles = HARDWARE_PROFILES[processor_type]
            if memory_profile in profiles:
                return profiles[memory_profile]
            else:
                # Use the most conservative profile
                return profiles[MemoryProfile.LOW_4GB]
        else:
            # Ultimate fallback - most conservative i3 profile
            return HARDWARE_PROFILES[ProcessorType.I3_8TH_GEN][MemoryProfile.LOW_4GB]

class PerformanceOptimizer:
    """Hardware-specific performance optimizations"""
    
    def __init__(self, profile: HardwareProfile):
        self.profile = profile
        
    def get_trading_engine_config(self) -> Dict[str, Any]:
        """Get trading engine configuration for hardware"""
        return {
            "max_symbols": self.profile.max_symbols,
            "update_frequency_ms": self.profile.update_frequency_ms,
            "memory_limit_gb": self.profile.max_memory_usage_gb,
            "cpu_limit_pct": self.profile.max_cpu_usage_pct,
            
            # Algorithm optimizations
            "use_vectorization": self.profile.enable_vectorization,
            "single_thread_mode": self.profile.use_single_thread,
            "reduced_precision": self.profile.processor_type == ProcessorType.I3_8TH_GEN,
            
            # Data management
            "compress_historical_data": self.profile.compress_data,
            "max_history_days": 15 if self.profile.reduced_history else 30,
            "aggressive_memory_cleanup": self.profile.aggressive_gc
        }
    
    def get_websocket_config(self) -> Dict[str, Any]:
        """Get WebSocket configuration for hardware"""
        return {
            "buffer_size": self.profile.websocket_buffer_size,
            "max_connections": min(100, self.profile.max_symbols * 10),
            "message_frequency_ms": max(self.profile.update_frequency_ms, 1000),
            "compression_enabled": self.profile.compress_data,
            "backpressure_aggressive": True
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration for hardware"""
        return {
            "metrics_retention_hours": self.profile.metrics_retention_hours,
            "scrape_interval_ms": max(5000, self.profile.update_frequency_ms),
            "reduce_metric_precision": self.profile.processor_type == ProcessorType.I3_8TH_GEN,
            "disable_heavy_metrics": self.profile.memory_profile == MemoryProfile.LOW_4GB
        }
    
    def get_docker_config(self) -> Dict[str, Any]:
        """Get Docker resource limits for hardware"""
        return {
            "memory_limit": f"{self.profile.max_memory_usage_gb}g",
            "cpu_limit": self.profile.max_cpu_usage_pct / 100,
            "swap_limit": "512m" if self.profile.memory_profile == MemoryProfile.LOW_4GB else "1g",
            "shm_size": "64m" if self.profile.memory_profile == MemoryProfile.LOW_4GB else "128m"
        }

# Global hardware optimizer instance
hardware_detector = HardwareDetector()
optimal_profile = hardware_detector.get_optimal_profile()
performance_optimizer = PerformanceOptimizer(optimal_profile)

# Print detected hardware info
print(f"🔧 Hardware Profile Detected:")
print(f"   Processor: {optimal_profile.processor_type.value}")
print(f"   Memory: {optimal_profile.memory_profile.value}")
print(f"   Max Symbols: {optimal_profile.max_symbols}")
print(f"   Update Freq: {optimal_profile.update_frequency_ms}ms")
print(f"   Target Latency: {optimal_profile.target_latency_ms}ms")

if optimal_profile.processor_type == ProcessorType.I3_8TH_GEN:
    print("⚡ i3 8th Gen Optimizations Active:")
    print(f"   Single Thread: {optimal_profile.use_single_thread}")
    print(f"   Aggressive GC: {optimal_profile.aggressive_gc}")
    print(f"   Data Compression: {optimal_profile.compress_data}")