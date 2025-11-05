"""
Neuromorphic Architecture - Synaptic Learning Foundation
Implements brain-inspired adaptive caching and pattern recognition
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import math
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SynapticConnection:
    """Represents a connection between two data access patterns"""
    source: str
    target: str
    weight: float = 0.0
    last_activation: float = 0.0
    activation_count: int = 0
    
    def strengthen(self, amount: float = 0.1):
        """Strengthen connection (Hebbian learning)"""
        self.weight = min(1.0, self.weight + amount)
        self.last_activation = time.time()
        self.activation_count += 1
    
    def decay(self, rate: float = 0.01):
        """Natural decay of unused connections"""
        time_since_activation = time.time() - self.last_activation
        decay_factor = math.exp(-rate * time_since_activation / 3600)  # hourly decay
        self.weight *= decay_factor


class SynapticNetwork:
    """Brain-inspired network for learning access patterns"""
    
    def __init__(self, learning_rate: float = 0.1, decay_rate: float = 0.01):
        self.connections: Dict[tuple, SynapticConnection] = {}
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.access_history: List[str] = []
        self.pattern_strength: Dict[tuple, float] = defaultdict(float)
    
    def fire_together(self, keys: List[str]):
        """Neurons that fire together wire together - Hebbian learning"""
        for i, key_a in enumerate(keys):
            for key_b in keys[i+1:]:
                conn_key = tuple(sorted([key_a, key_b]))
                
                if conn_key not in self.connections:
                    self.connections[conn_key] = SynapticConnection(
                        source=conn_key[0], target=conn_key[1]
                    )
                
                self.connections[conn_key].strengthen(self.learning_rate)
        
        # Track access patterns
        if len(keys) > 1:
            pattern = tuple(sorted(keys))
            self.pattern_strength[pattern] += 1
    
    def predict_next_access(self, current_key: str, top_k: int = 3) -> List[str]:
        """Predict likely next accesses based on learned patterns"""
        predictions = []
        
        for (source, target), conn in self.connections.items():
            if source == current_key:
                predictions.append((target, conn.weight))
            elif target == current_key:
                predictions.append((source, conn.weight))
        
        # Sort by connection strength and return top predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [pred[0] for pred in predictions[:top_k]]
    
    def decay_connections(self):
        """Apply natural decay to all connections"""
        for conn in self.connections.values():
            conn.decay(self.decay_rate)
        
        # Remove very weak connections
        self.connections = {
            key: conn for key, conn in self.connections.items() 
            if conn.weight > 0.001
        }


class AdaptiveCacheInterface(ABC):
    """Abstract interface for neuromorphic caching behavior"""
    
    @abstractmethod
    def learn_access_pattern(self, key: str, context: Dict[str, Any]):
        """Learn from access patterns to improve future performance"""
        pass
    
    @abstractmethod
    def predict_prefetch_candidates(self, current_key: str) -> List[str]:
        """Predict what data should be prefetched based on learned patterns"""
        pass
    
    @abstractmethod
    def adapt_eviction_strategy(self, cache_pressure: float) -> str:
        """Adapt cache eviction strategy based on current conditions"""
        pass


class NeuromorphicCacheManager(AdaptiveCacheInterface):
    """Neuromorphic cache manager with synaptic learning"""
    
    def __init__(self, capacity: int = 1000):
        self.synaptic_network = SynapticNetwork()
        self.capacity = capacity
        self.access_count: Dict[str, int] = defaultdict(int)
        self.temporal_patterns: Dict[int, List[str]] = {}  # hour -> access pattern
        
    def learn_access_pattern(self, key: str, context: Dict[str, Any]):
        """Learn from access patterns using neuromorphic principles"""
        self.access_count[key] += 1
        
        # Track temporal patterns (by hour)
        current_hour = int(time.time() // 3600)
        if current_hour not in self.temporal_patterns:
            self.temporal_patterns[current_hour] = []
        
        self.temporal_patterns[current_hour].append(key)
        
        # Learn co-access patterns
        recent_accesses = self.temporal_patterns.get(current_hour, [])[-5:]
        if len(recent_accesses) > 1:
            self.synaptic_network.fire_together(recent_accesses)
    
    def predict_prefetch_candidates(self, current_key: str) -> List[str]:
        """Use synaptic network to predict prefetch candidates"""
        return self.synaptic_network.predict_next_access(current_key)
    
    def adapt_eviction_strategy(self, cache_pressure: float) -> str:
        """Neuromorphic eviction strategy adaptation"""
        if cache_pressure > 0.9:
            return "aggressive_lru"  # Remove least recently used aggressively
        elif cache_pressure > 0.7:
            return "pattern_aware"   # Consider access patterns
        else:
            return "standard_lru"   # Standard LRU
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the synaptic network"""
        total_connections = len(self.synaptic_network.connections)
        avg_strength = sum(c.weight for c in self.synaptic_network.connections.values()) / max(1, total_connections)
        
        return {
            "total_synaptic_connections": total_connections,
            "average_connection_strength": avg_strength,
            "learned_patterns": len(self.synaptic_network.pattern_strength),
            "total_accesses": sum(self.access_count.values()),
            "unique_keys": len(self.access_count)
        }


__all__ = [
    'SynapticConnection',
    'SynapticNetwork', 
    'AdaptiveCacheInterface',
    'NeuromorphicCacheManager'
]
