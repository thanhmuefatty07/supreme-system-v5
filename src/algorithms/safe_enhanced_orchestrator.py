#!/usr/bin/env python3
"""
Safe Enhanced Orchestrator - C2a backward compatible
Basic improvements: timeout, memory tracking, logging
"""
import time
import logging
from typing import List, Dict, Callable, Any
from dataclasses import dataclass

@dataclass
class SafeAlgorithmConfig:
    name: str
    function: Callable[[Dict], Any]
    memory_estimate_mb: int
    timeout_seconds: int = 10

class SafeEnhancedOrchestrator:
    def __init__(self, total_memory_mb: int = 600):
        self.total_memory = total_memory_mb
        self.current_usage = 0
        self.algorithms: Dict[str, SafeAlgorithmConfig] = {}
        self.logger = logging.getLogger(__name__)
        
        try:
            import supreme_core
            self.memory_manager = supreme_core.SafeMemoryManager()
            self.has_rust = True
            self.logger.info("Rust core loaded")
        except ImportError:
            self.memory_manager = None
            self.has_rust = False
            self.logger.warning("Rust core not available, using Python tracking")
    
    def register_algorithm(self, config: SafeAlgorithmConfig) -> None:
        self.algorithms[config.name] = config
        self.logger.info(f"Registered: {config.name} (Mem: {config.memory_estimate_mb}MB)")
    
    def execute_safely(self, data: Dict, algorithms: List[str]) -> Dict[str, Any]:
        results = {}
        
        for algo_name in algorithms:
            if algo_name not in self.algorithms:
                self.logger.warning(f"Algorithm not found: {algo_name}")
                continue
            
            config = self.algorithms[algo_name]
            
            if not self._can_allocate(config.memory_estimate_mb):
                self.logger.warning(f"Insufficient memory for {algo_name}")
                results[algo_name] = {"error": "Insufficient memory"}
                continue
            
            try:
                start = time.time()
                result = self._execute_with_basic_timeout(config.function, data, config.timeout_seconds)
                elapsed = time.time() - start
                
                results[algo_name] = {
                    "result": result,
                    "execution_time": elapsed,
                    "memory_used_mb": config.memory_estimate_mb
                }
                self.logger.info(f"Executed {algo_name} in {elapsed:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Algorithm {algo_name} failed: {e}")
                results[algo_name] = {"error": str(e)}
        
        return results
    
    def _can_allocate(self, size_mb: int) -> bool:
        if self.has_rust:
            try:
                self.memory_manager.allocate(size_mb * 1024 * 1024)
                return True
            except:
                return False
        else:
            return (self.current_usage + size_mb) <= self.total_memory
    
    def _execute_with_basic_timeout(self, func: Callable, data: Dict, timeout: int) -> Any:
        # Basic execution without signal (Windows compatible)
        start = time.time()
        result = func(data)
        if time.time() - start > timeout:
            raise TimeoutError(f"Execution exceeded {timeout}s")
        return result
    
    def get_status(self) -> Dict[str, Any]:
        status = {
            "total_algorithms": len(self.algorithms),
            "has_rust_support": self.has_rust,
            "total_memory_mb": self.total_memory,
        }
        
        if self.has_rust:
            try:
                rust_stats = self.memory_manager.get_stats()
                status.update(rust_stats)
            except:
                pass
        else:
            status["current_usage_mb"] = self.current_usage
        
        return status
