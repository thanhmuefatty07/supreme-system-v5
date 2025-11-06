#!/usr/bin/env python3
"""
Realistic Algorithm Orchestrator — memory-aware, priority-based
Phase B implementation
"""
from __future__ import annotations
import asyncio
from typing import Dict, List, Callable, Any

class OrchestratorMetrics:
    def __init__(self) -> None:
        self.executed: int = 0
        self.skipped: int = 0
        self.memory_used_mb: float = 0.0
        self.latency_ms: float = 0.0

class MemoryAwareAlgorithmOrchestrator:
    def __init__(self, memory_budget_mb: int = 600) -> None:
        self.memory_budget_mb = memory_budget_mb
        self.algorithms: Dict[str, Callable[[Any], asyncio.Future]] = {}
        self.requirements_mb: Dict[str, int] = {}
        self.priority: Dict[str, int] = {}  # higher = more important

    def register(self, name: str, func: Callable[[Any], asyncio.Future], req_mb: int, prio: int) -> None:
        self.algorithms[name] = func
        self.requirements_mb[name] = req_mb
        self.priority[name] = prio

    async def execute(self, data: Any, selected: List[str]) -> Dict[str, Any]:
        # sort by priority desc, then by memory ascending
        plan = sorted(selected, key=lambda n: (-self.priority.get(n,0), self.requirements_mb.get(n, 0)))
        results: Dict[str, Any] = {}
        used = 0
        metrics = OrchestratorMetrics()

        for name in plan:
            need = self.requirements_mb.get(name, 0)
            if used + need > self.memory_budget_mb:
                metrics.skipped += 1
                continue
            fn = self.algorithms.get(name)
            if not fn:
                metrics.skipped += 1
                continue
            out = await fn(data)
            results[name] = out
            used += need
            metrics.executed += 1

        metrics.memory_used_mb = float(used)
        results["__metrics__"] = {
            "executed": metrics.executed,
            "skipped": metrics.skipped,
            "memory_used_mb": metrics.memory_used_mb,
        }
        return results
