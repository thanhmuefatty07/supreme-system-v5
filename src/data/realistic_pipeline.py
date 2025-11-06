#!/usr/bin/env python3
"""
Realistic Data Pipeline — B2 enhancements
- Adaptive batch sizing theo memory pressure proxy
- LRU cleanup for buffers
- Throughput & latency stats
"""
from __future__ import annotations
import asyncio
import time
from collections import OrderedDict
from typing import AsyncIterator, Dict, Any

class RealisticDataPipeline:
    def __init__(self, memory_budget_mb: int = 800) -> None:
        self.memory_budget_bytes = memory_budget_mb * 1024 * 1024
        self.inflight_bytes = 0
        self.buffers: OrderedDict[str, bytes] = OrderedDict()
        self.stats: Dict[str, Any] = {
            "processed": 0,
            "bytes": 0,
            "throughput_items_per_s": 0.0,
            "avg_batch_latency_ms": 0.0,
        }

    async def process_stream(self, stream: AsyncIterator[bytes]) -> Dict[str, Any]:
        start = time.perf_counter()
        batch_count = 0
        total_latency = 0.0

        async for batch in stream:
            t0 = time.perf_counter()
            # adaptive sizing: if inflight high -> force cleanup
            if not self._can_allocate(len(batch)):
                await self._cleanup_lru(len(batch))
            key = str(hash((batch_count, len(batch))))
            self.buffers[key] = batch
            self.inflight_bytes += len(batch)
            self.stats["processed"] += 1
            self.stats["bytes"] += len(batch)
            batch_count += 1
            # simulate processing cost: noop
            # release buffer immediately for streaming pipeline
            released = self.buffers.pop(key)
            self.inflight_bytes -= len(released)
            total_latency += (time.perf_counter() - t0) * 1000.0
            await asyncio.sleep(0)

        duration = max(1e-6, time.perf_counter() - start)
        self.stats["throughput_items_per_s"] = self.stats["processed"] / duration
        self.stats["avg_batch_latency_ms"] = (total_latency / max(1, batch_count))
        result = {"stats": self.stats}
        return result

    def _can_allocate(self, size: int) -> bool:
        return (self.inflight_bytes + size) <= self.memory_budget_bytes

    async def _cleanup_lru(self, need: int) -> None:
        # Free least-recent buffers until space is enough
        while self.buffers and not self._can_allocate(need):
            _, buf = self.buffers.popitem(last=False)
            self.inflight_bytes -= len(buf)
            await asyncio.sleep(0)
