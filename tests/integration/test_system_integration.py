#!/usr/bin/env python3
"""
Integration test — pipeline → orchestrator end-to-end
Phase B
"""
from __future__ import annotations
import asyncio
import json
from typing import AsyncIterator
from src.data.realistic_pipeline import RealisticDataPipeline
from src.algorithms.algorithm_orchestrator import MemoryAwareAlgorithmOrchestrator

async def fake_stream() -> AsyncIterator[bytes]:
    for i in range(50):
        yield b"x" * (10_000 + (i % 10) * 500)  # ~10KB per batch
        await asyncio.sleep(0)

async def dummy_algo(name: str, latency_ms: int):
    async def _run(data):
        await asyncio.sleep(latency_ms/1000.0)
        return {"name": name, "ok": True}
    return _run

async def main():
    pipeline = RealisticDataPipeline(memory_budget_mb=100)  # small budget for test
    data = await pipeline.process_stream(fake_stream())

    orch = MemoryAwareAlgorithmOrchestrator(memory_budget_mb=120)
    orch.register("ema", await dummy_algo("ema", 5), req_mb=50, prio=10)
    orch.register("rsi", await dummy_algo("rsi", 5), req_mb=40, prio=9)
    orch.register("macd", await dummy_algo("macd", 5), req_mb=60, prio=8)

    selected = ["ema", "rsi", "macd"]
    results = await orch.execute(data, selected)

    report = {
        "processed_items": len(data),
        "orchestrator": results.get("__metrics__", {}),
    }
    print(json.dumps(report))

if __name__ == "__main__":
    asyncio.run(main())
