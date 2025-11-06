#!/usr/bin/env python3
"""
Update integration test to validate adaptive pipeline & orchestrator limits
"""
from __future__ import annotations
import asyncio
import json
from typing import AsyncIterator
from src.data.realistic_pipeline import RealisticDataPipeline
from src.algorithms.algorithm_orchestrator import MemoryAwareAlgorithmOrchestrator

async def fake_stream() -> AsyncIterator[bytes]:
    # bigger, varied batches to trigger cleanup
    for i in range(120):
        yield b"x" * (20_000 + (i % 20) * 1_000)  # ~20–39KB per batch
        await asyncio.sleep(0)

async def algo(name: str, ms: int):
    async def _run(data):
        await asyncio.sleep(ms/1000.0)
        return {"name": name, "ok": True}
    return _run

async def main():
    pipeline = RealisticDataPipeline(memory_budget_mb=2)  # 2MB small to force LRU cleanup
    data = await pipeline.process_stream(fake_stream())

    orch = MemoryAwareAlgorithmOrchestrator(memory_budget_mb=120)
    orch.register("ema", await algo("ema", 5), req_mb=50, prio=10)
    orch.register("rsi", await algo("rsi", 5), req_mb=40, prio=9)
    orch.register("macd", await algo("macd", 5), req_mb=60, prio=8)

    selected = ["ema", "rsi", "macd"]
    results = await orch.execute(data, selected)

    report = {
        "pipeline": data.get("stats", {}),
        "orchestrator": results.get("__metrics__", {}),
    }
    print(json.dumps(report))

if __name__ == "__main__":
    asyncio.run(main())
