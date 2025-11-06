#!/usr/bin/env python3
"""
Checkpoint C — Performance Benchmarks for EMA/RSI/MACD
Targets: p95 latency < 150ms; improvement 1.5–2.5x vs baseline
Outputs: benchmark/perf_bench.json
"""
from __future__ import annotations
import json
import time
import numpy as np

def ema_scalar(data: np.ndarray, window: int) -> np.ndarray:
    alpha = 2.0 / (window + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

def rsi_scalar(data: np.ndarray, window: int) -> np.ndarray:
    gains = np.maximum(np.diff(data, prepend=data[0]), 0)
    losses = np.maximum(np.diff(data, prepend=data[0]) * -1, 0)
    out = np.zeros_like(data)
    for i in range(window, len(data)):
        avg_gain = gains[i-window+1:i+1].mean()
        avg_loss = losses[i-window+1:i+1].mean()
        rs = 100.0 if avg_loss == 0 else (avg_gain / avg_loss)
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out

def macd_scalar(data: np.ndarray) -> np.ndarray:
    ema12 = ema_scalar(data, 12)
    ema26 = ema_scalar(data, 26)
    return ema12 - ema26

# Placeholder optimized versions (vectorized where possible)
def ema_vectorized(data: np.ndarray, window: int) -> np.ndarray:
    # For now, reuse scalar; replace with Rust/Numba/SIMD later
    return ema_scalar(data, window)

def rsi_vectorized(data: np.ndarray, window: int) -> np.ndarray:
    return rsi_scalar(data, window)

def macd_vectorized(data: np.ndarray) -> np.ndarray:
    return macd_scalar(data)


def bench_once(fn, *args, repeats=3):
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        dt = (time.perf_counter() - t0) * 1000.0
        best = min(best, dt)
    return best


def main():
    np.random.seed(0)
    n = 1_000_000
    data = np.random.random(n).astype(np.float32)

    results = {}
    # EMA
    t_scalar = bench_once(ema_scalar, data, 20)
    t_opt = bench_once(ema_vectorized, data, 20)
    results['ema_ms'] = { 'scalar': t_scalar, 'optimized': t_opt, 'improvement': (t_scalar / max(1e-6, t_opt)) }

    # RSI
    t_scalar = bench_once(rsi_scalar, data, 14)
    t_opt = bench_once(rsi_vectorized, data, 14)
    results['rsi_ms'] = { 'scalar': t_scalar, 'optimized': t_opt, 'improvement': (t_scalar / max(1e-6, t_opt)) }

    # MACD
    t_scalar = bench_once(macd_scalar, data)
    t_opt = bench_once(macd_vectorized, data)
    results['macd_ms'] = { 'scalar': t_scalar, 'optimized': t_opt, 'improvement': (t_scalar / max(1e-6, t_opt)) }

    out = {
        'points': n,
        'results': results,
        'targets': { 'latency_p95_ms': 150.0, 'improvement_min': 1.5, 'improvement_max': 2.5 }
    }
    print(json.dumps(out))

if __name__ == '__main__':
    main()
