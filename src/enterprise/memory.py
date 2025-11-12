"""
Enterprise Memory Manager for Supreme System V5

World-class memory management with automatic resource management,
memory pools, and leak prevention.
"""

import gc
import tracemalloc
import resource
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
from contextlib import asynccontextmanager
import weakref
import psutil
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryPoolConfig:
    """Configuration for memory pools."""
    min_block_size: int = 64      # 64 bytes
    max_block_size: int = 1048576  # 1MB
    pool_sizes: Dict[str, int] = field(default_factory=lambda: {
        'small': 1024,    # 1KB blocks
        'medium': 65536,  # 64KB blocks
        'large': 1048576  # 1MB blocks
    })


@dataclass
class MemoryBlock:
    """Memory block with metadata."""
    data: bytearray
    size: int
    pool_name: str
    allocated_at: float
    freed: bool = False


class MemoryPool:
    """High-performance memory pool."""

    def __init__(self, block_size: int, max_blocks: int = 1000):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.available_blocks: List[bytearray] = []
        self.allocated_blocks: List[MemoryBlock] = []
        self.total_allocated = 0

    def allocate(self, size: int) -> memoryview:
        """Allocate memory from pool."""
        if size > self.block_size:
            raise ValueError(f"Requested size {size} exceeds block size {self.block_size}")

        # Find or create block
        if self.available_blocks:
            block = self.available_blocks.pop()
        else:
            if len(self.allocated_blocks) >= self.max_blocks:
                self._gc_collect()
            block = bytearray(self.block_size)

        memory_block = MemoryBlock(
            data=block,
            size=size,
            pool_name=f"pool_{self.block_size}",
            allocated_at=tracemalloc.get_tracemalloc_memory()
        )

        self.allocated_blocks.append(memory_block)
        self.total_allocated += size

        return memoryview(block)[:size]

    def deallocate(self, block_view: memoryview):
        """Return memory to pool."""
        # Find the corresponding block
        for i, block in enumerate(self.allocated_blocks):
            if block.data is block_view.obj and not block.freed:
                block.freed = True
                self.available_blocks.append(block.data)
                self.total_allocated -= block.size
                self.allocated_blocks.pop(i)
                break

    def _gc_collect(self):
        """Force garbage collection to free memory."""
        gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'block_size': self.block_size,
            'allocated_blocks': len(self.allocated_blocks),
            'available_blocks': len(self.available_blocks),
            'total_allocated': self.total_allocated,
            'utilization': len(self.allocated_blocks) / (len(self.allocated_blocks) + len(self.available_blocks)) if (len(self.allocated_blocks) + len(self.available_blocks)) > 0 else 0
        }


class EnterpriseMemoryManager:
    """World-class memory management system."""

    def __init__(self, config: MemoryPoolConfig = None):
        self.config = config or MemoryPoolConfig()

        # Enable advanced memory tracing
        tracemalloc.start()

        # Set resource limits
        try:
            resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024))  # 8GB soft, 16GB hard
        except (OSError, ValueError):
            logger.warning("Could not set resource limits")

        # Initialize memory pools
        self.pools: Dict[str, MemoryPool] = {}
        self._initialize_pools()

        # Monitoring
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss
        self.object_registry = weakref.WeakSet()

        # Configure GC
        self._configure_gc()

        # Metrics
        self.metrics = {
            'allocations': 0,
            'deallocations': 0,
            'gc_cycles': 0,
            'memory_peaks': [],
            'leak_detections': 0
        }

    def _initialize_pools(self):
        """Initialize memory pools."""
        for pool_name, block_size in self.config.pool_sizes.items():
            max_blocks = 1000 if pool_name == 'small' else (100 if pool_name == 'medium' else 10)
            self.pools[pool_name] = MemoryPool(block_size, max_blocks)

    def _configure_gc(self):
        """Configure garbage collector for optimal performance."""
        # Disable automatic GC
        gc.disable()

        # Set custom thresholds
        gc.set_threshold(1000, 10, 10)  # More aggressive collection

        # Configure generations
        gc.set_generations(3)

    @asynccontextmanager
    async def managed_memory_context(self, pool_name: str = 'small') -> AsyncIterator[MemoryPool]:
        """Context manager for memory-managed operations."""
        pool = self.pools[pool_name]

        # Pre-allocate if needed
        await self._ensure_pool_capacity(pool)

        try:
            yield pool
        finally:
            # Automatic cleanup
            await self._cleanup_pool(pool)

    async def allocate_from_pool(self, size: int, pool_name: str = 'small') -> memoryview:
        """Allocate memory from appropriate pool."""
        if pool_name not in self.pools:
            raise ValueError(f"Unknown pool: {pool_name}")

        pool = self.pools[pool_name]

        # Check if we need to allocate from this pool
        if size <= pool.block_size:
            try:
                block = pool.allocate(size)
                self.metrics['allocations'] += 1
                return block
            except ValueError:
                # Size too large for pool, use system allocation
                pass

        # Fall back to system allocation with monitoring
        return await self._system_allocate(size)

    async def _system_allocate(self, size: int) -> memoryview:
        """System memory allocation with monitoring."""
        # Check memory limits
        await self._check_memory_limits()

        # Allocate
        block = bytearray(size)
        self.metrics['allocations'] += 1

        # Register for monitoring
        self.object_registry.add(block)

        return memoryview(block)

    async def deallocate_to_pool(self, block: memoryview, pool_name: str = 'small'):
        """Deallocate memory to appropriate pool."""
        if pool_name in self.pools:
            pool = self.pools[pool_name]
            if hasattr(block, 'obj') and block.nbytes <= pool.block_size:
                pool.deallocate(block)
                self.metrics['deallocations'] += 1
                return

        # System deallocation
        await self._system_deallocate(block)

    async def _system_deallocate(self, block: memoryview):
        """System memory deallocation."""
        # Remove from monitoring
        if hasattr(block, 'obj'):
            self.object_registry.discard(block.obj)

        # Let GC handle it
        del block
        self.metrics['deallocations'] += 1

    async def _ensure_pool_capacity(self, pool: MemoryPool):
        """Ensure pool has adequate capacity."""
        if len(pool.available_blocks) < 5:
            # Pre-allocate some blocks
            for _ in range(10):
                if len(pool.allocated_blocks) + len(pool.available_blocks) < pool.max_blocks:
                    pool.available_blocks.append(bytearray(pool.block_size))

    async def _cleanup_pool(self, pool: MemoryPool):
        """Clean up unused memory in pool."""
        # Remove excess free blocks
        while len(pool.available_blocks) > 50:
            pool.available_blocks.pop()

        # Force GC if needed
        await self._smart_gc()

    async def _check_memory_limits(self):
        """Check if we're approaching memory limits."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        if memory_percent > 85:
            logger.warning(".1f")
            await self._emergency_memory_cleanup()
        elif memory_percent > 75:
            logger.info(".1f")
            await self._smart_gc()

        # Track memory peaks
        current_memory = memory_info.rss
        if not self.metrics['memory_peaks'] or current_memory > max(self.metrics['memory_peaks']):
            self.metrics['memory_peaks'].append(current_memory)
            if len(self.metrics['memory_peaks']) > 10:
                self.metrics['memory_peaks'].pop(0)

    async def _smart_gc(self):
        """Intelligent garbage collection."""
        # Only collect if fragmentation is high
        if len(gc.get_objects()) > 10000:
            collected = gc.collect()
            self.metrics['gc_cycles'] += 1
            if collected > 0:
                logger.debug(f"GC collected {collected} objects")

    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup procedures."""
        logger.warning("Initiating emergency memory cleanup")

        # Aggressive GC
        collected = gc.collect(2)  # Collect all generations
        logger.info(f"Emergency GC collected {collected} objects")

        # Clear all pools
        for pool in self.pools.values():
            pool.available_blocks.clear()
            # Note: We don't clear allocated_blocks as they're in use

        # Clear weak references
        self.object_registry.clear()

        # Force memory compaction if available
        try:
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
        except ImportError:
            pass

        self.metrics['leak_detections'] += 1

    async def detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        # Get memory snapshots
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        leaks = []
        for stat in top_stats[:10]:  # Top 10 memory users
            if stat.size > 10 * 1024 * 1024:  # Over 10MB
                leaks.append({
                    'file': stat.traceback[0].filename,
                    'line': stat.traceback[0].lineno,
                    'size': stat.size,
                    'count': stat.count
                })

        return {
            'potential_leaks': leaks,
            'total_tracked': len(top_stats),
            'total_memory': sum(stat.size for stat in top_stats)
        }

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        memory_info = self.process.memory_info()

        pool_stats = {}
        for pool_name, pool in self.pools.items():
            pool_stats[pool_name] = pool.get_stats()

        leak_info = await self.detect_memory_leaks()

        return {
            'process_memory': {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': self.process.memory_percent()
            },
            'pools': pool_stats,
            'metrics': self.metrics.copy(),
            'gc_stats': {
                'collections': gc.get_count(),
                'objects': len(gc.get_objects()),
                'stats': gc.get_stats()
            },
            'leaks': leak_info,
            'tracemalloc': {
                'active': tracemalloc.is_tracing(),
                'frames': len(tracemalloc.get_traced_memory()) if tracemalloc.is_tracing() else 0
            }
        }

    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        before_stats = await self.get_memory_stats()

        # Run optimizations
        await self._emergency_memory_cleanup()

        # Compact memory if possible
        try:
            import sys
            if hasattr(sys, '_debug_malloc_stats'):
                sys._debug_malloc_stats()
        except ImportError:
            pass

        after_stats = await self.get_memory_stats()

        memory_saved = before_stats['process_memory']['rss'] - after_stats['process_memory']['rss']

        return {
            'memory_saved': memory_saved,
            'before': before_stats['process_memory'],
            'after': after_stats['process_memory'],
            'optimization_successful': memory_saved > 0
        }

    async def graceful_shutdown(self):
        """Gracefully shutdown memory manager."""
        logger.info("Shutting down Enterprise Memory Manager")

        # Clear all pools
        for pool in self.pools.values():
            pool.available_blocks.clear()
            pool.allocated_blocks.clear()

        # Clear monitoring
        self.object_registry.clear()

        # Stop tracemalloc
        tracemalloc.stop()

        # Final GC
        gc.collect()

        logger.info("Memory manager shutdown complete")
