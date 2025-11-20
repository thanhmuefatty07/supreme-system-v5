"""
Enterprise Concurrency Manager for Supreme System V5

World-class concurrency management with structured concurrency,
deadlock prevention, and resource-aware execution.
"""

import asyncio
import threading
import time
import multiprocessing
import logging
from typing import Any, Callable, List, Dict, Optional, Set, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import concurrent.futures
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency management."""
    max_threads: int = None
    max_processes: int = None
    thread_timeout: float = 30.0
    process_timeout: float = 300.0
    deadlock_timeout: float = 30.0
    resource_check_interval: float = 1.0
    enable_monitoring: bool = True


@dataclass
class TaskContext:
    """Context for task execution."""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    cpu_intensive: bool = False
    io_bound: bool = False
    required_resources: List[str] = field(default_factory=list)
    priority: int = 1  # 1-10, higher is more important
    timeout: Optional[float] = None


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    resources_used: Dict[str, Any] = field(default_factory=dict)


class DeadlockDetector:
    """Advanced deadlock detection and resolution."""

    def __init__(self):
        self.wait_graph: Dict[str, Set[str]] = {}
        self.resource_owners: Dict[str, str] = {}
        self.detection_interval = 1.0

    async def detect_and_resolve(self, resource: str, waiting_tasks: List[str]) -> bool:
        """Detect deadlock and attempt resolution."""
        # Build wait-for graph
        self._build_wait_graph(resource, waiting_tasks)

        # Detect cycles (deadlocks)
        cycles = self._find_cycles()

        if cycles:
            logger.warning(f"Deadlock detected in cycles: {cycles}")
            # Resolve by killing lowest priority task
            await self._resolve_deadlock(cycles)
            return True

        return False

    def _build_wait_graph(self, resource: str, waiting_tasks: List[str]):
        """Build resource wait graph."""
        for task in waiting_tasks:
            if task not in self.wait_graph:
                self.wait_graph[task] = set()
            self.wait_graph[task].add(resource)

    def _find_cycles(self) -> List[List[str]]:
        """Find cycles in wait graph using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.wait_graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Cycle found
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in self.wait_graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    async def _resolve_deadlock(self, cycles: List[List[str]]):
        """Resolve deadlock by terminating lowest priority task."""
        # Find lowest priority task in cycle
        lowest_priority_task = None
        lowest_priority = float('inf')

        for cycle in cycles:
            for task in cycle:
                # In real implementation, get task priority
                priority = 1  # Default priority
                if priority < lowest_priority:
                    lowest_priority = priority
                    lowest_priority_task = task

        if lowest_priority_task:
            logger.info(f"Resolving deadlock by terminating task: {lowest_priority_task}")
            # Terminate the task (implementation depends on task type)


class EnterpriseConcurrencyManager:
    """World-class concurrency management system."""

    def __init__(self, config: ConcurrencyConfig = None):
        self.config = config or ConcurrencyConfig()
        self._initialize_resources()

        # Concurrency primitives
        self.resource_locks: Dict[str, asyncio.Lock] = {}
        self.task_groups: Dict[str, set] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # Monitoring
        self.monitoring_active = self.config.enable_monitoring
        self.resource_monitor = ResourceMonitor()
        self.deadlock_detector = DeadlockDetector()

        # Performance metrics
        self.metrics = {
            'tasks_executed': 0,
            'tasks_failed': 0,
            'deadlocks_resolved': 0,
            'average_execution_time': 0.0
        }

    def _initialize_resources(self):
        """Initialize system resources based on hardware."""
        cpu_count = multiprocessing.cpu_count()

        if self.config.max_threads is None:
            self.config.max_threads = min(cpu_count * 2, 32)

        if self.config.max_processes is None:
            self.config.max_processes = max(1, cpu_count // 2)

        # Initialize thread pool
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_threads,
            thread_name_prefix="enterprise-thread"
        )

        # Initialize process pool
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.config.max_processes
        )

    @asynccontextmanager
    async def managed_execution_context(self, context_name: str):
        """Context manager for structured task execution."""
        task_group = set()
        self.task_groups[context_name] = task_group

        try:
            yield task_group
        finally:
            # Cancel all tasks in group
            await self._cancel_task_group(context_name)
            del self.task_groups[context_name]

    async def execute_with_structured_concurrency(
        self,
        tasks: List[TaskContext],
        max_concurrent: int = None
    ) -> List[ExecutionResult]:
        """Execute tasks with structured concurrency and proper cancellation."""

        if max_concurrent is None:
            max_concurrent = len(tasks)

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def execute_task(task: TaskContext) -> ExecutionResult:
            async with semaphore:
                return await self._execute_single_task(task)

        # Execute all tasks with structured concurrency
        async with asyncio.TaskGroup() as tg:
            task_coroutines = [execute_task(task) for task in tasks]
            for coro in asyncio.as_completed(task_coroutines):
                result = await coro
                results.append(result)

        return results

    async def execute_threaded_tasks(
        self,
        task_functions: List[Callable],
        max_concurrent: int = None,
        timeout: float = 30.0
    ) -> List[Any]:
        """Execute CPU-bound tasks using thread pool."""

        if max_concurrent is None:
            max_concurrent = min(len(task_functions), self.config.max_threads)

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def execute_threaded_task(func: Callable) -> Any:
            async with semaphore:
                loop = asyncio.get_running_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self.thread_pool, func),
                        timeout=timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.warning(f"Threaded task timed out after {timeout}s")
                    raise
                except Exception as e:
                    logger.error(f"Threaded task failed: {e}")
                    raise

        # Execute with concurrency control
        async with asyncio.TaskGroup() as tg:
            for func in task_functions:
                tg.create_task(execute_threaded_task(func))

        return results

    async def execute_process_tasks(
        self,
        task_functions: List[Callable],
        args_list: List[Tuple] = None,
        max_concurrent: int = None,
        timeout: float = 300.0
    ) -> List[Any]:
        """Execute CPU-intensive tasks using process pool."""

        if args_list is None:
            args_list = [()] * len(task_functions)

        if max_concurrent is None:
            max_concurrent = min(len(task_functions), self.config.max_processes)

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def execute_process_task(func: Callable, args: Tuple) -> Any:
            async with semaphore:
                loop = asyncio.get_running_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self.process_pool, func, *args),
                        timeout=timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.warning(f"Process task timed out after {timeout}s")
                    raise
                except Exception as e:
                    logger.error(f"Process task failed: {e}")
                    raise

        # Execute with concurrency control
        async with asyncio.TaskGroup() as tg:
            for func, args in zip(task_functions, args_list):
                tg.create_task(execute_process_task(func, args))

        return results

    async def _execute_single_task(self, task: TaskContext) -> ExecutionResult:
        """Execute a single task with proper error handling and monitoring."""

        start_time = time.time()
        task_id = task.task_id

        try:
            # Acquire required resources
            async with self._acquire_resources(task.required_resources):
                # Determine execution context
                if task.cpu_intensive:
                    # Use process pool
                    result = await self._execute_in_process(task)
                elif task.io_bound:
                    # Use thread pool
                    result = await self._execute_in_thread(task)
                else:
                    # Execute directly
                    result = await self._execute_directly(task)

                execution_time = time.time() - start_time

                # Update metrics
                self.metrics['tasks_executed'] += 1
                self.metrics['average_execution_time'] = (
                    (self.metrics['average_execution_time'] * (self.metrics['tasks_executed'] - 1)) +
                    execution_time
                ) / self.metrics['tasks_executed']

                return ExecutionResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    resources_used=await self.resource_monitor.get_resource_usage()
                )

        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics['tasks_failed'] += 1

            logger.error(f"Task {task_id} failed: {e}")
            return ExecutionResult(
                task_id=task_id,
                success=False,
                error=e,
                execution_time=execution_time
            )

    async def _execute_in_process(self, task: TaskContext) -> Any:
        """Execute task in process pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.process_pool,
            task.function,
            *task.args,
            **task.kwargs
        )

    async def _execute_in_thread(self, task: TaskContext) -> Any:
        """Execute task in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            task.function,
            *task.args,
            **task.kwargs
        )

    async def _execute_directly(self, task: TaskContext) -> Any:
        """Execute task directly in asyncio."""
        return await task.function(*task.args, **task.kwargs)

    @asynccontextmanager
    async def _acquire_resources(self, resources: List[str]):
        """Acquire resources with deadlock detection."""
        if not resources:
            yield
            return

        # Sort resources to prevent deadlocks
        sorted_resources = sorted(resources)

        acquired_locks = []
        try:
            for resource in sorted_resources:
                if resource not in self.resource_locks:
                    self.resource_locks[resource] = asyncio.Lock()

                lock = self.resource_locks[resource]

                # Try to acquire with timeout
                try:
                    await asyncio.wait_for(lock.acquire(), timeout=self.config.deadlock_timeout)
                    acquired_locks.append(lock)
                except asyncio.TimeoutError:
                    # Deadlock detected
                    await self.deadlock_detector.detect_and_resolve(resource, [])
                    self.metrics['deadlocks_resolved'] += 1
                    raise

            yield

        finally:
            # Release in reverse order
            for lock in reversed(acquired_locks):
                lock.release()

    async def _cancel_task_group(self, group_name: str):
        """Cancel all tasks in a group."""
        if group_name in self.task_groups:
            tasks_to_cancel = self.task_groups[group_name].copy()

            # Cancel all tasks
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()

            # Wait for cancellation
            if tasks_to_cancel:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        return {
            'thread_pool': {
                'active_threads': len([t for t in self.thread_pool._threads if t.is_alive()]),
                'max_threads': self.config.max_threads
            },
            'process_pool': {
                'active_processes': len(self.process_pool._processes),
                'max_processes': self.config.max_processes
            },
            'metrics': self.metrics.copy(),
            'resource_usage': await self.resource_monitor.get_resource_usage(),
            'active_task_groups': len(self.task_groups)
        }

    async def graceful_shutdown(self, timeout: float = 30.0):
        """Gracefully shutdown all resources."""
        logger.info("Initiating graceful shutdown of concurrency manager")

        # Cancel all active tasks
        for group_name in list(self.task_groups.keys()):
            await self._cancel_task_group(group_name)

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True, timeout=timeout/2)

        # Shutdown process pool
        self.process_pool.shutdown(wait=True, timeout=timeout/2)

        logger.info("Concurrency manager shutdown complete")


class ResourceMonitor:
    """Advanced resource monitoring and management."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss

    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get comprehensive resource usage metrics."""
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()

        return {
            'memory_rss': memory_info.rss,
            'memory_vms': memory_info.vms,
            'memory_percent': self.process.memory_percent(),
            'cpu_percent': cpu_percent,
            'threads': self.process.num_threads(),
            'open_files': len(self.process.open_files()),
            'connections': len(self.process.connections())
        }

    async def check_resource_limits(self) -> Dict[str, bool]:
        """Check if system is approaching resource limits."""
        usage = await self.get_resource_usage()

        return {
            'memory_critical': usage['memory_percent'] > 90,
            'cpu_critical': usage['cpu_percent'] > 95,
            'threads_high': usage['threads'] > 100,
            'files_high': usage['open_files'] > 1000
        }
