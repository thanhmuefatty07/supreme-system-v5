"""
Memory-efficient circular buffer for price history.
Prevents memory leaks and maintains O(1) access time.
"""

from collections import deque
from typing import List, Optional
import array

class CircularBuffer:
    """
    Ultra-efficient circular buffer với fixed memory allocation.

    Performance Characteristics:
    - Memory: Fixed O(size) regardless of runtime
    - Access: O(1) for latest elements
    - CPU: Minimal overhead per operation
    - Cache: Excellent locality for sequential access
    """

    __slots__ = ('_data', '_size', '_index', '_is_full')

    def __init__(self, size: int, dtype: str = 'd'):
        """
        Initialize circular buffer.

        Args:
            size: Fixed buffer size
            dtype: Array data type ('d' for double, 'f' for float)
        """
        self._data = array.array(dtype, [0.0] * size)
        self._size = size
        self._index = 0
        self._is_full = False

    def append(self, value: float) -> None:
        """Add value to buffer (O(1) time)."""
        self._data[self._index] = value
        self._index = (self._index + 1) % self._size
        if self._index == 0:
            self._is_full = True

    def get_latest(self, n: int = 1) -> List[float]:
        """Get last n elements (O(n) but n typically small)."""
        if not self._is_full and self._index == 0:
            return []

        if not self._is_full:
            return self._data[:self._index][-n:]

        # Full buffer case
        result = []
        for i in range(n):
            idx = (self._index - 1 - i) % self._size
            if idx < 0:
                idx += self._size
            result.append(self._data[idx])

        return result[::-1]  # Reverse to chronological order

    def is_full(self) -> bool:
        """Check if buffer contains maximum elements."""
        return self._is_full

    def __len__(self) -> int:
        """Return current number of elements in buffer."""
        return self._size if self._is_full else self._index

    def clear(self) -> None:
        """Reset buffer state."""
        self._index = 0
        self._is_full = False

class RollingAverage:
    """
    Memory-efficient rolling average using circular buffer.

    Maintains running sum for O(1) average calculation.
    """

    __slots__ = ('_buffer', '_sum', '_count')

    def __init__(self, window: int):
        self._buffer = CircularBuffer(window)
        self._sum = 0.0
        self._count = 0

    def add(self, value: float) -> None:
        """Add value and update running statistics."""
        if self._buffer.is_full():
            # Remove oldest value from sum
            oldest = self._buffer.get_latest(1)[0]
            self._sum -= oldest
        else:
            self._count += 1

        self._buffer.append(value)
        self._sum += value

    def get_average(self) -> float:
        """Get current rolling average (O(1))."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def get_sum(self) -> float:
        """Get current sum."""
        return self._sum
