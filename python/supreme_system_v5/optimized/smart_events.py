"""
Intelligent event processing based on market significance + cadence control.
Adds enforced scalping cadence 30–60s with ±10% jitter and backstop of max_time_gap_seconds.
"""
from typing import Optional, Dict, Any
import time
import random

class SmartEventProcessor:
    __slots__ = ('_config','_last_price','_last_volume','_last_timestamp','_event_count','_skip_count','_volume_avg','_last_cadence_ts','_current_deadline')

    def __init__(self, config: Dict[str, Any]):
        self._config = {
            'min_price_change_pct': config.get('min_price_change_pct', 0.002),
            'min_volume_multiplier': config.get('min_volume_multiplier', 1.5),
            'max_time_gap_seconds': config.get('max_time_gap_seconds', 60),
            'volume_window': config.get('volume_window', 20),
            'scalping_min_interval': config.get('scalping_min_interval', 30),
            'scalping_max_interval': config.get('scalping_max_interval', 60),
            'cadence_jitter_pct': config.get('cadence_jitter_pct', 0.10)
        }
        self._last_price: Optional[float] = None
        self._last_volume: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._event_count = 0
        self._skip_count = 0
        from .circular_buffer import RollingAverage
        self._volume_avg = RollingAverage(self._config['volume_window'])
        self._last_cadence_ts = 0.0
        self._current_deadline = self._next_deadline()

    def _next_deadline(self) -> float:
        base = random.uniform(self._config['scalping_min_interval'], self._config['scalping_max_interval'])
        jitter = base * random.uniform(-self._config['cadence_jitter_pct'], self._config['cadence_jitter_pct'])
        return max(1.0, base + jitter)

    def should_process(self, price: float, volume: float, timestamp: Optional[float] = None) -> bool:
        if timestamp is None:
            timestamp = time.time()

        # First event always processes and initializes cadence
        if self._last_price is None:
            self._initialize_state(price, volume, timestamp)
            return True

        # Enforce cadence window (primary objective per requirements)
        elapsed_since_cadence = timestamp - self._last_cadence_ts
        if elapsed_since_cadence < self._current_deadline:
            # honor cadence unless backstop triggers
            backstop = (timestamp - self._last_timestamp) >= self._config['max_time_gap_seconds']
            if not backstop:
                self._update_state(price, volume, timestamp, processed=False)
                return False

        # Significance gates
        price_change_pct = abs(price - self._last_price) / max(self._last_price or 1.0, 1e-9)
        self._volume_avg.add(volume)
        avg_volume = self._volume_avg.get_average()
        volume_ratio = (volume / avg_volume) if avg_volume > 0 else 0.0

        price_sig = price_change_pct >= self._config['min_price_change_pct']
        vol_sig = volume_ratio >= self._config['min_volume_multiplier']
        time_sig = (timestamp - self._last_timestamp) >= self._config['max_time_gap_seconds']

        processed = price_sig or vol_sig or time_sig or (elapsed_since_cadence >= self._current_deadline)
        self._update_state(price, volume, timestamp, processed=processed)

        if processed:
            # reset cadence window
            self._last_cadence_ts = timestamp
            self._current_deadline = self._next_deadline()
        return processed

    def _initialize_state(self, price: float, volume: float, ts: float) -> None:
        self._last_price = price
        self._last_volume = volume
        self._last_timestamp = ts
        from .circular_buffer import RollingAverage
        self._volume_avg = RollingAverage(self._config['volume_window'])
        self._volume_avg.add(volume)
        self._event_count = 1
        self._skip_count = 0
        self._last_cadence_ts = ts
        self._current_deadline = self._next_deadline()

    def _update_state(self, price: float, volume: float, ts: float, processed: bool) -> None:
        self._last_price = price
        self._last_volume = volume
        self._last_timestamp = ts
        if processed:
            self._event_count += 1
        else:
            self._skip_count += 1

    def get_stats(self) -> Dict[str, Any]:
        total = self._event_count + self._skip_count
        skip_ratio = (self._skip_count / total) if total > 0 else 0.0
        return {
            'events_processed': self._event_count,
            'events_skipped': self._skip_count,
            'skip_ratio': skip_ratio,
            'cadence_deadline_s': self._current_deadline
        }

    def reset(self) -> None:
        self.__init__(self._config)
