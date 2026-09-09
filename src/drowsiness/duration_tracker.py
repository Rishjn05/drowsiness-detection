"""Tracks how long a boolean condition has been continuously true.

Used to require a state (e.g. "drowsy") to persist for a minimum duration
before acting on it, so a single borderline/flickering frame doesn't trigger
an action (like an alarm) immediately.
"""
from __future__ import annotations

from typing import Optional


class DurationTracker:
    def __init__(self, threshold_seconds: float) -> None:
        self.threshold_seconds = threshold_seconds
        self._start_time: Optional[float] = None

    def is_past_threshold(self, active: bool, timestamp: float) -> bool:
        """Feed in the current state and timestamp.

        Returns True once `active` has been continuously True for at least
        `threshold_seconds`. Any False sample resets the clock.
        """
        if not active:
            self._start_time = None
            return False

        if self._start_time is None:
            self._start_time = timestamp

        return (timestamp - self._start_time) >= self.threshold_seconds

    def current_duration(self, timestamp: float) -> float:
        """Seconds the condition has been continuously active right now.

        Returns 0.0 if the condition is not currently active.
        """
        if self._start_time is None:
            return 0.0
        return max(0.0, timestamp - self._start_time)

    def reset(self) -> None:
        self._start_time = None