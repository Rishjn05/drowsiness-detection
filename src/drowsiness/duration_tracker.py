"""Tracks how long a boolean condition (e.g. 'drowsy') has been
continuously true, driven by caller-supplied timestamps rather than
wall-clock time — so it behaves identically whether frames come from
a live webcam or are processed from a video file faster/slower than
real time.
"""
from __future__ import annotations

from typing import Optional


class DurationTracker:
    def __init__(self, threshold_seconds: float):
        self.threshold_seconds = threshold_seconds
        self._active_since: Optional[float] = None

    def update(self, active: bool, timestamp: float) -> float:
        """Record the active/inactive state at `timestamp`.

        Returns the number of seconds the active state has been held
        continuously up to and including this timestamp (0.0 if not
        currently active, or if this is the first moment it became
        active).
        """
        if not active:
            self._active_since = None
            return 0.0

        if self._active_since is None:
            self._active_since = timestamp

        return timestamp - self._active_since

    def is_past_threshold(self, active: bool, timestamp: float) -> bool:
        """Update state for this timestamp and report whether the active
        state has now persisted for at least `threshold_seconds`.
        """
        elapsed = self.update(active, timestamp)
        return active and elapsed >= self.threshold_seconds

    def reset(self) -> None:
        """Clear tracked state, as if no active period had ever started."""
        self._active_since = None
