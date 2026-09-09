"""Mouth-open duration tracking — a thin, name-preserving wrapper around
DurationTracker (see duration_tracker.py for the general-purpose logic).

Frame-count thresholds are fps-dependent; this instead requires the mouth
to stay open for a wall-clock/video-clock duration, so behavior is
consistent across cameras and video files regardless of frame rate.
"""
from __future__ import annotations

from .duration_tracker import DurationTracker


class YawnTracker(DurationTracker):
    def __init__(self, yawn_seconds: float = 4.0) -> None:
        super().__init__(threshold_seconds=yawn_seconds)

    @property
    def yawn_seconds(self) -> float:
        return self.threshold_seconds

    def update(self, mouth_open: bool, timestamp: float) -> float:
        return super().update(mouth_open, timestamp)

    def is_yawning(self, mouth_open: bool, timestamp: float) -> bool:
        return self.is_past_threshold(mouth_open, timestamp)