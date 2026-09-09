import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness.duration_tracker import DurationTracker


def test_not_past_threshold_before_duration_elapses():
    tracker = DurationTracker(threshold_seconds=2.0)
    tracker.update(active=True, timestamp=0.0)
    assert tracker.is_past_threshold(active=True, timestamp=1.0) is False


def test_past_threshold_once_duration_elapses():
    tracker = DurationTracker(threshold_seconds=2.0)
    tracker.update(active=True, timestamp=0.0)
    assert tracker.is_past_threshold(active=True, timestamp=2.0) is True


def test_going_inactive_resets_timer():
    tracker = DurationTracker(threshold_seconds=2.0)
    tracker.update(active=True, timestamp=0.0)
    tracker.update(active=False, timestamp=1.0)
    assert tracker.is_past_threshold(active=True, timestamp=1.1) is False


def test_reset_clears_state():
    tracker = DurationTracker(threshold_seconds=2.0)
    tracker.update(active=True, timestamp=0.0)
    tracker.reset()
    assert tracker.update(active=True, timestamp=2.1) == 0.0