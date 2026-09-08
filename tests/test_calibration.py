import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from drowsiness.calibration import calibrate


def test_calibrate_sets_threshold_below_baseline_mean():
    ear_samples = [0.30, 0.31, 0.29, 0.30, 0.32]
    mar_samples = [0.20, 0.22, 0.19, 0.21, 0.20]

    thresholds = calibrate(ear_samples, mar_samples)

    assert thresholds.ear_threshold < min(ear_samples)
    assert thresholds.mar_threshold > max(mar_samples)


def test_calibrate_raises_on_empty_samples():
    with pytest.raises(ValueError):
        calibrate([], [])


def test_calibrate_floor_on_ear_threshold():
    # Very low, noisy baseline shouldn't push threshold below the sane floor.
    thresholds = calibrate([0.12, 0.05, 0.20], [0.5, 0.5, 0.5])
    assert thresholds.ear_threshold >= 0.10
