"""Auto-calibrate EAR/MAR thresholds from a short baseline clip instead of
hardcoding 0.2 / 0.6 for every face and every camera."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import List


@dataclass
class Thresholds:
    ear_threshold: float
    mar_threshold: float


def calibrate(ear_samples: List[float], mar_samples: List[float]) -> Thresholds:
    """Given EAR/MAR readings collected while the subject's eyes were open
    and mouth closed (e.g. first N seconds of a session), derive
    person-specific thresholds a couple of standard deviations below/above
    the baseline mean.
    """
    if not ear_samples or not mar_samples:
        raise ValueError("Need at least one EAR and one MAR sample to calibrate")

    ear_mean = mean(ear_samples)
    ear_std = pstdev(ear_samples) if len(ear_samples) > 1 else 0.02
    mar_mean = mean(mar_samples)
    mar_std = pstdev(mar_samples) if len(mar_samples) > 1 else 0.05

    ear_threshold = max(0.10, ear_mean - 2.0 * ear_std)
    mar_threshold = mar_mean + 3.0 * mar_std

    return Thresholds(ear_threshold=round(ear_threshold, 3), mar_threshold=round(mar_threshold, 3))
