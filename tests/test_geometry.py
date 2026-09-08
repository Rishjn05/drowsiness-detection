import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from drowsiness.geometry import eye_aspect_ratio, mouth_aspect_ratio, head_tilt_angle


def test_ear_open_eye_is_larger_than_closed_eye():
    open_eye = [(0, 0), (1, -1), (2, -1), (3, 0), (2, 1), (1, 1)]
    closed_eye = [(0, 0), (1, -0.05), (2, -0.05), (3, 0), (2, 0.05), (1, 0.05)]

    open_ear = eye_aspect_ratio(open_eye)
    closed_ear = eye_aspect_ratio(closed_eye)

    assert open_ear > closed_ear
    assert closed_ear < 0.1


def test_ear_requires_six_points():
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0, 0), (1, 1)])


def test_mar_yawn_is_larger_than_closed_mouth():
    closed_mouth = [(0, 0), (1, -0.1), (2, -0.1), (3, 0), (2, 0.1), (1, 0.1)]
    open_mouth = [(0, 0), (1, -2), (2, -2), (3, 0), (2, 2), (1, 2)]

    assert mouth_aspect_ratio(open_mouth) > mouth_aspect_ratio(closed_mouth)


def test_head_tilt_angle_zero_when_level():
    assert head_tilt_angle((0, 0), (10, 0)) == pytest.approx(0.0)


def test_head_tilt_angle_positive_when_tilted_down():
    angle = head_tilt_angle((0, 0), (10, 10))
    assert angle == pytest.approx(45.0)
