"""Geometric metrics used for drowsiness detection.

Pure math, no dependency on any particular face-landmark library, so it can
be unit tested without installing mediapipe/dlib/opencv.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye: Sequence[Point]) -> float:
    """Standard 6-point EAR (Soukupova & Cech, 2016).

    eye must be ordered: [p1, p2, p3, p4, p5, p6] where p1-p4 is the
    horizontal (corner-to-corner) axis and p2-p6 / p3-p5 are the two
    vertical pairs.
    """
    if len(eye) != 6:
        raise ValueError(f"eye_aspect_ratio expects 6 points, got {len(eye)}")
    vertical_1 = _dist(eye[1], eye[5])
    vertical_2 = _dist(eye[2], eye[4])
    horizontal = _dist(eye[0], eye[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth: Sequence[Point]) -> float:
    """MAR from a 6-point mouth contour (outer corners + top/bottom lip mids).

    mouth must be ordered: [left_corner, top_1, top_2, right_corner,
    bottom_2, bottom_1] mirroring the eye convention so the same style of
    ratio (vertical opening / horizontal width) applies.
    """
    if len(mouth) != 6:
        raise ValueError(f"mouth_aspect_ratio expects 6 points, got {len(mouth)}")
    vertical_1 = _dist(mouth[1], mouth[5])
    vertical_2 = _dist(mouth[2], mouth[4])
    horizontal = _dist(mouth[0], mouth[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def head_tilt_angle(left_eye_outer: Point, right_eye_outer: Point) -> float:
    """Roll angle (degrees) of the eye line, useful for nod/head-drop cues."""
    dx = right_eye_outer[0] - left_eye_outer[0]
    dy = right_eye_outer[1] - left_eye_outer[1]
    return math.degrees(math.atan2(dy, dx))
