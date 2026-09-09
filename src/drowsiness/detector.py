"""Frame-by-frame drowsiness detector.

Replaces face_recognition/dlib (compiled dependency, slow CNN detector) with
MediaPipe Face Mesh (pure pip wheel, real-time on CPU).

Adds:
- PERCLOS-style rolling *time* window (not a fixed frame count), so it stays
  correct even if the real frame rate drifts from `fps_estimate`.
- Independently-tunable eye-closure (EAR) and yawn-duration (MAR) triggers,
  both expressed in seconds so they match what the CLI exposes.
- Graceful "no face found" handling.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from . import landmarks
from .geometry import eye_aspect_ratio, mouth_aspect_ratio

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


@dataclass
class FrameResult:
    face_found: bool
    ear: Optional[float] = None
    mar: Optional[float] = None
    eyes_closed: bool = False
    yawning: bool = False
    yawn_duration: float = 0.0
    perclos: float = 0.0
    drowsy: bool = False


@dataclass
class DetectorConfig:
    ear_threshold: float = 0.21
    mar_threshold: float = 0.6
    window_seconds: float = 6.0
    fps_estimate: int = 15  # kept for callers/logging; windowing itself is time-based
    perclos_drowsy_ratio: float = 0.4  # fraction of window with eyes closed
    yawn_seconds: float = 4.0  # seconds mouth must stay open before it counts as a yawn


class DrowsinessDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        if mp is None:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )
        self.config = config or DetectorConfig()
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # (timestamp, eyes_closed) samples, trimmed to window_seconds each frame
        self._eye_closed_history: Deque[Tuple[float, bool]] = deque()
        self._yawn_start: Optional[float] = None

    def close(self) -> None:
        self._mesh.close()

    def _extract_points(self, landmark_list, indices, w, h) -> List[Tuple[float, float]]:
        return [
            (landmark_list[i].x * w, landmark_list[i].y * h) for i in indices
        ]

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._eye_closed_history.append((timestamp, False))
            self._trim_history(timestamp)
            self._yawn_start = None
            return FrameResult(face_found=False, perclos=self._perclos())

        face_landmarks = results.multi_face_landmarks[0].landmark

        left_eye = self._extract_points(face_landmarks, landmarks.LEFT_EYE, w, h)
        right_eye = self._extract_points(face_landmarks, landmarks.RIGHT_EYE, w, h)
        mouth = self._extract_points(face_landmarks, landmarks.MOUTH, w, h)

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        eyes_closed = ear < self.config.ear_threshold
        self._eye_closed_history.append((timestamp, eyes_closed))
        self._trim_history(timestamp)

        mouth_open = mar > self.config.mar_threshold
        if mouth_open:
            if self._yawn_start is None:
                self._yawn_start = timestamp
            yawn_duration = timestamp - self._yawn_start
        else:
            self._yawn_start = None
            yawn_duration = 0.0

        yawning = yawn_duration >= self.config.yawn_seconds

        perclos = self._perclos()
        drowsy = perclos >= self.config.perclos_drowsy_ratio or yawning

        return FrameResult(
            face_found=True,
            ear=round(ear, 3),
            mar=round(mar, 3),
            eyes_closed=eyes_closed,
            yawning=yawning,
            yawn_duration=round(yawn_duration, 2),
            perclos=round(perclos, 3),
            drowsy=drowsy,
        )

    def _trim_history(self, now: float) -> None:
        cutoff = now - self.config.window_seconds
        while self._eye_closed_history and self._eye_closed_history[0][0] < cutoff:
            self._eye_closed_history.popleft()

    def _perclos(self) -> float:
        if not self._eye_closed_history:
            return 0.0
        closed = sum(1 for _, is_closed in self._eye_closed_history if is_closed)
        return closed / len(self._eye_closed_history)

    def __enter__(self) -> "DrowsinessDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()