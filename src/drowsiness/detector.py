"""Frame-by-frame drowsiness detector.

Replaces face_recognition/dlib (compiled dependency, slow CNN detector) with
MediaPipe Face Mesh (pure pip wheel, real-time on CPU).

Adds:
- PERCLOS-style rolling window instead of an unbounded up/down score.
- Separate, independently-tunable eye-closure and yawn-duration triggers.
- Graceful "no face found" handling.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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
    perclos: float = 0.0
    drowsy: bool = False


@dataclass
class DetectorConfig:
    ear_threshold: float = 0.21
    mar_threshold: float = 0.6
    window_seconds: float = 6.0
    fps_estimate: int = 15
    perclos_drowsy_ratio: float = 0.4  # fraction of window with eyes closed
    consecutive_yawn_frames: int = 8


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
        window_len = max(1, int(self.config.window_seconds * self.config.fps_estimate))
        self._eye_closed_history: Deque[bool] = deque(maxlen=window_len)
        self._yawn_run = 0

    def close(self) -> None:
        self._mesh.close()

    def _extract_points(self, landmark_list, indices, w, h) -> List[Tuple[float, float]]:
        return [
            (landmark_list[i].x * w, landmark_list[i].y * h) for i in indices
        ]

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._eye_closed_history.append(False)
            return FrameResult(face_found=False, perclos=self._perclos())

        face_landmarks = results.multi_face_landmarks[0].landmark

        left_eye = self._extract_points(face_landmarks, landmarks.LEFT_EYE, w, h)
        right_eye = self._extract_points(face_landmarks, landmarks.RIGHT_EYE, w, h)
        mouth = self._extract_points(face_landmarks, landmarks.MOUTH, w, h)

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        eyes_closed = ear < self.config.ear_threshold
        self._eye_closed_history.append(eyes_closed)

        if mar > self.config.mar_threshold:
            self._yawn_run += 1
        else:
            self._yawn_run = 0
        yawning = self._yawn_run >= self.config.consecutive_yawn_frames

        perclos = self._perclos()
        drowsy = perclos >= self.config.perclos_drowsy_ratio or yawning

        return FrameResult(
            face_found=True,
            ear=round(ear, 3),
            mar=round(mar, 3),
            eyes_closed=eyes_closed,
            yawning=yawning,
            perclos=round(perclos, 3),
            drowsy=drowsy,
        )

    def _perclos(self) -> float:
        if not self._eye_closed_history:
            return 0.0
        return sum(self._eye_closed_history) / len(self._eye_closed_history)

    def __enter__(self) -> "DrowsinessDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
