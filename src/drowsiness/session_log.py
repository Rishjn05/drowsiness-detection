"""Session logging: writes a per-frame CSV plus a summary of discrete
"drowsy events" (contiguous stretches where result.drowsy was True), so a
session can be reviewed after the fact instead of only watched live.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from .detector import FrameResult

CSV_FIELDS = ["timestamp", "face_found", "ear", "mar", "perclos", "eyes_closed", "yawning", "drowsy"]


@dataclass
class DrowsyEvent:
    start_time: float
    end_time: Optional[float] = None
    peak_perclos: float = 0.0

    @property
    def duration(self) -> float:
        if self.end_time is None:
            return 0.0
        return round(self.end_time - self.start_time, 2)


class SessionLogger:
    def __init__(self) -> None:
        self._rows: List[dict] = []
        self._events: List[DrowsyEvent] = []
        self._current_event: Optional[DrowsyEvent] = None

    def log_frame(self, timestamp: float, result: FrameResult) -> None:
        self._rows.append(
            {
                "timestamp": round(timestamp, 3),
                "face_found": result.face_found,
                "ear": result.ear,
                "mar": result.mar,
                "perclos": result.perclos,
                "eyes_closed": result.eyes_closed,
                "yawning": result.yawning,
                "drowsy": result.drowsy,
            }
        )

        if result.drowsy:
            if self._current_event is None:
                self._current_event = DrowsyEvent(start_time=timestamp)
            self._current_event.peak_perclos = max(self._current_event.peak_perclos, result.perclos)
        elif self._current_event is not None:
            self._current_event.end_time = timestamp
            self._events.append(self._current_event)
            self._current_event = None

    def close(self, final_timestamp: float) -> None:
        """Call once at the end of a session to flush any event still open."""
        if self._current_event is not None:
            self._current_event.end_time = final_timestamp
            self._events.append(self._current_event)
            self._current_event = None

    @property
    def events(self) -> List[DrowsyEvent]:
        return list(self._events)

    @property
    def rows(self) -> List[dict]:
        return list(self._rows)

    def summary(self) -> dict:
        total_duration = self._rows[-1]["timestamp"] if self._rows else 0.0
        drowsy_time = sum(e.duration for e in self._events)
        return {
            "total_frames": len(self._rows),
            "session_duration_s": total_duration,
            "drowsy_event_count": len(self._events),
            "total_drowsy_time_s": round(drowsy_time, 2),
            "drowsy_time_pct": round((drowsy_time / total_duration * 100), 2) if total_duration else 0.0,
            "events": [
                {"start_s": e.start_time, "end_s": e.end_time, "duration_s": e.duration, "peak_perclos": e.peak_perclos}
                for e in self._events
            ],
        }

    def write_csv(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)

    def write_summary_json(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.summary(), indent=2))

    def to_csv_string(self) -> str:
        """In-memory CSV text — used by the Streamlit dashboard's download button."""
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(self._rows)
        return buf.getvalue()
