"""Alarm playback with a cooldown so we don't spawn a new sound thread
on every single frame once the drowsy score crosses threshold (the bug
in the original script)."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional


class AlarmPlayer:
    def __init__(self, sound_path: str, cooldown_seconds: float = 3.0) -> None:
        self.sound_path = sound_path
        self.cooldown_seconds = cooldown_seconds
        self._last_played: float = 0.0
        self._lock = threading.Lock()
        self._active_thread: Optional[threading.Thread] = None

    def _play(self) -> None:
        try:
            from playsound import playsound

            playsound(self.sound_path)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[alarm] failed to play sound: {exc}")

    def trigger(self) -> bool:
        """Play the alarm if cooldown has elapsed. Returns True if it fired."""
        now = time.time()
        with self._lock:
            if now - self._last_played < self.cooldown_seconds:
                return False
            if self._active_thread is not None and self._active_thread.is_alive():
                return False
            if not Path(self.sound_path).exists():
                print(f"[alarm] sound file not found: {self.sound_path}")
                return False
            self._last_played = now
            self._active_thread = threading.Thread(target=self._play, daemon=True)
            self._active_thread.start()
            return True
