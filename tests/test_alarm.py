import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness.alarm import AlarmPlayer


def test_alarm_skips_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.wav"
    player = AlarmPlayer(str(missing), cooldown_seconds=0.1)
    assert player.trigger() is False


def test_alarm_respects_cooldown(tmp_path):
    sound = tmp_path / "alarm.wav"
    sound.write_bytes(b"fake-wav-bytes")
    player = AlarmPlayer(str(sound), cooldown_seconds=1.0)
    player._play = lambda: None  # skip real audio playback in tests

    first = player.trigger()
    second = player.trigger()  # should be blocked by cooldown

    assert first is True
    assert second is False


def test_alarm_fires_again_after_cooldown(tmp_path):
    sound = tmp_path / "alarm.wav"
    sound.write_bytes(b"fake-wav-bytes")
    player = AlarmPlayer(str(sound), cooldown_seconds=0.2)
    player._play = lambda: None

    assert player.trigger() is True
    time.sleep(0.25)
    assert player.trigger() is True
