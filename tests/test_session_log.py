import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness.detector import FrameResult
from drowsiness.session_log import SessionLogger


def _result(drowsy: bool, perclos: float = 0.0) -> FrameResult:
    return FrameResult(
        face_found=True, ear=0.2, mar=0.3, eyes_closed=drowsy,
        yawning=False, perclos=perclos, drowsy=drowsy,
    )


def test_logs_one_row_per_frame():
    logger = SessionLogger()
    logger.log_frame(0.0, _result(False))
    logger.log_frame(0.1, _result(False))
    assert len(logger.rows) == 2


def test_detects_single_drowsy_event():
    logger = SessionLogger()
    logger.log_frame(0.0, _result(False))
    logger.log_frame(1.0, _result(True, perclos=0.5))
    logger.log_frame(2.0, _result(True, perclos=0.8))
    logger.log_frame(3.0, _result(False))
    logger.close(4.0)

    events = logger.events
    assert len(events) == 1
    assert events[0].start_time == 1.0
    assert events[0].end_time == 3.0
    assert events[0].peak_perclos == 0.8


def test_open_event_is_closed_at_session_end():
    logger = SessionLogger()
    logger.log_frame(0.0, _result(True, perclos=0.6))
    logger.close(5.0)

    assert len(logger.events) == 1
    assert logger.events[0].end_time == 5.0


def test_summary_counts_events_and_duration():
    logger = SessionLogger()
    logger.log_frame(0.0, _result(False))
    logger.log_frame(1.0, _result(True, perclos=0.5))
    logger.log_frame(2.0, _result(False))
    logger.close(3.0)

    summary = logger.summary()
    assert summary["drowsy_event_count"] == 1
    assert summary["total_frames"] == 3


def test_write_csv_creates_file(tmp_path):
    logger = SessionLogger()
    logger.log_frame(0.0, _result(False))
    out = tmp_path / "nested" / "session.csv"
    logger.write_csv(str(out))
    assert out.exists()
    content = out.read_text()
    assert "timestamp" in content
    assert "drowsy" in content
