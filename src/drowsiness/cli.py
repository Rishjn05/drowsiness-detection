"""Command-line driver: run on webcam or a video file, no notebook cells."""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from .alarm import AlarmPlayer
from .calibration import calibrate
from .detector import DetectorConfig, DrowsinessDetector
from .session_log import SessionLogger


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time drowsiness detection")
    p.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0) or path to a video file",
    )
    p.add_argument("--alarm-file", default="assets/alarm.wav", help="Path to alarm sound")
    p.add_argument("--alarm-cooldown", type=float, default=3.0, help="Seconds between alarm triggers")
    p.add_argument("--calibrate-seconds", type=float, default=3.0, help="Baseline calibration window; 0 to skip")
    p.add_argument("--ear-threshold", type=float, default=None, help="Override EAR threshold (skips calibration)")
    p.add_argument("--mar-threshold", type=float, default=None, help="Override MAR threshold (skips calibration)")
    p.add_argument("--no-display", action="store_true", help="Run headless (no cv2.imshow window)")
    p.add_argument(
        "--log-dir",
        default=None,
        help="Directory to write a per-frame CSV + JSON event summary. Omit to skip logging.",
    )
    return p.parse_args(argv)


def _open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def run(argv=None) -> int:
    args = parse_args(argv)
    cap = _open_source(args.source)
    if not cap.isOpened():
        print(f"Could not open video source: {args.source}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    config = DetectorConfig(fps_estimate=int(fps) or 15)

    detector = DrowsinessDetector(config)
    alarm = AlarmPlayer(args.alarm_file, cooldown_seconds=args.alarm_cooldown)

    if args.ear_threshold is not None:
        config.ear_threshold = args.ear_threshold
    if args.mar_threshold is not None:
        config.mar_threshold = args.mar_threshold

    if args.calibrate_seconds > 0 and args.ear_threshold is None and args.mar_threshold is None:
        print(f"Calibrating for {args.calibrate_seconds:.0f}s — look at the camera normally...")
        ear_samples, mar_samples = [], []
        start = time.time()
        while time.time() - start < args.calibrate_seconds:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.process(frame)
            if result.face_found and result.ear is not None:
                ear_samples.append(result.ear)
                mar_samples.append(result.mar)
        if ear_samples:
            thresholds = calibrate(ear_samples, mar_samples)
            config.ear_threshold = thresholds.ear_threshold
            config.mar_threshold = thresholds.mar_threshold
            print(f"Calibrated: EAR<{config.ear_threshold} MAR>{config.mar_threshold}")
        else:
            print("No face detected during calibration; using defaults.")

    frame_count = 0
    drowsy_frames = 0
    start_time = time.time()
    logger = SessionLogger() if args.log_dir else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            frame = cv2.resize(frame, (800, 500))
            result = detector.process(frame)
            elapsed = time.time() - start_time

            if logger is not None:
                logger.log_frame(elapsed, result)

            if result.drowsy:
                drowsy_frames += 1
                alarm.trigger()

            if not args.no_display:
                _annotate(frame, result)
                cv2.imshow("Drowsiness detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                # Detect the window being closed via the titlebar X button —
                # waitKey alone doesn't see that, so the loop would otherwise
                # keep running (or spawn a "not responding" window) forever.
                if cv2.getWindowProperty("Drowsiness detection", cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    duration = time.time() - start_time
    pct = (drowsy_frames / frame_count * 100) if frame_count else 0.0
    print(f"Processed {frame_count} frames in {duration:.2f}s")
    print(f"Drowsy frames: {drowsy_frames} ({pct:.2f}%)")

    if logger is not None:
        logger.close(duration)
        from pathlib import Path
        import time as _time

        stamp = _time.strftime("%Y%m%d-%H%M%S")
        csv_path = Path(args.log_dir) / f"session-{stamp}.csv"
        json_path = Path(args.log_dir) / f"session-{stamp}-summary.json"
        logger.write_csv(str(csv_path))
        logger.write_summary_json(str(json_path))
        summary = logger.summary()
        print(f"Logged {summary['total_frames']} frames -> {csv_path}")
        print(f"Drowsy events: {summary['drowsy_event_count']} totalling {summary['total_drowsy_time_s']}s -> {json_path}")

    return 0


def _annotate(frame, result) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not result.face_found:
        cv2.putText(frame, "No face detected", (10, frame.shape[0] - 10), font, 0.8, (0, 165, 255), 2)
        return
    perclos_text = f"PERCLOS: {result.perclos:.2f}  EAR: {result.ear:.2f}  MAR: {result.mar:.2f}"
    cv2.putText(frame, perclos_text, (10, frame.shape[0] - 10), font, 0.6, (255, 255, 255), 1)
    if result.drowsy:
        cv2.putText(frame, "DROWSY", (frame.shape[1] - 160, 40), font, 1, (0, 0, 255), 2)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
