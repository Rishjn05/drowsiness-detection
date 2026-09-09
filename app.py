"""Streamlit dashboard — the thing the original README advertised but the
script never actually implemented (it just called cv2.imshow in a loop).

Run with: streamlit run app.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from src.drowsiness.alarm import AlarmPlayer
from src.drowsiness.detector import DetectorConfig, DrowsinessDetector
from src.drowsiness.session_log import SessionLogger

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("Driver Drowsiness Detection")

with st.sidebar:
    st.header("Settings")
    source_type = st.radio("Video source", ["Upload video", "Webcam"])
    ear_threshold = st.slider("EAR threshold (lower = stricter)", 0.10, 0.35, 0.21, 0.01)
    mar_threshold = st.slider("MAR (yawn) threshold", 0.3, 1.0, 0.6, 0.05)
    perclos_ratio = st.slider("PERCLOS drowsy ratio", 0.1, 0.9, 0.4, 0.05)
    alarm_cooldown = st.slider("Alarm cooldown (s)", 1.0, 10.0, 3.0, 0.5)
    alarm_file = st.text_input("Alarm sound path", "assets/alarm.wav")
    run_button = st.button("Start")

col_video, col_stats = st.columns([3, 1])
frame_placeholder = col_video.empty()
stats_placeholder = col_stats.empty()
chart_placeholder = st.empty()

video_path = None
if source_type == "Upload video":
    uploaded = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.close()
        video_path = tmp.name


def get_capture():
    if source_type == "Webcam":
        return cv2.VideoCapture(0)
    if video_path:
        return cv2.VideoCapture(video_path)
    return None


if run_button:
    cap = get_capture()
    if cap is None or not cap.isOpened():
        st.error("No video source available. Upload a video or enable webcam access.")
    else:
        config = DetectorConfig(
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold,
            perclos_drowsy_ratio=perclos_ratio,
        )
        detector = DrowsinessDetector(config)
        alarm = AlarmPlayer(alarm_file, cooldown_seconds=alarm_cooldown)
        logger = SessionLogger()
        session_start = time.time()

        perclos_history = []
        frame_count = 0
        drowsy_count = 0
        stop = st.sidebar.button("Stop")

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                frame = cv2.resize(frame, (800, 500))
                result = detector.process(frame)
                perclos_history.append(result.perclos)
                logger.log_frame(time.time() - session_start, result)

                if result.drowsy:
                    drowsy_count += 1
                    alarm.trigger()

                color = (0, 0, 255) if result.drowsy else (0, 200, 0)
                label = "DROWSY" if result.drowsy else "Alert"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                frame_placeholder.image(frame, channels="BGR")
                stats_placeholder.metric(
                    "PERCLOS", f"{result.perclos:.2f}",
                    delta="DROWSY" if result.drowsy else "alert",
                )
                if frame_count % 5 == 0:
                    chart_placeholder.line_chart(perclos_history[-150:])
                time.sleep(0.01)
        finally:
            detector.close()
            cap.release()

        logger.close(time.time() - session_start)
        pct = (drowsy_count / frame_count * 100) if frame_count else 0
        st.success(f"Done. {frame_count} frames processed, {pct:.1f}% flagged drowsy.")

        summary = logger.summary()
        st.subheader("Session summary")
        st.json(summary)
        st.download_button(
            "Download session log (CSV)",
            data=logger.to_csv_string(),
            file_name="drowsiness_session.csv",
            mime="text/csv",
        )
else:
    st.info("Choose a source in the sidebar and click Start.")
