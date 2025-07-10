🚗 Real-Time Driver Drowsiness Detection Dashboard
A real-time drowsiness detection web app that monitors a driver’s alertness level using facial landmarks, and raises an alarm if drowsiness is detected.
Built using Python, OpenCV, dlib, face_recognition, and Streamlit.

📋 Features
✅ Detects closed eyes & yawning using EAR (Eye Aspect Ratio) & MAR (Mouth Aspect Ratio).
✅ Streamlit web interface with interactive threshold controls.
✅ Works on uploaded video file.
✅ Displays live video with real-time overlay of score and drowsiness warning.
✅ Plays audible alarm when thresholds are exceeded.
✅ Logs performance metrics (number of frames, % of drowsy frames).

🧰 Tech Stack
Python 3.x
OpenCV
face_recognition (dlib-based)
SciPy
playsound
Streamlit

🖥️ Demo
📷 Video upload:
Upload a video of a person, and the app will analyze it in real time, highlighting when the person appears drowsy.

📊 Metrics logged:
Total frames processed
% of frames where drowsiness detected