import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import logging
import os
from datetime import datetime

# =========================
# ENVIRONMENT DETECTION
# =========================
IS_CLOUD = os.environ.get("STREAMLIT_CLOUD") == "true"

# =========================
# LOGGING SETUP
# =========================
LOG_FILE = "crowd_alerts.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log_and_print(message, level="INFO"):
    if level == "WARNING":
        logging.warning(message)
    else:
        logging.info(message)
    print(message)

# =========================
# THRESHOLDS
# =========================
VERY_LOW = 0.60
NORMAL = 1.20
ELEVATED = 2.50
HIGH = 4.00
CRITICAL = 6.00
SPIKE_THRESHOLD = 2.0

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Crowd Risk Detection", layout="wide")
st.title("🚨 Crowd Behavior Risk Detection System")

st.markdown("""
### 👤 Beginner-Friendly Overview
This system detects **unsafe crowd movement**, not people.

✔ No face detection  
✔ No personal tracking  
✔ Crowd motion only  

📌 **Logs are the MAIN output. Dashboard is visual help only.**
""")

# =========================
# INPUT SOURCE
# =========================
source = st.radio("📥 Select Input Source", ["Upload Video", "Use Webcam (Local Only)"])
video_box = st.empty()
alert_box = st.empty()
dashboard_placeholder = st.empty()

cap = None
webcam_allowed = True
uploaded_temp_file = None

if source == "Upload Video":
    uploaded_video = st.file_uploader("Upload CCTV / Crowd Video", type=["mp4", "avi"])
    if uploaded_video:
        uploaded_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        uploaded_temp_file.write(uploaded_video.read())
        uploaded_temp_file.close()
        cap = cv2.VideoCapture(uploaded_temp_file.name)
else:
    if IS_CLOUD:
        webcam_allowed = False
        st.warning("🚫 Webcam disabled on Streamlit Cloud. Upload a video instead.")
    else:
        cap = cv2.VideoCapture(0)

# =========================
# START DETECTION
# =========================
if st.button("▶ Start Detection") and cap is not None and webcam_allowed:
    ret, prev_frame = cap.read()
    if not ret:
        st.error("❌ Unable to read video source")
        st.stop()

    log_and_print("SYSTEM STARTED | Crowd analysis running")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_motion = 0.0
    motion_buffer = []

    frame_count = 0
    spike_count = 0
    alerts_count = 0
    low_risk_frames = 0
    risk_confidences = []
    start_time = time.time()

    # =========================
    # LIVE VIDEO LOOP
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = float(np.mean(mag))

        # Smooth motion
        motion_buffer.append(avg_motion)
        if len(motion_buffer) > 5:
            motion_buffer.pop(0)
        smooth_motion = np.mean(motion_buffer)

        # Spike detection
        spike = smooth_motion - prev_motion
        spike_detected = spike > SPIKE_THRESHOLD
        if spike_detected:
            spike_count += 1
        spike_explanation = "Sudden rush!" if spike_detected else "No spike."

        # Zone analysis
        h, w = gray.shape
        zones = {
            "Zone 1 (Top-Left)": np.mean(mag[:h//2, :w//2]),
            "Zone 2 (Top-Right)": np.mean(mag[:h//2, w//2:]),
            "Zone 3 (Bottom-Left)": np.mean(mag[h//2:, :w//2]),
            "Zone 4 (Bottom-Right)": np.mean(mag[h//2:, w//2:])
        }
        active_zone = max(zones, key=zones.get)

        # Risk classification
        if smooth_motion < VERY_LOW:
            risk, confidence, explanation = "VERY LOW", 0.95, "Crowd is calm."
        elif smooth_motion < NORMAL:
            risk, confidence, explanation = "NORMAL", 0.85, "Crowd moving normally."
        elif smooth_motion < ELEVATED:
            risk, confidence, explanation = "ELEVATED", 0.70, "Crowd getting dense."
        elif smooth_motion < HIGH:
            risk, confidence, explanation = "HIGH RISK", 0.60, "Fast moving crowd."
        else:
            risk, confidence, explanation = "CRITICAL", 0.90, "Dangerous movement!"

        if spike_detected:
            risk, confidence, explanation = "CRITICAL", 0.99, "Sudden rush detected!"

        risk_confidences.append(confidence)

        # Action based on risk
        if risk == "CRITICAL":
            action = "IMMEDIATE ACTION: Open exits, alert authorities"
        elif risk == "HIGH RISK":
            action = "PREVENTIVE ACTION: Deploy staff"
        elif risk == "ELEVATED":
            action = "MONITOR: Crowd density rising"
        else:
            action = "SAFE"

        # Logging
        log_message = f"{datetime.now()} | {risk} | {active_zone} | Motion={smooth_motion:.2f} | Spike={'YES' if spike_detected else 'NO'} | Confidence={confidence:.2f} | {action} | {explanation}"
        if risk in ["HIGH RISK", "CRITICAL"]:
            alerts_count += 1
            log_and_print(log_message, "WARNING")
            alert_box.warning(log_message)
        else:
            low_risk_frames += 1
            log_and_print(log_message)

        # Overlay text
        cv2.putText(frame, f"Risk: {risk} ({confidence:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Active Zone: {active_zone}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Spike: {'YES' if spike_detected else 'NO'}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frame live in Streamlit
        video_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Update dashboard
        with dashboard_placeholder.container():
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Frames", frame_count)
            c2.metric("Motion", f"{smooth_motion:.2f}")
            c3.metric("Spikes", spike_count)
            c4.metric("Alerts", alerts_count)
            c5.metric("Confidence", f"{np.mean(risk_confidences):.2f}")
            c6.metric("Active Zone", active_zone)

        prev_gray = gray
        prev_motion = smooth_motion
        frame_count += 1
        time.sleep(0.03)  # simulate ~30 FPS live playback

    cap.release()
    fps = frame_count / (time.time() - start_time)

    st.success(f"✅ Finished! Processed {frame_count} frames at {fps:.2f} FPS.")
    st.markdown(f"**Total Alerts:** {alerts_count}, **Spikes:** {spike_count}, **Avg Confidence:** {np.mean(risk_confidences):.2f}")

