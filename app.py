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
    """Logs to file AND prints to console"""
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
source = st.radio(
    "📥 Select Input Source",
    ["Upload Video", "Use Webcam (Local Only)"]
)

video_box = st.empty()
alert_box = st.empty()
dashboard_placeholder = st.empty()

# =========================
# VIDEO CAPTURE
# =========================
cap = None
webcam_allowed = True
uploaded_temp_file = None

if source == "Upload Video":
    uploaded_video = st.file_uploader("Upload CCTV / Crowd Video", type=["mp4", "avi"])
    if uploaded_video:
        uploaded_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        uploaded_temp_file.write(uploaded_video.read())
        cap = cv2.VideoCapture(uploaded_temp_file.name)
else:
    if IS_CLOUD:
        webcam_allowed = False
        st.warning("""
🚫 **Webcam Disabled on Streamlit Cloud**  
Use a video upload for demo or run locally for live camera.
""")
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

    # Rolling average buffer to smooth sudden spikes
    motion_buffer = []

    frame_count = 0
    spike_count = 0
    alerts_count = 0
    low_risk_frames = 0
    risk_confidences = []
    start_time = time.time()

    # Temp file to store processed video for Streamlit cloud
    processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_path, fourcc, 20.0,
                          (prev_frame.shape[1], prev_frame.shape[0]))

    # =========================
    # MAIN LOOP
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = float(np.mean(mag))

        # ---------- Smooth motion ----------
        motion_buffer.append(avg_motion)
        if len(motion_buffer) > 5:
            motion_buffer.pop(0)
        smooth_motion = np.mean(motion_buffer)

        # ---------- Spike Detection ----------
        spike = smooth_motion - prev_motion
        spike_detected = spike > SPIKE_THRESHOLD
        spike_explanation = "No sudden spike." if not spike_detected else "Sudden rush detected! High risk of panic or stampede."

        if spike_detected:
            spike_count += 1

        # ---------- Zone Analysis ----------
        h, w = gray.shape
        zones = {
            "Zone 1 (Top-Left)": np.mean(mag[:h//2, :w//2]),
            "Zone 2 (Top-Right)": np.mean(mag[:h//2, w//2:]),
            "Zone 3 (Bottom-Left)": np.mean(mag[h//2:, :w//2]),
            "Zone 4 (Bottom-Right)": np.mean(mag[h//2:, w//2:])
        }
        active_zone = max(zones, key=zones.get)

        # ---------- Risk Classification ----------
        if smooth_motion < VERY_LOW:
            risk = "VERY LOW"
            confidence = 0.95
            explanation = "Crowd is calm, safe movement."
        elif smooth_motion < NORMAL:
            risk = "NORMAL"
            confidence = 0.85
            explanation = "Crowd is moving normally."
        elif smooth_motion < ELEVATED:
            risk = "ELEVATED"
            confidence = 0.70
            explanation = "Crowd is getting dense, monitor carefully."
        elif smooth_motion < HIGH:
            risk = "HIGH RISK"
            confidence = 0.60
            explanation = "Crowd is moving fast, prepare staff."
        else:
            risk = "CRITICAL"
            confidence = 0.90
            explanation = "Crowd movement is dangerous, act immediately!"

        if spike_detected:
            risk = "CRITICAL"
            confidence = 0.99
            explanation = "Sudden rush detected! High risk of panic or stampede."

        risk_confidences.append(confidence)

        # ---------- Action ----------
        if risk == "CRITICAL":
            action = "IMMEDIATE ACTION: Open exits, stop inflow, alert authorities"
        elif risk == "HIGH RISK":
            action = "PREVENTIVE ACTION: Control entry, deploy staff"
        elif risk == "ELEVATED":
            action = "MONITOR: Crowd density rising"
        else:
            action = "SAFE"

        # ---------- Logging ----------
        log_message = (
            f"{datetime.now()} | {risk} | {active_zone} | "
            f"Motion={smooth_motion:.2f} | Spike={'YES' if spike_detected else 'NO'} | "
            f"Confidence={confidence:.2f} | {action} | Explanation: {explanation} | Spike Info: {spike_explanation}"
        )
        if risk in ["HIGH RISK", "CRITICAL"]:
            alerts_count += 1
            log_and_print(log_message, "WARNING")
            alert_box.warning(log_message)
        else:
            low_risk_frames += 1
            log_and_print(log_message, "INFO")

        # ---------- Overlay ----------
        cv2.putText(frame, f"Risk: {risk} ({confidence:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Active Zone: {active_zone}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Spike: {'YES' if spike_detected else 'NO'}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ---------- Write frame to output video ----------
        out.write(frame)

        prev_gray = gray
        prev_motion = smooth_motion
        frame_count += 1

    cap.release()
    out.release()
    fps = frame_count / (time.time() - start_time)
    avg_confidence = np.mean(risk_confidences) if risk_confidences else 0.0

    log_and_print(
        f"SYSTEM STOPPED | Frames={frame_count} | Alerts={alerts_count} | "
        f"LowRiskFrames={low_risk_frames} | AvgConfidence={avg_confidence:.2f} | FPS={fps:.2f}"
    )

    # =========================
    # PLAY PROCESSED VIDEO
    # =========================
    st.subheader("📹 Processed Crowd Video")
    st.video(processed_video_path)

    # =========================
    # FINAL BEGINNER-FRIENDLY SUMMARY WITH MORE EXPLANATION
    # =========================
    st.success(f"""
## ✅ Final Crowd Safety Report

### ❓ What happened?
The system watched **crowd movement over time** and detected:
• Fast movements → Measures how quickly the crowd is moving  
• Sudden spikes → Sudden rushes or panics detected via abrupt motion changes  
• Most active zones → Areas with highest crowd activity

### 📝 Alerts Explanation
- **VERY LOW / NORMAL:** Crowd is calm, no risk  
- **ELEVATED:** Crowd is getting denser, monitor closely  
- **HIGH RISK:** Fast moving or dense crowd, preventive actions needed  
- **CRITICAL:** Unsafe, immediate actions like opening exits or alerting authorities

### 🚨 Metrics Explained
- **Frames:** Number of video frames processed (higher = longer video analyzed)  
- **Motion:** Average crowd movement magnitude, higher means faster movement  
- **Spikes:** Number of sudden rushes detected (possible panic)  
- **Alerts:** High / critical risk frames logged  
- **Confidence:** System confidence (0–1) in risk estimation  
- **Active Zone:** The zone currently most crowded

### 🧭 Recommended Actions
1️⃣ Slow / stop entry if alerts are HIGH or CRITICAL  
2️⃣ Open exits and send staff to active zone  
3️⃣ Always follow instructions in **logs**

### 📁 Logs
• File: **{LOG_FILE}**  
• Contains timestamp, risk, active zone, motion, spike, confidence, action, explanation  
• Useful for audits or post-event analysis

### ⚙️ System Health
• Processing speed: **{fps:.2f} FPS**  
• Video display is now playable on Streamlit cloud
""")

# =========================
# LOG VIEWER
# =========================
st.markdown("---")
st.subheader("📄 System Logs")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", errors="replace") as f:
        log_text = f.read()
    st.text_area("Log Output (Read-Only)", log_text, height=250)
    st.download_button("⬇️ Download Logs", log_text, file_name="crowd_alerts.log")
else:
    st.info("No logs generated yet.")
