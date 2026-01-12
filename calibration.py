import cv2
import numpy as np
import os

# =========================
# VIDEO LIST (ADD ALL HERE)
# =========================
VIDEO_FOLDER = "videos"

VIDEO_FILES = [
    "normal1.mp4",
    "normal2.mp4",
    "normal3.mp4",
    "risk1.mp4",
    "risk2.mp4"
]

# =========================
# STORAGE
# =========================
all_avg_movements = []
all_max_movements = []

print("\n📊 CROWD CALIBRATION STARTED\n")

# =========================
# PROCESS EACH VIDEO
# =========================
for video_name in VIDEO_FILES:
    video_path = os.path.join(VIDEO_FOLDER, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open {video_name}")
        continue

    ret, prev_frame = cap.read()
    if not ret:
        print(f"❌ Could not read first frame of {video_name}")
        cap.release()
        continue

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    motion_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = np.mean(magnitude)

        motion_values.append(avg_motion)

        prev_gray = gray
        frame_count += 1

    cap.release()

    if len(motion_values) == 0:
        print(f"⚠️ No data for {video_name}")
        continue

    min_motion = np.min(motion_values)
    avg_motion = np.mean(motion_values)
    max_motion = np.max(motion_values)

    all_avg_movements.append(avg_motion)
    all_max_movements.append(max_motion)

    print(f"🎥 {video_name}")
    print(f"Frames analysed : {frame_count}")
    print(f"Min movement   : {min_motion:.2f}")
    print(f"Avg movement   : {avg_motion:.2f}")
    print(f"Max movement   : {max_motion:.2f}")
    print("-" * 40)

# =========================
# GLOBAL THRESHOLD CALCULATION
# =========================
print("\n✅ FINAL CALIBRATION SUMMARY\n")

global_avg = np.mean(all_avg_movements)
global_max = np.mean(all_max_movements)

VERY_LOW = global_avg * 0.5
NORMAL = global_avg
ELEVATED = global_avg * 2
HIGH = global_avg * 4
CRITICAL = global_max * 0.4

print("📌 GLOBAL STATISTICS")
print(f"Average of averages : {global_avg:.2f}")
print(f"Average of maximums : {global_max:.2f}\n")

print("🚦 SUGGESTED FINAL THRESHOLDS")
print(f"VERY_LOW   = {VERY_LOW:.2f}")
print(f"NORMAL     = {NORMAL:.2f}")
print(f"ELEVATED   = {ELEVATED:.2f}")
print(f"HIGH       = {HIGH:.2f}")
print(f"CRITICAL   = {CRITICAL:.2f}")

print("\n🎯 Calibration completed successfully.")
print("➡️ Use these thresholds in detection.py / app.py")
