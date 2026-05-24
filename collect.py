"""
collect.py — offline data collection
======================================
Run this on a video file of yourself doing squats/deadlifts.
It extracts 6 joint angles per frame using MediaPipe and saves them to a CSV.
You then manually add a "label" column (0=good, 1=knee_valgus, 2=rounded_back, 3=both).

Usage:
    python collect.py --video myvideo.mp4 --output data.csv

NOTE: The first run downloads pose_landmarker_full.task (~6 MB) automatically.
"""

import cv2
import csv
import numpy as np
import argparse
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Model download ────────────────────────────────────────────────────────────
# The new MediaPipe Tasks API requires a separate .task model file.
# We download it once and reuse it on subsequent runs.
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[collect] Downloading pose model (~6 MB) — one-time setup...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[collect] Model saved to {MODEL_PATH}")

# ── MediaPipe detector ────────────────────────────────────────────────────────
def make_detector():
    """
    RunningMode.IMAGE means each frame is processed independently —
    no tracking state carried between frames (same idea as static_image_mode=True before).
    """
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)

# ── Angle math ────────────────────────────────────────────────────────────────
def angle_at(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

def extract_angles(kp):
    return {
        "left_knee":  angle_at(kp[23], kp[25], kp[27]),
        "right_knee": angle_at(kp[24], kp[26], kp[28]),
        "back_angle": angle_at(kp[11], kp[23], kp[25]),
        "hip_depth":  float(kp[23][1] - kp[25][1]),
        "left_shin":  angle_at(kp[25], kp[27], kp[29]),
        "torso_lean": angle_at(kp[11], kp[23], [kp[23][0], 0.0]),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def process_video(video_path, output_csv, sample_every=3):
    ensure_model()
    detector = make_detector()

    cap = cv2.VideoCapture(video_path)
    rows = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            # MediaPipe Tasks API needs an mp.Image wrapping an RGB numpy array
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            # result.pose_landmarks is a list of people; [0] = first person
            if result.pose_landmarks:
                kp = [[lm.x, lm.y] for lm in result.pose_landmarks[0]]
                angles = extract_angles(kp)
                angles["frame"] = frame_idx
                angles["label"] = ""   # fill this in manually after
                rows.append(angles)

        frame_idx += 1

    cap.release()

    if rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[collect] Saved {len(rows)} frames to {output_csv}")
        print(f"[collect] Open the CSV and fill in the 'label' column:")
        print(f"          0 = good form")
        print(f"          1 = knee valgus (knees caving in)")
        print(f"          2 = rounded back")
        print(f"          3 = both faults")
    else:
        print("[collect] No poses detected — check that a person is visible in the video.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",        required=True,      help="Path to input video file")
    parser.add_argument("--output",       default="data.csv", help="Output CSV path")
    parser.add_argument("--sample-every", type=int, default=3, help="Extract every Nth frame")
    args = parser.parse_args()
    process_video(args.video, args.output, args.sample_every)
