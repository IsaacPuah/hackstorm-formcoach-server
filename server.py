"""
FormCoach — Flask inference server
===================================
The T5AI board POSTs a JPEG frame to /analyze every ~1.5s.
We run MediaPipe Pose on it, extract joint angles, classify form,
and return JSON that the board uses to color the skeleton overlay.

How MediaPipe Pose works (quick mental model):
  - It's a pretrained neural net from Google that finds 33 body landmarks
    in any image — things like left hip, right knee, left ankle, etc.
  - Each landmark comes back as (x, y) normalized to 0-1 across the image.
    So x=0.5, y=0.5 means dead center of the image.
  - We don't train MediaPipe — it's a frozen feature extractor.
    Our trained classifier sits on top, using the angles between those landmarks.

JSON response shape (what the firmware expects):
  {
    "keypoints":  [[x, y], ...],   // 33 pairs, normalized 0-1
    "label":      "good" | "knee_valgus" | "rounded_back" | "both" | "unknown",
    "issues":     ["left_knee", "spine", ...],   // joints to color red
    "confidence": 0.91
  }

NOTE: The first run downloads pose_landmarker_full.task (~6 MB) automatically.
"""

import cv2
import numpy as np
import os
import urllib.request
from flask import Flask, request, jsonify

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Model download ────────────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[FormCoach] Downloading pose model (~6 MB) — one-time setup...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[FormCoach] Model saved to {MODEL_PATH}")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
# RunningMode.IMAGE: each frame is independent, no tracking state between frames.
# This is what we want — each JPEG from the board is a standalone snapshot.
ensure_model()
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
pose_detector = vision.PoseLandmarker.create_from_options(options)

# ── Optional: load trained classifier if it exists ───────────────────────────
# The model won't exist until you run train.py, so we handle that gracefully.
# While the model is missing the server still works — it returns real keypoints
# but leaves "issues" empty (everything stays green on the board display).
MODEL_LOADED = False
classifier = None

try:
    import joblib
    classifier = joblib.load("formcoach_model.pkl")
    MODEL_LOADED = True
    print("[FormCoach] Classifier loaded — full form analysis active.")
except FileNotFoundError:
    print("[FormCoach] No classifier found — returning keypoints only.")
    print("[FormCoach] Run train.py once you have labeled data to enable classification.")

# ── Label and issue mappings ──────────────────────────────────────────────────
LABELS = {
    0: "good",
    1: "knee_valgus",
    2: "rounded_back",
    3: "both",
}

ISSUE_MAP = {
    0: [],
    1: ["left_knee", "right_knee"],
    2: ["spine"],
    3: ["left_knee", "right_knee", "spine"],
}

# ── Feature extraction ────────────────────────────────────────────────────────
# MediaPipe keypoint indices (the ones we care about):
#   11 = left shoulder    12 = right shoulder
#   23 = left hip         24 = right hip
#   25 = left knee        26 = right knee
#   27 = left ankle       28 = right ankle
#   29 = left foot        30 = right foot

def angle_at(a, b, c):
    """
    Returns the angle (degrees) at point b, between rays b->a and b->c.
    Think of b as the joint — e.g. knee — and a, c as the bones on either side.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

def extract_angles(kp):
    """
    kp: list of 33 [x, y] pairs from MediaPipe (normalized 0-1).
    Returns a list of 6 floats — the feature vector the classifier sees.
    """
    return [
        angle_at(kp[23], kp[25], kp[27]),            # left_knee:  hip -> knee -> ankle
        angle_at(kp[24], kp[26], kp[28]),            # right_knee: hip -> knee -> ankle
        angle_at(kp[11], kp[23], kp[25]),            # back_angle: shoulder -> hip -> knee
        float(kp[23][1] - kp[25][1]),                # hip_depth:  hip Y minus knee Y (squat depth)
        angle_at(kp[25], kp[27], kp[29]),            # left_shin:  knee -> ankle -> foot
        angle_at(kp[11], kp[23], [kp[23][0], 0.0])  # torso_lean: torso vs vertical axis
    ]

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main endpoint. The T5AI board sends a raw JPEG body (Content-Type: image/jpeg).
    We decode it, run pose estimation, classify form, and return JSON.
    """
    # Step 1: decode the raw JPEG bytes into an OpenCV image (numpy array)
    img_bytes = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "could not decode image"}), 400

    # Step 2: MediaPipe Tasks API needs an mp.Image wrapping an RGB numpy array
    # OpenCV loads BGR by default, so convert to RGB first
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose_detector.detect(mp_image)

    # Step 3: if no person detected, tell the board (it will clear the overlay)
    if not result.pose_landmarks:
        return jsonify({"error": "no pose detected"}), 200

    # Step 4: pull out the 33 keypoints as normalized [x, y] pairs
    # result.pose_landmarks is a list of people; [0] = first person detected
    kp = [[lm.x, lm.y] for lm in result.pose_landmarks[0]]

    # Step 5: classify if model is ready, otherwise return keypoints only
    if MODEL_LOADED:
        features   = extract_angles(kp)
        label_idx  = int(classifier.predict([features])[0])
        confidence = float(classifier.predict_proba([features]).max())
        label      = LABELS[label_idx]
        issues     = ISSUE_MAP.get(label_idx, [])
    else:
        label      = "unknown"
        issues     = []
        confidence = 0.0

    return jsonify({
        "keypoints":  kp,
        "label":      label,
        "issues":     issues,
        "confidence": round(confidence, 3),
    })

@app.route("/health", methods=["GET"])
def health():
    """Quick check — open http://localhost:5000/health in a browser to verify it's running."""
    return jsonify({
        "status":           "ok",
        "model_loaded":     MODEL_LOADED,
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    args = parser.parse_args()
    print(f"[FormCoach] Server starting on http://0.0.0.0:{args.port}")
    print(f"[FormCoach] POST a JPEG to /analyze — GET /health to check status")
    app.run(host="0.0.0.0", port=args.port, debug=False)
