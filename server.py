"""
FormCoach — Flask inference server
"""

import cv2
import numpy as np
import pandas as pd
import os
import urllib.request
import base64
from flask import Flask, request, jsonify

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Response
import time

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

ensure_model()
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
pose_detector = vision.PoseLandmarker.create_from_options(options)

MODEL_LOADED = False
classifier = None

try:
    import joblib
    classifier = joblib.load("formcoach_model.pkl")
    MODEL_LOADED = True
    print("[FormCoach] Classifier loaded — full form analysis active.")
except FileNotFoundError:
    print("[FormCoach] No classifier found — returning keypoints only.")

LABELS = {0: "good", 1: "knee_valgus", 2: "rounded_back", 3: "both"}
ISSUE_MAP = {
    0: [],
    1: ["left_knee", "right_knee"],
    2: ["spine"],
    3: ["left_knee", "right_knee", "spine"],
}

SKELETON_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]

def angle_at(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

def extract_angles(kp):
    return [
        angle_at(kp[23], kp[25], kp[27]),
        angle_at(kp[24], kp[26], kp[28]),
        angle_at(kp[11], kp[23], kp[25]),
        float(kp[23][1] - kp[25][1]),
        angle_at(kp[25], kp[27], kp[29]),
        angle_at(kp[11], kp[23], [kp[23][0], 0.0])
    ]

app = Flask(__name__)
latest_annotated = None
latest_label = "unknown"

@app.route("/analyze", methods=["POST"])
def analyze():
    global latest_annotated, latest_label

    img_bytes = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "could not decode image"}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose_detector.detect(mp_image)

    if not result.pose_landmarks:
        return jsonify({"error": "no pose detected"}), 200

    kp = [[lm.x, lm.y] for lm in result.pose_landmarks[0]]

    if MODEL_LOADED:
        FEATURE_NAMES = ["left_knee","right_knee","back_angle","hip_depth","left_shin","torso_lean"]
        features   = pd.DataFrame([extract_angles(kp)], columns=FEATURE_NAMES)
        label_idx  = int(classifier.predict(features)[0])
        confidence = float(classifier.predict_proba(features).max())
        label      = LABELS[label_idx]
        issues     = ISSUE_MAP.get(label_idx, [])
    else:
        label      = "unknown"
        issues     = []
        confidence = 0.0

    latest_label = label

    # Draw skeleton on image
    h, w = img.shape[:2]
    bad_joints = set()
    if "left_knee" in issues:
        bad_joints.update([25, 27])
    if "right_knee" in issues:
        bad_joints.update([26, 28])
    if "spine" in issues:
        bad_joints.update([11, 12, 23, 24])

    for a, b in SKELETON_CONNECTIONS:
        x1, y1 = int(kp[a][0]*w), int(kp[a][1]*h)
        x2, y2 = int(kp[b][0]*w), int(kp[b][1]*h)
        color = (0, 0, 255) if (a in bad_joints or b in bad_joints) else (0, 255, 0)
        cv2.line(img, (x1,y1), (x2,y2), color, 2)

    for i, (x, y) in enumerate(kp):
        cx, cy = int(x*w), int(y*h)
        color = (0, 0, 255) if i in bad_joints else (0, 255, 0)
        cv2.circle(img, (cx, cy), 5, color, -1)

    # Label text on image
    label_color = (0, 255, 0) if label == "good" else (0, 0, 255)
    cv2.putText(img, label.upper().replace("_", " "), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)
    cv2.putText(img, f"conf: {confidence:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    latest_annotated = img.copy()

    return jsonify({
        "keypoints":  kp,
        "label":      label,
        "issues":     issues,
        "confidence": round(confidence, 3),
    })

@app.route("/latest")
def latest():
    global latest_annotated, latest_label
    if latest_annotated is None:
        return "<h1>No frame yet — waiting for board...</h1>", 404
    _, buf = cv2.imencode('.jpg', latest_annotated)
    img_b64 = base64.b64encode(buf).decode()
    label_color = "lime" if latest_label == "good" else "red"
    return f'''
    <html>
    <head>
        <meta http-equiv="refresh" content="2">
        <style>
            body {{ background: #111; margin: 0; font-family: sans-serif; }}
            .container {{ display: flex; flex-direction: column; align-items: center; }}
            img {{ max-width: 100%; max-height: 85vh; object-fit: contain; margin-top: 10px; }}
            h1 {{ color: {label_color}; font-size: 2em; margin: 10px; }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1>FormCoach — {latest_label.upper().replace("_", " ")}</h1>
        <img src="data:image/jpeg;base64,{img_b64}">
    </div>
    </body>
    </html>
    '''

@app.route('/stream')
def stream():
    def generate_frames():
        while True:
            if latest_annotated is not None:
                _, buf = cv2.imencode('.jpg', latest_annotated)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.1)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_LOADED})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    args = parser.parse_args()
    print(f"[FormCoach] Server starting on http://0.0.0.0:{args.port}")
    print(f"[FormCoach] Open http://0.0.0.0:{args.port}/latest in browser for live feed")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)