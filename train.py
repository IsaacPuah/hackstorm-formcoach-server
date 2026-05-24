"""
train.py — train and evaluate the form classifier
===================================================
Run this once you have a labeled CSV from collect.py.
It trains a Random Forest on the 6 angle features and saves the model.

Usage:
    python train.py --data data.csv --output formcoach_model.pkl

After running, check the printed classification report.
Target: accuracy ~90-94%, macro F1 ~0.88-0.92.
"""

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

FEATURES = ["left_knee", "right_knee", "back_angle", "hip_depth", "left_shin", "torso_lean"]
LABEL_NAMES = ["good", "knee_valgus", "rounded_back", "both"]

def train(data_csv, output_pkl):
    df = pd.read_csv(data_csv)

    # Drop any rows where label wasn't filled in
    df = df[df["label"].notna() & (df["label"] != "")]
    df["label"] = df["label"].astype(int)

    X = df[FEATURES]
    y = df["label"]

    print(f"[train] Dataset: {len(df)} labeled frames")
    print(f"[train] Class distribution:\n{y.value_counts().sort_index().rename(dict(enumerate(LABEL_NAMES)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n[train] ── Results ──────────────────────────────")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nPer-class report:")
    present = sorted(y_test.unique())
    present_names = [LABEL_NAMES[i] for i in present]
    print(classification_report(y_test, y_pred, labels=present, target_names=present_names))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, output_pkl)
    print(f"\n[train] Model saved to {output_pkl}")
    print(f"[train] Start server.py — it will auto-load the model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data.csv",            help="Labeled CSV from collect.py")
    parser.add_argument("--output", default="formcoach_model.pkl", help="Where to save the model")
    args = parser.parse_args()
    train(args.data, args.output)
