"""Train a speech-emotion classifier on RAVDESS speech audio only.

The trainer recursively searches the supplied dataset folder, but ONLY accepts
RAVDESS speech recordings whose first two filename fields are 03-01.
Song recordings (03-02) are ignored automatically.
"""
from pathlib import Path
import argparse
import json
from collections import Counter
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from app.audio_features import extract_features

CODES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def collect(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root.resolve()}")
    rows = []
    ignored_songs = 0
    for p in sorted(root.rglob("*.wav")):
        parts = p.stem.split("-")
        # RAVDESS: modality- vocal_channel- emotion- intensity- statement- repetition- actor
        if len(parts) >= 7 and parts[0] == "03" and parts[1] == "01":
            if parts[2] in CODES:
                rows.append((p, CODES[parts[2]]))
        elif len(parts) >= 2 and parts[0] == "03" and parts[1] == "02":
            ignored_songs += 1
    return rows, ignored_songs

def main():
    ap = argparse.ArgumentParser(description="Train RAVDESS speech emotion model")
    ap.add_argument("--dataset", default="data/ravdess")
    ap.add_argument("--output", default="models/emotion_model.joblib")
    ap.add_argument("--test-size", type=float, default=.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, ignored_songs = collect(args.dataset)
    counts = Counter(label for _, label in rows)
    print("\nRAVDESS speech-only dataset scan")
    print(f"Dataset root: {Path(args.dataset).resolve()}")
    print(f"Speech WAV files found: {len(rows)}")
    print(f"Song WAV files ignored: {ignored_songs}")
    print("Emotion distribution:")
    for label in CODES.values():
        print(f"  {label:10s}: {counts.get(label, 0)}")

    missing = [name for name in CODES.values() if counts.get(name, 0) == 0]
    if missing:
        raise SystemExit("Missing emotion classes: " + ", ".join(missing) + ". Check that the RAVDESS speech dataset is extracted correctly.")
    if len(rows) < 100:
        raise SystemExit(f"Found only {len(rows)} speech WAV files. Expected about 1,440 RAVDESS speech files.")

    print(f"\nExtracting MFCC/audio features from {len(rows)} speech files...")
    X, y = [], []
    for i, (p, label) in enumerate(rows, 1):
        try:
            X.append(extract_features(str(p)))
            y.append(label)
        except Exception as e:
            print(f"Skipping {p}: {e}")
        if i % 100 == 0 or i == len(rows):
            print(f"  processed {i}/{len(rows)}")

    X = np.asarray(X)
    y = np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=args.seed,
            n_jobs=-1, max_features="sqrt"
        ))
    ])
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    accuracy = accuracy_score(yte, pred)
    labels = list(CODES.values())

    print("\n================ EVALUATION ================")
    print("Accuracy:", round(accuracy, 4))
    print(classification_report(yte, pred, labels=labels, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(yte, pred, labels=labels))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "RAVDESS",
        "dataset_type": "speech_only",
        "speech_file_count": int(len(rows)),
        "ignored_song_file_count": int(ignored_songs),
        "feature_count": int(X.shape[1]),
        "classes": labels,
        "test_accuracy": float(accuracy),
    }
    joblib.dump({"model": model, "metadata": metadata}, out)
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSaved model:", out.resolve())
    print("Saved metadata:", out.with_suffix(".json").resolve())

if __name__ == "__main__":
    main()
