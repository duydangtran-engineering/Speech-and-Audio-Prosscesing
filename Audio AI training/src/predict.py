"""Kiểm tra nhanh: đoán nhãn của 1 file audio bất kỳ (m4a hoặc wav)."""
import sys

import joblib
import librosa
import numpy as np

import audio_io
from config import MODEL_DIR, SAMPLE_RATE
from features import extract
from step1_split import detect_words, to_fixed_length


def main(path):
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    model = bundle["model"]

    y = audio_io.load(path, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    segments, _ = detect_words(y)
    segments = segments or [(0, len(y))]

    for i, (s, e) in enumerate(segments):
        feat = extract(to_fixed_length(y[s:e])).reshape(1, -1)
        proba = model.predict_proba(feat)[0]
        best = int(np.argmax(proba))
        print(f"[{s/SAMPLE_RATE:5.2f}s] {model.classes_[best]:6s} "
              f"({proba[best]:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("cách dùng: python predict.py <file audio>")
    main(sys.argv[1])
