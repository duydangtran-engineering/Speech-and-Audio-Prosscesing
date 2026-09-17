"""Bước 1c: kiểm tra "sức khỏe" của các clip đã cắt trước khi mất công train.

Một người nói mà clip ngắn hơn hẳn người khác thì từ của họ đang bị cắt cụt
mất 1 nửa; người mà clip dài hơn hẳn thì các từ đang bị dính liền vào nhau.
Cả 2 trường hợp đều trông bình thường nếu chỉ nhìn số lượng clip, đó là cách
1 bản ghi hỏng vẫn sống sót đến tận lúc ra 1 con số accuracy khó hiểu.

Chạy ngay sau step1_split.py:
    python step1c_validate.py
"""
import numpy as np

import audio_io
from config import CLIPS_DIR, LABELS, SAMPLE_RATE

AUG_MARK = "__aug"
SHORT_RATIO = 0.75      # gắn cờ người nói dưới 75% median của nhãn đó
LONG_RATIO = 1.35       # hoặc trên 135% median đó


def voiced_ms(path):
    """Độ dài phần thực sự có tiếng, tính bằng mili giây."""
    import librosa
    y = audio_io.load(path, sr=SAMPLE_RATE)
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    voiced = np.where(rms > max(rms.max() * 0.06, 1e-6))[0]
    if len(voiced) < 3:
        return 0.0
    return (voiced[-1] - voiced[0] + 1) * 160 / SAMPLE_RATE * 1000


def main():
    problems = []

    for label in LABELS:
        folder = CLIPS_DIR / label
        if not folder.exists():
            print(f"[thiếu] {label}")
            continue

        by_speaker = {}
        for wav in sorted(folder.glob("*.wav")):
            if AUG_MARK in wav.stem:
                continue
            by_speaker.setdefault(wav.stem.split("_")[0], []).append(wav)

        if not by_speaker:
            continue

        durations = {s: np.array([voiced_ms(f) for f in files])
                     for s, files in by_speaker.items()}
        label_median = np.median(np.concatenate(list(durations.values())))

        print(f"\n{label}  (median {label_median:.0f} ms trên "
              f"{len(by_speaker)} người nói)")
        print(f"  {'người nói':10s} {'clip':>6} {'median':>8} {'p10':>7} "
              f"{'p90':>7}  ghi chú")

        for speaker, d in sorted(durations.items()):
            med = np.median(d)
            ratio = med / label_median if label_median else 1.0
            note = ""
            if ratio < SHORT_RATIO:
                note = "NGẮN - từ đang bị cắt cụt mất 1 nửa"
            elif ratio > LONG_RATIO:
                note = "DÀI - các từ có thể đang bị dính liền nhau"
            if np.percentile(d, 90) > label_median * 2.2:
                note = (note + " | ") if note else ""
                note += "một số clip chứa hơn 1 từ"
            if note:
                problems.append(f"{label}/{speaker}: {note}")
            print(f"  {speaker:10s} {len(d):6d} {med:7.0f}ms {np.percentile(d,10):6.0f}ms "
                  f"{np.percentile(d,90):6.0f}ms  {note}")

    print("\n" + "=" * 60)
    if problems:
        print(f"Tìm thấy {len(problems)} vấn đề:")
        for p in problems:
            print(f"  - {p}")
        print("\nNghe lại các clip đó trước khi train. Một người nói bị cắt "
              "sai từ sẽ làm dữ liệu gán nhãn sai lan vào mọi fold, chứ không "
              "chỉ riêng phần của họ.")
    else:
        print("Tất cả người nói đều ổn định.")


if __name__ == "__main__":
    main()
