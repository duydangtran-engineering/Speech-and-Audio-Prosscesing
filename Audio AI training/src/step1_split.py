"""Bước 1: cắt các bản ghi .m4a dài thành các clip .wav 1 từ, dài 1 giây.

Cấu trúc thư mục đầu vào mong muốn:
    data/raw/tang/vinh_01.m4a
    data/raw/tang/nam_01.m4a
    data/raw/giam/...
    data/raw/noise/...      (tạp âm nền, chặt thành khúc mà không cần phát hiện từ)

Đầu ra:
    data/clips/tang/vinh_01_000.wav ...
"""
import sys

import librosa
import numpy as np
import soundfile as sf

import audio_io
from config import (CLIPS_DIR, CLIP_SAMPLES, ENERGY_PERCENTILE, FRAME_LENGTH,
                    HOP_LENGTH, LABELS, MAX_WORD_MS, MIN_SILENCE_MS,
                    MIN_WORD_MS, PAD_MS,
                    RAW_DIR, SAMPLE_RATE, SPLIT_OVERRIDES)

AUDIO_EXT = {".m4a", ".mp4", ".wav", ".mp3", ".aac", ".ogg", ".flac"}


def detect_words(y, energy_percentile=ENERGY_PERCENTILE,
                 min_word_ms=MIN_WORD_MS, min_silence_ms=MIN_SILENCE_MS,
                 pad_ms=PAD_MS, max_word_ms=MAX_WORD_MS):
    """Trả về danh sách (start, end) - chỉ số sample - cho mỗi từ được nói."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH,
                              hop_length=HOP_LENGTH)[0]
    threshold = np.percentile(rms, energy_percentile) * 0.35
    threshold = max(threshold, rms.max() * 0.08)
    active = rms > threshold

    min_word = int(min_word_ms / 1000 * SAMPLE_RATE / HOP_LENGTH)
    min_sil = int(min_silence_ms / 1000 * SAMPLE_RATE / HOP_LENGTH)

    segments = []
    start = None
    silence = 0
    for i, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_sil:
                end = i - silence
                if end - start >= min_word:
                    segments.append((start, end))
                start = None
                silence = 0
    if start is not None and len(active) - start >= min_word:
        segments.append((start, len(active)))

    pad = int(pad_ms / 1000 * SAMPLE_RATE)
    limit = max_word_ms / 1000 * SAMPLE_RATE
    out, dropped = [], 0
    for s, e in segments:
        s = max(0, s * HOP_LENGTH - pad)
        e = min(len(y), e * HOP_LENGTH + pad)
        # Đoạn dài như thế này là 2 từ mà bộ phát hiện không tách ra được.
        # Nếu giữ lại sẽ đưa 1 clip gán nhãn sai vào mọi fold khi train.
        if e - s > limit:
            dropped += 1
            continue
        out.append((s, e))
    return out, dropped


def to_fixed_length(chunk):
    """Căn giữa 1 đoạn có độ dài thay đổi vào trong buffer 1 giây."""
    if len(chunk) >= CLIP_SAMPLES:
        centre = len(chunk) // 2
        half = CLIP_SAMPLES // 2
        return chunk[max(0, centre - half): max(0, centre - half) + CLIP_SAMPLES]
    buf = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    offset = (CLIP_SAMPLES - len(chunk)) // 2
    buf[offset:offset + len(chunk)] = chunk
    return buf


def split_noise(y):
    """File tạp âm chỉ đơn giản là chặt thành các khung 1 giây liên tiếp."""
    step = CLIP_SAMPLES // 2
    return [y[i:i + CLIP_SAMPLES] for i in range(0, len(y) - CLIP_SAMPLES, step)]


def process_label(label):
    src_dir = RAW_DIR / label
    dst_dir = CLIPS_DIR / label
    if not src_dir.exists():
        print(f"[bỏ qua] không tìm thấy {src_dir}")
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for path in sorted(src_dir.iterdir()):
        if path.suffix.lower() not in AUDIO_EXT:
            continue
        y = audio_io.load(path, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y)

        dropped = 0
        if label in ("noise", "silence"):
            chunks = split_noise(y)
        else:
            opts = SPLIT_OVERRIDES.get(f"{label}/{path.stem}", {})
            if opts:
                print(f"    (override: {opts})")
            segments, dropped = detect_words(y, **opts)
            chunks = [to_fixed_length(y[s:e]) for s, e in segments]

        for idx, chunk in enumerate(chunks):
            out = dst_dir / f"{path.stem}_{idx:03d}.wav"
            sf.write(out, to_fixed_length(np.asarray(chunk)), SAMPLE_RATE,
                     subtype="PCM_16")
        note = f"  ({dropped} bị loại vì quá dài)" if dropped else ""
        print(f"  {path.name}: {len(chunks)} clip{note}")
        total += len(chunks)
    print(f"[{label}] {total} clip")
    return total


def main():
    labels = sys.argv[1:] or LABELS
    grand = sum(process_label(label) for label in labels)
    print(f"\nTổng số clip: {grand}")
    print(f"Nghe lại bằng tai trong {CLIPS_DIR} trước khi train.")


if __name__ == "__main__":
    main()
