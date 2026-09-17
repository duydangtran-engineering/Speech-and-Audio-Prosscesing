"""Đọc audio, chạy được với file .m4a trên mọi hệ điều hành.

soundfile không giải mã được m4a/aac, và các bản librosa gần đây đã bỏ
fallback qua audioread, nên file nào không phải wav/flac/ogg thuần sẽ được
giải mã bằng cách pipe qua ffmpeg.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE

NATIVE_EXT = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def _ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe is None:
        sys.exit("Không tìm thấy ffmpeg trong PATH. Cài ffmpeg rồi mở lại VSCode.")
    return exe


def load(path, sr=SAMPLE_RATE):
    """Trả về mảng float32 mono, đã resample về `sr`."""
    path = Path(path)

    if path.suffix.lower() in NATIVE_EXT:
        y, native_sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if native_sr != sr:
            import librosa
            y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
        return np.ascontiguousarray(y, dtype=np.float32)

    cmd = [
        _ffmpeg_bin(), "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg lỗi khi xử lý {path.name}:\n"
                           f"{proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
