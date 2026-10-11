"""Look at a clip: waveform + log-Mel spectrogram.

    python src/inspect_clip.py data/raw/len/len_sp01_0000.wav
    python src/inspect_clip.py --label len --n 6      # grid of random samples

This is the sanity check that catches "my word is cut off" and
"my mic records nothing but hum" before you train on 3000 bad files.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from config import DATA_RAW, SAMPLE_RATE

N_FFT = 400      # 25 ms window
HOP = 160        # 10 ms hop
N_MELS = 40


def log_mel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=20, fmax=SAMPLE_RATE // 2,
    )
    return librosa.power_to_db(mel, ref=np.max)


def plot_one(path: Path, axes) -> None:
    audio, _ = sf.read(path, dtype="float32")
    ax_wave, ax_spec = axes

    ax_wave.plot(np.linspace(0, len(audio) / SAMPLE_RATE, len(audio)), audio, lw=0.6)
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_title(path.name, fontsize=9)
    ax_wave.set_ylabel("amp")

    librosa.display.specshow(
        log_mel(audio), sr=SAMPLE_RATE, hop_length=HOP,
        x_axis="time", y_axis="mel", ax=ax_spec, cmap="magma",
    )
    ax_spec.set_ylabel("mel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()

    if args.path:
        paths = [args.path]
    elif args.label:
        candidates = sorted((DATA_RAW / args.label).glob("*.wav"))
        if not candidates:
            print(f"No clips found for label '{args.label}'.")
            return
        paths = random.sample(candidates, min(args.n, len(candidates)))
    else:
        parser.error("give a file path or --label")

    fig, axes = plt.subplots(2, len(paths), figsize=(4 * len(paths), 5),
                             squeeze=False)
    for column, path in enumerate(paths):
        plot_one(path, (axes[0][column], axes[1][column]))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
