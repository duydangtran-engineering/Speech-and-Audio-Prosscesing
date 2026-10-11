"""Shared audio helpers: quality checks, level meters, trimming, and the
clip store.

Everything here works on arrays and files only - no audio device - so
tools that import recordings from disk do not need a microphone present.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from config import (
    CLIPPING_PEAK,
    DATA_RAW,
    MIN_PEAK,
    MIN_RMS,
    SAMPLE_RATE,
    SUBTYPE,
)

META_FILE = DATA_RAW / "metadata.csv"


def peak(x: np.ndarray) -> float:
    """Maximum absolute amplitude of the signal."""
    return float(np.max(np.abs(x))) if x.size else 0.0


def rms(x: np.ndarray) -> float:
    """Root mean square = perceived loudness of the signal."""
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def check_quality(x: np.ndarray) -> tuple[bool, list[str]]:
    """Return (is_ok, warnings) for a recorded clip.

    Catches the two failure modes that quietly ruin a dataset:
    clips recorded too quietly, and clips that hit the ceiling
    and got distorted.
    """
    warnings: list[str] = []
    p, r = peak(x), rms(x)

    if p >= CLIPPING_PEAK:
        warnings.append(f"CLIPPING (peak={p:.3f}) - move away from the mic")
    if p < MIN_PEAK:
        warnings.append(f"SILENT (peak={p:.3f}) - nothing was captured")
    elif r < MIN_RMS:
        warnings.append(f"TOO QUIET (rms={r:.4f}) - speak up or move closer")

    return (len(warnings) == 0, warnings)


def level_bar(x: np.ndarray, width: int = 40) -> str:
    """ASCII meter so the speaker can see their level while recording."""
    p = peak(x)
    filled = int(min(p, 1.0) * width)
    bar = "#" * filled + "-" * (width - filled)
    flag = "  <-- CLIPPING" if p >= CLIPPING_PEAK else ""
    return f"[{bar}] peak={p:.3f} rms={rms(x):.4f}{flag}"


def center_on_energy(x: np.ndarray, target_len: int) -> np.ndarray:
    """Crop or pad `x` to `target_len` samples, centered on the loudest part.

    Recording is slightly longer than the model window so a word said a bit
    early or late is not cut off. This finds where the energy actually is
    and keeps that region.
    """
    if x.size == target_len:
        return x
    if x.size < target_len:
        pad = target_len - x.size
        return np.pad(x, (pad // 2, pad - pad // 2))

    # Slide a window of target_len and keep the one with the most energy.
    energy = np.square(x)
    cumulative = np.concatenate([[0.0], np.cumsum(energy)])
    window_energy = cumulative[target_len:] - cumulative[:-target_len]
    start = int(np.argmax(window_energy))
    return x[start : start + target_len]


# ----------------------------------------------------------------------
# Clip store
# ----------------------------------------------------------------------
def next_index(label: str, speaker: str) -> int:
    """Next free index for this label+speaker, so reruns never overwrite."""
    folder = DATA_RAW / label
    folder.mkdir(parents=True, exist_ok=True)
    existing = list(folder.glob(f"{label}_{speaker}_*.wav"))
    if not existing:
        return 0
    indices = []
    for path in existing:
        try:
            indices.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=-1) + 1


def save_clip(audio: np.ndarray, label: str, speaker: str, index: int,
              note: str = "") -> Path:
    folder = DATA_RAW / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label}_{speaker}_{index:04d}.wav"
    sf.write(path, audio, SAMPLE_RATE, subtype=SUBTYPE)
    append_metadata(path, label, speaker, audio, note)
    return path


def append_metadata(path: Path, label: str, speaker: str,
                    audio: np.ndarray, note: str) -> None:
    """One row per clip. Useful later for per-speaker statistics."""
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not META_FILE.exists()
    with META_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                ["filename", "label", "speaker", "duration_s",
                 "peak", "rms", "recorded_at", "note"]
            )
        writer.writerow([
            path.relative_to(DATA_RAW).as_posix(),
            label,
            speaker,
            f"{len(audio) / SAMPLE_RATE:.3f}",
            f"{float(np.max(np.abs(audio))):.4f}",
            f"{float(np.sqrt(np.mean(np.square(audio)))):.5f}",
            datetime.now().isoformat(timespec="seconds"),
            note,
        ])


# ----------------------------------------------------------------------
# Word segmentation
# ----------------------------------------------------------------------
def _detect_runs(audio: np.ndarray, min_word: float, min_gap: float,
                 threshold_ratio: float) -> list[tuple[int, int]]:
    """Energy-based speech runs, as (start, end) sample pairs.

    The threshold is relative to this recording's own noise floor rather
    than absolute, so it adapts to a quiet room or a loud one without the
    speaker having to match a fixed level.
    """
    frame = int(0.02 * SAMPLE_RATE)
    n_frames = len(audio) // frame
    if n_frames < 3:
        return []
    frames = audio[:n_frames * frame].reshape(n_frames, frame)
    energy = np.sqrt(np.mean(frames ** 2, axis=1))

    floor = max(float(np.percentile(energy, 20)), 1e-5)
    peak = float(np.percentile(energy, 99))
    # Halfway (geometric) between floor and peak, but never below a few
    # times the floor: catches soft words without chasing noise.
    threshold = max(floor * threshold_ratio, np.sqrt(floor * peak) * 0.5)
    active = energy > threshold

    gap_frames = max(1, int(min_gap / 0.02))
    runs: list[list[int]] = []
    for i, is_active in enumerate(active):
        if is_active:
            if runs and i - runs[-1][1] <= gap_frames:
                runs[-1][1] = i
            else:
                runs.append([i, i])

    min_frames = max(1, int(min_word / 0.02))
    return [(a * frame, min((b + 1) * frame, len(audio)))
            for a, b in runs if b - a + 1 >= min_frames]


def segment_words(audio: np.ndarray, min_word: float = 0.12,
                  min_gap: float = 0.12, threshold_ratio: float = 3.5,
                  max_word: float = 0.85
                  ) -> tuple[list[tuple[int, int]], int]:
    """Split a continuous recording into single spoken words.

    Returns (segments, dropped). A speaker who does not pause enough
    produces runs holding several words; saving those as one clip teaches
    the model the wrong thing, so each over-long run is re-examined with a
    stricter threshold, which usually finds the dip between the words.
    Runs that still cannot be resolved are dropped rather than guessed at
    - losing a few clips out of fifty costs nothing, mislabelling them is
    expensive.
    """
    runs = _detect_runs(audio, min_word, min_gap, threshold_ratio)
    segments: list[tuple[int, int]] = []
    dropped = 0

    for start, end in runs:
        if (end - start) / SAMPLE_RATE <= max_word:
            segments.append((start, end))
            continue

        piece = audio[start:end]
        for ratio in (5.0, 7.0, 10.0, 14.0):
            sub = _detect_runs(piece, min_word, min_gap * 0.6, ratio)
            if sub and all((b - a) / SAMPLE_RATE <= max_word for a, b in sub):
                segments.extend((start + a, start + b) for a, b in sub)
                break
        else:
            dropped += 1

    return segments, dropped
