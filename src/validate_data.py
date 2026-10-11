"""Dataset health check. Run after every recording session.

    python src/validate_data.py
    python src/validate_data.py --delete-bad      # remove broken clips

Finds the problems that silently destroy a keyword-spotting dataset:
wrong sample rate, wrong length, empty or clipped clips, filenames that
lost their speaker id, and classes with too few distinct speakers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_utils import check_quality
from config import (
    ALL_LABELS,
    CLIP_SAMPLES,
    COMMAND_LABELS,
    DATA_RAW,
    SAMPLE_RATE,
    SPEAKER_ALIASES,
    TARGET_PER_COMMAND,
)

MIN_SPEAKERS = 10  # see roadmap W2: single-speaker data will fail the demo


def parse_speaker(path: Path) -> str | None:
    """Speaker id, with top-up sessions resolved back to the person.

    Must match dataset.load_all_clips, otherwise this report claims more
    speakers than the train/val/test split actually sees.
    """
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    return SPEAKER_ALIASES.get(parts[-2], parts[-2])


def validate() -> tuple[list[tuple[Path, str]], dict]:
    problems: list[tuple[Path, str]] = []
    per_label_count: Counter[str] = Counter()
    per_label_speakers: dict[str, set[str]] = defaultdict(set)
    per_speaker_count: Counter[str] = Counter()

    for label in ALL_LABELS:
        folder = DATA_RAW / label
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.wav")):
            speaker = parse_speaker(path)
            if speaker is None:
                problems.append((path, "filename has no speaker id"))
                continue

            try:
                audio, sample_rate = sf.read(path, dtype="float32")
            except Exception as error:  # noqa: BLE001
                problems.append((path, f"unreadable: {error}"))
                continue

            if audio.ndim > 1:
                problems.append((path, f"not mono ({audio.ndim} channels)"))
                continue
            if sample_rate != SAMPLE_RATE:
                problems.append((path, f"sample rate {sample_rate} != {SAMPLE_RATE}"))
                continue
            if len(audio) != CLIP_SAMPLES:
                problems.append((path, f"length {len(audio)} != {CLIP_SAMPLES}"))
                continue

            ok, warnings = check_quality(audio)
            # Silence clips are supposed to be quiet; only flag clipping there.
            if label == "_silence_":
                warnings = [w for w in warnings if "CLIPPING" in w]
                ok = not warnings
            if not ok:
                problems.append((path, "; ".join(warnings)))
                continue

            per_label_count[label] += 1
            per_label_speakers[label].add(speaker)
            per_speaker_count[speaker] += 1

    stats = {
        "count": per_label_count,
        "speakers": per_label_speakers,
        "per_speaker": per_speaker_count,
    }
    return problems, stats


def report(problems, stats, target: int) -> None:
    print("\n" + "=" * 68)
    print("DATASET REPORT")
    print("=" * 68)
    print(f"{'label':<12}{'clips':>8}{'target':>8}{'speakers':>10}  status")
    print("-" * 68)

    for label in ALL_LABELS:
        count = stats["count"][label]
        speakers = len(stats["speakers"][label])
        goal = target if label in COMMAND_LABELS else 200

        flags = []
        if count < goal:
            flags.append(f"need {goal - count} more")
        if label in COMMAND_LABELS and speakers < MIN_SPEAKERS:
            flags.append(f"only {speakers} speakers")
        status = "OK" if not flags else " | ".join(flags)
        print(f"{label:<12}{count:>8}{goal:>8}{speakers:>10}  {status}")

    total = sum(stats["count"].values())
    print("-" * 68)
    print(f"{'TOTAL':<12}{total:>8}")

    # Class balance matters: a class with far fewer samples will be
    # under-predicted unless it is weighted during training.
    counts = [stats["count"][l] for l in COMMAND_LABELS if stats["count"][l]]
    if counts:
        ratio = max(counts) / min(counts)
        note = "balanced" if ratio < 1.5 else "IMBALANCED - use class weights"
        print(f"Command class balance ratio: {ratio:.2f}  ({note})")

    if stats["per_speaker"]:
        print("\nClips per speaker:")
        for speaker, count in stats["per_speaker"].most_common():
            print(f"  {speaker:<10} {count:>6}")
        dominant = stats["per_speaker"].most_common(1)[0]
        share = dominant[1] / total if total else 0
        if share > 0.5:
            print(f"\n  WARNING: '{dominant[0]}' is {share:.0%} of the dataset."
                  "\n  The model will overfit to this voice. Recruit more speakers.")

    if problems:
        print(f"\n{len(problems)} PROBLEM FILES:")
        for path, reason in problems[:40]:
            print(f"  {path.name:<32} {reason}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
    else:
        print("\nNo problem files.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-bad", action="store_true")
    parser.add_argument("--target", type=int, default=TARGET_PER_COMMAND)
    args = parser.parse_args()

    problems, stats = validate()
    report(problems, stats, args.target)

    if args.delete_bad and problems:
        confirm = input(f"Delete {len(problems)} problem files? [y/N] ").strip().lower()
        if confirm == "y":
            for path, _ in problems:
                path.unlink(missing_ok=True)
            print(f"Deleted {len(problems)} files.")


if __name__ == "__main__":
    main()
