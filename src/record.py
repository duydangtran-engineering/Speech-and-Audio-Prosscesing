"""Interactive recorder for the voice-command dataset.

Run from a real terminal (not a notebook) so microphone access works:

    python src/record.py                     # guided menu
    python src/record.py --label len --speaker sp01 --count 30
    python src/record.py --stream --speaker sp01 --minutes 3 --label _unknown_

Files are saved as:
    data/raw/<label>/<label>_<speaker>_<index>.wav

The speaker id in the filename is what makes a speaker-independent
train/val/test split possible later. Never drop it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from audio_utils import (
    center_on_energy,
    check_quality,
    level_bar,
    next_index,
    save_clip,
    segment_words,
)
from config import (
    ALL_LABELS,
    CLIP_SAMPLES,
    DATA_RAW,
    LABEL_PROMPT,
    RECORD_DURATION,
    SAMPLE_RATE,
)



# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------
def countdown(seconds: int = 2) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining}...", end="\r", flush=True)
        time.sleep(0.6)
    print("  SPEAK NOW      ", end="\r", flush=True)


def record_clip(duration: float = RECORD_DURATION) -> np.ndarray:
    """Capture `duration` seconds of mono audio as float32 in [-1, 1]."""
    frames = int(duration * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def play(audio: np.ndarray) -> None:
    sd.play(audio, SAMPLE_RATE)
    sd.wait()


# ----------------------------------------------------------------------
# Mode 1: one clip at a time, with review
# ----------------------------------------------------------------------
def session_clips(label: str, speaker: str, count: int,
                  review: bool = True, auto: bool = False,
                  gap: float = 1.2) -> None:
    prompt = LABEL_PROMPT.get(label, label)
    index = next_index(label, speaker)

    print(f"\n{'=' * 62}")
    print(f"  label   : {label}")
    print(f"  say     : {prompt}")
    print(f"  speaker : {speaker}")
    print(f"  target  : {count} clips (starting at index {index})")
    print(f"{'=' * 62}")
    if auto:
        print(f"  AUTO mode: records every {gap:.1f}s with no key press.")
        print("  Vary your delivery: loud/soft, fast/slow, near/far.")
        print("  Ctrl+C to stop. Clips already saved are kept.\n")
        input("  Press ENTER to start... ")
    else:
        print("  ENTER = record   |   s = skip   |   q = quit\n")

    saved = 0
    while saved < count:
        if auto:
            print(f"[{saved + 1}/{count}] get ready...", end="\r", flush=True)
            time.sleep(gap)
        else:
            action = input(f"[{saved + 1}/{count}] ready > ").strip().lower()
            if action == "q":
                break
            if action == "s":
                continue

        countdown(1 if auto else 2)
        audio = record_clip()
        audio = center_on_energy(audio, CLIP_SAMPLES)
        print(f"  {level_bar(audio)}          ")

        ok, warnings = check_quality(audio)
        for warning in warnings:
            print(f"  ! {warning}")

        # In auto mode a bad clip is dropped silently and retried,
        # so the session never stops to wait for an answer.
        if auto and not ok:
            print("  discarded (auto), retrying\n")
            continue

        if review and not ok:
            choice = input("  keep anyway? [y/N/p=play] ").strip().lower()
            if choice == "p":
                play(audio)
                choice = input("  keep? [y/N] ").strip().lower()
            if choice != "y":
                print("  discarded\n")
                continue

        path = save_clip(audio, label, speaker, index)
        print(f"  saved -> {path.name}\n")
        index += 1
        saved += 1

    print(f"Done: {saved} clips saved for '{label}' by '{speaker}'.\n")


# ----------------------------------------------------------------------
# Mode 3: rapid - speak the word over and over, segment afterwards
# ----------------------------------------------------------------------
def session_rapid(label: str, speaker: str, seconds: float,
                  review: bool = True) -> None:
    """Record one long take, then cut it into individual clips.

    Roughly four times faster than prompting per clip, and the delivery
    stays more natural: the speaker settles into a rhythm instead of
    resetting between every prompt.
    """
    prompt = LABEL_PROMPT.get(label, label)
    index = next_index(label, speaker)

    print(f"\n{'=' * 62}")
    print(f"  RAPID MODE - say '{prompt}' over and over")
    print(f"  speaker : {speaker}   duration: {seconds:.0f}s")
    print(f"{'=' * 62}")
    print("  Leave a short pause between words, about half a second.")
    print("  Vary it as you go: louder, softer, faster, nearer, further.\n")
    input("  Press ENTER to start... ")

    frames = int(seconds * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    start = time.time()
    while sd.get_stream().active:
        elapsed = time.time() - start
        print(f"  recording {elapsed:5.1f}s / {seconds:.0f}s", end="\r", flush=True)
        time.sleep(0.2)
    sd.wait()
    audio = audio.flatten()

    segments, dropped = segment_words(audio)
    print(f"\n  found {len(segments)} words                    ")

    if not segments:
        print("  Nothing detected. Speak louder or closer to the mic.\n")
        return

    lengths = [(b - a) / SAMPLE_RATE for a, b in segments]
    print(f"  word length: min {min(lengths):.2f}s  "
          f"mean {np.mean(lengths):.2f}s  max {max(lengths):.2f}s")
    if dropped:
        print(f"  ! {dropped} run(s) held several words and could not be "
              f"split - dropped.")
        print("    Leave a clearer pause between repetitions.")

    if review:
        answer = input("  Save these? [Y/n] ").strip().lower()
        if answer == "n":
            print("  Discarded.\n")
            return

    saved = skipped = 0
    for begin, end in segments:
        middle = (begin + end) // 2
        half = CLIP_SAMPLES // 2
        clip_start = max(0, min(middle - half, len(audio) - CLIP_SAMPLES))
        clip = audio[clip_start:clip_start + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))

        ok, _ = check_quality(clip)
        if not ok:
            skipped += 1
            continue
        save_clip(clip, label, speaker, index, note="rapid")
        index += 1
        saved += 1

    print(f"  saved {saved} clips, skipped {skipped} (low quality)\n")


# ----------------------------------------------------------------------
# Mode 2: continuous stream, auto-sliced
# ----------------------------------------------------------------------
def session_stream(label: str, speaker: str, minutes: float) -> None:
    """Record continuously and slice into 1-second clips.

    This is the fast way to collect _unknown_ (just talk normally for a few
    minutes) and _silence_ (leave the mic running in a quiet room).
    """
    total_seconds = minutes * 60.0
    index = next_index(label, speaker)

    print(f"\n{'=' * 62}")
    print(f"  STREAM MODE - label '{label}', speaker '{speaker}'")
    print(f"  duration: {minutes:.1f} min -> about {int(total_seconds)} clips")
    if label == "_unknown_":
        print("  Talk normally. Read a book out loud. Do NOT say the commands.")
    elif label == "_silence_":
        print("  Stay quiet. Let room noise, fan, keyboard, traffic be captured.")
    print(f"{'=' * 62}\n")
    input("Press ENTER to start... ")

    frames = int(total_seconds * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")

    start = time.time()
    while sd.get_stream().active:
        elapsed = time.time() - start
        print(f"  recording {elapsed:5.1f}s / {total_seconds:.0f}s",
              end="\r", flush=True)
        time.sleep(0.25)
    sd.wait()
    audio = audio.flatten()
    print("\n  slicing...")

    saved = skipped = 0
    for start_sample in range(0, len(audio) - CLIP_SAMPLES + 1, CLIP_SAMPLES):
        clip = audio[start_sample : start_sample + CLIP_SAMPLES]
        ok, _ = check_quality(clip)
        # For _silence_ we deliberately keep quiet clips; that is the point.
        if not ok and label != "_silence_":
            skipped += 1
            continue
        save_clip(clip, label, speaker, index, note="stream")
        index += 1
        saved += 1

    print(f"Done: {saved} clips saved, {skipped} skipped (low quality).\n")


# ----------------------------------------------------------------------
# Menu / CLI
# ----------------------------------------------------------------------
def show_devices() -> None:
    print("\nAudio input devices:")
    for i, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default}")
    print()


def show_progress() -> None:
    print("\nCurrent dataset:")
    total = 0
    for label in ALL_LABELS:
        files = list((DATA_RAW / label).glob("*.wav")) if (DATA_RAW / label).exists() else []
        speakers = {f.stem.split("_")[-2] for f in files} if files else set()
        total += len(files)
        print(f"  {label:<12} {len(files):>5} clips   {len(speakers):>2} speakers")
    print(f"  {'TOTAL':<12} {total:>5} clips\n")


def menu() -> None:
    show_progress()
    speaker = input("Speaker id (e.g. sp01) > ").strip() or "sp00"

    print("\nLabels:")
    for i, label in enumerate(ALL_LABELS):
        print(f"  [{i}] {label:<12} {LABEL_PROMPT.get(label, '')}")
    choice = input("Label number > ").strip()
    try:
        label = ALL_LABELS[int(choice)]
    except (ValueError, IndexError):
        print("Invalid label.")
        return

    if label in ("_unknown_", "_silence_"):
        mode = input("Mode: [1] one-by-one  [2] stream (recommended) > ").strip()
        if mode == "2":
            minutes = float(input("Minutes to record > ").strip() or "3")
            session_stream(label, speaker, minutes)
            return

    count = int(input("How many clips > ").strip() or "20")
    session_clips(label, speaker, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record voice command dataset")
    parser.add_argument("--label", choices=ALL_LABELS)
    parser.add_argument("--speaker", help="speaker id, e.g. sp01")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--stream", action="store_true",
                        help="continuous recording, auto-sliced into 1s clips")
    parser.add_argument("--minutes", type=float, default=3.0)
    parser.add_argument("--device", type=int, help="input device index")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-review", action="store_true",
                        help="do not ask before discarding low-quality clips")
    parser.add_argument("--auto", action="store_true",
                        help="record back-to-back without pressing ENTER")
    parser.add_argument("--rapid", action="store_true",
                        help="one long take, cut into words automatically")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="length of the rapid take")
    parser.add_argument("--gap", type=float, default=1.2,
                        help="seconds of pause between clips in auto mode")
    args = parser.parse_args()

    if args.list_devices:
        show_devices()
        return
    if args.progress:
        show_progress()
        return
    if args.device is not None:
        sd.default.device = (args.device, None)

    try:
        if args.label and args.speaker:
            if args.rapid:
                session_rapid(args.label, args.speaker, args.seconds,
                              review=not args.no_review)
            elif args.stream:
                session_stream(args.label, args.speaker, args.minutes)
            else:
                session_clips(args.label, args.speaker, args.count,
                              review=not args.no_review,
                              auto=args.auto, gap=args.gap)
        else:
            menu()
    except KeyboardInterrupt:
        print("\nInterrupted. Clips already saved are kept.")
        sys.exit(0)


if __name__ == "__main__":
    main()
