"""Record once, then sweep the real-time parameters offline.

    python src/tune_realtime.py --name clean3            # record + sweep
    python src/tune_realtime.py --name clean3 --replay    # sweep saved audio
    python src/tune_realtime.py --name aug3 --replay      # compare models

Re-recording eight utterances per word for every candidate threshold is
slow and inconsistent. Instead this records each utterance once, saves
it, and replays the detector over the saved audio for every combination
of threshold and smoothing.

Two numbers matter and they pull in opposite directions:

  hit rate    - a command is spoken and the right one fires
  false accept- ordinary speech fires a command that was never intended

A threshold that maximizes hit rate alone produces a game that reacts to
every word the player says. The table reports both so the trade-off is
explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from config import COMMAND_LABELS, LABEL_PROMPT, ROOT, SAMPLE_RATE, SUBTYPE
from features import LogMel
from live_demo import load_model
from test_live import detect

TEST_DIR = ROOT / "data" / "live_test"
TRIAL_SECONDS = 2.0
DISTRACTOR_SECONDS = 5.0

THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SMOOTHINGS = [1, 2, 3]


# ----------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------
def record_trials(trials: int, distractors: int) -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPART 1 - commands. Say the word ONCE, as you would while playing.\n")
    for label in COMMAND_LABELS:
        folder = TEST_DIR / label
        folder.mkdir(exist_ok=True)
        print(f"--- {LABEL_PROMPT.get(label, label)} ---")
        for trial in range(trials):
            input(f"  [{trial + 1}/{trials}] ENTER then say it > ")
            audio = sd.rec(int(TRIAL_SECONDS * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            sf.write(folder / f"{trial:03d}.wav", audio.flatten(),
                     SAMPLE_RATE, subtype=SUBTYPE)
        print()

    print("PART 2 - distractors. Talk normally for 5 seconds each time.")
    print("Read something aloud. Do NOT say any of the six commands.")
    print("Every command that fires here is a false accept.\n")
    folder = TEST_DIR / "_distractor_"
    folder.mkdir(exist_ok=True)
    for trial in range(distractors):
        input(f"  [{trial + 1}/{distractors}] ENTER then talk > ")
        audio = sd.rec(int(DISTRACTOR_SECONDS * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        sf.write(folder / f"{trial:03d}.wav", audio.flatten(),
                 SAMPLE_RATE, subtype=SUBTYPE)
    print(f"\nSaved to {TEST_DIR}\n")


def load_trials() -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    commands: dict[str, list[np.ndarray]] = {}
    for label in COMMAND_LABELS:
        folder = TEST_DIR / label
        if not folder.exists():
            continue
        commands[label] = [sf.read(p, dtype="float32")[0]
                           for p in sorted(folder.glob("*.wav"))]
    distractor_dir = TEST_DIR / "_distractor_"
    distractors = ([sf.read(p, dtype="float32")[0]
                    for p in sorted(distractor_dir.glob("*.wav"))]
                   if distractor_dir.exists() else [])
    return commands, distractors


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
def sweep(commands, distractors, model, log_mel, labels, device,
          hop: float, vad: float):
    rows = []
    for smoothing in SMOOTHINGS:
        for threshold in THRESHOLDS:
            hits = trials = wrong = 0
            per_word: dict[str, tuple[int, int]] = {}

            for label, clips in commands.items():
                word_hits = 0
                for audio in clips:
                    got, _, _, _, _ = detect(
                        audio, model, log_mel, labels, device,
                        hop=hop, smoothing=smoothing,
                        threshold=threshold, vad=vad)
                    trials += 1
                    if got == label:
                        hits += 1
                        word_hits += 1
                    elif got is not None:
                        wrong += 1
                per_word[label] = (word_hits, len(clips))

            false_accepts = 0
            for audio in distractors:
                got, _, n_fires, _, _ = detect(
                    audio, model, log_mel, labels, device,
                    hop=hop, smoothing=smoothing,
                    threshold=threshold, vad=vad)
                false_accepts += n_fires

            rows.append({
                "smoothing": smoothing,
                "threshold": threshold,
                "hit_rate": hits / max(trials, 1),
                "wrong": wrong,
                "false_accepts": false_accepts,
                "per_word": per_word,
                "latency_ms": smoothing * hop * 1000,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="clean3")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=6)
    parser.add_argument("--replay", action="store_true",
                        help="reuse the saved recordings, do not record again")
    parser.add_argument("--hop", type=float, default=0.2)
    parser.add_argument("--vad", type=float, default=0.012)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    if args.device is not None:
        sd.default.device = (args.device, None)

    if not args.replay:
        record_trials(args.trials, args.distractors)

    commands, distractors = load_trials()
    if not commands:
        raise SystemExit(f"No recordings in {TEST_DIR}. Run without --replay.")

    n_clips = sum(len(v) for v in commands.values())
    print(f"replaying {n_clips} command clips and "
          f"{len(distractors)} distractor clips\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels = load_model(args.name, device)
    log_mel = LogMel().to(device)

    rows = sweep(commands, distractors, model, log_mel, labels, device,
                 args.hop, args.vad)

    print("\n" + "=" * 78)
    print(f"PARAMETER SWEEP - model {args.name}, hop {args.hop}s")
    print("=" * 78)
    print(f"{'smooth':>7}{'thresh':>8}{'hit rate':>11}{'wrong':>7}"
          f"{'false acc':>11}{'latency':>10}")
    print("-" * 78)
    for row in rows:
        print(f"{row['smoothing']:>7}{row['threshold']:>8.2f}"
              f"{row['hit_rate']:>10.0%} {row['wrong']:>6}"
              f"{row['false_accepts']:>11}{row['latency_ms']:>8.0f}ms")

    # A usable configuration: no wrong command, and distractor speech
    # almost never fires. Among those, take the highest hit rate, and
    # break ties with the lower latency.
    usable = [r for r in rows if r["wrong"] == 0 and r["false_accepts"] <= 1]
    if not usable:
        usable = [r for r in rows if r["wrong"] == 0]
    if usable:
        best = max(usable, key=lambda r: (r["hit_rate"], -r["latency_ms"]))
        print("\n" + "=" * 78)
        print("RECOMMENDED")
        print("=" * 78)
        print(f"  --smoothing {best['smoothing']} "
              f"--threshold {best['threshold']:.2f}")
        print(f"  hit rate {best['hit_rate']:.0%}, "
              f"{best['wrong']} wrong commands, "
              f"{best['false_accepts']} false accepts, "
              f"~{best['latency_ms']:.0f} ms decision latency")
        print("\n  per word:")
        for label, (hits, total) in best["per_word"].items():
            flag = "  <- weak" if hits / max(total, 1) < 0.8 else ""
            print(f"    {label:<10} {hits}/{total}{flag}")
    print()


if __name__ == "__main__":
    main()
