"""Measure real-time hit rate per command word.

    python src/test_live.py --name clean3
    python src/test_live.py --name clean3 --trials 10 --labels xuong len

The offline confusion matrix scores single, pre-trimmed clips. It cannot
tell you that one word wins 95% of the time while another wins 40%, which
is what decides whether the game feels responsive or broken.

Each trial records a fixed window, then replays the exact realtime
pipeline over it (sliding windows, VAD, smoothing, threshold) and records
whether the right command would have fired.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd
import torch

from config import CLIP_SAMPLES, COMMAND_LABELS, LABEL_PROMPT, SAMPLE_RATE
from features import LogMel
from live_demo import load_model

TRIAL_SECONDS = 2.0


def detect(audio: np.ndarray, model, log_mel, labels, device, *,
           hop: float, smoothing: int, threshold: float, vad: float,
           refractory: float = 0.7):
    """Replay the realtime detector over a recorded window.

    This mirrors live_demo.py exactly, including the ring buffer: audio
    arrives in hop-sized blocks, and firing a command both clears the
    posterior history and wipes the buffer. Slicing the array directly
    instead would report double-fires that the live system never makes.

    Returns (fired_label, confidence, n_fires, best_label, best_confidence)
    where `best_*` is the strongest command hypothesis regardless of the
    threshold — the difference between the two tells you whether a miss
    was a misclassification or merely low confidence.
    """
    hop_samples = max(1, int(hop * SAMPLE_RATE))
    history: list[np.ndarray] = []
    fired: list[tuple[str, float]] = []
    best_label, best_confidence = None, 0.0
    command_ids = {i for i, l in enumerate(labels) if l in COMMAND_LABELS}

    ring = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    locked_until = -1.0   # minimum gap between commands
    armed = True          # re-armed only after a silent window

    for start in range(0, len(audio) - hop_samples + 1, hop_samples):
        block = audio[start:start + hop_samples]
        ring[:-hop_samples] = ring[hop_samples:]
        ring[-hop_samples:] = block
        now = (start + hop_samples) / SAMPLE_RATE

        if float(np.sqrt(np.mean(ring ** 2))) < vad:
            history.clear()
            # A silent window marks the end of an utterance. This, not the
            # timer, is what guarantees one command per spoken word: a long
            # word stays in the buffer well past any fixed refractory.
            armed = True
            continue

        with torch.no_grad():
            tensor = torch.from_numpy(ring.copy()).unsqueeze(0).to(device)
            probs = torch.softmax(model(log_mel(tensor)), dim=1)[0].cpu().numpy()

        history.append(probs)
        if len(history) > max(1, smoothing):
            history.pop(0)
        smoothed = np.mean(history, axis=0)

        for i in command_ids:
            if smoothed[i] > best_confidence:
                best_confidence, best_label = float(smoothed[i]), labels[i]

        index = int(np.argmax(smoothed))
        if (index in command_ids and smoothed[index] >= threshold
                and armed and now >= locked_until):
            fired.append((labels[index], float(smoothed[index])))
            history.clear()
            ring[:] = 0.0
            armed = False
            locked_until = now + refractory

    if fired:
        return fired[0][0], fired[0][1], len(fired), best_label, best_confidence
    return None, 0.0, 0, best_label, best_confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="clean3")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--labels", nargs="*", default=COMMAND_LABELS)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("--hop", type=float, default=0.2)
    parser.add_argument("--vad", type=float, default=0.012)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    if args.device is not None:
        sd.default.device = (args.device, None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels = load_model(args.name, device)
    log_mel = LogMel().to(device)

    print(f"\n{args.trials} trials per word, threshold {args.threshold}, "
          f"smoothing {args.smoothing}, hop {args.hop}s")
    print("Say the word ONCE per trial, the way you would while playing.\n")

    results: dict[str, dict] = {}

    for label in args.labels:
        print(f"--- {LABEL_PROMPT.get(label, label)} ---")
        hits = doubles = 0
        confidences: list[float] = []
        peaks: list[float] = []
        wrong: dict[str, int] = {}

        for trial in range(1, args.trials + 1):
            input(f"  [{trial}/{args.trials}] ENTER then say it > ")
            audio = sd.rec(int(TRIAL_SECONDS * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()

            got, conf, n_fires, best, best_conf = detect(
                audio, model, log_mel, labels, device,
                hop=args.hop, smoothing=args.smoothing,
                threshold=args.threshold, vad=args.vad)
            peaks.append(best_conf)

            if got == label:
                hits += 1
                confidences.append(conf)
                extra = "  (FIRED TWICE)" if n_fires > 1 else ""
                if n_fires > 1:
                    doubles += 1
                print(f"      hit   {conf:.2f}{extra}")
            elif got is None:
                print(f"      miss  (no fire; best guess {best} {best_conf:.2f})")
            else:
                wrong[got] = wrong.get(got, 0) + 1
                print(f"      WRONG -> {got} {conf:.2f}")

        results[label] = {
            "hits": hits,
            "doubles": doubles,
            "mean_conf": float(np.mean(confidences)) if confidences else 0.0,
            "mean_peak": float(np.mean(peaks)),
            "wrong": wrong,
        }
        print()

    # --- report -------------------------------------------------------
    print("=" * 72)
    print(f"REAL-TIME HIT RATE - model {args.name}")
    print("=" * 72)
    print(f"{'word':<10}{'hit rate':>12}{'mean conf':>12}"
          f"{'mean peak':>12}{'doubles':>10}  confused with")
    print("-" * 72)
    for label in args.labels:
        r = results[label]
        rate = r["hits"] / args.trials
        confused = ", ".join(f"{k}({v})" for k, v in r["wrong"].items()) or "-"
        print(f"{label:<10}{r['hits']}/{args.trials} ({rate:>5.0%}) "
              f"{r['mean_conf']:>11.2f}{r['mean_peak']:>12.2f}"
              f"{r['doubles']:>10}  {confused}")

    total_hits = sum(r["hits"] for r in results.values())
    total = args.trials * len(args.labels)
    print("-" * 72)
    print(f"overall: {total_hits}/{total} ({total_hits / total:.0%})")

    weak = [l for l in args.labels if results[l]["hits"] / args.trials < 0.8]
    if weak:
        print(f"\nBelow 80%: {', '.join(weak)}")
        for label in weak:
            r = results[label]
            if r["mean_peak"] >= args.threshold * 0.8 and not r["wrong"]:
                print(f"  {label}: recognized but under-confident "
                      f"(peak {r['mean_peak']:.2f}) -> lower the threshold "
                      f"or reduce smoothing")
            elif r["wrong"]:
                print(f"  {label}: genuinely confused with "
                      f"{', '.join(r['wrong'])} -> record more of both")
            else:
                print(f"  {label}: not detected at all "
                      f"(peak {r['mean_peak']:.2f}) -> more training data, "
                      f"or consider a different word")
    print()


if __name__ == "__main__":
    main()
