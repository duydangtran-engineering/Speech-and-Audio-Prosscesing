"""Real-time keyword spotting from the microphone.

    python src/live_demo.py                    # uses models/clean3.pt
    python src/live_demo.py --name aug3
    python src/live_demo.py --threshold 0.85 --device 1

This is the bridge between "the model scores well offline" and "the game
is playable". Offline accuracy says nothing about the three things that
actually decide playability:

  1. how often normal speech triggers a command (false accepts),
  2. how long it takes from finishing a word to getting a decision,
  3. whether one spoken word produces one command or five.

Those are what the VAD gate, the confidence threshold, the posterior
smoothing and the refractory period below are for.
"""

from __future__ import annotations

import argparse
import queue
import time

import numpy as np
import sounddevice as sd
import torch

from config import CLIP_SAMPLES, COMMAND_LABELS, MODELS_DIR, SAMPLE_RATE
from features import LogMel
from model import build_model

HOP_SECONDS = 0.2          # how often a decision is made
SMOOTHING_WINDOWS = 3      # posteriors averaged over this many windows
REFRACTORY_SECONDS = 0.7   # silence after firing, so one word = one command
VAD_RMS = 0.012            # below this the window is treated as silence


class RingBuffer:
    """Keeps the most recent CLIP_SAMPLES samples."""

    def __init__(self, size: int):
        self.buffer = np.zeros(size, dtype=np.float32)

    def push(self, block: np.ndarray) -> None:
        n = len(block)
        if n >= len(self.buffer):
            self.buffer[:] = block[-len(self.buffer):]
        else:
            self.buffer[:-n] = self.buffer[n:]
            self.buffer[-n:] = block

    def read(self) -> np.ndarray:
        return self.buffer.copy()

    def clear(self) -> None:
        """Forget the audio held so far.

        Called right after a command fires. Without this, a long word is
        still sitting in the 1-second window when the refractory period
        ends, and the same utterance fires a second time.
        """
        self.buffer[:] = 0.0


def load_model(name: str, device: torch.device):
    path = MODELS_DIR / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}. Train a model first.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    labels = checkpoint["labels"]
    model = build_model(checkpoint["model"], len(labels)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"loaded {path.name}  ({checkpoint['model']}, "
          f"val_acc {checkpoint.get('val_acc', float('nan')):.4f})")
    return model, labels


def bar(value: float, width: int = 24) -> str:
    filled = int(max(0.0, min(1.0, value)) * width)
    return "#" * filled + "-" * (width - filled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="clean3")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="minimum smoothed confidence to fire a command")
    parser.add_argument("--device", type=int, help="input device index")
    parser.add_argument("--show-all", action="store_true",
                        help="print every window, not just fired commands")
    parser.add_argument("--diagnose", action="store_true",
                        help="print the top-3 classes for every active window")
    parser.add_argument("--smoothing", type=int, default=SMOOTHING_WINDOWS,
                        help="windows averaged before deciding (1 = off)")
    parser.add_argument("--vad", type=float, default=VAD_RMS,
                        help="RMS below which a window counts as silence")
    parser.add_argument("--hop", type=float, default=HOP_SECONDS)
    parser.add_argument("--refractory", type=float, default=REFRACTORY_SECONDS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels = load_model(args.name, device)
    log_mel = LogMel().to(device)
    command_ids = {i for i, label in enumerate(labels) if label in COMMAND_LABELS}

    if args.device is not None:
        sd.default.device = (args.device, None)

    ring = RingBuffer(CLIP_SAMPLES)
    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio status: {status}]")
        blocks.put(indata[:, 0].copy())

    history: list[np.ndarray] = []
    last_fired = 0.0
    armed = True          # re-armed only after a silent window
    latencies: list[float] = []
    fired_counts: dict[str, int] = {}

    peak_confidence: dict[str, float] = {}

    print(f"\nthreshold {args.threshold}  |  hop {args.hop}s  |  "
          f"smoothing {args.smoothing}  |  refractory {args.refractory}s  |  "
          f"vad {args.vad}")
    print("Speak a command. Ctrl+C to stop and see the summary.\n")

    block_size = int(args.hop * SAMPLE_RATE)
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=block_size,
                            callback=callback):
            while True:
                block = blocks.get()
                ring.push(block)
                window = ring.read()
                start = time.perf_counter()

                rms = float(np.sqrt(np.mean(window ** 2)))
                if rms < args.vad:
                    history.clear()
                    # A silent window ends the utterance and re-arms the
                    # detector. The timer alone is not enough: a long word
                    # is still in the buffer after the refractory expires,
                    # which is what made one spoken word fire twice.
                    armed = True
                    if args.show_all or args.diagnose:
                        print(f"  [silence] rms={rms:.4f}          ", end="\r")
                    continue

                with torch.no_grad():
                    tensor = torch.from_numpy(window).unsqueeze(0).to(device)
                    probs = torch.softmax(model(log_mel(tensor)), dim=1)[0]
                latencies.append((time.perf_counter() - start) * 1000)

                history.append(probs.cpu().numpy())
                if len(history) > max(1, args.smoothing):
                    history.pop(0)
                smoothed = np.mean(history, axis=0)

                index = int(np.argmax(smoothed))
                confidence = float(smoothed[index])
                label = labels[index]

                # Track how high each class ever got, even when it never
                # crossed the threshold. This is what tells you whether a
                # word is being misclassified or merely under-confident.
                for i, value in enumerate(smoothed):
                    name = labels[i]
                    if value > peak_confidence.get(name, 0.0):
                        peak_confidence[name] = float(value)

                if args.diagnose:
                    top = np.argsort(smoothed)[::-1][:3]
                    parts = " | ".join(
                        f"{labels[i]}={smoothed[i]:.2f}" for i in top)
                    print(f"  rms={rms:.3f}  {parts}                ", end="\r")
                elif args.show_all:
                    print(f"  {label:>10} {confidence:.2f} [{bar(confidence)}]",
                          end="\r")

                now = time.time()
                if (index in command_ids
                        and confidence >= args.threshold
                        and armed
                        and now - last_fired >= args.refractory):
                    # Trailing spaces overwrite the leftover \r diagnose line.
                    print(f"  >> {label.upper():<8} {confidence:.2f} "
                          f"[{bar(confidence)}]" + " " * 30)
                    fired_counts[label] = fired_counts.get(label, 0) + 1
                    last_fired = now
                    armed = False
                    history.clear()
                    ring.clear()   # one utterance must produce one command

    except KeyboardInterrupt:
        print("\n\n" + "=" * 52)
        print("SESSION SUMMARY")
        print("=" * 52)
        if latencies:
            # Skip the first window: it includes CUDA kernel warm-up and is
            # not representative of steady-state inference.
            array = np.array(latencies[1:] if len(latencies) > 1 else latencies)
            print(f"inference latency: mean {array.mean():.1f} ms, "
                  f"p95 {np.percentile(array, 95):.1f} ms, "
                  f"max {array.max():.1f} ms")
            print(f"windows processed: {len(latencies)}")

            # What the player actually feels: the model cannot decide until
            # the word has been seen by `smoothing` consecutive windows.
            pipeline_ms = args.hop * 1000 * args.smoothing + array.mean()
            print(f"end-to-end decision latency: about {pipeline_ms:.0f} ms "
                  f"({args.smoothing} x {args.hop * 1000:.0f} ms hop "
                  f"+ {array.mean():.1f} ms inference)")
        print("\nper-command: times fired, and the highest smoothed"
              " confidence ever reached")
        for label in COMMAND_LABELS:
            count = fired_counts.get(label, 0)
            peak = peak_confidence.get(label, 0.0)
            note = ""
            if count == 0 and peak >= args.threshold * 0.75:
                note = "  <- recognized but under threshold"
            elif count == 0:
                note = "  <- never recognized"
            print(f"  {label:<10} fired {count:>3}   peak {peak:.2f}{note}")

        if not fired_counts:
            print("\nNo command fired. Lower --threshold or check the mic.")
        print()


if __name__ == "__main__":
    main()
