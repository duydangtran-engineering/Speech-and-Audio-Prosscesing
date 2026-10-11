"""Export a trained checkpoint to ONNX for the browser.

    python src/export_onnx.py --name final

Writes into web/:
  model.onnx            the classifier, input (1, 1, 40, 101) log-mel
  model_meta.json       labels and the feature settings the page must use
  parity_reference.json one waveform with the log-mel and logits Python
                        produced for it, so the browser can prove its own
                        front end agrees before anyone trusts a demo

Feature extraction stays in JavaScript rather than being folded into the
graph: onnxruntime-web has no FFT op, and keeping the front end explicit
makes the mismatch testable instead of invisible.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from config import ALL_LABELS, CLIP_SAMPLES, MODELS_DIR, ROOT, SAMPLE_RATE
from features import (
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_MELS,
    WIN_LENGTH,
    LogMel,
)
from model import build_model, count_parameters

WEB_DIR = ROOT / "web"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="final")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint_path = MODELS_DIR / f"{args.name}.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"No checkpoint at {checkpoint_path}")

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    labels = checkpoint["labels"]
    model = build_model(checkpoint["model"], len(labels)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"loaded {checkpoint_path.name}  "
          f"({checkpoint['model']}, {count_parameters(model):,} params)")

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = WEB_DIR / "model.onnx"

    dummy = torch.zeros(1, 1, N_MELS, N_FRAMES)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["logmel"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}},
    )
    size_kb = onnx_path.stat().st_size / 1024
    print(f"exported -> web/model.onnx  ({size_kb:.0f} KB)")

    meta = {
        "labels": labels,
        "commandLabels": [l for l in labels if not l.startswith("_")],
        "features": {
            "sampleRate": SAMPLE_RATE,
            "clipSamples": CLIP_SAMPLES,
            "nFft": N_FFT,
            "winLength": WIN_LENGTH,
            "hopLength": HOP_LENGTH,
            "nMels": N_MELS,
            "fMin": F_MIN,
            "fMax": F_MAX,
            "nFrames": N_FRAMES,
            "logEpsilon": 1e-6,
        },
        # Defaults chosen in tune_realtime.py; the page may override them.
        "runtime": {"threshold": 0.80, "smoothing": 1, "hop": 0.2,
                    "vadRms": 0.012, "refractory": 0.3},
        "checkpoint": {"name": args.name, "model": checkpoint["model"],
                       "valAcc": float(checkpoint.get("val_acc", 0.0))},
    }
    (WEB_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")
    print("exported -> web/model_meta.json")

    # --- reference case for the browser parity check -------------------
    rng = np.random.default_rng(7)
    t = np.arange(CLIP_SAMPLES) / SAMPLE_RATE
    tone = (np.sin(2 * np.pi * 130 * t) + 0.5 * np.sin(2 * np.pi * 260 * t)
            + 0.3 * np.sin(2 * np.pi * 520 * t))
    envelope = np.exp(-((t - 0.45) ** 2) / (2 * 0.1 ** 2))
    waveform = (tone * envelope * 0.3
                + rng.normal(0, 0.005, CLIP_SAMPLES)).astype(np.float32)

    log_mel = LogMel()
    with torch.no_grad():
        spec = log_mel(torch.from_numpy(waveform).unsqueeze(0))
        logits = model(spec)[0].numpy()

    (WEB_DIR / "parity_reference.json").write_text(json.dumps({
        "waveform": waveform.astype(float).tolist(),
        "mel": spec[0, 0].numpy().astype(float).tolist(),
        "logits": logits.astype(float).tolist(),
        "labels": labels,
    }), encoding="utf-8")
    print("exported -> web/parity_reference.json")

    # --- verify the ONNX graph matches PyTorch -------------------------
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nonnxruntime not installed - skipping graph check.")
        print("  pip install onnxruntime")
        return

    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"logmel": spec.numpy()})[0][0]
    difference = float(np.max(np.abs(onnx_logits - logits)))
    print(f"\nPyTorch vs ONNX max abs diff: {difference:.3e}  "
          f"{'OK' if difference < 1e-4 else 'MISMATCH'}")
    print(f"predicted class: {labels[int(np.argmax(onnx_logits))]}")


if __name__ == "__main__":
    main()
