"""Leave-one-speaker-out cross-validation.

    python src/cross_validate.py                 # augmented, all speakers
    python src/cross_validate.py --no-augment
    python src/cross_validate.py --epochs 200

A single speaker-held-out run answers "how well does it work on THIS
person", and with five speakers that answer swings by ten points or more
depending on who happened to be drawn. Comparing two models across runs
where the test speaker changed compares two different exams.

This trains once per speaker, each time holding that speaker out, and
reports the mean and spread. The spread is the number that says how much
to trust any single run.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ALL_LABELS, COMMAND_LABELS, MODELS_DIR
from dataset import (
    SPECIAL_LABEL_NAMES,
    Augment,
    KeywordDataset,
    class_weights,
    load_all_clips,
    load_noise_bank,
    split_by_speaker,
)
from features import LogMel, SpecAugment
from model import build_model


def run_fold(clips, test_speaker, val_speaker, args, device):
    train_clips, val_clips, test_clips = split_by_speaker(
        clips, force_test=test_speaker, force_val=val_speaker, quiet=True)

    augment = None
    if not args.no_augment:
        augment = Augment(load_noise_bank(train_clips), seed=args.seed)

    train_loader = DataLoader(KeywordDataset(train_clips, augment),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(KeywordDataset(val_clips), batch_size=args.batch_size)
    test_loader = DataLoader(KeywordDataset(test_clips), batch_size=args.batch_size)

    log_mel = LogMel().to(device)
    spec_augment = SpecAugment().to(device)
    model = build_model(args.model, len(ALL_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_clips, len(ALL_LABELS)).to(device),
        label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train(); spec_augment.train()
        for waveform, target in train_loader:
            waveform, target = waveform.to(device), target.to(device)
            spec = log_mel(waveform)
            if not args.no_augment:
                spec = spec_augment(spec)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(spec), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = seen = 0
        with torch.no_grad():
            for waveform, target in val_loader:
                waveform, target = waveform.to(device), target.to(device)
                correct += (model(log_mel(waveform)).argmax(1) == target).sum().item()
                seen += target.size(0)
        accuracy = correct / max(seen, 1)
        if accuracy > best_val:
            best_val = accuracy
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch + 1}/{args.epochs}  val {accuracy:.4f}",
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for waveform, target in test_loader:
            waveform = waveform.to(device)
            preds.append(model(log_mel(waveform)).argmax(1).cpu().numpy())
            trues.append(target.numpy())
    pred, true = np.concatenate(preds), np.concatenate(trues)
    return float((pred == true).mean()), pred, true


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dscnn", choices=["dscnn", "cnn"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clips = load_all_clips()
    speakers = sorted({c.speaker for c in clips
                       if c.label_name not in SPECIAL_LABEL_NAMES})
    if len(speakers) < 4:
        raise SystemExit(f"Need 4+ command speakers, found {len(speakers)}.")

    tag = args.tag or ("clean" if args.no_augment else "aug")
    print(f"Leave-one-speaker-out over {len(speakers)} speakers: {speakers}")
    print(f"model {args.model}, {args.epochs} epochs, "
          f"augment {'off' if args.no_augment else 'on'}\n")

    scores = []
    confusion = np.zeros((len(ALL_LABELS), len(ALL_LABELS)), dtype=int)

    for i, held_out in enumerate(speakers):
        val_speaker = speakers[(i + 1) % len(speakers)]
        print(f"[{i + 1}/{len(speakers)}] test={held_out}  val={val_speaker}")
        accuracy, pred, true = run_fold(clips, held_out, val_speaker, args, device)
        scores.append(accuracy)
        for t, p in zip(true, pred):
            confusion[t, p] += 1
        print(f"    -> test accuracy {accuracy:.4f}\n", flush=True)

    scores = np.array(scores)
    print("=" * 62)
    print(f"CROSS-VALIDATION - {tag}")
    print("=" * 62)
    for speaker, score in zip(speakers, scores):
        print(f"  held out {speaker:<8} {score:.4f}")
    print("-" * 62)
    print(f"  mean {scores.mean():.4f}   std {scores.std():.4f}   "
          f"min {scores.min():.4f}   max {scores.max():.4f}")
    print(f"\n  A difference smaller than {2 * scores.std():.3f} between two "
          f"models\n  is inside the speaker-to-speaker noise of this dataset.")

    print("\nPer-word recall, pooled over all folds:")
    for i, label in enumerate(ALL_LABELS):
        total = confusion[i].sum()
        if total:
            print(f"  {label:<12} {confusion[i, i] / total:.3f}  "
                  f"({confusion[i, i]}/{total})")

    print("\nTop confusions, pooled:")
    pairs = [(confusion[i, j], ALL_LABELS[i], ALL_LABELS[j])
             for i in range(len(ALL_LABELS))
             for j in range(len(ALL_LABELS)) if i != j and confusion[i, j]]
    for count, actual, predicted in sorted(pairs, reverse=True)[:8]:
        print(f"  {actual:>10} -> {predicted:<12} {count}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / f"cv_{tag}.json").write_text(json.dumps({
        "tag": tag, "speakers": speakers, "scores": scores.tolist(),
        "mean": float(scores.mean()), "std": float(scores.std()),
        "confusion": confusion.tolist(), "labels": ALL_LABELS,
        "args": vars(args),
    }, indent=2), encoding="utf-8")
    print(f"\nsaved -> models/cv_{tag}.json\n")


if __name__ == "__main__":
    main()
