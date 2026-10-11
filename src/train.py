"""Train a keyword-spotting model.

    python src/train.py                          # dscnn, 40 epochs, augmented
    python src/train.py --model cnn
    python src/train.py --no-augment             # ablation baseline
    python src/train.py --epochs 60 --lr 2e-3

Writes models/<run-name>.pt and models/<run-name>_history.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ALL_LABELS, MODELS_DIR
from dataset import (
    Augment,
    KeywordDataset,
    class_weights,
    load_all_clips,
    load_noise_bank,
    split_by_speaker,
)
from features import LogMel, SpecAugment
from model import build_model, count_parameters


def evaluate(model, log_mel, loader, device, criterion):
    model.eval()
    total_loss = correct = seen = 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for waveform, target in loader:
            waveform, target = waveform.to(device), target.to(device)
            logits = model(log_mel(waveform))
            total_loss += criterion(logits, target).item() * target.size(0)
            pred = logits.argmax(1)
            correct += (pred == target).sum().item()
            seen += target.size(0)
            all_pred.append(pred.cpu())
            all_true.append(target.cpu())
    return (total_loss / max(seen, 1), correct / max(seen, 1),
            torch.cat(all_pred).numpy(), torch.cat(all_true).numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dscnn", choices=["dscnn", "cnn"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-specaugment", action="store_true")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-shift", action="store_true")
    parser.add_argument("--no-speed", action="store_true")
    parser.add_argument("--name", help="run name, defaults to the model name")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # --- data ---------------------------------------------------------
    clips = load_all_clips()
    if not clips:
        print("No clips found. Record data first.")
        return
    print(f"loaded {len(clips)} clips, {len(ALL_LABELS)} classes")

    train_clips, val_clips, test_clips = split_by_speaker(clips, seed=args.seed)
    print(f"train {len(train_clips)} | val {len(val_clips)} | test {len(test_clips)}")

    augment = None
    if not args.no_augment:
        augment = Augment(
            load_noise_bank(train_clips),
            time_shift=not args.no_shift,
            add_noise=not args.no_noise,
            speed=not args.no_speed,
            seed=args.seed,
        )

    loaders = {
        "train": DataLoader(KeywordDataset(train_clips, augment),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=False),
        "val": DataLoader(KeywordDataset(val_clips),
                          batch_size=args.batch_size, num_workers=0),
        "test": DataLoader(KeywordDataset(test_clips),
                           batch_size=args.batch_size, num_workers=0),
    }

    # --- model --------------------------------------------------------
    log_mel = LogMel().to(device)
    spec_augment = SpecAugment().to(device)
    model = build_model(args.model, len(ALL_LABELS), dropout=args.dropout).to(device)
    print(f"model: {args.model}, {count_parameters(model):,} parameters")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_clips, len(ALL_LABELS)).to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # --- train --------------------------------------------------------
    run_name = args.name or args.model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{run_name}.pt"
    history_path = MODELS_DIR / f"{run_name}_history.csv"

    best_val = 0.0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        spec_augment.train()
        running_loss = correct = seen = 0

        for waveform, target in loaders["train"]:
            waveform, target = waveform.to(device), target.to(device)
            spec = log_mel(waveform)
            if not args.no_specaugment:
                spec = spec_augment(spec)

            optimizer.zero_grad(set_to_none=True)
            logits = model(spec)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item() * target.size(0)
            correct += (logits.argmax(1) == target).sum().item()
            seen += target.size(0)

        scheduler.step()
        train_loss, train_acc = running_loss / seen, correct / seen
        val_loss, val_acc, _, _ = evaluate(model, log_mel, loaders["val"],
                                           device, criterion)
        history.append((epoch, train_loss, train_acc, val_loss, val_acc))

        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": args.model,
                "state_dict": model.state_dict(),
                "labels": ALL_LABELS,
                "epoch": epoch,
                "val_acc": val_acc,
                "args": vars(args),
            }, checkpoint_path)
            marker = "  <- best, saved"

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train {train_loss:.4f}/{train_acc:.4f}  "
              f"val {val_loss:.4f}/{val_acc:.4f}{marker}")

    elapsed = time.time() - start
    print(f"\ntrained in {elapsed:.1f}s ({elapsed / args.epochs:.2f}s per epoch)")

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(history)

    # --- final test with the best checkpoint --------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc, pred, true = evaluate(model, log_mel, loaders["test"],
                                               device, criterion)
    print(f"\nbest val acc : {best_val:.4f} (epoch {checkpoint['epoch']})")
    print(f"TEST accuracy: {test_acc:.4f}")

    np.savez(MODELS_DIR / f"{run_name}_test_predictions.npz",
             pred=pred, true=true, labels=np.array(ALL_LABELS))
    (MODELS_DIR / f"{run_name}_summary.json").write_text(
        json.dumps({"run": run_name, "model": args.model,
                    "params": count_parameters(model),
                    "best_val_acc": best_val, "test_acc": test_acc,
                    "train_seconds": elapsed, "args": vars(args)}, indent=2),
        encoding="utf-8")
    print(f"saved -> {checkpoint_path.name}")


if __name__ == "__main__":
    main()
