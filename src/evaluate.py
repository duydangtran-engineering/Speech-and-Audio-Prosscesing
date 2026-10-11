"""Evaluation report: per-class metrics, confusion matrix, learning curves.

    python src/evaluate.py                 # uses models/dscnn.*
    python src/evaluate.py --name cnn
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import MODELS_DIR


def load_history(name: str):
    path = MODELS_DIR / f"{name}_history.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.array([float(row[key]) for row in rows])
        for key in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="dscnn")
    parser.add_argument("--save", action="store_true", help="save figures to models/")
    args = parser.parse_args()

    data = np.load(MODELS_DIR / f"{args.name}_test_predictions.npz",
                   allow_pickle=True)
    pred, true, labels = data["pred"], data["true"], list(data["labels"])

    print("\n" + "=" * 60)
    print(f"CLASSIFICATION REPORT - {args.name}")
    print("=" * 60)
    print(classification_report(true, pred, target_names=labels,
                                digits=3, zero_division=0))

    matrix = confusion_matrix(true, pred, labels=range(len(labels)))

    # The pairs the model actually confuses. This is the table that tells
    # you which command word to replace.
    print("Most confused pairs:")
    errors = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and matrix[i, j] > 0:
                errors.append((matrix[i, j], labels[i], labels[j]))
    for count, actual, predicted in sorted(errors, reverse=True)[:8]:
        print(f"  {actual:>10} -> {predicted:<10} {count} times")
    if not errors:
        print("  none")

    history = load_history(args.name)
    n_panels = 3 if history else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    axes = np.atleast_1d(axes)

    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(normalized, annot=matrix, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                cbar_kws={"label": "recall"}, vmin=0, vmax=1)
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("actual")
    axes[0].set_title(f"Confusion matrix - {args.name}")

    if history:
        axes[1].plot(history["epoch"], history["train_loss"], label="train")
        axes[1].plot(history["epoch"], history["val_loss"], label="val")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
        axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)

        axes[2].plot(history["epoch"], history["train_acc"], label="train")
        axes[2].plot(history["epoch"], history["val_acc"], label="val")
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("accuracy")
        axes[2].set_title("Accuracy"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if args.save:
        out = MODELS_DIR / f"{args.name}_report.png"
        plt.savefig(out, dpi=150)
        print(f"\nsaved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
