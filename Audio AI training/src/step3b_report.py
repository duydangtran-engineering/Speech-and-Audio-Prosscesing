"""Báo cáo cho 1 người nói: model đoán đúng cái gì, với độ tự tin bao nhiêu.

    python step3b_report.py laptop          # trung thực: train lại, bỏ laptop ra
    python step3b_report.py laptop --saved  # dùng luôn model.joblib
    python step3b_report.py --list          # xem ai có trong dataset

Mặc định sẽ train lại model sau khi bỏ người nói đó ra, rồi chấm điểm trên
clip của họ. model.joblib đã được fit trên mọi clip, kể cả clip của người đó,
nên chấm điểm bằng nó chẳng khác nào chấm bài bằng chính đáp án - `--saved`
làm đúng như vậy và chỉ hữu ích để xem model đã xuất xưởng hoạt động ra sao
trên dữ liệu nó đã "học thuộc".

Bảng độ tự tin ở cuối mới là thứ hữu ích thực tế: nó cho biết nên truyền giá
trị --threshold nào cho live_demo.py, dựa trên đo đạc thay vì đoán mò.
"""
import argparse
import sys

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import ACTION_MAP, LABELS, MODEL_DIR


def load():
    path = MODEL_DIR / "dataset.npz"
    if not path.exists():
        sys.exit(f"Không thấy {path} - chạy step2_dataset.py trước.")
    d = np.load(path, allow_pickle=True)
    is_aug = d["is_aug"] if "is_aug" in d else np.zeros(len(d["y"]), bool)
    return d["X"], d["y"].astype(str), d["speakers"].astype(str), is_aug


def fit_without(speaker, X, y, speakers):
    from step3_train import MODELS
    model = clone(MODELS["random_forest"])
    train = speakers != speaker
    print(f"Đang train trên {train.sum()} clip của "
          f"{len(set(speakers[train]))} người nói khác...")
    return model.fit(X[train], y[train])


def confidence_table(y_true, proba, classes):
    """Mỗi ngưỡng sẽ đổi lại được gì, đo trên đúng người nói này."""
    best = classes[np.argmax(proba, axis=1)]
    top = proba.max(axis=1)
    is_command = y_true != "noise"
    is_command &= y_true != "silence"

    print("\nngưỡng     kích hoạt   đúng      bỏ sót   sai")
    print("            (% số lệnh đã nói)")
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
        fired = (top >= t) & (best != "noise") & (best != "silence")
        cmd = is_command
        hit = (fired & cmd & (best == y_true)).sum()
        miss = (cmd & ~fired).sum()
        bad = (fired & cmd & (best != y_true)).sum()
        false_fire = (fired & ~cmd).sum()
        n = cmd.sum()
        print(f"  {t:.2f}      {fired.sum():5d}   {100*hit/n:6.1f}%  "
              f"{100*miss/n:6.1f}%  {100*bad/n:5.1f}%"
              + (f"   (+{false_fire} báo động giả trên noise/silence)"
                 if false_fire else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker", nargs="?")
    ap.add_argument("--saved", action="store_true",
                    help="dùng model.joblib thay vì train lại (bị lộ đề)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    X, y, speakers, is_aug = load()

    if args.list or not args.speaker:
        print("Người nói có trong dataset:")
        for s in sorted(set(speakers)):
            n = (speakers == s).sum()
            labels = sorted(set(y[speakers == s]))
            print(f"  {s:10s} {n:5d} clip   {', '.join(labels)}")
        return

    if args.speaker not in set(speakers):
        sys.exit(f"Không có người nói '{args.speaker}'. Dùng --list để xem danh sách.")

    test = (speakers == args.speaker) & ~is_aug
    if args.saved:
        import joblib
        model = joblib.load(MODEL_DIR / "model.joblib")["model"]
        print("CẢNH BÁO: model.joblib đã được train trên cả những clip này. "
              "Con số này bị thổi phồng.\n")
    else:
        model = fit_without(args.speaker, X, y, speakers)

    proba = model.predict_proba(X[test])
    pred = model.classes_[np.argmax(proba, axis=1)]
    truth = y[test]

    labels = [l for l in LABELS if l in set(truth)]
    print(f"\n=== {args.speaker}: {test.sum()} clip ===")
    print(f"accuracy {accuracy_score(truth, pred):.3f}")
    print(classification_report(truth, pred, labels=labels, digits=3,
                                zero_division=0))

    act_true = np.array([ACTION_MAP.get(v, "IGNORE") for v in truth])
    act_pred = np.array([ACTION_MAP.get(v, "IGNORE") for v in pred])
    print(f"action level {accuracy_score(act_true, act_pred):.3f}")

    print("\nMỗi nhãn thực tế bị đoán thành gì (hàng = đã nói, cột = đoán được):")
    cm = confusion_matrix(truth, pred, labels=labels)
    width = max(len(l) for l in labels) + 1
    print(" " * width + "".join(f"{l[:6]:>7}" for l in labels))
    for name, row in zip(labels, cm):
        print(f"{name:<{width}}" + "".join(f"{v:7d}" for v in row))

    confidence_table(truth, proba, model.classes_)
    print("\nChọn ngưỡng ở chỗ 'sai' vẫn gần bằng 0 mà 'đúng' đã ngừng tăng, "
          "rồi truyền ngưỡng đó cho live_demo.py.")


if __name__ == "__main__":
    main()
