"""Bước 3: train và so sánh Random Forest với SVM, rồi lưu model tốt hơn.

Các quy tắc đánh giá giúp con số đáng tin cậy:

1. Held-out speaker (giữ riêng 1 người nói ra ngoài). Mỗi fold test trên 1
   giọng hoàn toàn vắng mặt lúc train, nên điểm số phản ánh đúng trải nghiệm
   của 1 người lạ dùng thiết bị - chứ không phải những người đã thu dữ liệu.
2. Clip augment chỉ dùng để train, không bao giờ dùng để test. Một bản sao
   trộn tạp âm của 1 clip không phải là bằng chứng mới: chấm điểm trên nó sẽ
   hoặc làm đẹp điểm số (vì bản gốc sạch của nó đã có trong tập train) hoặc
   phạt oan model vì 1 đề thi khó hơn. Fold test chỉ chứa bản ghi gốc, nên
   điểm số vẫn so sánh được dù có chạy step1b_augment.py hay không.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import ACTION_MAP, LABELS, MODEL_DIR, RANDOM_SEED, REPORT_DIR

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    "svm_rbf": make_pipeline(
        StandardScaler(),
        SVC(C=10, gamma="scale", kernel="rbf", class_weight="balanced",
            random_state=RANDOM_SEED)),
}


def plot_confusion(y_true, y_pred, labels, name):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d",
        xticks_rotation=45)
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def speaker_folds(speakers, is_aug):
    """Mỗi fold ứng với 1 người nói: train trên mọi người khác, test trên clip gốc của họ."""
    for held_out in sorted(set(speakers)):
        train = np.where(speakers != held_out)[0]
        test = np.where((speakers == held_out) & ~is_aug)[0]
        if len(test):
            yield held_out, train, test


def evaluate(model, X, y, speakers, is_aug):
    y_true, y_pred, per_speaker = [], [], {}
    for name, train, test in speaker_folds(speakers, is_aug):
        fitted = clone(model).fit(X[train], y[train])
        pred = fitted.predict(X[test])
        per_speaker[name] = accuracy_score(y[test], pred)
        y_true.append(y[test])
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred), per_speaker


def main():
    data = np.load(MODEL_DIR / "dataset.npz", allow_pickle=True)
    X, y, speakers = data["X"], data["y"], data["speakers"]
    is_aug = data["is_aug"] if "is_aug" in data else np.zeros(len(y), bool)
    labels = [l for l in LABELS if l in set(y)]

    print(f"{len(y)} clip, {(~is_aug).sum()} gốc, {is_aug.sum()} augment")
    print(f"người nói: {', '.join(sorted(set(speakers)))}\n")

    scores = {}
    for name, model in MODELS.items():
        y_true, y_pred, per_speaker = evaluate(model, X, y, speakers, is_aug)
        acc = accuracy_score(y_true, y_pred)
        scores[name] = acc

        print(f"===== {name} | accuracy held-out-speaker = {acc:.3f} =====")
        print("  " + "  ".join(f"{s}={v:.3f}" for s, v in per_speaker.items()))
        print(classification_report(y_true, y_pred, labels=labels, digits=3))
        plot_confusion(y_true, y_pred, labels, name)

        words = y_true != "noise"
        print(f"--- chỉ tính lệnh ({words.sum()} clip) = "
              f"{accuracy_score(y_true[words], y_pred[words]):.3f}")

        # Nhầm "on" thành "bat" vẫn kích đúng mã IR, nên đây mới là con số
        # thiết bị thực sự làm đúng.
        act_true = np.array([ACTION_MAP.get(v, "IGNORE") for v in y_true])
        act_pred = np.array([ACTION_MAP.get(v, "IGNORE") for v in y_pred])
        print(f"--- action level = {accuracy_score(act_true, act_pred):.3f}\n")
        plot_confusion(act_true, act_pred, sorted(set(act_true)),
                       f"{name}_action")

    best = max(scores, key=scores.get)
    print(f"Model tốt nhất: {best} ({scores[best]:.3f})")

    final = clone(MODELS[best]).fit(X, y)
    joblib.dump({"model": final, "labels": labels}, MODEL_DIR / "model.joblib")
    print(f"Đã lưu -> {MODEL_DIR / 'model.joblib'}")

    if best == "random_forest":
        names = data["feature_names"]
        order = np.argsort(final.feature_importances_)[::-1][:10]
        print("\nĐặc trưng quan trọng nhất:")
        for i in order:
            print(f"  {names[i]:16s} {final.feature_importances_[i]:.4f}")


if __name__ == "__main__":
    main()
