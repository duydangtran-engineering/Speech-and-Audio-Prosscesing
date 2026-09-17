"""Bước 4: chuyển Random Forest đã train thành file header C cho ESP32.

emlearn chỉ chuyển đổi phần bộ phân lớp (classifier). Phần trích đặc trưng
(MFCC + ZCR + RMS + pitch) phải được viết lại bằng C trên thiết bị, với
đúng tham số như bên Python, nếu không các con số đưa vào cây quyết định sẽ
không khớp với dữ liệu lúc train.
"""
import joblib
import numpy as np

from config import ACTION_MAP, MODEL_DIR
from features import FEATURE_NAMES

OUT = MODEL_DIR / "voice_model.h"


def main():
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    model, labels = bundle["model"], bundle["labels"]

    if not hasattr(model, "estimators_"):
        raise SystemExit("emlearn export chỉ hỗ trợ Random Forest. "
                         "Train lại với random_forest là model tốt nhất.")

    import emlearn
    cmodel = emlearn.convert(model, method="inline")
    cmodel.save(file=str(OUT), name="voice_model")

    header = OUT.read_text()
    meta = [
        "",
        "// ---- metadata sinh tự động ----",
        f"#define VOICE_N_FEATURES {len(FEATURE_NAMES)}",
        f"#define VOICE_N_CLASSES {len(labels)}",
        "static const char *VOICE_LABELS[] = {"
        + ", ".join(f'"{l}"' for l in labels) + "};",
        "",
        "typedef enum { ACT_IGNORE, ACT_POWER_ON, ACT_POWER_OFF,",
        "               ACT_VOLUME_UP, ACT_VOLUME_DOWN, ACT_MODE_SWITCH } voice_action_t;",
        "static const voice_action_t VOICE_ACTIONS[] = {"
        + ", ".join(f"ACT_{ACTION_MAP.get(l, 'IGNORE')}" for l in labels) + "};",
        "// thứ tự đặc trưng: " + ", ".join(FEATURE_NAMES),
    ]
    OUT.write_text(header + "\n".join(meta) + "\n")

    print(f"Đã lưu -> {OUT}")
    print("Copy file này vào cạnh sketch .ino rồi gọi:")
    print('  int idx = voice_model_predict(features, VOICE_N_FEATURES);')


if __name__ == "__main__":
    main()
