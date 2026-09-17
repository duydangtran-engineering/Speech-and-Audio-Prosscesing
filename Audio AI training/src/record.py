"""Thu dữ liệu training bằng đúng chiếc micro mà bản demo sẽ dùng để nghe.

Vì sao việc này quan trọng: model chỉ train trên bản ghi từ điện thoại sẽ
nghe micro laptop như 1 thứ chưa từng gặp bao giờ. Độ tự tin sụp đổ ở mọi
lớp và dự đoán trôi dạt về phía lớp nào đó tình cờ nằm gần nhất trong không
gian đặc trưng. Thêm bản ghi thu bằng đúng micro mục tiêu sẽ sửa được việc
này, và không có lượng dữ liệu điện thoại nào bù lại được.

    python record.py --name laptop                     # tất cả nhãn
    python record.py --name laptop --labels tang tat    # chỉ 2 nhãn này
    python record.py --name laptop --seconds 40         # thu dài hơn
    python record.py --name inmp441 --device 3          # chọn micro cụ thể

Mỗi lượt thu được ghi vào data/raw/<nhãn>/<tên>.wav, nên cái tên bạn truyền
vào sẽ trở thành speaker id cho GroupKFold. Dùng 1 tên chưa ai dùng, và dùng
ĐÚNG cùng 1 tên cho mọi nhãn trong 1 buổi thu.
"""
import argparse
import sys
import time

import numpy as np
import soundfile as sf

from config import LABELS, RAW_DIR, SAMPLE_RATE

PROMPT = {
    "tang": "TANG  - nói, nghỉ, lặp lại",
    "giam": "GIAM  - nói, nghỉ, lặp lại",
    "bat": "BAT   - nói, nghỉ, lặp lại",
    "tat": "TAT   - nói, nghỉ, lặp lại",
    "on": "ON    - nói, nghỉ, lặp lại",
    "off": "OFF   - nói, nghỉ, lặp lại",
    "mode": "MODE  - nói, nghỉ, lặp lại",
    "noise": "NOISE - nói chuyện bình thường, đọc gì đó, nói các từ gần "
             "giống lệnh (tan, tam, bac, bang) - bất cứ gì trừ 1 câu lệnh",
    "silence": "SILENCE - không nói gì cả, chỉ để phòng được ghi âm tự nhiên",
}


def countdown(seconds=3):
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  BẮT ĐẦU  ")


def record(seconds, device):
    import sounddevice as sd
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32", device=device)
    for remaining in range(seconds, 0, -1):
        print(f"  đang thu... còn {remaining:2d}s", end="\r", flush=True)
        time.sleep(1)
    sd.wait()
    print(" " * 40, end="\r")
    return audio[:, 0]


def check(audio, gap_expected):
    """Cảnh báo 2 lỗi thường gặp làm hỏng 1 lượt thu."""
    peak = float(np.abs(audio).max())
    quiet = float(np.mean(np.abs(audio) < peak * 1e-4)) * 100

    notes = []
    if peak < 0.05:
        notes.append(f"quá nhỏ (đỉnh {peak:.3f}) - nói gần micro hơn")
    if peak > 0.99:
        notes.append("bị clip (vỡ tiếng) - lùi ra xa hoặc hạ input level")
    if quiet > 15:
        notes.append(f"{quiet:.0f}% lượt thu này là im lặng tuyệt đối - "
                     "noise suppression đang bật, tắt nó trong sound settings")
    if gap_expected and quiet < 2:
        notes.append("gần như không có khoảng lặng giữa các từ - hãy nghỉ "
                     "rõ ràng hơn, nếu không bộ tách từ sẽ không tìm được "
                     "ranh giới giữa các từ")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="speaker id, ví dụ laptop hoặc inmp441")
    ap.add_argument("--labels", nargs="*", default=LABELS)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    try:
        import sounddevice  # noqa: F401
    except ImportError:
        sys.exit("pip install sounddevice")

    if "_" in args.name:
        sys.exit("Dùng tên không có dấu '_' - nó là ký tự phân tách speaker id.")

    print(f"Thu với tên '{args.name}', {args.seconds}s mỗi nhãn.")
    print("Nhắm khoảng ~20 lần lặp mỗi lượt thu. Nghỉ rõ ràng giữa các từ.\n")

    for label in args.labels:
        folder = RAW_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{args.name}.wav"

        if out.exists():
            if input(f"{out.name} đã tồn tại trong {label}/. Ghi đè? [y/N] ").lower() != "y":
                continue

        print(f"\n=== {PROMPT.get(label, label.upper())}")
        input("    Nhấn Enter khi sẵn sàng...")
        countdown()
        audio = record(args.seconds, args.device)

        for note in check(audio, gap_expected=label not in ("noise", "silence")):
            print(f"    CẢNH BÁO: {note}")

        sf.write(out, audio, SAMPLE_RATE, subtype="PCM_16")
        print(f"    đã lưu {out}")

    print("\nXong. Giờ chạy:")
    print("  Remove-Item ..\\data\\clips -Recurse -Force")
    print("  python step1_split.py")
    print("  python step1c_validate.py")
    print("  python step2_dataset.py")
    print("  python step3_train.py")


if __name__ == "__main__":
    main()
