"""Demo trực tiếp: nói 1 câu lệnh và nó in ra từ đó. Còn lại thì im lặng.

    python live_demo.py                     # nghe qua micro mặc định
    python live_demo.py --list-devices      # liệt kê thiết bị input
    python live_demo.py --device 2          # chọn 1 thiết bị cụ thể
    python live_demo.py --threshold 0.55    # model phải chắc chắn đến mức nào
    python live_demo.py --debug             # hiện cả những gì bị từ chối
    python live_demo.py --file some.m4a     # chạy trên 1 bản ghi thay vì mic

Cần thêm 1 package:
    pip install sounddevice

Cách nó quyết định giữ im lặng:
  1. Âm thanh phải đủ to và đủ dài để trông giống 1 từ.
  2. Model không được phân loại nó là noise hoặc silence.
  3. Lớp thắng phải vượt --threshold, và vượt lớp á quân ít nhất --margin.
     1 từ mà model còn đang phân vân thì không phải là 1 lệnh.
"""
import argparse
import queue
import sys
import time

import joblib
import numpy as np

from config import (ACTION_MAP, CLIP_SAMPLES, MAX_WORD_MS, MIN_WORD_MS,
                    MODEL_DIR, SAMPLE_RATE)
from features import extract

BLOCK_MS = 20                       # tần suất nhìn vào input
SILENCE_MS = 150                    # khoảng lặng để kết thúc 1 từ
PRE_ROLL_MS = 120                   # audio giữ lại từ trước khi từ bắt đầu
REFRACTORY_S = 0.6                  # bỏ qua từ mới ngay sau khi vừa kích hoạt
CALIBRATE_S = 1.0                   # đo tiếng nền phòng trước khi bắt đầu nghe
NOISE_MULTIPLE = 3.0                # giọng nói = to hơn tiếng phòng chừng này lần
FLOOR = 0.003                       # không bao giờ tin ngưỡng thấp hơn mức này

# Các từ ngắn, nhỏ tiếng - "bat", "tat", "off" - là những từ hay bị cổng
# (gate) nuốt mất: chúng dài 240-330 ms và kết thúc bằng phụ âm vô thanh.
# Nếu chúng không bao giờ kích hoạt, vấn đề là cổng đang đặt quá cao, không
# phải model đoán sai. Theo dõi bằng --debug: 1 từ đến được model sẽ in ra lý
# do; 1 từ không bao giờ xuất hiện nghĩa là chưa từng được "nghe thấy" -
# --gate là tham số để chỉnh việc đó.

IGNORED = {"noise", "silence"}


def load_model():
    path = MODEL_DIR / "model.joblib"
    if not path.exists():
        sys.exit(f"Không thấy {path} - chạy step3_train.py trước.")
    bundle = joblib.load(path)
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        sys.exit("Model đã lưu không báo được độ tự tin. Train lại với "
                 "random_forest, hoặc thêm probability=True cho SVM.")
    return model


def classify(model, audio, threshold, margin):
    """Trả về (label, confidence, reason) - label là None nghĩa là im lặng."""
    proba = model.predict_proba(extract(audio).reshape(1, -1))[0]
    order = np.argsort(proba)[::-1]
    best, second = model.classes_[order[0]], model.classes_[order[1]]
    top, runner_up = proba[order[0]], proba[order[1]]

    if best in IGNORED:
        return None, top, f"bị phân loại là {best}"
    if top < threshold:
        return None, top, f"{best} chỉ {top:.2f}, dưới mức {threshold:.2f}"
    if top - runner_up < margin:
        return None, top, f"{best} {top:.2f} vs {second} {runner_up:.2f} - quá sát nhau"
    return best, top, ""


def report(label, confidence):
    action = ACTION_MAP.get(label, "IGNORE")
    bar = "#" * int(confidence * 20)
    print(f"  {label:<6} {confidence:.2f} [{bar:<20}]  -> {action}")


def run_file(model, path, threshold, margin, debug):
    """Áp dụng đúng logic quyết định, nhưng chạy trên 1 bản ghi thay vì micro."""
    import audio_io
    import librosa
    from step1_split import detect_words, to_fixed_length

    y = librosa.util.normalize(audio_io.load(path, sr=SAMPLE_RATE))
    segments, _ = detect_words(y)
    print(f"Tìm thấy {len(segments)} đoạn có tiếng trong {path}\n")
    for start, end in segments:
        label, confidence, reason = classify(
            model, to_fixed_length(y[start:end]), threshold, margin)
        stamp = f"[{start / SAMPLE_RATE:6.2f}s]"
        if label:
            print(stamp, end="")
            report(label, confidence)
        elif debug:
            print(f"{stamp}  (im lặng: {reason})")


def run_mic(model, device, threshold, margin, debug, gate_override=None,
            meter=False):
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("pip install sounddevice")

    block = int(SAMPLE_RATE * BLOCK_MS / 1000)
    blocks = queue.Queue()

    def callback(indata, frames, time_info, status):
        blocks.put(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=block, device=device, callback=callback)
    with stream:
        print("Đang đo tiếng nền phòng, giữ im lặng...")
        room = []
        deadline = time.time() + CALIBRATE_S
        while time.time() < deadline:
            room.append(np.sqrt(np.mean(blocks.get() ** 2)))
        room_level = float(np.median(room))
        gate = gate_override or max(room_level * NOISE_MULTIPLE, FLOOR)
        print(f"Mức nền phòng {room_level:.4f}, ngưỡng giọng nói {gate:.4f}"
              + ("  (đặt tay)" if gate_override else ""))

        if meter:
            print("\nĐồng hồ mức âm lượng. Nói mỗi lệnh vài lần và quan sát "
                  "đỉnh.\nNếu 1 từ đạt đỉnh dưới ngưỡng, nó sẽ không bao giờ "
                  "đến được model - hạ --gate xuống.\nCtrl+C khi xong.\n")
            while True:
                peak = max(np.sqrt(np.mean(blocks.get() ** 2))
                           for _ in range(10))          # cửa sổ 200 ms
                mark = "<== vượt ngưỡng" if peak > gate else ""
                bar = "#" * min(40, int(peak / max(gate, 1e-6) * 10))
                print(f"  {peak:.4f} [{bar:<40}] {mark}")

        print("Đang nghe. Nói 1 câu lệnh. Ctrl+C để dừng.\n")

        pre_roll = int(PRE_ROLL_MS / BLOCK_MS)
        need_silence = int(SILENCE_MS / BLOCK_MS)
        recent, word, quiet_for = [], [], 0
        in_word = False
        last_fired = 0.0

        while True:
            chunk = blocks.get()
            loud = np.sqrt(np.mean(chunk ** 2)) > gate

            if not in_word:
                recent.append(chunk)
                del recent[:-pre_roll]
                if loud and time.time() - last_fired > REFRACTORY_S:
                    in_word = True
                    word = recent + [chunk]
                    recent, quiet_for = [], 0
                continue

            word.append(chunk)
            quiet_for = 0 if loud else quiet_for + 1
            too_long = len(word) * BLOCK_MS > MAX_WORD_MS + PRE_ROLL_MS

            if quiet_for >= need_silence or too_long:
                in_word = False
                audio = np.concatenate(word)
                voiced_ms = (len(word) - quiet_for) * BLOCK_MS
                word = []

                if voiced_ms < MIN_WORD_MS:
                    if debug:
                        peak = float(np.sqrt(np.mean(audio ** 2)))
                        print(f"  (im lặng: chỉ {voiced_ms}ms có tiếng, "
                              f"mức {peak:.4f} so với ngưỡng {gate:.4f})")
                    continue

                if len(audio) > CLIP_SAMPLES:
                    centre = len(audio) // 2
                    half = CLIP_SAMPLES // 2
                    audio = audio[centre - half:centre - half + CLIP_SAMPLES]

                label, confidence, reason = classify(model, audio, threshold,
                                                     margin)
                if label:
                    report(label, confidence)
                    last_fired = time.time()
                elif debug:
                    print(f"  (im lặng: {reason})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="độ tự tin tối thiểu để chấp nhận 1 lệnh")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="lớp thắng phải vượt lớp á quân bao nhiêu")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--file", help="chạy trên 1 bản ghi thay vì micro")
    ap.add_argument("--debug", action="store_true",
                    help="in cả lý do vì sao 1 âm thanh bị từ chối")
    ap.add_argument("--gate", type=float, default=None,
                    help="đặt tay ngưỡng âm lượng (xem --meter)")
    ap.add_argument("--meter", action="store_true",
                    help="chỉ hiện mức âm lượng input, để chọn --gate")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    model = load_model()
    print(f"Model biết các lớp: {', '.join(model.classes_)}")
    print(f"threshold {args.threshold}  margin {args.margin}\n")

    if args.file:
        run_file(model, args.file, args.threshold, args.margin, args.debug)
    else:
        try:
            run_mic(model, args.device, args.threshold, args.margin,
                    args.debug, args.gate, args.meter)
        except KeyboardInterrupt:
            print("\nĐã dừng.")


if __name__ == "__main__":
    main()
