"""Bước 1d (tùy chọn): xếp hạng các clip nghi là tiếng ồn/gió bị bộ tách từ
cắt nhầm thành 1 "từ", để bạn chỉ cần nghe 1 danh sách ngắn thay vì mò qua
hàng nghìn clip.

CHỈ xét trong 7 nhãn từ lệnh (tang/giam/bat/tat/on/off/mode). "noise" và
"silence" KHÔNG được xếp hạng ở đây, vì 2 lớp đó vốn dĩ được thiết kế để
chứa tạp âm/im lặng - nếu tính chung, chúng sẽ chiếm gần hết top nghi ngờ
(vì đúng bản chất chúng đã "giống ồn") và che mất các clip lỗi thật sự nằm
trong 7 từ lệnh mà công cụ này cần tìm.

Không có công cụ nào đoán chắc 100% "đây là gió, đây là từ thật" chỉ bằng số
liệu - nên script này KHÔNG tự xoá gì cả. Nó chỉ tính điểm nghi ngờ cho mỗi
clip dựa trên 3 dấu hiệu của tiếng ồn nền (gió, quạt, tiếng cà micro...) so
với 1 từ nói bình thường:

  - flatness (độ "phẳng" của phổ): tiếng ồn (gió, hiss) trải đều năng lượng
    trên mọi tần số -> flatness cao. Giọng nói có formant rõ -> flatness thấp.
  - peakiness (độ "nhọn" của đường bao năng lượng): 1 từ nói ra có 1 đỉnh rõ
    rệt rồi tắt; gió/tiếng cà micro thường kêu đều đều suốt cả đoạn -> đường
    bao phẳng, không có đỉnh rõ.
  - duty (tỉ lệ % của cả clip 1 giây có tiếng to): tiếng ồn kéo dài lấp gần
    hết clip; 1 từ ngắn (150-1000 ms) thường chỉ chiếm 1 phần của clip.

Cách dùng:
    python step1d_flag_noisy.py
        -> ghi bảng xếp hạng đầy đủ vào reports/suspect_noise_clips.csv
        -> copy (không xoá bản gốc) TOP_N clip nghi ngờ nhất vào
           data/clips_review/<nhãn>/, xếp theo đúng thứ tự nghi ngờ

    Mở data/clips_review/, nghe qua bằng Media Player/VLC (mở cả thư mục,
    bấm next liên tục sẽ nhanh hơn mở từng file). File nào đúng là tiếng ồn/
    gió thì XOÁ NGAY TRONG THƯ MỤC clips_review đó (không đụng vào data/clips).

    python step1d_flag_noisy.py --apply
        -> so sánh với danh sách đã copy lúc trước (manifest), file nào bạn
           đã xoá trong clips_review thì bị xoá luôn khỏi data/clips.
           File bạn giữ lại trong clips_review thì không bị đụng tới.

Chạy lại step2_dataset.py sau khi --apply để dataset không còn các clip đó.
"""
import argparse
import shutil
from pathlib import Path

import librosa

import audio_io
from config import (ACTION_MAP, CLIPS_DIR, FEAT_HOP, FEAT_WIN, LABELS, N_FFT,
                    REPORT_DIR, SAMPLE_RATE, TRIM_REL_THRESHOLD)

# "noise" và "silence" có ACTION_MAP là IGNORE và vốn dĩ được thu để giống
# tạp âm - loại chúng ra, chỉ xếp hạng trong các từ lệnh thật sự.
COMMAND_LABELS = [l for l in LABELS if ACTION_MAP.get(l) != "IGNORE"]

REVIEW_DIR = CLIPS_DIR.parent / "clips_review"
MANIFEST = REVIEW_DIR / "_manifest.txt"
TOP_N = 150     # số clip nghi ngờ nhất được copy ra để nghe


def score_clip(y):
    rms = librosa.feature.rms(y=y, frame_length=FEAT_WIN, hop_length=FEAT_HOP)[0]
    threshold = max(rms.max() * TRIM_REL_THRESHOLD, 1e-6)
    active = rms > threshold
    if active.sum() < 3:
        return None      # gần như im lặng hoàn toàn, không phải trường hợp này

    flat = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT,
                                             hop_length=FEAT_HOP,
                                             win_length=FEAT_WIN)[0]
    flat_active = float(flat[active].mean())
    peakiness = float((rms.max() - rms[active].mean()) / rms.max())
    duty = float(active.mean())

    # phổ càng phẳng, đường bao càng ít nhọn, càng kéo dài -> càng giống ồn
    score = flat_active - peakiness + 0.5 * duty
    return dict(score=score, flat=flat_active, peakiness=peakiness, duty=duty)


def rank():
    rows = []
    for label in COMMAND_LABELS:
        folder = CLIPS_DIR / label
        if not folder.exists():
            continue
        for wav in sorted(folder.glob("*.wav")):
            y = audio_io.load(wav, sr=SAMPLE_RATE)
            info = score_clip(y)
            if info is None:
                continue
            rows.append((info["score"], label, wav, info))

    rows.sort(key=lambda r: -r[0])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / "suspect_noise_clips.csv"
    with open(report, "w", encoding="utf-8") as f:
        f.write("score,label,file,flatness,peakiness,duty\n")
        for score, label, wav, info in rows:
            f.write(f"{score:.4f},{label},{wav},{info['flat']:.4f},"
                    f"{info['peakiness']:.4f},{info['duty']:.4f}\n")
    print(f"Đã ghi bảng xếp hạng đầy đủ ({len(rows)} clip, chỉ trong "
          f"{', '.join(COMMAND_LABELS)}) -> {report}")
    return rows


def copy_for_review(rows):
    if REVIEW_DIR.exists():
        # ignore_errors=True: 1 thư mục/file có thể đang bị Windows khoá (do
        # Explorer đang xem trước, hoặc Media Player còn mở file từ lần nghe
        # trước) - bỏ qua phần không xoá được thay vì dừng hẳn chương trình.
        # Thường gặp nhất là thư mục "noise"/"silence" sót lại từ phiên bản
        # code cũ (trước khi giới hạn chỉ còn 7 nhãn lệnh).
        shutil.rmtree(REVIEW_DIR, ignore_errors=True)
        if REVIEW_DIR.exists():
            print(f"Lưu ý: không xoá hết được {REVIEW_DIR} cũ (có file đang bị "
                  "khoá) - đóng hết Explorer/Media Player đang mở thư mục đó "
                  "rồi chạy lại nếu muốn dọn sạch. Vẫn tiếp tục ghi đè.")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    top = rows[:TOP_N]
    manifest_lines = []
    for rank_i, (score, label, wav, info) in enumerate(top, start=1):
        dst_dir = REVIEW_DIR / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        # số thứ tự ở đầu tên để trình phát nhạc mở theo đúng thứ tự nghi ngờ
        dst = dst_dir / f"{rank_i:03d}__{wav.name}"
        shutil.copy2(wav, dst)
        manifest_lines.append(f"{dst.relative_to(REVIEW_DIR)}\t{wav}")

    MANIFEST.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Đã copy {len(top)} clip nghi ngờ nhất vào {REVIEW_DIR}")
    print("\nBước tiếp theo:")
    print(f"  1. Mở {REVIEW_DIR}, nghe qua từng nhãn (VLC/Media Player mở cả")
    print("     thư mục rồi bấm next sẽ nhanh hơn mở từng file).")
    print("  2. File nào đúng là gió/tiếng ồn thì XOÁ NGAY trong thư mục đó")
    print("     (không đụng vào data/clips).")
    print("  3. Chạy: python step1d_flag_noisy.py --apply")


def apply():
    if not MANIFEST.exists():
        raise SystemExit("Chưa có manifest - chạy step1d_flag_noisy.py "
                         "(không có --apply) trước.")

    removed, kept = 0, 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        review_rel, original = line.split("\t")
        review_path = REVIEW_DIR / review_rel
        if review_path.exists():
            kept += 1
            continue
        # bạn đã xoá bản copy trong clips_review -> xoá luôn bản gốc
        original_path = Path(original)
        if original_path.exists():
            original_path.unlink()
            removed += 1

    print(f"Đã xoá {removed} clip khỏi {CLIPS_DIR} (giữ lại {kept}).")
    print("Chạy lại step2_dataset.py rồi step3_train.py để cập nhật model.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="đồng bộ những gì đã xoá trong clips_review vào data/clips")
    args = ap.parse_args()

    if args.apply:
        apply()
    else:
        copy_for_review(rank())
