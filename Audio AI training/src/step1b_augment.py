"""Bước 1b (tùy chọn): tăng số clip lớp lệnh bằng cách trộn thêm tạp âm nền.

Chỉ áp dụng trộn tạp âm (noise mixing). Đã thử đổi pitch và đổi tốc độ trên
chính bộ dữ liệu của dự án này, cả hai đều làm accuracy held-out-speaker TỆ
HƠN:

    không augment          0.957
    đổi pitch +-1.5 nửa cung 0.918
    đổi pitch +-3 nửa cung   0.932
    đổi tốc độ 0.9 / 1.1     0.943
    hạ âm lượng (gain)       0.948
    trộn tạp âm              0.963   <- cách duy nhất giúp ích

Tạp âm được lấy từ data/clips/noise nên phần nền được thêm vào là tiếng ồn
thật trong phòng của chính các bản ghi nhóm, không phải noise tổng hợp.

File augment được ghi cạnh file gốc với tên <tên>__augN.wav. Chúng vẫn giữ
tiền tố tên người nói, nên GroupKFold vẫn giữ mọi bản sao của 1 người nói
trong cùng 1 fold, không có clip augment nào rò rỉ sang fold test.

Chạy SAU step1_split.py và TRƯỚC step2_dataset.py:
    python step1b_augment.py          # tạo clip đã augment
    python step1b_augment.py --clean  # xóa chúng đi
"""
import sys

import numpy as np
import soundfile as sf

import audio_io
from config import CLIPS_DIR, LABELS, RANDOM_SEED, SAMPLE_RATE

AUG_MARK = "__aug"
# Đã thử 5 dB trước tiên và bị hại: ở mức đó clip lệnh bắt đầu nghe giống
# chính lớp noise, khiến model nhầm lẫn giữa 2 lớp.
SNR_DB = [20, 12]         # mỗi mức SNR tạo 1 bản sao
GAUSSIAN_SIGMA = 0.005    # dùng khi không có clip noise nào để lấy mẫu


def load_noise_pool(rng):
    """Nền phải là tiếng ồn phòng, tuyệt đối không phải lớp 'noise'.

    Lớp 'noise' phần lớn là người nói chuyện - đó chính là mục đích của nó.
    Trộn nó vào clip lệnh sẽ đặt thêm 1 giọng nói thứ 2 dưới từ khóa, khiến
    model học rằng lệnh có chứa giọng nói ở nền, làm nhòe ranh giới giữa lệnh
    và noise. Chỉ có lớp 'silence' (quạt, tiếng xe, phòng trống) là an toàn
    để trộn.
    """
    folder = CLIPS_DIR / "silence"
    files = [f for f in sorted(folder.glob("*.wav")) if AUG_MARK not in f.stem] \
        if folder.exists() else []
    if not files:
        print("Không thấy data/clips/silence - dùng tạm gaussian noise.\n"
              "Nên thu 2-3 phút tiếng nền yên tĩnh mỗi người nói để có nền "
              "tốt hơn hiss tổng hợp.")
        return []
    picks = rng.choice(files, min(60, len(files)), replace=False)
    return [audio_io.load(f, sr=SAMPLE_RATE) for f in picks]


def mix(clip, noise, snr_db, rng):
    if noise is None:
        return np.clip(clip + rng.normal(0, GAUSSIAN_SIGMA, len(clip)), -1, 1)

    if len(noise) < len(clip):
        noise = np.tile(noise, int(np.ceil(len(clip) / len(noise))))
    start = rng.integers(0, len(noise) - len(clip) + 1)
    noise = noise[start:start + len(clip)]

    clip_power = np.mean(clip ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(clip_power / (noise_power * 10 ** (snr_db / 10)))
    return np.clip(clip + noise * scale, -1, 1)


def clean():
    removed = 0
    for label in LABELS:
        folder = CLIPS_DIR / label
        if not folder.exists():
            continue
        for f in folder.glob(f"*{AUG_MARK}*.wav"):
            f.unlink()
            removed += 1
    print(f"Đã xóa {removed} clip augment.")


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    pool = load_noise_pool(rng)

    for label in LABELS:
        if label in ("noise", "silence"):
            continue                      # 2 lớp này vốn đã là nền rồi
        folder = CLIPS_DIR / label
        if not folder.exists():
            continue

        originals = [f for f in sorted(folder.glob("*.wav"))
                     if AUG_MARK not in f.stem]
        made = 0
        for path in originals:
            clip = audio_io.load(path, sr=SAMPLE_RATE)
            for i, snr in enumerate(SNR_DB):
                noise = pool[rng.integers(len(pool))] if pool else None
                out = folder / f"{path.stem}{AUG_MARK}{i}.wav"
                sf.write(out, mix(clip, noise, snr, rng), SAMPLE_RATE,
                         subtype="PCM_16")
                made += 1
        print(f"{label:6s}: {len(originals)} gốc -> {made} clip augment")


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean()
    else:
        main()
