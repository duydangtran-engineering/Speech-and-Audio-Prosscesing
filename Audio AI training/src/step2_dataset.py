"""Bước 2: biến các clip 1 giây thành ma trận đặc trưng (X, y, speaker)."""
import numpy as np

import audio_io
from config import CLIPS_DIR, LABELS, MODEL_DIR, RANDOM_SEED, SAMPLE_RATE
from features import FEATURE_NAMES, extract

AUG_MARK = "__aug"

# Giới hạn số clip mỗi nhãn. None nghĩa là dùng hết. Giới hạn này quan trọng
# khi 1 lớp áp đảo các lớp còn lại: "noise" tăng nhanh vì file noise bị chặt
# bằng sliding window, và 1 lớp lớn gấp 3 lần các lớp lệnh khác sẽ khiến rừng
# cây thiên vị trả lời "noise".
MAX_PER_LABEL = 1000


def pick(wavs, limit, rng):
    """Rút gọn 1 nhãn xuống còn `limit` clip, lấy đều từ mỗi người nói.

    Lấy mẫu ngẫu nhiên thuần túy sẽ làm mỏng mỗi người nói theo cùng 1 tỉ lệ,
    điều này gây hại nhiều nhất cho người có ít clip nhất - và người ít clip
    nhất ở đây chính là micro mục tiêu, người nói mà bản demo thực sự phụ
    thuộc vào. Nên chia đều ngân sách cho mỗi người nói, để người có ít clip
    giữ nguyên hết, và dùng phần dư cho những người còn nhiều clip để lấy thêm.
    """
    by_speaker = {}
    for wav in wavs:
        by_speaker.setdefault(wav.stem.split("_")[0], []).append(wav)

    kept, pools, budget = [], dict(by_speaker), limit
    while pools and budget > 0:
        share = max(1, budget // len(pools))
        for speaker in list(pools):
            if budget <= 0:
                break
            take = min(share, len(pools[speaker]), budget)
            picked = set(rng.choice(np.array(pools[speaker], dtype=object),
                                    take, replace=False))
            kept.extend(picked)
            budget -= take
            pools[speaker] = [w for w in pools[speaker] if w not in picked]
            if not pools[speaker]:
                del pools[speaker]
    return sorted(kept)


def build():
    rng = np.random.default_rng(RANDOM_SEED)
    X, y, speakers, files, is_aug = [], [], [], [], []

    for label in LABELS:
        folder = CLIPS_DIR / label
        if not folder.exists():
            continue

        wavs = sorted(folder.glob("*.wav"))
        total = len(wavs)
        if MAX_PER_LABEL and total > MAX_PER_LABEL:
            wavs = pick(wavs, MAX_PER_LABEL, rng)

        for wav in wavs:
            audio = audio_io.load(wav, sr=SAMPLE_RATE)
            X.append(extract(audio))
            y.append(label)
            # quy ước tên file: <người nói>_<lần thu>_<chỉ số>.wav
            speakers.append(wav.stem.split("_")[0])
            files.append(str(wav))
            is_aug.append(AUG_MARK in wav.stem)

        note = f"  (đã giới hạn từ {total})" if total > len(wavs) else ""
        print(f"{label:8s}: {len(wavs)} mẫu{note}")

    X = np.vstack(X)
    y = np.array(y)
    speakers = np.array(speakers)
    is_aug = np.array(is_aug)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_DIR / "dataset.npz", X=X, y=y, speakers=speakers,
             files=np.array(files), is_aug=is_aug,
             feature_names=np.array(FEATURE_NAMES))
    print(f"\nX shape = {X.shape}  ({is_aug.sum()} augmented) "
          f"-> {MODEL_DIR / 'dataset.npz'}")


if __name__ == "__main__":
    build()
