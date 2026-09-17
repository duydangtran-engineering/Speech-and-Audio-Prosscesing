"""Trích đặc trưng, dùng chung cho cả lúc train và lúc inference.

Các quyết định thiết kế dưới đây (đều đã đo đạc, không phải đoán mò):

1. Clip 1 giây được cắt lại về phần có tiếng nói trước tiên. Lấy trung bình
   MFCC trên cả phần im lặng bị đệm vào sẽ khiến vector mô tả "clip này im
   lặng bao nhiêu" - tức tốc độ nói của người nói - thay vì mô tả từ đó là gì.
2. Các frame MFCC được pool vào N_BINS bin thời gian thay vì lấy 1 trung bình
   toàn cục. Trung bình toàn cục làm mất hình dạng của từ, khiến "tang" và
   "tat" chồng lên nhau.
3. Chuẩn hóa trung bình cepstral theo từng clip (cepstral mean normalisation -
   trừ đi trung bình MFCC của chính clip đó) loại bỏ độ lệch do micro và độ
   lớn giọng nói, trong khi vẫn giữ được hình dạng (contour) của từ. Việc này
   chạy được trên thiết bị mà không cần đăng ký giọng người dùng trước.
4. Pitch (F0) được thêm vào theo thang log2, chuẩn hóa theo từng clip. MFCC
   mô tả bao phổ (formant, âm sắc) nhưng gần như "mù" với tần số cơ bản, nên
   gần như không thấy được thanh điệu tiếng Việt. "tăng" (thanh ngang - bằng
   phẳng) và "tắt" (thanh sắc - đi lên) là cặp từ mà chính README của dự án
   này đã ghi nhận là dễ nhầm nhất - đó là vấn đề về thanh điệu, không phải
   về âm sắc, nên đặc trưng này được thêm vào để giải quyết đúng vấn đề đó.
5. RMS energy được thêm theo đúng cách ZCR đã có sẵn (pool theo từng bin thời
   gian). Trước đây RMS chỉ tồn tại bên trong trim_silence() rồi bị bỏ đi;
   giờ đường bao năng lượng của clip (nguyên âm to, phụ âm nhỏ) được giữ lại.

Đo trên tập chia theo người nói - held-out-speaker (tang vs mode, 4 người):
    trung bình/std toàn cục trên clip đã đệm    0.743   <- thiết kế cũ
    chia bin thời gian, chưa chuẩn hóa          0.903
    chia bin + CMN theo từng clip               0.951
    + delta + ZCR theo bin                      0.957   <- bản trước của file này
    + RMS theo bin + pitch contour              ???     <- chạy lại step2_dataset.py
                                                             và step3_train.py, điền
                                                             số vào đây, và xem lại ô
                                                             tang/tat trong
                                                             reports/confusion_*.png

Cấu trúc vector đặc trưng:
    [mfcc_bin_means  (N_MFCC*N_BINS),
     mfcc_std        (N_MFCC),
     delta_bin_means (N_MFCC*N_BINS),
     zcr_bin_means   (N_BINS),
     rms_bin_means   (N_BINS),
     pitch_bin_means (N_BINS),   # log2(f0/ref), đã trừ trung bình theo clip -> contour thanh điệu
     voiced_duration (1)]
"""
import numpy as np
import librosa

from config import (CLIP_SAMPLES, FEAT_HOP, FEAT_WIN, N_BINS, N_FFT, N_MFCC,
                    PITCH_FMAX, PITCH_FMIN, PITCH_REF_HZ, SAMPLE_RATE,
                    TRIM_MARGIN_FRAMES, TRIM_REL_THRESHOLD)

FEATURE_NAMES = (
    [f"mfcc{c}_bin{b}" for b in range(N_BINS) for c in range(N_MFCC)]
    + [f"mfcc{c}_std" for c in range(N_MFCC)]
    + [f"delta{c}_bin{b}" for b in range(N_BINS) for c in range(N_MFCC)]
    + [f"zcr_bin{b}" for b in range(N_BINS)]
    + [f"rms_bin{b}" for b in range(N_BINS)]
    + [f"pitch_bin{b}" for b in range(N_BINS)]
    + ["voiced_duration"]
)


def pad_or_trim(y):
    if len(y) > CLIP_SAMPLES:
        return y[:CLIP_SAMPLES]
    if len(y) < CLIP_SAMPLES:
        return np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y


def trim_silence(y):
    """Cắt clip lại chỉ còn phần frame thực sự mang tiếng của từ."""
    rms = librosa.feature.rms(y=y, frame_length=FEAT_WIN, hop_length=FEAT_HOP)[0]
    threshold = max(rms.max() * TRIM_REL_THRESHOLD, 1e-6)
    voiced = np.where(rms > threshold)[0]
    if len(voiced) < 3:
        return y
    start = max(0, (voiced[0] - TRIM_MARGIN_FRAMES) * FEAT_HOP)
    end = min(len(y), (voiced[-1] + TRIM_MARGIN_FRAMES + 1) * FEAT_HOP)
    return y[start:end]


def _pool(matrix, n_bins):
    """Lấy trung bình mỗi hàng theo n_bins đoạn thời gian bằng nhau."""
    return np.concatenate([b.mean(axis=1)
                           for b in np.array_split(matrix, n_bins, axis=1)])


def _pitch_contour(y, sr):
    """F0 theo từng frame, đưa về thang log2 và trừ trung bình của chính clip.

    Thang log2 + trừ trung bình theo từng clip sẽ loại bỏ cao độ tuyệt đối
    (vốn chủ yếu phản ánh người nói là ai - nam hay nữ) và giữ lại HÌNH DẠNG
    của contour (lên / xuống / bằng phẳng), đây mới là thứ mang thông tin
    thanh điệu tiếng Việt.

    librosa.yin luôn trả về 1 ước lượng, kể cả ở frame không có tiếng/gần im
    lặng (khác với pyin, nó không có cờ voiced/unvoiced), nhưng clip đã được
    trim_silence() cắt về vùng có tiếng trước khi hàm này chạy, nên các frame
    đó chỉ là thiểu số, và việc lấy trung bình theo bin trong _pool() giúp vài
    frame nhiễu không lấn át cả 1 bin.
    """
    f0 = librosa.yin(y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr,
                     frame_length=FEAT_WIN, hop_length=FEAT_HOP)
    f0 = np.maximum(f0, 1e-6)
    log_f0 = np.log2(f0 / PITCH_REF_HZ)
    log_f0 -= log_f0.mean()
    return log_f0.reshape(1, -1)


def extract(y, sr=SAMPLE_RATE):
    y = trim_silence(pad_or_trim(np.asarray(y, dtype=np.float32)))

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                               hop_length=FEAT_HOP, win_length=FEAT_WIN)
    mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)     # CMN theo từng clip

    delta = (librosa.feature.delta(mfcc) if mfcc.shape[1] >= 9
             else np.zeros_like(mfcc))
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FEAT_WIN,
                                             hop_length=FEAT_HOP)
    rms = librosa.feature.rms(y=y, frame_length=FEAT_WIN, hop_length=FEAT_HOP)
    pitch = _pitch_contour(y, sr)

    return np.concatenate([
        _pool(mfcc, N_BINS),
        mfcc.std(axis=1),
        _pool(delta, N_BINS),
        _pool(zcr, N_BINS),
        _pool(rms, N_BINS),
        _pool(pitch, N_BINS),
        [len(y) / sr],
    ]).astype(np.float32)
