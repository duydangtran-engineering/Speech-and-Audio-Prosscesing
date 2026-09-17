# Voice → IR: ML pipeline (tang / giam / bat / tat / noise)

## 1. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`librosa` đọc `.m4a` qua `audioread` → cần **ffmpeg** trong PATH
(`winget install ffmpeg` hoặc `choco install ffmpeg`).

## 2. Bỏ file thu âm vào đúng thư mục

```
data/raw/tang/vinh_01.m4a      <- tên file: <tên người>_<lần thu>.m4a
data/raw/tang/nam_01.m4a
data/raw/giam/...
data/raw/bat/...
data/raw/tat/...
data/raw/noise/vinh_01.m4a     <- tạp âm: quạt, TV, nói chuyện, im lặng
```

Tên người ở đầu file là BẮT BUỘC — script dùng nó để chia train/test theo
người nói (speaker-independent), đây mới là con số accuracy trung thực.

## 3. Chạy 4 bước

```bash
cd src
python step1_split.py      # cắt m4a dài -> clip 1 giây
python step2_dataset.py    # trích MFCC + delta + ZCR + RMS + pitch -> dataset.npz
python step3_train.py      # train + so sánh RandomForest vs SVM
python step4_export.py     # xuất models/voice_model.h cho ESP32
```

Test nhanh 1 file bất kỳ: `python predict.py ../data/raw/tang/vinh_02.m4a`

> **Đã sửa `features.py` (thêm RMS + pitch/F0)?** Bắt buộc chạy lại `step2_dataset.py`
> rồi `step3_train.py` — `dataset.npz` và `model.joblib` cũ được trích theo vector
> đặc trưng cũ (thiếu 2 nhóm này) nên không tương thích với code `features.py` mới.

## 4. Sau step1 — PHẢI nghe lại

Mở `data/clips/<label>/` nghe vài chục file. Nếu bị cắt cụt từ hoặc dính 2 từ
vào 1 clip, chỉnh trong `config.py`:

| Vấn đề | Sửa |
|---|---|
| Cắt sót từ (ít clip quá) | giảm `ENERGY_PERCENTILE` (75 → 60) |
| Bắt cả tiếng ồn (nhiều clip rác) | tăng `ENERGY_PERCENTILE` (75 → 85) |
| 2 từ dính vào 1 clip | giảm `MIN_SILENCE_MS` (120 → 80) và nói cách xa hơn |
| 1 từ bị tách làm 2 | tăng `MIN_SILENCE_MS` (120 → 200) |
| Mất phụ âm đầu/cuối | tăng `PAD_MS` (60 → 100) |

Cách thu để cắt dễ: nói từ đó lặp lại, **nghỉ rõ ràng ~0.5 giây giữa mỗi lần**,
mỗi file ~20 lần. Mục tiêu ≥ 200 clip/lớp, ít nhất 4–5 người, có cả file thu
trong môi trường ồn.

### 4b. (tùy chọn) Lọc bớt clip là tiếng ồn/gió bị cắt nhầm thành "từ"

Bộ tách từ ở step1 dựa trên năng lượng (RMS), nên 1 cơn gió thổi vào micro
hoặc tiếng "phù" hơi cũng đủ to để bị hiểu nhầm là 1 từ và bị cắt ra thành
clip — những clip này là nhiễu thật sự trong tập train. Nghe thủ công vài
nghìn clip để tìm chúng rất mất công, nên có `step1d_flag_noisy.py` để
xếp hạng trước:

```bash
python step1d_flag_noisy.py          # xếp hạng, copy ~150 clip nghi ngờ nhất
                                      # ra data/clips_review/ (không đụng bản gốc)
#  ... nghe qua data/clips_review/, xoá file nào đúng là gió/ồn ngay trong đó ...
python step1d_flag_noisy.py --apply  # đồng bộ: xoá đúng những file đó khỏi data/clips
```

Script chỉ xếp hạng theo độ "giống tiếng ồn" (phổ phẳng, đường bao năng lượng
không có đỉnh rõ, kéo dài gần hết clip) — **không tự xoá gì**, vì chỉ dựa vào
vài chỉ số thống kê sẽ có trường hợp đoán nhầm (ví dụ 1 từ nói to/gằn giọng có
thể bị chấm điểm nghi ngờ cao dù là giọng thật). Luôn nghe lại trước khi xoá.

## 5. Đọc kết quả (phần "phân tích mô hình" của đề bài)

`step3_train.py` in ra precision/recall/f1 từng lớp và lưu confusion matrix vào
`reports/confusion_*.png`. Hai điều gần như chắc chắn sẽ thấy:

- **tang ↔ tat** dễ nhầm nhất (cùng nguyên âm /a/, chỉ khác thanh điệu và phụ âm cuối).
  MFCC trung bình làm mất thông tin thanh điệu → `features.py` nay đã thêm đặc trưng
  pitch contour (`librosa.yin`, chuẩn hoá log2 theo từng clip) chính vì lý do này.
  Sau khi chạy lại step2/step3, so sánh confusion matrix mới với bản cũ ở ô
  tang/tat — nếu số nhầm giảm rõ, ghi lại con số trước/sau vào báo cáo, đó là
  bằng chứng cụ thể cho phần "model tuning".
- **noise** thường có recall cao nhất — đó là lớp dễ, đừng lấy nó để khoe accuracy tổng.

Random Forest gần như luôn thắng ở đây vì `emlearn` xuất cây quyết định ra C rất
gọn, còn SVM RBF phải mang theo support vectors → nặng cho ESP32-C3.

Sau khi train xong, xem khối "Top features" cuối log của `step3_train.py`
(chỉ in khi Random Forest thắng): nếu các `pitch_bin*`/`rms_bin*` lọt vào top
10, chúng thực sự đóng góp và đáng để port sang C ở bước 6. Nếu không lọt vào
top và accuracy tổng thể gần như không đổi so với bản cũ, có thể bỏ pitch khi
triển khai lên ESP32-C3 để đỡ tốn tài nguyên (YIN nặng hơn MFCC khá nhiều),
miễn là accuracy trên tập test không giảm.

## 6. Nối sang phần nhúng

`models/voice_model.h` chỉ chứa **bộ phân lớp**. Trên ESP32 bạn vẫn phải tự viết
lại phần trích đặc trưng bằng C với ĐÚNG tham số trong `config.py`
(16 kHz, n_mfcc=13, n_fft=512, hop=160, win=400) và đúng thứ tự vector ghi ở cuối
file .h. Sai thứ tự hoặc sai tham số → model chạy được nhưng đoán bừa.

Nếu giữ pitch/RMS trong bản deploy cuối (xem mục 5), cần thêm trên ESP32:
- RMS: dễ, chỉ là căn bậc hai trung bình bình phương từng frame — rẻ.
- Pitch (YIN, `PITCH_FMIN=75`, `PITCH_FMAX=400`, `PITCH_REF_HZ=100` trong
  `config.py`): thuật toán YIN tốn CPU hơn MFCC nhiều (autocorrelation theo
  từng lag ứng với dải tần 75–400 Hz), cân nhắc trước khi cam kết port sang
  ESP32-C3 — đây là lý do bước 5 khuyên kiểm tra "Top features" trước.

Kiểm chứng: in vector đặc trưng của 1 clip ở Python và cùng clip đó trên ESP32,
2 dãy số phải khớp tới ~2 chữ số thập phân trước khi tin vào kết quả predict.