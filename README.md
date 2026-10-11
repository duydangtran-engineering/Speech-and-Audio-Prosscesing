# Voice-Controlled Web Game (Keyword Spotting)

A browser game controlled by Vietnamese voice commands (`len`, `xuong`, `trai`, `phai`, `ban`, `dung`). Course project for Speech & Audio Processing.

A small depthwise-separable CNN (~22.9k parameters, PyTorch) classifies 1-second audio into 6 commands plus `_silence_` and `_unknown_`. It is trained with noise, time-shift, speed and SpecAugment augmentation and a **speaker-independent split** (reported test accuracy: **89.8%**). The model is exported to ONNX and runs fully in the browser with ONNX Runtime Web, so no audio leaves the device.

## Layout

```
src/        recording, validation, features, training, evaluation, ONNX export
web/        the game (index.html, game.js, voice.js, features.js, model.onnx)
data/raw/metadata.csv   one row per clip (speaker ids are anonymised: sp01, sp02, ...)
```

The audio clips themselves are not published (they are other people's voices).

## Run the game

```bash
python src/fetch_ort.py          # downloads ONNX Runtime Web into web/vendor/
cd web && python -m http.server 8000
# open http://localhost:8000 and allow the microphone
```

## Train your own model

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python src/record.py --label len --speaker sp01 --count 30   # record clips
python src/validate_data.py                                  # check data health
python src/train.py                                          # train
python src/evaluate.py                                       # per-class report
python src/export_onnx.py                                    # -> web/model.onnx
```

Every clip carries a speaker id (`len_sp01_0007.wav`) because train/val/test are split **by speaker**; this is what makes the accuracy number honest.

`web/parity_reference.json` holds reference features so the JavaScript feature extractor can be checked against the Python one.

## Stack

Python, PyTorch, librosa, ONNX / ONNX Runtime Web, JavaScript
