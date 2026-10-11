"""Import a phone recording as training clips.

    python src/import_recording.py trai_minh.m4a --label trai --speaker sp07
    python src/import_recording.py folder/ --speaker sp07        # names from files
    python src/import_recording.py x.m4a --label len --speaker sp07 --dry-run

Carrying a PC around to collect voices does not scale. This lets anyone
record on a phone - Voice Memos, Zalo, anything - and send one file per
command word. The file is converted to 16 kHz mono, split into individual
utterances with the same energy segmentation the rapid recorder uses, and
saved under the project's naming convention.

Phone recordings differ from the desktop microphone in level, room and
codec, so the importer reports RMS and SNR before anything is written:
a session that is too quiet is worth redoing, not training on.

Needs ffmpeg for compressed formats (m4a, mp3). Plain .wav works without.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from config import ALL_LABELS, CLIP_SAMPLES, DATA_RAW, SAMPLE_RATE, SUBTYPE
from audio_utils import check_quality, next_index, save_clip, segment_words

AUDIO_SUFFIXES = {".wav", ".m4a", ".mp3", ".aac", ".ogg", ".opus",
                  ".flac", ".wma", ".mp4", ".3gp", ".amr", ".caf"}


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def load_audio(path: Path) -> np.ndarray:
    """Return mono float32 at SAMPLE_RATE, converting if needed."""
    if path.suffix.lower() == ".wav":
        try:
            audio, rate = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if rate == SAMPLE_RATE:
                return audio.astype(np.float32)
        except Exception:
            pass   # fall through to ffmpeg

    if not have_ffmpeg():
        raise SystemExit(
            f"{path.name} needs conversion but ffmpeg was not found.\n"
            "  Install it with:  winget install Gyan.FFmpeg\n"
            "  then reopen the terminal so PATH picks it up.\n"
            "  (Plain 16 kHz .wav files import without ffmpeg.)")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "converted.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-f", "wav", str(out)],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"ffmpeg failed on {path.name}:\n"
                             f"{result.stderr[-600:]}")
        audio, _ = sf.read(out, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def measure(audio: np.ndarray) -> tuple[float, float, float]:
    """Overall RMS, noise floor and SNR in dB for a whole recording."""
    frame = 400
    n = len(audio) // frame
    if n < 2:
        return 0.0, 0.0, 0.0
    frames = audio[:n * frame].reshape(n, frame)
    energies = np.sqrt(np.mean(frames ** 2, axis=1))
    speech = float(np.percentile(energies, 90))
    floor = max(float(np.percentile(energies, 10)), 1e-6)
    return speech, floor, 20 * float(np.log10(speech / floor))


def label_from_name(path: Path) -> str | None:
    """Guess the command from the filename, e.g. 'trai_minh.m4a' -> 'trai'."""
    stem = path.stem.lower()
    for label in ALL_LABELS:
        if stem == label or stem.startswith(label + "_") or label in stem.split("_"):
            return label
    return None


def import_file(path: Path, label: str, speaker: str, dry_run: bool) -> int:
    audio = load_audio(path)
    duration = len(audio) / SAMPLE_RATE
    speech, floor, snr = measure(audio)

    print(f"\n{path.name}  ->  label '{label}', speaker '{speaker}'")
    print(f"  {duration:.1f}s  ·  rms {speech:.4f}  ·  noise {floor:.4f}"
          f"  ·  SNR {snr:.1f} dB")

    if snr < 15:
        print("  ! LOW SNR - recorded far from the phone or in a noisy room.")
        print("    Ask for it again: phone 20-30 cm away, quiet room.")
    if speech < 0.02:
        print("  ! VERY QUIET - the model will see mostly amplified noise.")

    segments, dropped = segment_words(audio)
    if not segments:
        print("  ! No words detected. Check the file actually contains speech.")
        return 0

    lengths = [(b - a) / SAMPLE_RATE for a, b in segments]
    print(f"  found {len(segments)} words  ·  length min {min(lengths):.2f}s"
          f" mean {np.mean(lengths):.2f}s max {max(lengths):.2f}s")
    if dropped:
        print(f"  ! {dropped} run(s) held several words run together and "
              f"could not be split - dropped.")
        print("    A few lost out of fifty is fine; many means the speaker "
              "did not pause.")

    if dry_run:
        print("  (dry run - nothing written)")
        return 0

    index = next_index(label, speaker)
    saved = skipped = 0
    for begin, end in segments:
        middle = (begin + end) // 2
        half = CLIP_SAMPLES // 2
        start = max(0, min(middle - half, len(audio) - CLIP_SAMPLES))
        clip = audio[start:start + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        ok, _ = check_quality(clip)
        if not ok:
            skipped += 1
            continue
        save_clip(clip, label, speaker, index, note=f"import:{path.name}")
        index += 1
        saved += 1

    print(f"  saved {saved} clips, skipped {skipped}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import phone recordings as training clips")
    parser.add_argument("path", type=Path, help="audio file or folder")
    parser.add_argument("--label", choices=ALL_LABELS,
                        help="command word; guessed from the filename if omitted")
    parser.add_argument("--speaker", required=True,
                        help="speaker id for the whole import, e.g. sp07")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen, write nothing")
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"Not found: {args.path}")

    files = ([args.path] if args.path.is_file()
             else sorted(p for p in args.path.rglob("*")
                         if p.suffix.lower() in AUDIO_SUFFIXES))
    if not files:
        raise SystemExit(f"No audio files in {args.path}")

    total = 0
    unlabelled = []
    for path in files:
        label = args.label or label_from_name(path)
        if label is None:
            unlabelled.append(path)
            continue
        total += import_file(path, label, args.speaker, args.dry_run)

    if unlabelled:
        print("\nSkipped - could not tell which command these are.")
        print("Name them after the word (trai.m4a, len_minh.m4a) or pass --label:")
        for path in unlabelled:
            print(f"  {path.name}")

    print(f"\nTotal: {total} clips imported for speaker '{args.speaker}'.")
    if total and not args.dry_run:
        print("Now run:  python src/validate_data.py")


if __name__ == "__main__":
    main()
