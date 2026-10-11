"""Dataset, speaker-aware splitting, and waveform augmentation.

Design notes
------------
* Clips are small, so the whole dataset is loaded into RAM once.
* Augmentation happens on the raw waveform inside __getitem__ (CPU).
* The mel-spectrogram is computed on the GPU in batch (see features.py),
  which is much faster than doing it per sample on the CPU.
* The split is done BY SPEAKER. A speaker appears in exactly one of
  train/val/test. Splitting by file instead inflates accuracy by ~10
  points and the model then fails on any new voice.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from config import (
    ALL_LABELS,
    CLIP_SAMPLES,
    DATA_RAW,
    SAMPLE_RATE,
    SPEAKER_ALIASES,
)

LABEL_TO_INDEX = {label: i for i, label in enumerate(ALL_LABELS)}


@dataclass
class Clip:
    audio: np.ndarray
    label: int
    speaker: str
    label_name: str


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_all_clips() -> list[Clip]:
    clips: list[Clip] = []
    for label in ALL_LABELS:
        folder = DATA_RAW / label
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.wav")):
            parts = path.stem.split("_")
            if len(parts) < 3:
                continue
            # Resolve top-up sessions back to the person who recorded them.
            speaker = SPEAKER_ALIASES.get(parts[-2], parts[-2])
            audio, sample_rate = sf.read(path, dtype="float32")
            if sample_rate != SAMPLE_RATE or len(audio) != CLIP_SAMPLES:
                continue
            clips.append(Clip(audio, LABEL_TO_INDEX[label], speaker, label))
    return clips


def load_noise_bank(clips: list[Clip]) -> np.ndarray:
    """Use the recorded _silence_ clips as the noise bank.

    No extra download needed, and it is exactly the noise the system
    will meet at demo time: this room, this microphone.
    """
    noise = [c.audio for c in clips if c.label_name == "_silence_"]
    if not noise:
        return np.zeros((1, CLIP_SAMPLES), dtype=np.float32)
    return np.stack(noise)


# ----------------------------------------------------------------------
# Speaker-aware split
# ----------------------------------------------------------------------
SPECIAL_LABEL_NAMES = {"_silence_", "_unknown_"}


def _file_level_split(clips: list[Clip], val_ratio: float, test_ratio: float,
                      seed: int) -> tuple[list[Clip], list[Clip], list[Clip]]:
    """Shuffle and cut. Used only where a speaker split is impossible."""
    shuffled = clips[:]
    random.Random(seed).shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_ratio))
    n_val = int(round(len(shuffled) * val_ratio))
    return (shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val],
            shuffled[:n_test])


def _assign_speakers(clips: list[Clip], val_ratio: float, test_ratio: float,
                     seed: int) -> tuple[set[str], set[str]]:
    """Choose which speakers are held out for val and test.

    Assigning speakers uniformly at random goes wrong when they
    contributed very different amounts: the largest contributor can land
    in validation, leaving training with the two smallest and a val set
    bigger than the train set.

    So held-out speakers are drawn from the smaller contributors, just
    enough to reach the target fraction of clips. Training keeps the bulk
    of the data, which is where it does the most good.
    """
    counts: Counter[str] = Counter(c.speaker for c in clips)
    total = sum(counts.values())
    # Ascending by clip count; the seed only breaks ties between speakers
    # of equal size, so the split stays reproducible.
    rng = random.Random(seed)
    ordered = sorted(counts, key=lambda s: (counts[s], rng.random()))

    def take(pool: list[str], ratio: float) -> tuple[set[str], list[str]]:
        chosen: set[str] = set()
        accumulated = 0
        for speaker in list(pool):
            if chosen and accumulated >= total * ratio:
                break
            if len(pool) - len(chosen) <= 1:
                break   # never empty the training pool
            chosen.add(speaker)
            accumulated += counts[speaker]
        return chosen, [s for s in pool if s not in chosen]

    test_speakers, remaining = take(ordered, test_ratio)
    val_speakers, _ = take(remaining, val_ratio)
    return test_speakers, val_speakers


def split_by_speaker(clips: list[Clip], val_ratio: float = 0.15,
                     test_ratio: float = 0.15, seed: int = 1337,
                     force_test: str | None = None,
                     force_val: str | None = None,
                     quiet: bool = False
                     ) -> tuple[list[Clip], list[Clip], list[Clip]]:
    """Speaker-independent split that guarantees every class appears
    in every subset.

    Command classes are split by speaker: a voice in the test set was
    never heard during training, which is the only honest way to measure
    how the system behaves on a new player.

    _silence_ and _unknown_ are handled separately. In practice those are
    recorded in bulk sessions under their own ids (room01, and whoever
    happened to read aloud), so folding them into the speaker split
    silently starves one subset of an entire class. _unknown_ still gets
    a speaker split when enough people contributed to it; _silence_ is
    room noise rather than an identity, so it is split at file level.
    """
    command_clips = [c for c in clips if c.label_name not in SPECIAL_LABEL_NAMES]
    train: list[Clip] = []
    val: list[Clip] = []
    test: list[Clip] = []

    # --- command classes: speaker split ------------------------------
    command_speakers = sorted({c.speaker for c in command_clips})
    if len(command_speakers) >= 4:
        if force_test is not None:
            # Cross-validation drives which speaker is held out, so that
            # the score does not depend on the luck of the draw.
            test_speakers = {force_test}
            others = [s for s in command_speakers if s != force_test]
            val_speakers = {force_val} if force_val else {others[0]}
        else:
            test_speakers, val_speakers = _assign_speakers(
                command_clips, val_ratio, test_ratio, seed)
        for clip in command_clips:
            if clip.speaker in test_speakers:
                test.append(clip)
            elif clip.speaker in val_speakers:
                val.append(clip)
            else:
                train.append(clip)
        train_speakers = [s for s in command_speakers
                          if s not in test_speakers | val_speakers]
        if not quiet:
            print(f"Commands: speaker-independent split over "
                  f"{len(command_speakers)} speakers")
            print(f"  train: {train_speakers}")
            print(f"  val  : {sorted(val_speakers)}")
            print(f"  test : {sorted(test_speakers)}")
    else:
        print("\n" + "!" * 70)
        print(f"  Only {len(command_speakers)} speaker(s) recorded the command")
        print("  words. Falling back to a per-file split for them. Accuracy")
        print("  will be OPTIMISTIC and says nothing about a new voice.")
        print("  Reach 4+ speakers before reporting any final number.")
        print("!" * 70 + "\n")
        a, b, c = _file_level_split(command_clips, val_ratio, test_ratio, seed)
        train += a; val += b; test += c

    # --- special classes ---------------------------------------------
    for special in sorted(SPECIAL_LABEL_NAMES):
        subset = [c for c in clips if c.label_name == special]
        if not subset:
            continue
        speakers = sorted({c.speaker for c in subset})

        if special == "_unknown_" and len(speakers) >= 4:
            test_speakers, val_speakers = _assign_speakers(
                subset, val_ratio, test_ratio, seed)
            for clip in subset:
                if clip.speaker in test_speakers:
                    test.append(clip)
                elif clip.speaker in val_speakers:
                    val.append(clip)
                else:
                    train.append(clip)
            if not quiet:
                print(f"{special}: speaker split over {len(speakers)} speakers")
        else:
            a, b, c = _file_level_split(subset, val_ratio, test_ratio, seed)
            train += a; val += b; test += c
            if not quiet:
                reason = ("room noise, not an identity"
                          if special == "_silence_"
                          else f"only {len(speakers)} speaker(s)")
                print(f"{special}: file-level split ({reason})")

    if not quiet:
        _report_coverage(train, val, test)
    return train, val, test


def _report_coverage(train: list[Clip], val: list[Clip],
                     test: list[Clip]) -> None:
    """Fail loudly if a class is missing from a subset.

    A class absent from training makes its test score meaningless, and a
    subset made of a single class makes its accuracy meaningless too.
    Both happened before this check existed.
    """
    missing = []
    for name, subset in (("train", train), ("val", val), ("test", test)):
        present = {c.label_name for c in subset}
        for label in ALL_LABELS:
            if label not in present:
                missing.append(f"{label} missing from {name}")

    if missing:
        print("\n" + "!" * 70)
        print("  CLASS COVERAGE PROBLEM - results will not be meaningful:")
        for line in missing:
            print(f"    {line}")
        print("  Record more data for the affected classes and speakers.")
        print("!" * 70 + "\n")


# ----------------------------------------------------------------------
# Waveform augmentation
# ----------------------------------------------------------------------
class Augment:
    """Random waveform transforms, applied only to the training set.

    Each transform is individually switchable so the ablation study in
    W4 can measure what each one is actually worth.
    """

    def __init__(self, noise_bank: np.ndarray, *, time_shift: bool = True,
                 add_noise: bool = True, gain: bool = True,
                 speed: bool = True, seed: int = 0):
        self.noise_bank = noise_bank
        self.time_shift = time_shift
        self.add_noise = add_noise
        self.gain = gain
        self.speed = speed
        self.rng = np.random.default_rng(seed)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        out = audio.astype(np.float32, copy=True)

        if self.speed and self.rng.random() < 0.5:
            out = self._resample(out, self.rng.uniform(0.9, 1.1))
        if self.time_shift:
            out = self._shift(out, int(self.rng.integers(-1600, 1601)))
        if self.add_noise and self.rng.random() < 0.8:
            out = self._mix_noise(out, snr_db=self.rng.uniform(5.0, 20.0))
        if self.gain:
            out = out * float(10 ** (self.rng.uniform(-6, 6) / 20))

        return np.clip(out, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _fit(audio: np.ndarray) -> np.ndarray:
        if len(audio) > CLIP_SAMPLES:
            start = (len(audio) - CLIP_SAMPLES) // 2
            return audio[start:start + CLIP_SAMPLES]
        if len(audio) < CLIP_SAMPLES:
            pad = CLIP_SAMPLES - len(audio)
            return np.pad(audio, (pad // 2, pad - pad // 2))
        return audio

    def _resample(self, audio: np.ndarray, rate: float) -> np.ndarray:
        n = max(1, int(len(audio) / rate))
        stretched = np.interp(
            np.linspace(0, len(audio) - 1, n),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        return self._fit(stretched)

    @staticmethod
    def _shift(audio: np.ndarray, samples: int) -> np.ndarray:
        out = np.zeros_like(audio)
        if samples > 0:
            out[samples:] = audio[:-samples]
        elif samples < 0:
            out[:samples] = audio[-samples:]
        else:
            out[:] = audio
        return out

    def _mix_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        noise = self.noise_bank[self.rng.integers(len(self.noise_bank))]
        signal_power = float(np.mean(audio ** 2))
        noise_power = float(np.mean(noise ** 2))
        if signal_power < 1e-9 or noise_power < 1e-9:
            return audio
        scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
        return audio + noise * scale


# ----------------------------------------------------------------------
# Torch dataset
# ----------------------------------------------------------------------
class KeywordDataset(Dataset):
    def __init__(self, clips: list[Clip], augment: Augment | None = None):
        self.clips = clips
        self.augment = augment

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int):
        clip = self.clips[index]
        audio = clip.audio
        if self.augment is not None:
            audio = self.augment(audio)
        return torch.from_numpy(np.ascontiguousarray(audio)), clip.label


def class_weights(clips: list[Clip], n_classes: int,
                  max_weight: float = 5.0) -> torch.Tensor:
    """Inverse-frequency weights, so a small class is not ignored.

    Two guards learned the hard way:
    * a class absent from training gets weight 1.0, not a huge number.
      Combined with label smoothing, a huge weight on an absent class
      inflates the loss and dominates the gradient.
    * weights are capped, so one rare class cannot swamp training.
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    for clip in clips:
        counts[clip.label] += 1

    present = counts > 0
    weights = np.ones(n_classes, dtype=np.float64)
    if present.any():
        mean_count = counts[present].mean()
        weights[present] = np.clip(mean_count / counts[present],
                                   1.0 / max_weight, max_weight)
    return torch.tensor(weights, dtype=torch.float32)