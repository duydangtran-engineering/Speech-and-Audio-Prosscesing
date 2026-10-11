"""Central configuration for the voice-controlled game project.

Every script imports from here so that audio parameters stay consistent
across recording, preprocessing, training and inference.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_NOISE = ROOT / "data" / "noise"
MODELS_DIR = ROOT / "models"

# --- Audio parameters --------------------------------------------------
SAMPLE_RATE = 16_000          # Hz. Speech energy lives below 8 kHz -> 16k is enough.
CHANNELS = 1                  # mono
CLIP_DURATION = 1.0           # seconds, the fixed window the model sees
RECORD_DURATION = 1.3         # seconds actually captured, trimmed down later
SUBTYPE = "PCM_16"            # 16-bit WAV

CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# --- Label set ---------------------------------------------------------
# Command classes: the words the game reacts to.
COMMAND_LABELS = [
    "len",      # "len"   -> turn up
    "xuong",    # "xuong" -> turn down
    "trai",     # "trai"  -> turn left
    "phai",     # "phai"  -> turn right
    "khien",    # "khien" -> shield, brief invulnerability
    "luot",     # "luot"  -> dash, burst of speed
]

# Special classes: required so the model can stay quiet during normal
# speech and silence instead of firing a random command.
SPECIAL_LABELS = [
    "_silence_",   # background noise, no speech
    "_unknown_",   # speech that is not one of the commands
]

ALL_LABELS = COMMAND_LABELS + SPECIAL_LABELS

# Vietnamese prompt shown to the person being recorded.
LABEL_PROMPT = {
    "len": 'LEN      (say: "len")',
    "xuong": 'XUONG    (say: "xuong")',
    "trai": 'TRAI     (say: "trai")',
    "phai": 'PHAI     (say: "phai")',
    "khien": 'KHIEN    (say: "khien" - shield)',
    "luot": 'LUOT     (say: "luot" - dash)',
    "_silence_": "(stay silent - capturing background noise)",
    "_unknown_": "(say any non-command word)",
}

# --- Speaker aliases ---------------------------------------------------
# Extra recording sessions by someone already in the dataset. Mapping the
# session id back to the person keeps the speaker-independent split
# honest: without this, a top-up session by the same voice can land in
# the test set while the original session trains the model, which leaks
# the voice across the split and inflates the score.
SPEAKER_ALIASES = {
    "sp01b": "sp01",
}

# --- Quality thresholds ------------------------------------------------
MIN_PEAK = 0.02        # below this the clip is basically silent
CLIPPING_PEAK = 0.99   # at or above this the microphone was overdriven
MIN_RMS = 0.005        # below this the speech is too quiet to be useful

# Target number of samples per class (see roadmap W2).
TARGET_PER_COMMAND = 350
TARGET_SILENCE = 200
TARGET_UNKNOWN = 300
