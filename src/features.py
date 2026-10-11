"""Log-mel front end, computed on the GPU in batch.

The exact numbers here (n_fft, hop, n_mels, fmin, fmax) must be
reproduced bit-for-bit in JavaScript at W6. Any mismatch between the
Python front end and the browser front end shows up as a model that
scores 93% offline and 40% in the game.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from config import SAMPLE_RATE

WIN_LENGTH = 400   # 25 ms of actual signal
N_FFT = 512        # zero-padded to a power of two
HOP_LENGTH = 160   # 10 ms
N_MELS = 40
F_MIN = 20.0
F_MAX = 7600.0
N_FRAMES = 101     # 1.0 s at 10 ms hop, center-padded

# N_FFT is 512 rather than 400 on purpose. The analysis window stays at
# 400 samples; the frame is zero-padded to 512 before the transform.
# A radix-2 FFT is a few dozen lines in JavaScript, while a 400-point
# transform needs Bluestein or a slow DFT. Since the browser has to
# reproduce this front end bit-for-bit at inference time, the cheaper
# thing to change is the number here.


class LogMel(nn.Module):
    """Waveform (B, 16000) -> log-mel (B, 1, 40, 101)."""

    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel(waveform)                    # (B, 40, T)
        spec = torch.log(spec + 1e-6)                # log compression
        # Per-utterance normalization: makes the model robust to overall
        # recording level, so a quiet speaker is not a different problem.
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        spec = (spec - mean) / std
        return spec.unsqueeze(1)                     # (B, 1, 40, T)


class SpecAugment(nn.Module):
    """Mask random frequency bands and time steps.

    Cheap, strong regularizer for keyword spotting: the model is forced
    to use the whole time-frequency pattern rather than one lucky band.
    Training only.
    """

    def __init__(self, freq_mask: int = 8, time_mask: int = 16,
                 n_freq: int = 2, n_time: int = 2):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        batch, _, n_mels, n_frames = spec.shape
        out = spec.clone()

        for _ in range(self.n_freq):
            width = torch.randint(0, self.freq_mask + 1, (batch,), device=spec.device)
            start = (torch.rand(batch, device=spec.device)
                     * (n_mels - width).clamp_min(1)).long()
            idx = torch.arange(n_mels, device=spec.device)[None, :]
            mask = (idx >= start[:, None]) & (idx < (start + width)[:, None])
            out[mask[:, None, :, None].expand_as(out)] = 0.0

        for _ in range(self.n_time):
            width = torch.randint(0, self.time_mask + 1, (batch,), device=spec.device)
            start = (torch.rand(batch, device=spec.device)
                     * (n_frames - width).clamp_min(1)).long()
            idx = torch.arange(n_frames, device=spec.device)[None, :]
            mask = (idx >= start[:, None]) & (idx < (start + width)[:, None])
            out[mask[:, None, None, :].expand_as(out)] = 0.0

        return out
