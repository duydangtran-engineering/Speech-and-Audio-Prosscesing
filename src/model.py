"""Keyword-spotting models.

Two architectures so W4 has something to compare:

  cnn    - plain convolutional baseline
  dscnn  - depthwise-separable CNN, the standard strong/cheap choice for
           keyword spotting (Zhang et al., "Hello Edge", 2017)

Both take a log-mel image (B, 1, 40, 101) and output class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.block(x))


class BaselineCNN(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 128, pool=False),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SeparableBlock(nn.Module):
    """Depthwise 3x3 followed by pointwise 1x1.

    Roughly 8x fewer multiplications than a dense 3x3 convolution at the
    same channel count, which is what keeps inference fast enough to run
    in a browser on every 200 ms window.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSCNN(nn.Module):
    def __init__(self, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, (10, 4), stride=(2, 2), padding=(5, 1), bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            SeparableBlock(width, width),
            SeparableBlock(width, width),
            SeparableBlock(width, width),
            SeparableBlock(width, width),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


def build_model(name: str, n_classes: int, **kwargs) -> nn.Module:
    models = {"cnn": BaselineCNN, "dscnn": DSCNN}
    if name not in models:
        raise ValueError(f"unknown model '{name}', choose from {list(models)}")
    return models[name](n_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
