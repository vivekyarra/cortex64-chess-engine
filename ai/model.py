from __future__ import annotations

import torch
from torch import nn


DEFAULT_IN_CHANNELS = 18


class ChessCNN(nn.Module):
    """Simple CNN that maps a board tensor to a scalar evaluation.

    Input:  (batch, C, 8, 8) where C is typically 18 (piece planes + state planes).
    Output: (batch,) float in [-1, 1] (tanh), positive = advantage for White.
    """

    def __init__(self, in_channels: int = DEFAULT_IN_CHANNELS) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)

