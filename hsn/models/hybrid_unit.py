from __future__ import annotations

import torch
import torch.nn as nn


class FeatureHybridUnit(nn.Module):
    """Learnable HU from SNN feature space to ANN feature increment space."""
    def __init__(self, channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, snn_feat: torch.Tensor) -> torch.Tensor:
        return self.net(snn_feat)
