from __future__ import annotations

import torch
import torch.nn as nn

from .lif import LIFSpike, reset_lif_states


class SpikingConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, padding: int | None = None,
                 threshold: float = 1.0, decay: float = 0.5, spike: bool = True):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.spike = LIFSpike(threshold, decay) if spike else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spike(self.bn(self.conv(x)))


class SNNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1, threshold: float = 1.0, decay: float = 0.5):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = SpikingConvBN(in_ch, mid_ch, 1, stride=1, padding=0, threshold=threshold, decay=decay)
        self.conv2 = SpikingConvBN(mid_ch, mid_ch, 3, stride=stride, padding=1, threshold=threshold, decay=decay)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.out_spike = LIFSpike(threshold, decay)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.out_spike(out + identity)


class SNNResNet22(nn.Module):
    """Spiking ResNet-22-style branch. Input [B,S,C,H,W], output [B,512,H/8,W/8]."""

    def __init__(self, in_channels: int = 3, threshold: float = 1.0, decay: float = 0.5, readout: str = 'last'):
        super().__init__()
        self.readout = readout
        self.conv1 = SpikingConvBN(in_channels, 64, 7, stride=2, padding=3, threshold=threshold, decay=decay)
        self.conv2 = self._make_layer(64, 64, blocks=3, stride=2, threshold=threshold, decay=decay)
        self.conv3 = self._make_layer(256, 128, blocks=4, stride=2, threshold=threshold, decay=decay)
        self.out_channels = 512

    def _make_layer(self, in_ch: int, mid_ch: int, blocks: int, stride: int, threshold: float, decay: float) -> nn.Sequential:
        layers = [SNNBottleneck(in_ch, mid_ch, stride=stride, threshold=threshold, decay=decay)]
        out_ch = mid_ch * SNNBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(SNNBottleneck(out_ch, mid_ch, stride=1, threshold=threshold, decay=decay))
        return nn.Sequential(*layers)

    def reset_state(self) -> None:
        reset_lif_states(self)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.reset_state()
        outs = []
        for t in range(x_seq.shape[1]):
            outs.append(self.step(x_seq[:, t]))
        if self.readout == 'mean':
            return torch.stack(outs, dim=1).mean(dim=1)
        return outs[-1]
