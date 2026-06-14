from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, padding: int | None = None):
        if padding is None:
            padding = k // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ANNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = ConvBNReLU(in_ch, mid_ch, 1, stride=1, padding=0)
        self.conv2 = ConvBNReLU(mid_ch, mid_ch, 3, stride=stride, padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
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
        return self.relu(out + identity)


class ANNResNet22(nn.Module):
    """ResNet-22-style ANN branch from HSN supplementary table.

    Conv1: 7x7,64,stride2
    Conv2: bottleneck mid=64 x3, first stride2 -> 256 channels
    Conv3: bottleneck mid=128 x4, first stride2 -> 512 channels
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, 64, 7, stride=2, padding=3)
        self.conv2 = self._make_layer(64, 64, blocks=3, stride=2)
        self.conv3 = self._make_layer(256, 128, blocks=4, stride=2)
        self.out_channels = 512

    def _make_layer(self, in_ch: int, mid_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ANNBottleneck(in_ch, mid_ch, stride=stride)]
        out_ch = mid_ch * ANNBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(ANNBottleneck(out_ch, mid_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
