from __future__ import annotations

from typing import Iterable, Tuple

import torch


class AnchorGenerator:
    def __init__(self, stride: int = 8, scales: Iterable[float] = (32, 64), ratios: Iterable[float] = (0.5, 1.0, 2.0)):
        self.stride = float(stride)
        self.scales = list(scales)
        self.ratios = list(ratios)
        self.num_anchors = len(self.scales) * len(self.ratios)

    def grid_anchors(self, feat_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
        h, w = feat_hw
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * self.stride
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * self.stride
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        centers = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        anchors = []
        for scale in self.scales:
            area = scale * scale
            for ratio in self.ratios:
                aw = (area / ratio) ** 0.5
                ah = aw * ratio
                x1 = centers[:, 0] - aw / 2
                y1 = centers[:, 1] - ah / 2
                x2 = centers[:, 0] + aw / 2
                y2 = centers[:, 1] + ah / 2
                anchors.append(torch.stack([x1, y1, x2, y2], dim=-1))
        return torch.stack(anchors, dim=1).reshape(-1, 4)
