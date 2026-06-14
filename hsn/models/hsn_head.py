from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSNTrackingHead(nn.Module):
    """HU/RPN-like tracking head.

    This head merges target/template feature with search/predicted feature, then uses four
    3x3 convs following the supplementary description:
      3x3,512 -> 3x3,512 -> 3x3,512*6*2 and 3x3,512*6*4.
    """
    def __init__(self, channels: int = 512, num_anchors: int = 6, stride: int = 8, template_context: float = 0.25):
        super().__init__()
        self.channels = channels
        self.num_anchors = num_anchors
        self.stride = stride
        self.template_context = template_context
        self.template_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Practical RPN interpretation of the supplementary table:
        # classification has 6 anchors x 2 classes, regression has 6 anchors x 4 coords.
        self.cls = nn.Conv2d(channels, num_anchors * 2, 3, padding=1)
        self.reg = nn.Conv2d(channels, num_anchors * 4, 3, padding=1)

    def _template_vector(self, template_feat: torch.Tensor, template_boxes: torch.Tensor) -> torch.Tensor:
        b, c, h, w = template_feat.shape
        vecs = []
        for i in range(b):
            box = template_boxes[i].detach().float() / self.stride
            x1, y1, x2, y2 = box.tolist()
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            x1 -= bw * self.template_context
            y1 -= bh * self.template_context
            x2 += bw * self.template_context
            y2 += bh * self.template_context
            ix1 = int(max(0, min(w - 1, round(x1))))
            iy1 = int(max(0, min(h - 1, round(y1))))
            ix2 = int(max(ix1 + 1, min(w, round(x2))))
            iy2 = int(max(iy1 + 1, min(h, round(y2))))
            roi = template_feat[i:i + 1, :, iy1:iy2, ix1:ix2]
            vecs.append(F.adaptive_avg_pool2d(roi, 1).flatten(1))
        return torch.cat(vecs, dim=0)

    def forward(self, template_feat: torch.Tensor, search_feat: torch.Tensor, template_boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        template_vec = self._template_vector(template_feat, template_boxes)
        gate = self.template_gate(template_vec).view(template_feat.shape[0], self.channels, 1, 1)
        x = search_feat * gate + search_feat
        x = self.conv1(x)
        x = self.conv2(x)
        cls = self.cls(x)
        reg = self.reg(x)
        return cls, reg
