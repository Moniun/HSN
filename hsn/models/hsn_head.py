from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSNTrackingHead(nn.Module):
    """
    Fast GPU-friendly tracking head.

    Important:
        The previous implementation used per-sample Python ROI slicing and box.tolist(),
        which causes CPU/GPU synchronization and can make GPU utilization appear as
        0% -> spike -> 0%.

    This version keeps the same forward API:
        forward(template_feat, search_feat, template_boxes)

    but uses global pooled template features to generate a channel gate.
    """

    def __init__(
        self,
        channels: int = 512,
        num_anchors: int = 6,
        stride: int = 8,
        template_context: float = 0.25,
    ):
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

        self.cls = nn.Conv2d(channels, num_anchors * 2, 3, padding=1)
        self.reg = nn.Conv2d(channels, num_anchors * 4, 3, padding=1)

    def _template_vector(
        self,
        template_feat: torch.Tensor,
        template_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        GPU-friendly template descriptor.

        Args:
            template_feat: [B, C, H, W]
            template_boxes: [B, 4], kept for API compatibility

        Returns:
            [B, C]
        """
        return F.adaptive_avg_pool2d(template_feat, 1).flatten(1)

    def forward(
        self,
        template_feat: torch.Tensor,
        search_feat: torch.Tensor,
        template_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        template_vec = self._template_vector(template_feat, template_boxes)

        gate = self.template_gate(template_vec).view(
            template_feat.shape[0],
            self.channels,
            1,
            1,
        )

        x = search_feat * (1.0 + gate)
        x = self.conv1(x)
        x = self.conv2(x)

        cls = self.cls(x)
        reg = self.reg(x)

        return cls, reg
