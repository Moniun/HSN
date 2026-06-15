from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSNTrackingHead(nn.Module):
    """
    Target-aware tracking head.

    It uses template_boxes to build a soft target ROI mask on the template feature map,
    then correlates target descriptor with search feature.
    """

    def __init__(
        self,
        channels: int = 512,
        num_anchors: int = 15,
        stride: int = 8,
        template_context: float = 0.35,
    ):
        super().__init__()

        self.channels = int(channels)
        self.num_anchors = int(num_anchors)
        self.stride = float(stride)
        self.template_context = float(template_context)

        self.template_proj = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, self.channels),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(self.channels + 1, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.cls = nn.Conv2d(self.channels, self.num_anchors * 2, 3, padding=1)
        self.reg = nn.Conv2d(self.channels, self.num_anchors * 4, 3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.cls.bias is not None:
            nn.init.constant_(self.cls.bias, 0.0)
        if self.reg.bias is not None:
            nn.init.constant_(self.reg.bias, 0.0)

    def _make_soft_roi_mask(
        self,
        feat: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W]
            boxes: [B, 4] padded-image xyxy coordinates

        Returns:
            mask: [B, 1, H, W], sum=1
        """
        b, _, h, w = feat.shape
        device = feat.device
        dtype = feat.dtype

        boxes = boxes.to(device=device, dtype=dtype)

        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        bw = (x2 - x1).clamp(min=self.stride)
        bh = (y2 - y1).clamp(min=self.stride)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        sx = (bw * (0.5 + self.template_context)).clamp(min=self.stride * 0.5)
        sy = (bh * (0.5 + self.template_context)).clamp(min=self.stride * 0.5)

        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * self.stride
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * self.stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

        cx = cx.view(b, 1, 1)
        cy = cy.view(b, 1, 1)
        sx = sx.view(b, 1, 1)
        sy = sy.view(b, 1, 1)

        dist = ((xx - cx) ** 2) / (2 * sx ** 2 + 1e-6) + ((yy - cy) ** 2) / (2 * sy ** 2 + 1e-6)
        mask = torch.exp(-dist).unsqueeze(1)
        mask = mask / mask.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)

        return mask

    def _template_vector(
        self,
        template_feat: torch.Tensor,
        template_boxes: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._make_soft_roi_mask(template_feat, template_boxes)
        vec = (template_feat * mask).sum(dim=(2, 3))
        return vec

    def forward(
        self,
        template_feat: torch.Tensor,
        search_feat: torch.Tensor,
        template_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        template_vec = self._template_vector(template_feat, template_boxes)

        gate = self.template_proj(template_vec).view(
            search_feat.shape[0],
            self.channels,
            1,
            1,
        )

        t_norm = F.normalize(template_vec, dim=1).view(search_feat.shape[0], self.channels, 1, 1)
        s_norm = F.normalize(search_feat, dim=1)
        sim = (s_norm * t_norm).sum(dim=1, keepdim=True)

        x = search_feat * (1.0 + gate)
        x = torch.cat([x, sim], dim=1)

        x = self.fuse(x)
        x = self.conv1(x)
        x = self.conv2(x)

        cls = self.cls(x)
        reg = self.reg(x)

        return cls, reg