from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchors import AnchorGenerator
from .utils import box_iou, encode_boxes, flatten_cls_reg


class HSNLoss(nn.Module):
    def __init__(self, cfg: Dict, anchor_gen: AnchorGenerator):
        super().__init__()

        self.cfg = cfg
        self.loss_cfg = cfg["loss"]
        self.anchor_gen = anchor_gen

        self.pos_iou = float(self.loss_cfg.get("pos_iou", 0.25))
        self.neg_iou = float(self.loss_cfg.get("neg_iou", 0.10))
        self.pos_weight = float(self.loss_cfg.get("pos_weight", 3.0))
        self.neg_pos_ratio = int(self.loss_cfg.get("neg_pos_ratio", 3))
        self.min_neg = int(self.loss_cfg.get("min_neg", 128))
        self.reg_beta = float(self.loss_cfg.get("reg_beta", 1.0))

    def build_targets(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        b = gt_boxes.shape[0]
        n = anchors.shape[0]

        labels = torch.full(
            (b, n),
            -1,
            dtype=torch.long,
            device=gt_boxes.device,
        )
        reg_targets = torch.zeros(
            (b, n, 4),
            dtype=torch.float32,
            device=gt_boxes.device,
        )
        pos_mask = torch.zeros(
            (b, n),
            dtype=torch.bool,
            device=gt_boxes.device,
        )

        max_ious = []

        for i in range(b):
            ious = box_iou(anchors, gt_boxes[i:i + 1]).squeeze(1)

            labels[i, ious < self.neg_iou] = 0
            labels[i, ious >= self.pos_iou] = 1

            best = torch.argmax(ious)
            labels[i, best] = 1

            pos_mask[i] = labels[i] == 1

            matched = gt_boxes[i].view(1, 1, 4).expand(1, n, 4)
            reg_targets[i] = encode_boxes(anchors, matched)[0]

            max_ious.append(float(ious[best].detach().cpu()))

        stats = {
            "num_pos": float(pos_mask.sum().detach().cpu()),
            "num_neg": float((labels == 0).sum().detach().cpu()),
            "num_ignore": float((labels < 0).sum().detach().cpu()),
            "max_iou_mean": float(sum(max_ious) / max(1, len(max_ious))),
            "max_iou_min": float(min(max_ious) if max_ious else 0.0),
            "max_iou_max": float(max(max_ious) if max_ious else 0.0),
        }

        return labels, reg_targets, pos_mask, stats

    def _balanced_cls_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        neg_mask = labels == 0

        ce = F.cross_entropy(
            logits.reshape(-1, 2),
            labels.reshape(-1).clamp(min=0),
            reduction="none",
        ).view_as(labels)

        pos_loss = ce[pos_mask]
        neg_loss = ce[neg_mask]

        if pos_loss.numel() > 0:
            pos_loss = pos_loss.mean()
        else:
            pos_loss = logits.sum() * 0.0

        if neg_loss.numel() > 0:
            num_pos = int(pos_mask.sum().detach().cpu())
            max_neg = max(self.min_neg, num_pos * self.neg_pos_ratio)
            max_neg = min(max_neg, neg_loss.numel())

            neg_loss = torch.topk(neg_loss, k=max_neg, largest=True).values.mean()
        else:
            neg_loss = logits.sum() * 0.0

        return self.pos_weight * pos_loss + neg_loss

    def cls_reg_loss(
        self,
        cls: torch.Tensor,
        reg: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        anchors = self.anchor_gen.grid_anchors(cls.shape[-2:], cls.device)
        cls_f, reg_f = flatten_cls_reg(cls, reg, self.anchor_gen.num_anchors)

        labels, reg_targets, pos_mask, stat = self.build_targets(anchors, gt_boxes)

        loss_cls = self._balanced_cls_loss(cls_f, labels, pos_mask)

        if pos_mask.any():
            loss_reg = F.smooth_l1_loss(
                reg_f[pos_mask],
                reg_targets[pos_mask],
                reduction="mean",
                beta=self.reg_beta,
            )
        else:
            loss_reg = reg.sum() * 0.0

        return loss_cls, loss_reg, stat

    def feature_loss(
        self,
        pred_feat: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:
        if target_feat is None:
            return pred_feat.sum() * 0.0

        return F.mse_loss(
            F.normalize(pred_feat, dim=1),
            F.normalize(target_feat.detach(), dim=1),
        )