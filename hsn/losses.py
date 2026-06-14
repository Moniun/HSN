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
        self.loss_cfg = cfg['loss']
        self.anchor_gen = anchor_gen

    def build_targets(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = gt_boxes.shape[0]
        n = anchors.shape[0]
        labels = torch.full((b, n), -1, dtype=torch.long, device=gt_boxes.device)
        reg_targets = torch.zeros((b, n, 4), dtype=torch.float32, device=gt_boxes.device)
        pos_mask = torch.zeros((b, n), dtype=torch.bool, device=gt_boxes.device)
        for i in range(b):
            ious = box_iou(anchors, gt_boxes[i:i + 1]).squeeze(1)
            labels[i, ious < self.loss_cfg['neg_iou']] = 0
            labels[i, ious >= self.loss_cfg['pos_iou']] = 1
            best = int(torch.argmax(ious))
            labels[i, best] = 1
            pos_mask[i] = labels[i] == 1
            matched = gt_boxes[i].view(1, 1, 4).expand(1, n, 4)
            reg_targets[i] = encode_boxes(anchors, matched)[0]
        return labels, reg_targets, pos_mask

    def cls_reg_loss(self, cls: torch.Tensor, reg: torch.Tensor, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        anchors = self.anchor_gen.grid_anchors(cls.shape[-2:], cls.device)
        cls_f, reg_f = flatten_cls_reg(cls, reg, self.anchor_gen.num_anchors)
        labels, reg_targets, pos_mask = self.build_targets(anchors, gt_boxes)
        valid = labels >= 0
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[labels == 1] = self.loss_cfg['pos_weight']
        cls_loss_all = F.cross_entropy(cls_f.reshape(-1, 2), labels.reshape(-1).clamp(min=0), reduction='none').view_as(labels)
        loss_cls = (cls_loss_all[valid] * weights[valid]).mean() if valid.any() else cls.sum() * 0
        if pos_mask.any():
            loss_reg = F.smooth_l1_loss(reg_f[pos_mask], reg_targets[pos_mask], reduction='mean')
        else:
            loss_reg = reg.sum() * 0
        stat = {
            'num_pos': float(pos_mask.sum().detach().cpu()),
            'num_valid': float(valid.sum().detach().cpu()),
        }
        return loss_cls, loss_reg, stat

    def feature_loss(self, pred_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(F.normalize(pred_feat, dim=1), F.normalize(target_feat.detach(), dim=1))
