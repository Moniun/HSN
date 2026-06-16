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
        self.pos_weight = float(self.loss_cfg.get("pos_weight", 1.0))
        self.neg_pos_ratio = int(self.loss_cfg.get("neg_pos_ratio", 10))
        self.min_neg = int(self.loss_cfg.get("min_neg", 512))
        self.reg_beta = float(self.loss_cfg.get("reg_beta", 1.0))

        # Optional ranking loss. This directly matches ANN eval behavior:
        # GT-near positive anchors should rank above hard background anchors.
        # Set rank_weight=0.0 in config if you want only CE + OHEM.
        self.rank_weight = float(self.loss_cfg.get("rank_weight", 0.2))
        self.rank_margin = float(self.loss_cfg.get("rank_margin", 0.2))
        self.rank_topk_neg = int(self.loss_cfg.get("rank_topk_neg", 64))

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

            # Always keep the best-matching anchor positive.
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
        """
        Per-sample hard negative mining.

        logits: [B, N, 2]
        labels: [B, N], 1=positive, 0=negative, -1=ignore
        pos_mask: [B, N]
        """
        neg_mask = labels == 0

        ce = F.cross_entropy(
            logits.reshape(-1, 2),
            labels.reshape(-1).clamp(min=0),
            reduction="none",
        ).view_as(labels)

        batch_losses = []

        for i in range(labels.shape[0]):
            pos_i = pos_mask[i]
            neg_i = neg_mask[i]

            if pos_i.any():
                pos_loss = ce[i][pos_i].mean()
                num_pos = int(pos_i.sum().detach().cpu())
            else:
                pos_loss = logits[i].sum() * 0.0
                num_pos = 0

            if neg_i.any():
                neg_values = ce[i][neg_i]

                # min_neg is interpreted per image, not per batch.
                k = max(self.min_neg, max(1, num_pos) * self.neg_pos_ratio)
                k = min(k, neg_values.numel())

                neg_loss = torch.topk(neg_values, k=k, largest=True).values.mean()
            else:
                neg_loss = logits[i].sum() * 0.0

            batch_losses.append(self.pos_weight * pos_loss + neg_loss)

        return torch.stack(batch_losses).mean()

    def _ranking_cls_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ranking loss for ANN top-1 selection.

        It uses the foreground-vs-background logit margin as the anchor score:
            score = logit_fg - logit_bg

        For each image, force the best positive anchor score to be higher than
        hard negative anchor scores by rank_margin.
        """
        if self.rank_weight <= 0:
            return logits.sum() * 0.0

        neg_mask = labels == 0
        scores = logits[..., 1] - logits[..., 0]  # [B, N]

        losses = []

        for i in range(labels.shape[0]):
            pos_i = pos_mask[i]
            neg_i = neg_mask[i]

            if not pos_i.any() or not neg_i.any():
                losses.append(logits[i].sum() * 0.0)
                continue

            best_pos_score = scores[i][pos_i].max()
            neg_scores = scores[i][neg_i]

            k = min(max(1, self.rank_topk_neg), neg_scores.numel())
            hard_neg_scores = torch.topk(neg_scores, k=k, largest=True).values

            losses.append(F.relu(self.rank_margin + hard_neg_scores - best_pos_score).mean())

        return torch.stack(losses).mean()

    def cls_reg_loss(
        self,
        cls: torch.Tensor,
        reg: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        anchors = self.anchor_gen.grid_anchors(cls.shape[-2:], cls.device)
        cls_f, reg_f = flatten_cls_reg(cls, reg, self.anchor_gen.num_anchors)

        labels, reg_targets, pos_mask, stat = self.build_targets(anchors, gt_boxes)

        ce_ohem_loss = self._balanced_cls_loss(cls_f, labels, pos_mask)
        rank_loss = self._ranking_cls_loss(cls_f, labels, pos_mask)
        loss_cls = ce_ohem_loss + self.rank_weight * rank_loss

        if pos_mask.any():
            loss_reg = F.smooth_l1_loss(
                reg_f[pos_mask],
                reg_targets[pos_mask],
                reduction="mean",
                beta=self.reg_beta,
            )
        else:
            loss_reg = reg.sum() * 0.0

        stat["cls_ce_ohem"] = float(ce_ohem_loss.detach().cpu())
        stat["cls_rank"] = float(rank_loss.detach().cpu())
        stat["cls_rank_weight"] = float(self.rank_weight)

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
