from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .anchors import AnchorGenerator
from .models.ann_resnet22 import ANNResNet22
from .models.snn_resnet22 import SNNResNet22
from .models.hybrid_unit import FeatureHybridUnit
from .models.hsn_head import HSNTrackingHead
from .utils import decode_boxes, flatten_cls_reg


class TianmoucHSN(nn.Module):
    """Paper-oriented HSN with Tianmouc COP/AOP inputs."""

    def __init__(self, cfg: Dict):
        super().__init__()
        m = cfg['model']
        self.cfg = cfg
        self.ann = ANNResNet22(in_channels=m['cop_channels'])
        self.snn = SNNResNet22(
            in_channels=m['aop_channels'],
            threshold=m['spike_threshold'],
            decay=m['spike_decay'],
            readout='last',
        )
        self.feature_hu = FeatureHybridUnit(channels=m['feature_channels'])
        self.anchor_gen = AnchorGenerator(
            stride=m['anchor_stride'],
            scales=m['anchor_scales'],
            ratios=m['anchor_ratios'],
        )
        self.head = HSNTrackingHead(
            channels=m['feature_channels'],
            num_anchors=self.anchor_gen.num_anchors,
            stride=m['anchor_stride'],
            template_context=m.get('template_context', 0.25),
        )

    @property
    def num_anchors(self) -> int:
        return self.anchor_gen.num_anchors

    def encode_cop(self, x: torch.Tensor) -> torch.Tensor:
        return self.ann(x)

    def encode_aop(self, aop: torch.Tensor) -> torch.Tensor:
        return self.snn(aop)

    def forward_ann(self, template: torch.Tensor, template_box: torch.Tensor, search: torch.Tensor) -> Dict[str, torch.Tensor]:
        template_feat = self.encode_cop(template)
        search_feat = self.encode_cop(search)
        cls, reg = self.head(template_feat, search_feat, template_box)
        return {
            'cls': cls,
            'reg': reg,
            'template_feat': template_feat,
            'search_feat': search_feat,
            'pred_feat': search_feat,
        }

    def anchors_for(self, cls: torch.Tensor) -> torch.Tensor:
        return self.anchor_gen.grid_anchors(cls.shape[-2:], cls.device)

    def forward_hsn_sequence(
        self,
        template: torch.Tensor,
        template_box: torch.Tensor,
        ref: torch.Tensor,
        aop: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> Dict[str, object]:
        """
        High-frequency HSN forward.

        Args:
            template:     [B, 3, H, W]
            template_box: [B, 4]
            ref:          [B, 3, H, W]
            aop:          [B, K, 3, H, W]
            target:       [B, 3, H, W] or None

        Returns:
            cls_seq:       list length K, each [B, A*2, Hf, Wf]
            reg_seq:       list length K, each [B, A*4, Hf, Wf]
            pred_feat_seq: list length K, each [B, C, Hf, Wf]
        """
        template_feat = self.ann(template)
        ref_feat = self.ann(ref)

        cls_seq = []
        reg_seq = []
        pred_feat_seq = []

        K = aop.shape[1]

        for k in range(K):
            # 用从当前 COP 到第 k 个 AOP 的全部前缀输入 SNN
            aop_prefix = aop[:, :k + 1]

            snn_feat = self.snn(aop_prefix)
            delta_feat = self.feature_hu(snn_feat)
            pred_feat = ref_feat + delta_feat

            cls, reg = self.head(template_feat, pred_feat, template_box)

            cls_seq.append(cls)
            reg_seq.append(reg)
            pred_feat_seq.append(pred_feat)

        target_feat = self.ann(target) if target is not None else None

        return {
            "cls_seq": cls_seq,
            "reg_seq": reg_seq,
            "pred_feat_seq": pred_feat_seq,
            "target_feat": target_feat,
            "template_feat": template_feat,
            "ref_feat": ref_feat,
        }


    def forward_hsn(
        self,
        template: torch.Tensor,
        template_box: torch.Tensor,
        ref: torch.Tensor,
        aop: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible final-step HSN forward.
        """
        out = self.forward_hsn_sequence(
            template=template,
            template_box=template_box,
            ref=ref,
            aop=aop,
            target=target,
        )

        return {
            "cls": out["cls_seq"][-1],
            "reg": out["reg_seq"][-1],
            "pred_feat": out["pred_feat_seq"][-1],
            "target_feat": out["target_feat"],
            "template_feat": out["template_feat"],
            "ref_feat": out["ref_feat"],
        }

    @torch.no_grad()
    def decode(
        self,
        cls: torch.Tensor,
        reg: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode single-step anchor cls/reg output.

        Args:
            cls: [B, A*2, Hf, Wf]
            reg: [B, A*4, Hf, Wf]
            image_hw: [H, W] of padded input image

        Returns:
            boxes:  [B, 4]
            scores: [B]
        """
        anchors = self.anchors_for(cls)

        cls_f, reg_f = flatten_cls_reg(cls, reg, self.num_anchors)

        prob = cls_f.softmax(dim=-1)[..., 1]
        scores, idx = prob.max(dim=1)

        batch_indices = torch.arange(cls.shape[0], device=cls.device)

        deltas = reg_f[batch_indices, idx]
        boxes_all = decode_boxes(anchors, deltas)

        boxes = boxes_all[batch_indices, idx]

        h, w = image_hw
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h - 1)

        return boxes, scores

    @torch.no_grad()
    def decode_sequence(
        self,
        cls_seq: list[torch.Tensor],
        reg_seq: list[torch.Tensor],
        image_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode every AOP-step prediction.

        Returns:
            boxes_seq:  [B, K, 4]
            scores_seq: [B, K]
        """
        boxes_all = []
        scores_all = []

        for cls, reg in zip(cls_seq, reg_seq):
            boxes, scores = self.decode(cls, reg, image_hw)
            boxes_all.append(boxes)
            scores_all.append(scores)

        boxes_seq = torch.stack(boxes_all, dim=1)
        scores_seq = torch.stack(scores_all, dim=1)

        return boxes_seq, scores_seq
