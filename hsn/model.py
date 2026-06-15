from __future__ import annotations

from typing import Dict, Optional, Tuple

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

        m = cfg["model"]

        self.cfg = cfg

        self.ann = ANNResNet22(in_channels=m["cop_channels"])

        self.snn = SNNResNet22(
            in_channels=m["aop_channels"],
            threshold=m["spike_threshold"],
            decay=m["spike_decay"],
            readout="last",
        )

        self.feature_hu = FeatureHybridUnit(channels=m["feature_channels"])

        self.anchor_gen = AnchorGenerator(
            stride=m["anchor_stride"],
            scales=m["anchor_scales"],
            ratios=m["anchor_ratios"],
        )

        self.head = HSNTrackingHead(
            channels=m["feature_channels"],
            num_anchors=self.anchor_gen.num_anchors,
            stride=m["anchor_stride"],
            template_context=m.get("template_context", 0.25),
        )

    @property
    def num_anchors(self) -> int:
        return self.anchor_gen.num_anchors

    @property
    def anchor_generator(self):
        return self.anchor_gen

    @property
    def hu(self):
        return self.feature_hu

    def encode_cop(self, x: torch.Tensor) -> torch.Tensor:
        return self.ann(x)

    def encode_aop(self, aop: torch.Tensor) -> torch.Tensor:
        return self.snn(aop)

    def forward(
        self,
        mode: str,
        template: torch.Tensor,
        template_box: torch.Tensor,
        search: Optional[torch.Tensor] = None,
        ref: Optional[torch.Tensor] = None,
        aop: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ):
        if mode == "ann":
            if search is None:
                if target is None:
                    raise ValueError("mode='ann' requires search or target")
                search = target

            return self.forward_ann(
                template=template,
                template_box=template_box,
                search=search,
            )

        if mode == "hsn":
            if ref is None or aop is None:
                raise ValueError("mode='hsn' requires ref and aop")

            return self.forward_hsn(
                template=template,
                template_box=template_box,
                ref=ref,
                aop=aop,
                target=target,
            )

        if mode == "hsn_sequence":
            if ref is None or aop is None:
                raise ValueError("mode='hsn_sequence' requires ref and aop")

            return self.forward_hsn_sequence(
                template=template,
                template_box=template_box,
                ref=ref,
                aop=aop,
                target=target,
            )

        raise ValueError(f"Unknown forward mode: {mode}")

    def forward_ann(
        self,
        template: torch.Tensor,
        template_box: torch.Tensor,
        search: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        template_feat = self.ann(template)
        search_feat = self.ann(search)

        cls, reg = self.head(
            template_feat,
            search_feat,
            template_box,
        )

        return {
            "cls": cls,
            "reg": reg,
            "template_feat": template_feat,
            "search_feat": search_feat,
            "pred_feat": search_feat,
        }

    def forward_hsn_sequence(
        self,
        template: torch.Tensor,
        template_box: torch.Tensor,
        ref: torch.Tensor,
        aop: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        template_feat = self.ann(template)
        ref_feat = self.ann(ref)

        snn_feat_seq = self.snn(
            aop,
            return_sequence=True,
        )

        cls_seq = []
        reg_seq = []
        pred_feat_seq = []

        K = snn_feat_seq.shape[1]

        for k in range(K):
            snn_feat_k = snn_feat_seq[:, k]

            delta_feat = self.feature_hu(snn_feat_k)
            pred_feat = ref_feat + delta_feat

            cls, reg = self.head(
                template_feat,
                pred_feat,
                template_box,
            )

            cls_seq.append(cls)
            reg_seq.append(reg)
            pred_feat_seq.append(pred_feat)

        target_feat = None
        if target is not None:
            with torch.no_grad():
                target_feat = self.ann(target)

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
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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

    def anchors_for(self, cls: torch.Tensor) -> torch.Tensor:
        return self.anchor_gen.grid_anchors(cls.shape[-2:], cls.device)

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
        anchors = self.anchors_for(cls)  # [N, 4]

        if anchors.ndim == 3:
            anchors = anchors[0]

        cls_f, reg_f = flatten_cls_reg(cls, reg, self.num_anchors)
        # cls_f: [B, N, 2]
        # reg_f: [B, N, 4]

        prob = cls_f.softmax(dim=-1)[..., 1]  # [B, N]
        scores, idx = prob.max(dim=1)         # [B]

        batch_indices = torch.arange(cls.shape[0], device=cls.device)

        if anchors.shape[0] != reg_f.shape[1]:
            raise RuntimeError(
                f"Anchor number mismatch: anchors={anchors.shape}, "
                f"reg_f={reg_f.shape}. Check AnchorGenerator/grid size."
            )

        # Correct logic:
        # select the best anchor for each batch item,
        # then decode its corresponding delta.
        selected_anchors = anchors[idx]                    # [B, 4]
        selected_deltas = reg_f[batch_indices, idx]        # [B, 4]

        boxes = decode_boxes(selected_anchors, selected_deltas)

        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

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
