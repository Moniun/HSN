from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def select_device(name: str = 'cuda') -> torch.device:
    if name == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(name)


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area(boxes1)[:, None]
    area2 = box_area(boxes2)[None, :]
    return inter / (area1 + area2 - inter + 1e-6)


def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Encode gt boxes relative to anchors. anchors [N,4], gt [B,N,4]."""
    a = xyxy_to_cxcywh(anchors)
    g = xyxy_to_cxcywh(gt_boxes)
    ax, ay, aw, ah = a.unbind(-1)
    gx, gy, gw, gh = g.unbind(-1)
    tx = (gx - ax) / aw.clamp(min=1e-6)
    ty = (gy - ay) / ah.clamp(min=1e-6)
    tw = torch.log(gw.clamp(min=1e-6) / aw.clamp(min=1e-6))
    th = torch.log(gh.clamp(min=1e-6) / ah.clamp(min=1e-6))
    return torch.stack([tx, ty, tw, th], dim=-1)


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Decode deltas relative to anchors. anchors [N,4], deltas [...,N,4]."""
    a = xyxy_to_cxcywh(anchors)
    ax, ay, aw, ah = a.unbind(-1)
    dx, dy, dw, dh = deltas.unbind(-1)
    px = dx * aw + ax
    py = dy * ah + ay
    pw = torch.exp(dw.clamp(max=4.0)) * aw
    ph = torch.exp(dh.clamp(max=4.0)) * ah
    return cxcywh_to_xyxy(torch.stack([px, py, pw, ph], dim=-1))


def center_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pc = (pred[..., :2] + pred[..., 2:]) / 2
    tc = (target[..., :2] + target[..., 2:]) / 2
    return torch.linalg.norm(pc - tc, dim=-1)


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def flatten_cls_reg(cls: torch.Tensor, reg: torch.Tensor, num_anchors: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """cls [B,A*2,H,W] -> [B,H*W*A,2], reg [B,A*4,H,W] -> [B,H*W*A,4]."""
    b, _, h, w = cls.shape
    cls = cls.view(b, num_anchors, 2, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 2)
    reg = reg.view(b, num_anchors, 4, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
    return cls, reg
