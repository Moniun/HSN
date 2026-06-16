from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsn.data import TianmoucHSNDataset  # noqa: E402
from hsn.model import TianmoucHSN  # noqa: E402
from hsn.utils import box_iou, decode_boxes, flatten_cls_reg  # noqa: E402


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: torch.nn.Module, checkpoint: str | Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[checkpoint] loaded: {checkpoint}")
    print(f"[checkpoint] missing keys: {len(missing)}")
    print(f"[checkpoint] unexpected keys: {len(unexpected)}")

    if missing:
        print("[checkpoint] missing examples:", missing[:10])
    if unexpected:
        print("[checkpoint] unexpected examples:", unexpected[:10])

    return ckpt if isinstance(ckpt, dict) else {}


def set_eval_mode(model: nn.Module, bn_mode: str) -> None:
    """
    bn_mode:
      eval     : normal eval mode, use BN running mean/var
      train_bn : use batch statistics for BN, but do not update running stats
    """
    if bn_mode == "eval":
        model.eval()
        return

    if bn_mode == "train_bn":
        model.train()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.momentum = 0.0
        return

    raise ValueError(f"Unsupported bn_mode: {bn_mode}")


def clamp_boxes(boxes: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h - 1)
    return boxes


@torch.no_grad()
def decode_top1_outputs(
    model: TianmoucHSN,
    cls: torch.Tensor,
    reg: torch.Tensor,
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      pred_boxes: [B, 4]
      scores: [B]
      top1_idx: [B]
      anchors: [N, 4]
      cls_f: [B, N, 2]
      reg_f: [B, N, 4]
    """
    anchors = model.anchor_gen.grid_anchors(cls.shape[-2:], cls.device)
    cls_f, reg_f = flatten_cls_reg(cls, reg, model.num_anchors)

    if anchors.shape[0] != cls_f.shape[1]:
        raise RuntimeError(
            f"Anchor number mismatch: anchors={anchors.shape}, "
            f"cls_f={cls_f.shape}, reg_f={reg_f.shape}"
        )

    prob = cls_f.softmax(dim=-1)[..., 1]
    scores, top1_idx = prob.max(dim=1)

    batch_indices = torch.arange(cls.shape[0], device=cls.device)
    selected_anchors = anchors[top1_idx]
    selected_deltas = reg_f[batch_indices, top1_idx]

    pred_boxes = decode_boxes(selected_anchors, selected_deltas)
    if pred_boxes.ndim == 1:
        pred_boxes = pred_boxes.unsqueeze(0)

    pred_boxes = clamp_boxes(pred_boxes, image_hw)
    return pred_boxes, scores, top1_idx, anchors, cls_f, reg_f


@torch.no_grad()
def compute_oracle_diagnostics(
    anchors: torch.Tensor,
    reg_f: torch.Tensor,
    cls_f: torch.Tensor,
    top1_idx: torch.Tensor,
    target_box: torch.Tensor,
    pred_boxes: torch.Tensor,
    image_hw: Tuple[int, int],
    topk_list: List[int],
) -> List[Dict[str, float]]:
    """
    For each sample, compute:
      - top1_anchor_iou: IoU between selected top-score anchor and GT
      - best_anchor_iou: max IoU between any anchor and GT
      - best_anchor_score: foreground score at best-GT-IoU anchor
      - best_anchor_rank: score rank of best-GT-IoU anchor, 1 means highest score
      - oracle_iou: decode reg at best-GT-IoU anchor and compute IoU
      - top{k}_anchor_iou_max: among top-k scored anchors, max anchor IoU with GT

    Interpretation:
      If oracle_iou is decent but normal pred_iou is poor, classification selection is failing.
      If oracle_iou is also poor, regression/decode/target learning is also failing.
    """
    prob = cls_f.softmax(dim=-1)[..., 1]
    batch_size, num_anchors = prob.shape

    diagnostics: List[Dict[str, float]] = []

    for i in range(batch_size):
        gt = target_box[i : i + 1]

        anchor_ious = box_iou(anchors, gt).squeeze(1)
        best_anchor_idx = torch.argmax(anchor_ious)

        top1_anchor_iou = anchor_ious[top1_idx[i]]
        best_anchor_iou = anchor_ious[best_anchor_idx]

        best_anchor_score = prob[i, best_anchor_idx]
        best_anchor_rank = int((prob[i] > best_anchor_score).sum().detach().cpu()) + 1

        oracle_box = decode_boxes(
            anchors[best_anchor_idx].unsqueeze(0),
            reg_f[i, best_anchor_idx].unsqueeze(0),
        )
        oracle_box = clamp_boxes(oracle_box, image_hw)

        oracle_iou = box_iou(oracle_box, gt).squeeze()
        oracle_center_err = center_error_torch(oracle_box, gt).squeeze()

        pred_iou = box_iou(pred_boxes[i : i + 1], gt).squeeze()
        pred_center_err = center_error_torch(pred_boxes[i : i + 1], gt).squeeze()

        item: Dict[str, float] = {
            "pred_iou_torch": float(pred_iou.detach().cpu()),
            "pred_center_error_torch": float(pred_center_err.detach().cpu()),
            "top1_anchor_iou": float(top1_anchor_iou.detach().cpu()),
            "best_anchor_iou": float(best_anchor_iou.detach().cpu()),
            "best_anchor_score": float(best_anchor_score.detach().cpu()),
            "best_anchor_rank": float(best_anchor_rank),
            "oracle_iou": float(oracle_iou.detach().cpu()),
            "oracle_center_error": float(oracle_center_err.detach().cpu()),
            "score_gap_top1_minus_best_anchor": float(
                (prob[i, top1_idx[i]] - best_anchor_score).detach().cpu()
            ),
        }

        for k in topk_list:
            kk = min(int(k), num_anchors)
            topk_idx = torch.topk(prob[i], k=kk, largest=True).indices
            topk_anchor_iou_max = anchor_ious[topk_idx].max()
            item[f"top{k}_anchor_iou_max"] = float(topk_anchor_iou_max.detach().cpu())

        diagnostics.append(item)

    return diagnostics


def center_error_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = (a[..., :2] + a[..., 2:]) * 0.5
    bc = (b[..., :2] + b[..., 2:]) * 0.5
    return torch.linalg.norm(ac - bc, dim=-1)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    ix1 = np.maximum(a[:, 0], b[:, 0])
    iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 2], b[:, 2])
    iy2 = np.minimum(a[:, 3], b[:, 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])

    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def center_error_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = (a[:, :2] + a[:, 2:]) * 0.5
    bc = (b[:, :2] + b[:, 2:]) * 0.5
    return np.linalg.norm(ac - bc, axis=1)


def success_auc_np(ious: np.ndarray) -> float:
    if len(ious) == 0:
        return 0.0

    thresholds = np.linspace(0.0, 1.0, 21)
    success = [(ious >= t).mean() for t in thresholds]
    return float(np.mean(success))


def precision_at_np(errors: np.ndarray, thr: float = 20.0) -> float:
    if len(errors) == 0:
        return 0.0
    return float((errors <= thr).mean())


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.array(values, dtype=np.float32)))


def safe_median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.array(values, dtype=np.float32)))


def safe_percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def summarize_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {
            "num_samples": 0,
            "mean_iou": 0.0,
            "success_auc": 0.0,
            "precision_20px": 0.0,
            "mean_center_error": 0.0,
            "median_center_error": 0.0,
            "mean_score": 0.0,
        }

    ious = np.array([r["iou"] for r in records], dtype=np.float32)
    errors = np.array([r["center_error"] for r in records], dtype=np.float32)
    scores = np.array([r["score"] for r in records], dtype=np.float32)

    return {
        "num_samples": int(len(records)),
        "mean_iou": float(np.mean(ious)),
        "success_auc": success_auc_np(ious),
        "precision_20px": precision_at_np(errors, 20.0),
        "mean_center_error": float(np.mean(errors)),
        "median_center_error": float(np.median(errors)),
        "mean_score": float(np.mean(scores)),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p90": float(np.percentile(scores, 90)),
    }


def summarize_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {}

    keys = [
        "top1_anchor_iou",
        "best_anchor_iou",
        "best_anchor_score",
        "best_anchor_rank",
        "oracle_iou",
        "oracle_center_error",
        "score_gap_top1_minus_best_anchor",
    ]

    out: Dict[str, float] = {}

    for key in keys:
        vals = [float(r[key]) for r in records if key in r]
        out[f"{key}_mean"] = safe_mean(vals)
        out[f"{key}_p50"] = safe_median(vals)
        out[f"{key}_p90"] = safe_percentile(vals, 90)

    dynamic_topk_keys = sorted(
        {
            k
            for r in records
            for k in r.keys()
            if k.startswith("top") and k.endswith("_anchor_iou_max")
        }
    )

    for key in dynamic_topk_keys:
        vals = [float(r[key]) for r in records if key in r]
        out[f"{key}_mean"] = safe_mean(vals)
        out[f"{key}_p50"] = safe_median(vals)
        out[f"{key}_p90"] = safe_percentile(vals, 90)

    oracle_ious = np.array([r["oracle_iou"] for r in records if "oracle_iou" in r], dtype=np.float32)
    pred_ious = np.array([r["iou"] for r in records if "iou" in r], dtype=np.float32)
    ranks = np.array([r["best_anchor_rank"] for r in records if "best_anchor_rank" in r], dtype=np.float32)

    if len(oracle_ious) > 0:
        out["oracle_success_auc"] = success_auc_np(oracle_ious)
        out["oracle_precision_iou_0p3"] = float((oracle_ious >= 0.3).mean())
        out["oracle_precision_iou_0p5"] = float((oracle_ious >= 0.5).mean())

    if len(pred_ious) > 0:
        out["pred_precision_iou_0p3"] = float((pred_ious >= 0.3).mean())
        out["pred_precision_iou_0p5"] = float((pred_ious >= 0.5).mean())

    if len(ranks) > 0:
        out["best_anchor_rank_le_1"] = float((ranks <= 1).mean())
        out["best_anchor_rank_le_5"] = float((ranks <= 5).mean())
        out["best_anchor_rank_le_20"] = float((ranks <= 20).mean())
        out["best_anchor_rank_le_100"] = float((ranks <= 100).mean())

    return out


def group_by_seq(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        seq = str(r["seq_name"])
        groups.setdefault(seq, []).append(r)

    return {seq: summarize_metrics(items) for seq, items in groups.items()}


def group_diag_by_seq(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        seq = str(r["seq_name"])
        groups.setdefault(seq, []).append(r)

    return {seq: summarize_diagnostics(items) for seq, items in groups.items()}


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: str | Path, records: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    base_fields = [
        "seq_name",
        "obj_id",
        "template_t",
        "target_t",
        "iou",
        "center_error",
        "score",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "top1_anchor_iou",
        "best_anchor_iou",
        "best_anchor_score",
        "best_anchor_rank",
        "oracle_iou",
        "oracle_center_error",
        "score_gap_top1_minus_best_anchor",
    ]

    extra_fields = sorted(
        {
            k
            for r in records
            for k in r.keys()
            if k not in base_fields
        }
    )

    fields = base_fields + extra_fields

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})


def parse_topk(value: str) -> List[int]:
    out = []
    for x in value.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return sorted(set(out))


def get_batch_item(value: Any, idx: int, default: Any = "") -> Any:
    if value is None:
        return default

    if torch.is_tensor(value):
        v = value.detach().cpu()
        if v.ndim == 0:
            return v.item()
        return v[idx].item()

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value[idx]

    if isinstance(value, (list, tuple)):
        return value[idx]

    return value


def print_interpretation(diag: Dict[str, float]) -> None:
    print("=" * 80)
    print("[DIAGNOSIS HINT]")

    top1_anchor_iou = diag.get("top1_anchor_iou_mean", 0.0)
    best_anchor_iou = diag.get("best_anchor_iou_mean", 0.0)
    best_rank_p50 = diag.get("best_anchor_rank_p50", 0.0)
    oracle_iou = diag.get("oracle_iou_mean", 0.0)
    pred_iou03 = diag.get("pred_precision_iou_0p3", 0.0)
    oracle_iou03 = diag.get("oracle_precision_iou_0p3", 0.0)

    print(f"top1_anchor_iou_mean : {top1_anchor_iou:.4f}")
    print(f"best_anchor_iou_mean : {best_anchor_iou:.4f}")
    print(f"best_anchor_rank_p50 : {best_rank_p50:.1f}")
    print(f"oracle_iou_mean      : {oracle_iou:.4f}")
    print(f"pred IoU>=0.3 ratio  : {pred_iou03:.4f}")
    print(f"oracle IoU>=0.3 ratio: {oracle_iou03:.4f}")

    if top1_anchor_iou < 0.10 and best_rank_p50 > 20 and oracle_iou >= 0.25:
        print(
            "[likely] classification selection failed: "
            "GT-near anchors are not ranked high, but regression near GT is usable."
        )
    elif top1_anchor_iou < 0.10 and oracle_iou < 0.15:
        print(
            "[likely] classification and regression both failed, "
            "or regression target/decode/anchor assignment needs checking."
        )
    elif top1_anchor_iou >= 0.25 and oracle_iou < 0.15:
        print(
            "[likely] top-score anchor is near GT, but decoded box is bad. "
            "Check regression output, decode_boxes, and encode/decode consistency."
        )
    else:
        print(
            "[mixed] no single obvious failure mode. "
            "Check per-sequence diagnostics in the saved json/csv."
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/hsn_tianmouc.yaml")
    parser.add_argument("--data-root", default="data/nfs")
    parser.add_argument("--checkpoint", default="runs/tianmouc_hsn_reproduce/ann_best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val"])

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--save-json", default="runs/eval/ann_eval.json")
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--print-worst", type=int, default=20)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--bn-mode", default="eval", choices=["eval", "train_bn"])
    parser.add_argument("--topk", default="1,5,20,100")

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.batch_size is None:
        args.batch_size = int(cfg["train"].get("ann_batch_size", 64))

    topk_list = parse_topk(args.topk)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 80)
    print("[CONFIG]")
    print(f"config     : {args.config}")
    print(f"data_root  : {args.data_root}")
    print(f"split      : {args.split}")
    print(f"checkpoint : {args.checkpoint}")
    print(f"batch_size : {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"device     : {device}")
    print(f"bn_mode    : {args.bn_mode}")
    print(f"topk       : {topk_list}")
    print("=" * 80)

    dataset = TianmoucHSNDataset(
        cfg,
        split=args.split,
        data_root=args.data_root,
        mode="ann",
    )

    if args.max_samples is not None and args.max_samples > 0:
        n = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
        print(f"[dataset] using subset: {n} samples")

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(**loader_kwargs)

    model = TianmoucHSN(cfg).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)

    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print(f"[checkpoint] epoch: {ckpt['epoch']}")
        if "metrics" in ckpt:
            print(f"[checkpoint] metrics: {ckpt['metrics']}")

    set_eval_mode(model, args.bn_mode)

    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="eval_ann", ncols=120)

        for batch in pbar:
            template = batch["template"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            template_box = batch["template_box"].to(device, non_blocking=True)
            target_box = batch["target_box"].to(device, non_blocking=True)

            image_hw = (int(target.shape[-2]), int(target.shape[-1]))

            if args.amp and device.type == "cuda":
                amp_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                amp_ctx = nullcontext()

            with amp_ctx:
                try:
                    out = model(
                        mode="ann",
                        template=template,
                        template_box=template_box,
                        search=target,
                    )
                except TypeError:
                    out = model.forward_ann(
                        template=template,
                        template_box=template_box,
                        search=target,
                    )

                pred_boxes, scores, top1_idx, anchors, cls_f, reg_f = decode_top1_outputs(
                    model=model,
                    cls=out["cls"],
                    reg=out["reg"],
                    image_hw=image_hw,
                )

            diag_items = compute_oracle_diagnostics(
                anchors=anchors,
                reg_f=reg_f,
                cls_f=cls_f,
                top1_idx=top1_idx,
                target_box=target_box,
                pred_boxes=pred_boxes,
                image_hw=image_hw,
                topk_list=topk_list,
            )

            pred_np = pred_boxes.detach().cpu().numpy()
            gt_np = target_box.detach().cpu().numpy()
            score_np = scores.detach().cpu().numpy()

            ious = box_iou_np(pred_np, gt_np)
            errors = center_error_np(pred_np, gt_np)

            seq_names = batch.get("seq_name", [""] * len(ious))
            obj_ids = batch.get("obj_id", None)
            template_ts = batch.get("template_t", None)
            target_ts = batch.get("target_t", None)

            for i in range(len(ious)):
                record: Dict[str, Any] = {
                    "seq_name": str(get_batch_item(seq_names, i, "")),
                    "obj_id": int(get_batch_item(obj_ids, i, -1)),
                    "template_t": int(get_batch_item(template_ts, i, -1)),
                    "target_t": int(get_batch_item(target_ts, i, -1)),
                    "iou": float(ious[i]),
                    "center_error": float(errors[i]),
                    "score": float(score_np[i]),
                    "pred_x1": float(pred_np[i, 0]),
                    "pred_y1": float(pred_np[i, 1]),
                    "pred_x2": float(pred_np[i, 2]),
                    "pred_y2": float(pred_np[i, 3]),
                    "gt_x1": float(gt_np[i, 0]),
                    "gt_y1": float(gt_np[i, 1]),
                    "gt_x2": float(gt_np[i, 2]),
                    "gt_y2": float(gt_np[i, 3]),
                }

                record.update(diag_items[i])
                records.append(record)

            cur = summarize_metrics(records)
            cur_diag = summarize_diagnostics(records)

            pbar.set_postfix(
                {
                    "mIoU": f"{cur['mean_iou']:.3f}",
                    "AUC": f"{cur['success_auc']:.3f}",
                    "P20": f"{cur['precision_20px']:.3f}",
                    "score": f"{cur['mean_score']:.3f}",
                    "t1A": f"{cur_diag.get('top1_anchor_iou_mean', 0.0):.3f}",
                    "orIoU": f"{cur_diag.get('oracle_iou_mean', 0.0):.3f}",
                    "rank": f"{cur_diag.get('best_anchor_rank_p50', 0.0):.0f}",
                }
            )

    overall = summarize_metrics(records)
    diagnostics = summarize_diagnostics(records)

    per_seq = group_by_seq(records)
    per_seq_diagnostics = group_diag_by_seq(records)

    payload = {
        "config": args.config,
        "data_root": args.data_root,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "bn_mode": args.bn_mode,
        "topk": topk_list,
        "overall": overall,
        "diagnostics": diagnostics,
        "per_seq": per_seq,
        "per_seq_diagnostics": per_seq_diagnostics,
    }

    print("=" * 80)
    print("[ANN EVAL RESULT]")
    for k, v in overall.items():
        print(f"{k}: {v}")

    print("=" * 80)
    print("[ANN ORACLE DIAGNOSTICS]")
    for k, v in diagnostics.items():
        print(f"{k}: {v}")

    print_interpretation(diagnostics)

    if args.print_worst > 0 and records:
        print("=" * 80)
        print(f"[WORST {args.print_worst}]")
        worst = sorted(records, key=lambda x: x["iou"])[: args.print_worst]

        for r in worst:
            print(
                f"seq={r['seq_name']}, "
                f"target_t={r['target_t']}, "
                f"iou={r['iou']:.4f}, "
                f"err={r['center_error']:.2f}, "
                f"score={r['score']:.4f}, "
                f"top1_anchor_iou={r['top1_anchor_iou']:.4f}, "
                f"best_anchor_iou={r['best_anchor_iou']:.4f}, "
                f"best_anchor_rank={r['best_anchor_rank']:.0f}, "
                f"oracle_iou={r['oracle_iou']:.4f}"
            )

    if args.save_json:
        save_json(args.save_json, payload)
        print(f"[saved] json: {args.save_json}")

    if args.save_csv:
        save_csv(args.save_csv, records)
        print(f"[saved] csv: {args.save_csv}")


if __name__ == "__main__":
    main()