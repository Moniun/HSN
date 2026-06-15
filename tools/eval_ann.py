from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsn.data import TianmoucHSNDataset  # noqa: E402
from hsn.model import TianmoucHSN  # noqa: E402
from hsn.utils import decode_boxes, flatten_cls_reg  # noqa: E402


def load_yaml(path: str | Path) -> Dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: torch.nn.Module, checkpoint: str | Path, device: torch.device) -> None:
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


@torch.no_grad()
def decode_outputs(
    model: TianmoucHSN,
    cls: torch.Tensor,
    reg: torch.Tensor,
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Local robust decode, independent of model.decode implementation.

    Args:
        cls: [B, A*2, Hf, Wf]
        reg: [B, A*4, Hf, Wf]
        image_hw: padded image size [H, W]

    Returns:
        boxes: [B, 4]
        scores: [B]
    """
    anchors = model.anchor_gen.grid_anchors(cls.shape[-2:], cls.device)

    cls_f, reg_f = flatten_cls_reg(cls, reg, model.num_anchors)
    # cls_f: [B, N, 2]
    # reg_f: [B, N, 4]
    # anchors: [N, 4]

    if anchors.shape[0] != cls_f.shape[1]:
        raise RuntimeError(
            f"Anchor number mismatch: anchors={anchors.shape}, cls_f={cls_f.shape}, reg_f={reg_f.shape}"
        )

    prob = cls_f.softmax(dim=-1)[..., 1]
    scores, idx = prob.max(dim=1)

    batch_indices = torch.arange(cls.shape[0], device=cls.device)

    selected_anchors = anchors[idx]
    selected_deltas = reg_f[batch_indices, idx]

    boxes = decode_boxes(selected_anchors, selected_deltas)

    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)

    h, w = image_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h - 1)

    return boxes, scores


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Args:
        a: [N, 4]
        b: [N, 4]

    Returns:
        iou: [N]
    """
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


def summarize_metrics(records: List[Dict]) -> Dict[str, float]:
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


def group_by_seq(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict]] = {}

    for r in records:
        seq = str(r["seq_name"])
        groups.setdefault(seq, []).append(r)

    return {seq: summarize_metrics(items) for seq, items in groups.items()}


def save_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: str | Path, records: List[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
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
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hsn_tianmouc.yaml")
    parser.add_argument("--data-root", default="data/nfs")
    parser.add_argument("--checkpoint", default="runs/tianmouc_hsn_reproduce/ann_best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-json", default="runs/eval/ann_eval.json")
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--print-worst", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 80)
    print("[CONFIG]")
    print(f"config      : {args.config}")
    print(f"data_root   : {args.data_root}")
    print(f"split       : {args.split}")
    print(f"checkpoint  : {args.checkpoint}")
    print(f"batch_size  : {args.batch_size}")
    print(f"num_workers : {args.num_workers}")
    print(f"device      : {device}")
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

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=False,
    )

    model = TianmoucHSN(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    image_hw = tuple(cfg["data"].get("padded_size", [232, 296]))

    records: List[Dict] = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="eval_ann", ncols=120)

        for batch in pbar:
            template = batch["template"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            template_box = batch["template_box"].to(device, non_blocking=True)
            target_box = batch["target_box"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
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

            pred_boxes, scores = decode_outputs(
                model=model,
                cls=out["cls"],
                reg=out["reg"],
                image_hw=image_hw,
            )

            pred_np = pred_boxes.detach().cpu().numpy()
            gt_np = target_box.detach().cpu().numpy()
            score_np = scores.detach().cpu().numpy()

            ious = box_iou_np(pred_np, gt_np)
            errors = center_error_np(pred_np, gt_np)

            seq_names = batch.get("seq_name", [""] * len(ious))
            obj_ids = batch.get("obj_id", torch.zeros(len(ious), dtype=torch.long))
            template_ts = batch.get("template_t", torch.zeros(len(ious), dtype=torch.long))
            target_ts = batch.get("target_t", torch.zeros(len(ious), dtype=torch.long))

            if torch.is_tensor(obj_ids):
                obj_ids = obj_ids.detach().cpu().numpy()
            if torch.is_tensor(template_ts):
                template_ts = template_ts.detach().cpu().numpy()
            if torch.is_tensor(target_ts):
                target_ts = target_ts.detach().cpu().numpy()

            for i in range(len(ious)):
                records.append(
                    {
                        "seq_name": str(seq_names[i]),
                        "obj_id": int(obj_ids[i]),
                        "template_t": int(template_ts[i]),
                        "target_t": int(target_ts[i]),
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
                )

            cur = summarize_metrics(records)
            pbar.set_postfix(
                {
                    "mIoU": f"{cur['mean_iou']:.3f}",
                    "AUC": f"{cur['success_auc']:.3f}",
                    "P20": f"{cur['precision_20px']:.3f}",
                    "score": f"{cur['mean_score']:.3f}",
                }
            )

    overall = summarize_metrics(records)
    per_seq = group_by_seq(records)

    payload = {
        "config": args.config,
        "data_root": args.data_root,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "overall": overall,
        "per_seq": per_seq,
    }

    print("=" * 80)
    print("[ANN EVAL RESULT]")
    for k, v in overall.items():
        print(f"{k}: {v}")
    print("=" * 80)

    if args.print_worst > 0 and records:
        print(f"[WORST {args.print_worst}]")
        worst = sorted(records, key=lambda x: x["iou"])[: args.print_worst]
        for r in worst:
            print(
                f"seq={r['seq_name']}, target_t={r['target_t']}, "
                f"iou={r['iou']:.4f}, err={r['center_error']:.2f}, score={r['score']:.4f}"
            )

    if args.save_json:
        save_json(args.save_json, payload)
        print(f"[saved] json: {args.save_json}")

    if args.save_csv:
        save_csv(args.save_csv, records)
        print(f"[saved] csv: {args.save_csv}")


if __name__ == "__main__":
    main()
