from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsn.model import TianmoucHSN
from hsn.data import TianmoucHSNDataset


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, device: torch.device) -> Dict[str, Any]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[checkpoint] loaded: {ckpt_path}")
    print(f"[checkpoint] missing keys: {len(missing)}")
    print(f"[checkpoint] unexpected keys: {len(unexpected)}")
    if missing:
        print("[checkpoint] missing examples:", missing[:10])
    if unexpected:
        print("[checkpoint] unexpected examples:", unexpected[:10])
    return ckpt if isinstance(ckpt, dict) else {}


def to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def box_iou_xyxy(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred[None, :]
    if gt.ndim == 1:
        gt = gt[None, :]

    x1 = np.maximum(pred[:, 0], gt[:, 0])
    y1 = np.maximum(pred[:, 1], gt[:, 1])
    x2 = np.minimum(pred[:, 2], gt[:, 2])
    y2 = np.minimum(pred[:, 3], gt[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_p = np.maximum(0.0, pred[:, 2] - pred[:, 0]) * np.maximum(0.0, pred[:, 3] - pred[:, 1])
    area_g = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    union = area_p + area_g - inter
    return inter / np.maximum(union, 1e-6)


def center_xy(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    return np.stack([(box[..., 0] + box[..., 2]) * 0.5, (box[..., 1] + box[..., 3]) * 0.5], axis=-1)


def center_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    cp = center_xy(pred)
    cg = center_xy(gt)
    return np.linalg.norm(cp - cg, axis=-1)


def safe_mean(x: Iterable[float]) -> Optional[float]:
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def safe_median(x: Iterable[float]) -> Optional[float]:
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def success_curve(ious: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if ious.size == 0:
        return np.zeros_like(thresholds, dtype=np.float64)
    return np.asarray([(ious >= th).mean() for th in thresholds], dtype=np.float64)


def precision_curve(errors: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if errors.size == 0:
        return np.zeros_like(thresholds, dtype=np.float64)
    return np.asarray([(errors <= th).mean() for th in thresholds], dtype=np.float64)


def summarize_records(records: List[Dict[str, Any]], name: str, out_dir: Path) -> Dict[str, Any]:
    ious = np.asarray([r["iou"] for r in records], dtype=np.float64)
    errors = np.asarray([r["center_error"] for r in records], dtype=np.float64)
    scores = np.asarray([r.get("score", np.nan) for r in records], dtype=np.float64)

    overlap_th = np.linspace(0.0, 1.0, 101)
    precision_th = np.arange(0.0, 51.0, 1.0)
    succ = success_curve(ious, overlap_th)
    prec = precision_curve(errors, precision_th)

    curve_path = out_dir / f"{name}_success_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["overlap_threshold", "success_rate"])
        for th, sr in zip(overlap_th, succ):
            w.writerow([float(th), float(sr)])

    precision_path = out_dir / f"{name}_precision_curve.csv"
    with precision_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["center_error_threshold_px", "precision_rate"])
        for th, pr in zip(precision_th, prec):
            w.writerow([float(th), float(pr)])

    return {
        "name": name,
        "num_predictions": int(len(records)),
        "mean_iou_mIoU": safe_mean(ious),
        "median_iou": safe_median(ious),
        "success_auc_overlap_0_1": float(succ.mean()) if len(records) else None,
        "success_rate_iou_0p3": float((ious >= 0.3).mean()) if len(records) else None,
        "success_rate_iou_0p5": float((ious >= 0.5).mean()) if len(records) else None,
        "mean_center_error_px": safe_mean(errors),
        "median_center_error_px": safe_median(errors),
        "precision_20px": float((errors <= 20.0).mean()) if len(records) else None,
        "precision_10px": float((errors <= 10.0).mean()) if len(records) else None,
        "mean_score": safe_mean(scores),
        "score_p50": safe_median(scores),
        "success_curve_csv": str(curve_path),
        "precision_curve_csv": str(precision_path),
    }


def parse_latency_steps(s: str) -> List[int]:
    steps = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        steps.append(int(p))
    if not steps:
        steps = [0]
    if min(steps) < 0:
        raise ValueError("latency steps must be >= 0")
    return sorted(set(steps))


def parse_speed_bins(s: str) -> List[float]:
    vals = []
    for p in s.split(","):
        p = p.strip().lower()
        if p in ("inf", "+inf", "infinity"):
            vals.append(float("inf"))
        elif p:
            vals.append(float(p))
    if len(vals) < 2:
        raise ValueError("speed bins must contain at least two edges")
    if vals[0] != 0.0:
        vals = [0.0] + vals
    if not math.isinf(vals[-1]):
        vals.append(float("inf"))
    if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError(f"speed bins must be increasing: {vals}")
    return vals


def write_predictions_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    keys = [
        "metric_scope", "seq_name", "obj_id", "template_t", "ref_t", "target_t",
        "aop_idx", "latency_steps", "speed_px_per_aop_step", "score", "iou", "center_error",
        "pred_x1", "pred_y1", "pred_x2", "pred_y2", "gt_x1", "gt_y1", "gt_x2", "gt_y2",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in keys}
            w.writerow(row)


def add_record(
    records: List[Dict[str, Any]],
    metric_scope: str,
    sample: Dict[str, Any],
    pred_box: np.ndarray,
    gt_box: np.ndarray,
    score: float,
    aop_idx: Optional[int] = None,
    latency_steps: int = 0,
    speed_px_per_aop_step: Optional[float] = None,
) -> None:
    pred_box = np.asarray(pred_box, dtype=np.float32).reshape(4)
    gt_box = np.asarray(gt_box, dtype=np.float32).reshape(4)
    iou = float(box_iou_xyxy(pred_box, gt_box)[0])
    err = float(center_error(pred_box, gt_box))

    def scalar(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.item()
            return to_np(x).tolist()
        return x

    seq = sample.get("seq_name", "")
    if isinstance(seq, (list, tuple)):
        seq = seq[0]

    records.append({
        "metric_scope": metric_scope,
        "seq_name": str(seq),
        "obj_id": int(scalar(sample.get("obj_id", -1))),
        "template_t": int(scalar(sample.get("template_t", -1))),
        "ref_t": int(scalar(sample.get("ref_t", -1))) if "ref_t" in sample else "",
        "target_t": int(scalar(sample.get("target_t", -1))) if "target_t" in sample else "",
        "aop_idx": int(aop_idx) if aop_idx is not None else "",
        "latency_steps": int(latency_steps),
        "speed_px_per_aop_step": float(speed_px_per_aop_step) if speed_px_per_aop_step is not None else "",
        "score": float(score),
        "iou": iou,
        "center_error": err,
        "pred_x1": float(pred_box[0]),
        "pred_y1": float(pred_box[1]),
        "pred_x2": float(pred_box[2]),
        "pred_y2": float(pred_box[3]),
        "gt_x1": float(gt_box[0]),
        "gt_y1": float(gt_box[1]),
        "gt_x2": float(gt_box[2]),
        "gt_y2": float(gt_box[3]),
    })


def gpu_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def eval_ann(
    model: TianmoucHSN,
    cfg: Dict[str, Any],
    split: str,
    data_root: Optional[str],
    device: torch.device,
    max_samples: Optional[int],
    out_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset = TianmoucHSNDataset(cfg, split=split, data_root=data_root, mode="ann")
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    image_hw = tuple(cfg["data"].get("padded_size", [232, 296]))
    records: List[Dict[str, Any]] = []

    total_time = 0.0
    total_updates = 0

    for i in range(n):
        sample = dataset[i]
        template = sample["template"].unsqueeze(0).to(device, non_blocking=True)
        search = sample["target"].unsqueeze(0).to(device, non_blocking=True)
        template_box = sample["template_box"].unsqueeze(0).to(device, non_blocking=True)
        gt_box = to_np(sample["target_box"])

        gpu_sync_if_needed(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(mode="ann", template=template, template_box=template_box, search=search)
            pred_box, score = model.decode(out["cls"], out["reg"], image_hw=image_hw)
        gpu_sync_if_needed(device)
        total_time += time.perf_counter() - t0
        total_updates += 1

        add_record(
            records,
            metric_scope="ann_offline_cop_rate",
            sample=sample,
            pred_box=to_np(pred_box[0]),
            gt_box=gt_box,
            score=float(score[0].detach().cpu()),
            latency_steps=0,
        )

        if (i + 1) % 200 == 0 or i == 0 or i + 1 == n:
            print(f"[ann] {i + 1}/{n}")

    summary = summarize_records(records, "ann_offline_cop_rate", out_dir)
    summary.update({
        "wall_time_s": total_time,
        "cop_updates_per_second_fps": total_updates / max(total_time, 1e-9),
        "avg_latency_ms_per_cop_update": 1000.0 * total_time / max(total_updates, 1),
    })
    return summary, records


def gt_speed_for_index(gt_seq: np.ndarray, idx: int) -> float:
    if len(gt_seq) <= 1:
        return 0.0
    idx = int(np.clip(idx, 0, len(gt_seq) - 1))
    c = center_xy(gt_seq)
    if idx == 0:
        return float(np.linalg.norm(c[1] - c[0]))
    return float(np.linalg.norm(c[idx] - c[idx - 1]))


def eval_hsn(
    model: TianmoucHSN,
    cfg: Dict[str, Any],
    split: str,
    data_root: Optional[str],
    device: torch.device,
    max_samples: Optional[int],
    latency_steps: List[int],
    out_dir: Path,
    speed_bins: List[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset = TianmoucHSNDataset(cfg, split=split, data_root=data_root, mode="hsn")
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    image_hw = tuple(cfg["data"].get("padded_size", [232, 296]))

    cop_records: List[Dict[str, Any]] = []
    stream_records_by_latency: Dict[int, List[Dict[str, Any]]] = {l: [] for l in latency_steps}

    total_time = 0.0
    total_intervals = 0
    total_aop_predictions = 0

    for i in range(n):
        sample = dataset[i]
        template = sample["template"].unsqueeze(0).to(device, non_blocking=True)
        ref = sample["ref"].unsqueeze(0).to(device, non_blocking=True)
        target = sample["target"].unsqueeze(0).to(device, non_blocking=True)
        aop = sample["aop"].unsqueeze(0).to(device, non_blocking=True)
        template_box = sample["template_box"].unsqueeze(0).to(device, non_blocking=True)
        target_box = to_np(sample["target_box"])
        target_boxes_seq = to_np(sample["target_boxes_seq"]).astype(np.float32)
        aop_indices = to_np(sample["aop_frame_indices"]).astype(np.int64)

        gpu_sync_if_needed(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                mode="hsn_sequence",
                template=template,
                template_box=template_box,
                ref=ref,
                aop=aop,
                target=target,
            )
            boxes_seq, scores_seq = model.decode_sequence(out["cls_seq"], out["reg_seq"], image_hw=image_hw)
        gpu_sync_if_needed(device)
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        total_intervals += 1

        boxes_seq_np = to_np(boxes_seq[0]).astype(np.float32)  # [K,4]
        scores_seq_np = to_np(scores_seq[0]).astype(np.float32)  # [K]
        K = boxes_seq_np.shape[0]
        total_aop_predictions += K

        # COP-rate HSN output: final AOP-step prediction for this COP interval.
        interval_speed = 0.0
        if len(target_boxes_seq) >= 2:
            interval_speed = float(np.linalg.norm(center_xy(target_boxes_seq[-1]) - center_xy(target_boxes_seq[0])) / max(len(target_boxes_seq) - 1, 1))
        add_record(
            cop_records,
            metric_scope="hsn_offline_cop_rate_final_step",
            sample=sample,
            pred_box=boxes_seq_np[-1],
            gt_box=target_box,
            score=float(scores_seq_np[-1]),
            aop_idx=int(aop_indices[-1]) if len(aop_indices) else None,
            latency_steps=0,
            speed_px_per_aop_step=interval_speed,
        )

        # AOP-rate streaming outputs. If latency=L, compare y_hat[k] with GT[k+L].
        for L in latency_steps:
            if K <= L:
                continue
            for k in range(K - L):
                gt_idx = k + L
                speed = gt_speed_for_index(target_boxes_seq, gt_idx)
                add_record(
                    stream_records_by_latency[L],
                    metric_scope=f"hsn_streaming_aop_rate_latency_{L}",
                    sample=sample,
                    pred_box=boxes_seq_np[k],
                    gt_box=target_boxes_seq[gt_idx],
                    score=float(scores_seq_np[k]),
                    aop_idx=int(aop_indices[k]) if k < len(aop_indices) else None,
                    latency_steps=L,
                    speed_px_per_aop_step=speed,
                )

        if (i + 1) % 100 == 0 or i == 0 or i + 1 == n:
            print(f"[hsn] {i + 1}/{n}, K={K}")

    summaries: Dict[str, Any] = {}
    cop_summary = summarize_records(cop_records, "hsn_offline_cop_rate_final_step", out_dir)
    cop_summary.update({
        "wall_time_s": total_time,
        "cop_intervals_per_second_fps": total_intervals / max(total_time, 1e-9),
        "avg_latency_ms_per_cop_interval": 1000.0 * total_time / max(total_intervals, 1),
        "aop_predictions_per_second_fps": total_aop_predictions / max(total_time, 1e-9),
        "avg_latency_ms_per_aop_update": 1000.0 * total_time / max(total_aop_predictions, 1),
        "total_aop_predictions": int(total_aop_predictions),
    })
    summaries["hsn_offline_cop_rate_final_step"] = cop_summary

    all_records = list(cop_records)
    for L, records in stream_records_by_latency.items():
        name = f"hsn_streaming_aop_rate_latency_{L}"
        summary = summarize_records(records, name, out_dir)
        summary.update({
            "wall_time_s_shared_with_hsn_eval": total_time,
            "aop_predictions_per_second_fps_shared": total_aop_predictions / max(total_time, 1e-9),
        })
        summaries[name] = summary
        all_records.extend(records)
        write_speed_bin_csv(records, out_dir / f"{name}_speed_bins.csv", speed_bins)

    write_speed_bin_csv(cop_records, out_dir / "hsn_offline_cop_rate_final_step_speed_bins.csv", speed_bins)
    return summaries, all_records


def write_speed_bin_csv(records: List[Dict[str, Any]], path: Path, bins: List[float]) -> None:
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        subset = []
        for r in records:
            s = r.get("speed_px_per_aop_step", "")
            if s == "" or s is None:
                continue
            s = float(s)
            if s >= lo and s < hi:
                subset.append(r)
        ious = np.asarray([r["iou"] for r in subset], dtype=np.float64)
        errors = np.asarray([r["center_error"] for r in subset], dtype=np.float64)
        rows.append({
            "speed_bin_lo_px_per_aop_step": lo,
            "speed_bin_hi_px_per_aop_step": hi,
            "num_predictions": len(subset),
            "mean_iou_mIoU": safe_mean(ious),
            "success_rate_iou_0p5": float((ious >= 0.5).mean()) if len(subset) else None,
            "mean_center_error_px": safe_mean(errors),
            "precision_20px": float((errors <= 20.0).mean()) if len(subset) else None,
        })

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parts = {}
    for name in ["ann", "snn", "feature_hu", "head"]:
        if hasattr(model, name):
            m = getattr(model, name)
            parts[name] = int(sum(p.numel() for p in m.parameters()))
    return {"total": int(total), "trainable_current_mode": int(trainable), "parts": parts}


def main() -> None:
    parser = argparse.ArgumentParser(
        "Evaluate HSN/ANN tracking with paper-aligned metrics: mIoU, success plot/AUC, streaming accuracy, speed bins, FPS."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--mode", default="hsn", choices=["ann", "hsn", "both"])
    parser.add_argument("--out-dir", default="runs/eval_paper_metrics")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--latency-steps", default="0,1,2", help="Comma-separated AOP-step latency values for streaming accuracy.")
    parser.add_argument("--speed-bins", default="0,1,2,5,10,20,inf", help="Speed bin edges in px/AOP-step.")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TianmoucHSN(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    latency_steps = parse_latency_steps(args.latency_steps)
    speed_bins = parse_speed_bins(args.speed_bins)

    summary: Dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "data_root": args.data_root or cfg.get("data", {}).get("root", "./data/nfs"),
        "split": args.split,
        "mode": args.mode,
        "device": str(device),
        "max_samples": args.max_samples,
        "parameter_count": count_parameters(model),
        "notes": {
            "energy_per_inference_uJ": "not measured by this script; requires hardware power measurement or a calibrated hardware model.",
            "mac_or_operation_count": "not measured here; use a profiler/FLOP counter if needed.",
            "streaming_accuracy_definition": "For latency L, compare prediction at AOP step k with ground truth at AOP step k+L.",
        },
        "metrics": {},
    }
    all_records: List[Dict[str, Any]] = []

    if args.mode in ("ann", "both"):
        ann_summary, ann_records = eval_ann(model, cfg, args.split, args.data_root, device, args.max_samples, out_dir)
        summary["metrics"]["ann_offline_cop_rate"] = ann_summary
        all_records.extend(ann_records)

    if args.mode in ("hsn", "both"):
        hsn_summaries, hsn_records = eval_hsn(model, cfg, args.split, args.data_root, device, args.max_samples, latency_steps, out_dir, speed_bins)
        summary["metrics"].update(hsn_summaries)
        all_records.extend(hsn_records)

    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.save_predictions:
        write_predictions_csv(out_dir / "predictions.csv", all_records)

    print("\n[done]")
    print(f"summary: {summary_path}")
    if args.save_predictions:
        print(f"predictions: {out_dir / 'predictions.csv'}")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
