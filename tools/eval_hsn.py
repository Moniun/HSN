from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch
from tqdm import tqdm

from common import add_common_args, prepare, make_loader, load_checkpoint
from hsn.model import TianmoucHSN
from hsn.utils import batch_to_device, box_iou, center_error


def _cat_or_empty(xs: List[torch.Tensor]) -> torch.Tensor:
    if not xs:
        return torch.empty(0)
    return torch.cat(xs, dim=0)


def _success_auc(ious: torch.Tensor) -> float:
    if ious.numel() == 0:
        return 0.0
    thresholds = torch.linspace(0, 1, 21, device=ious.device)
    success = torch.stack([(ious >= t).float().mean() for t in thresholds]).mean()
    return float(success.detach().cpu())


def _basic_metrics(
    ious: torch.Tensor,
    center_errors: torch.Tensor,
    scores: torch.Tensor | None = None,
    prefix: str = "",
) -> Dict[str, float]:
    if ious.numel() == 0:
        return {
            f"{prefix}mean_iou": 0.0,
            f"{prefix}success_auc": 0.0,
            f"{prefix}precision_20px": 0.0,
            f"{prefix}mean_center_error": 0.0,
            f"{prefix}mean_score": 0.0,
        }

    out = {
        f"{prefix}mean_iou": float(ious.mean().detach().cpu()),
        f"{prefix}success_auc": _success_auc(ious),
        f"{prefix}precision_20px": float((center_errors <= 20).float().mean().detach().cpu()),
        f"{prefix}mean_center_error": float(center_errors.mean().detach().cpu()),
    }

    if scores is not None and scores.numel() > 0:
        out[f"{prefix}mean_score"] = float(scores.mean().detach().cpu())
    else:
        out[f"{prefix}mean_score"] = 0.0

    return out


def _center(boxes: torch.Tensor) -> torch.Tensor:
    cx = (boxes[..., 0] + boxes[..., 2]) * 0.5
    cy = (boxes[..., 1] + boxes[..., 3]) * 0.5
    return torch.stack([cx, cy], dim=-1)


def _speed_from_gt_seq(gt_seq: torch.Tensor) -> torch.Tensor:
    """
    gt_seq: [B, K, 4]
    return: [B, K], first speed is copied from second.
    """
    c = _center(gt_seq)
    diff = c[:, 1:] - c[:, :-1]
    speed = torch.norm(diff, dim=-1)

    if speed.shape[1] == 0:
        return torch.zeros(gt_seq.shape[:2], device=gt_seq.device)

    first = speed[:, :1]
    speed = torch.cat([first, speed], dim=1)
    return speed


def evaluate_offline(model, loader, device, image_hw):
    """
    高频离线评估：
    pred[k] 对比 target_boxes_seq[k]
    """
    model.eval()

    ious_all = []
    ces_all = []
    scores_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval_offline_hf", ncols=120):
            batch = batch_to_device(batch, device)

            out = model.forward_hsn_sequence(
                template=batch["template"],
                template_box=batch["template_box"],
                ref=batch["ref"],
                aop=batch["aop"],
                target=None,
            )

            pred_boxes_seq, scores_seq = model.decode_sequence(
                out["cls_seq"],
                out["reg_seq"],
                image_hw=image_hw,
            )

            gt_boxes_seq = batch["target_boxes_seq"]

            B, K, _ = gt_boxes_seq.shape

            pred_flat = pred_boxes_seq.reshape(B * K, 4)
            gt_flat = gt_boxes_seq.reshape(B * K, 4)
            scores_flat = scores_seq.reshape(B * K)

            iou = box_iou(pred_flat, gt_flat).diag()
            ce = center_error(pred_flat, gt_flat)

            ious_all.append(iou.detach().cpu())
            ces_all.append(ce.detach().cpu())
            scores_all.append(scores_flat.detach().cpu())

    ious = _cat_or_empty(ious_all)
    ces = _cat_or_empty(ces_all)
    scores = _cat_or_empty(scores_all)

    return _basic_metrics(ious, ces, scores, prefix="offline_")


def evaluate_streaming(model, loader, device, image_hw, latency_steps: int):
    """
    Streaming latency-aware evaluation.

    pred[k] 对比 gt[k + latency_steps]。

    当前版本只在一个 COP window 内比较。如果 k + latency_steps 超出当前 window，
    就跳过。更严格的跨 window streaming 可以后续再做全序列状态缓存。
    """
    model.eval()

    ious_all = []
    ces_all = []
    scores_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval_streaming_lag{latency_steps}", ncols=120):
            batch = batch_to_device(batch, device)

            out = model.forward_hsn_sequence(
                template=batch["template"],
                template_box=batch["template_box"],
                ref=batch["ref"],
                aop=batch["aop"],
                target=None,
            )

            pred_boxes_seq, scores_seq = model.decode_sequence(
                out["cls_seq"],
                out["reg_seq"],
                image_hw=image_hw,
            )

            gt_boxes_seq = batch["target_boxes_seq"]

            B, K, _ = gt_boxes_seq.shape

            if latency_steps <= 0:
                pred_use = pred_boxes_seq
                gt_use = gt_boxes_seq
                score_use = scores_seq
            else:
                if latency_steps >= K:
                    continue
                pred_use = pred_boxes_seq[:, :-latency_steps]
                gt_use = gt_boxes_seq[:, latency_steps:]
                score_use = scores_seq[:, :-latency_steps]

            pred_flat = pred_use.reshape(-1, 4)
            gt_flat = gt_use.reshape(-1, 4)
            score_flat = score_use.reshape(-1)

            iou = box_iou(pred_flat, gt_flat).diag()
            ce = center_error(pred_flat, gt_flat)

            ious_all.append(iou.detach().cpu())
            ces_all.append(ce.detach().cpu())
            scores_all.append(score_flat.detach().cpu())

    ious = _cat_or_empty(ious_all)
    ces = _cat_or_empty(ces_all)
    scores = _cat_or_empty(scores_all)

    return _basic_metrics(
        ious,
        ces,
        scores,
        prefix=f"stream_lag{latency_steps}_",
    )


def evaluate_speed_bins(model, loader, device, image_hw, speed_bins):
    """
    按 GT 运动速度分组评估。

    speed 单位：padded 输入坐标系中的 px / AOP-step。
    """
    model.eval()

    bin_stats = {
        i: {"ious": [], "ces": [], "scores": []}
        for i in range(len(speed_bins) - 1)
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval_speed_bins", ncols=120):
            batch = batch_to_device(batch, device)

            out = model.forward_hsn_sequence(
                template=batch["template"],
                template_box=batch["template_box"],
                ref=batch["ref"],
                aop=batch["aop"],
                target=None,
            )

            pred_boxes_seq, scores_seq = model.decode_sequence(
                out["cls_seq"],
                out["reg_seq"],
                image_hw=image_hw,
            )

            gt_boxes_seq = batch["target_boxes_seq"]

            B, K, _ = gt_boxes_seq.shape

            pred_flat = pred_boxes_seq.reshape(B * K, 4)
            gt_flat = gt_boxes_seq.reshape(B * K, 4)
            scores_flat = scores_seq.reshape(B * K)

            ious = box_iou(pred_flat, gt_flat).diag()
            ces = center_error(pred_flat, gt_flat)

            speeds = _speed_from_gt_seq(gt_boxes_seq).reshape(B * K)

            for i in range(len(speed_bins) - 1):
                lo = float(speed_bins[i])
                hi = float(speed_bins[i + 1])
                mask = (speeds >= lo) & (speeds < hi)

                if mask.any():
                    bin_stats[i]["ious"].append(ious[mask].detach().cpu())
                    bin_stats[i]["ces"].append(ces[mask].detach().cpu())
                    bin_stats[i]["scores"].append(scores_flat[mask].detach().cpu())

    results = {}

    for i in range(len(speed_bins) - 1):
        lo = speed_bins[i]
        hi = speed_bins[i + 1]

        ious = _cat_or_empty(bin_stats[i]["ious"])
        ces = _cat_or_empty(bin_stats[i]["ces"])
        scores = _cat_or_empty(bin_stats[i]["scores"])

        prefix = f"speed_{lo}_{hi}_"
        results.update(_basic_metrics(ious, ces, scores, prefix=prefix))
        results[f"{prefix}num_samples"] = int(ious.numel())

    return results


def profile_runtime(model, loader, device, image_hw, warmup: int = 10, iters: int = 50):
    """
    统计 AutoDL 上的推理速度。
    不能等价于原文 Tianjic 芯片能耗，只作为 GPU/CPU runtime reference。
    """
    model.eval()

    times = []
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()

            out = model.forward_hsn_sequence(
                template=batch["template"],
                template_box=batch["template_box"],
                ref=batch["ref"],
                aop=batch["aop"],
                target=None,
            )
            _ = model.decode_sequence(out["cls_seq"], out["reg_seq"], image_hw=image_hw)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t1 = time.time()

            if count >= warmup:
                times.append(t1 - t0)

            count += 1

            if len(times) >= iters:
                break

    if not times:
        return {
            "runtime_avg_latency_ms_per_batch": 0.0,
            "runtime_fps_windows": 0.0,
            "runtime_note": "not enough iterations",
        }

    avg = sum(times) / len(times)

    return {
        "runtime_avg_latency_ms_per_batch": avg * 1000.0,
        "runtime_windows_per_second": 1.0 / avg,
        "runtime_energy_note": "Tianjic hardware energy is unavailable on AutoDL; report GPU/CPU latency only.",
    }


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode",
        default="all",
        choices=["offline", "streaming", "speed", "profile", "all"],
    )
    parser.add_argument(
        "--latency-steps",
        type=int,
        default=1,
        help="Streaming: compare pred[k] with gt[k+latency_steps].",
    )
    parser.add_argument(
        "--speed-bins",
        type=float,
        nargs="+",
        default=[0, 2, 5, 10, 20, 1e9],
    )
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=50)

    args = parser.parse_args()

    cfg, device = prepare(args)

    model = TianmoucHSN(cfg).to(device)
    load_checkpoint(model, args.checkpoint, strict=True, map_location=device)
    model.eval()

    batch_size = cfg["train"].get("hsn_batch_size", 1)
    loader = make_loader(
        cfg,
        cfg["data"]["val_split"],
        batch_size,
        False,
    )

    image_hw = tuple(cfg["data"]["padded_size"])

    results = {}

    if args.mode in ["offline", "all"]:
        results.update(evaluate_offline(model, loader, device, image_hw))

    if args.mode in ["streaming", "all"]:
        results.update(evaluate_streaming(model, loader, device, image_hw, args.latency_steps))

    if args.mode in ["speed", "all"]:
        results.update(evaluate_speed_bins(model, loader, device, image_hw, args.speed_bins))

    if args.mode in ["profile", "all"]:
        results.update(
            profile_runtime(
                model,
                loader,
                device,
                image_hw,
                warmup=args.profile_warmup,
                iters=args.profile_iters,
            )
        )

    print(results)


if __name__ == "__main__":
    main()