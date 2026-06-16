from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import yaml
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsn.model import TianmoucHSN
from hsn.data import (
    _to_hwc_cop,
    _to_td_t_hw,
    _to_sd_t_hw2,
    _to_boxes_m_t_4,
    _resize_pad_img_chw,
    _resize_pad_map,
    _scale_box_to_padded,
    _valid_box,
)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[checkpoint] loaded: {path}")
    print(f"[checkpoint] missing keys: {len(missing)}")
    print(f"[checkpoint] unexpected keys: {len(unexpected)}")
    if missing:
        print("[checkpoint] missing examples:", missing[:10])
    if unexpected:
        print("[checkpoint] unexpected examples:", unexpected[:10])


def find_npz(data_root: Path, split_dir: str, seq: Optional[str]) -> Path:
    root = data_root / split_dir
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {root}")

    if seq is None:
        print(f"[seq] no --seq specified, using first file: {files[0].stem}")
        return files[0]

    exact = root / f"{seq}.npz"
    if exact.exists():
        return exact

    matches = [p for p in files if seq in p.stem]
    if matches:
        print(f"[seq] exact not found, using matched file: {matches[0].stem}")
        return matches[0]

    raise FileNotFoundError(f"Cannot find seq={seq} in {root}")


def tensor_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)

    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    vals = x[finite]
    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)

    if hi - lo < eps:
        lo = vals.min()
        hi = vals.max()

    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def heatmap_rgb(x: np.ndarray, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    x01 = normalize01(x)
    gray = (x01 * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def td_rgb(td_map: np.ndarray) -> np.ndarray:
    """
    TD 可视化：
    - 如果 TD 有正负值：正值偏红，负值偏蓝。
    - 如果 TD 全非负：使用热力图。
    """
    td_map = td_map.astype(np.float32)

    if td_map.min() >= 0:
        return heatmap_rgb(td_map)

    pos = np.clip(td_map, 0, None)
    neg = np.clip(-td_map, 0, None)
    mag = np.abs(td_map)

    pos01 = normalize01(pos)
    neg01 = normalize01(neg)
    mag01 = normalize01(mag)

    rgb = np.zeros((*td_map.shape, 3), dtype=np.float32)
    rgb[..., 0] = pos01
    rgb[..., 1] = 0.35 * mag01
    rgb[..., 2] = neg01

    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def sd_rgb(sd0: np.ndarray, sd1: np.ndarray) -> np.ndarray:
    mag = np.sqrt(sd0.astype(np.float32) ** 2 + sd1.astype(np.float32) ** 2)
    return heatmap_rgb(mag)


def resize_pad_map_np(
    x: np.ndarray,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> np.ndarray:
    return _resize_pad_map(x, content_size, padded_size).astype(np.float32)


def make_aop_tensor(
    td: np.ndarray,
    sd: np.ndarray,
    a0: int,
    a1: int,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Return [K, 3, H, W]
    channel = TD, SD0, SD1
    """
    frames = []
    for k in range(a0, a1):
        td_k = resize_pad_map_np(td[k], content_size, padded_size)
        sd0 = resize_pad_map_np(sd[k, :, :, 0], content_size, padded_size)
        sd1 = resize_pad_map_np(sd[k, :, :, 1], content_size, padded_size)
        frames.append(np.stack([td_k, sd0, sd1], axis=0))

    return torch.from_numpy(np.stack(frames, axis=0)).float()


def make_aop_visual_frame(
    td: np.ndarray,
    sd: np.ndarray,
    aop_idx: int,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
    view: str,
) -> np.ndarray:
    td_map = resize_pad_map_np(td[aop_idx], content_size, padded_size)
    sd0 = resize_pad_map_np(sd[aop_idx, :, :, 0], content_size, padded_size)
    sd1 = resize_pad_map_np(sd[aop_idx, :, :, 1], content_size, padded_size)

    td_img = td_rgb(td_map)
    sd_img = sd_rgb(sd0, sd1)

    if view == "td":
        return td_img
    if view == "sd":
        return sd_img
    if view == "tdsd":
        return np.concatenate([td_img, sd_img], axis=1)

    raise ValueError(f"unknown view: {view}")


def draw_box(img: np.ndarray, box: np.ndarray, color: Tuple[int, int, int], label: str) -> None:
    """
    img: RGB
    color: RGB
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(float).tolist()

    x1 = int(np.clip(round(x1), 0, w - 1))
    y1 = int(np.clip(round(y1), 0, h - 1))
    x2 = int(np.clip(round(x2), 0, w - 1))
    y2 = int(np.clip(round(y2), 0, h - 1))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if label:
        cv2.putText(
            img,
            label,
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def put_text(img: np.ndarray, text: str, y: int) -> None:
    cv2.putText(
        img,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def scale_box_np(
    box: np.ndarray,
    raw_cop_size: Tuple[int, int],
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> np.ndarray:
    return (
        _scale_box_to_padded(box, raw_cop_size, content_size, padded_size)
        .detach()
        .cpu()
        .numpy()
    )


def first_valid_template(boxes: np.ndarray, obj_id: int, min_box_size: float) -> int:
    for t in range(boxes.shape[1]):
        if _valid_box(boxes[obj_id, t], min_box_size):
            return t
    raise RuntimeError(f"No valid template for obj_id={obj_id}")


@torch.no_grad()
def forward_interval(
    model: TianmoucHSN,
    template: torch.Tensor,
    template_box: torch.Tensor,
    ref: torch.Tensor,
    aop: torch.Tensor,
    target: torch.Tensor,
    image_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    输入一个 COP interval 的 AOP window。
    返回每个 AOP step 的预测：
      boxes_seq: [K, 4]
      scores_seq: [K]
    """
    out = model(
        mode="hsn_sequence",
        template=template.unsqueeze(0).to(device),
        template_box=template_box.unsqueeze(0).to(device),
        ref=ref.unsqueeze(0).to(device),
        aop=aop.unsqueeze(0).to(device),
        target=target.unsqueeze(0).to(device),
    )

    boxes_seq, scores_seq = model.decode_sequence(
        out["cls_seq"],
        out["reg_seq"],
        image_hw=image_hw,
    )

    return (
        boxes_seq[0].detach().cpu().numpy(),
        scores_seq[0].detach().cpu().numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/hsn_tianmouc.yaml")
    parser.add_argument("--checkpoint", default="runs/tianmouc_hsn/hsn_best.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--seq", default=None)
    parser.add_argument("--obj-id", type=int, default=0)

    parser.add_argument("--out-rgb", default="runs/vis/tracking_rgb_cop_rate.mp4")
    parser.add_argument("--out-aop", default="runs/vis/tracking_aop_rate.mp4")

    parser.add_argument("--rgb-fps", type=float, default=10.0)
    parser.add_argument("--aop-fps", type=float, default=100.0)

    parser.add_argument("--aop-view", default="tdsd", choices=["td", "sd", "tdsd"])

    # 注意：默认使用完整序列，不人为指定帧数。
    # 这两个参数只用于调试长视频时截取片段。
    parser.add_argument("--start-cop", type=int, default=None)
    parser.add_argument("--end-cop", type=int, default=None)

    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    data_root = Path(args.data_root or data_cfg.get("root", "./data/nfs"))
    split_dir = data_cfg.get("train_split", "train") if args.split == "train" else data_cfg.get("val_split", "val")

    raw_cop_size = tuple(data_cfg.get("raw_cop_size", [320, 640]))
    content_size = tuple(data_cfg.get("content_size", [128, 256]))
    padded_size = tuple(data_cfg.get("padded_size", [232, 296]))
    min_box_size = float(data_cfg.get("min_box_size", 2))

    ph, pw = padded_size
    image_hw = (ph, pw)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    npz_path = find_npz(data_root, split_dir, args.seq)
    print(f"[data] loading: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)

    cop = _to_hwc_cop(d["cop"])
    td = _to_td_t_hw(d["td"])
    sd = _to_sd_t_hw2(d["sd"])
    boxes = _to_boxes_m_t_4(d["boxes"]).astype(np.float32)
    boxes_all = _to_boxes_m_t_4(d["boxes_all"]).astype(np.float32)
    cop_indices = d["cop_indices"].astype(np.int64)

    obj_id = args.obj_id
    if obj_id < 0 or obj_id >= boxes.shape[0]:
        raise ValueError(f"obj_id={obj_id} out of range, num_objects={boxes.shape[0]}")

    model = TianmoucHSN(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    template_t = first_valid_template(boxes, obj_id, min_box_size)

    template = _resize_pad_img_chw(cop[template_t], content_size, padded_size)
    template_box = _scale_box_to_padded(
        boxes[obj_id, template_t],
        raw_cop_size,
        content_size,
        padded_size,
    )

    start_cop = template_t if args.start_cop is None else max(template_t, args.start_cop)
    end_cop = len(cop) - 1 if args.end_cop is None else min(args.end_cop, len(cop) - 1)

    if end_cop <= start_cop:
        raise ValueError(f"bad cop range: start={start_cop}, end={end_cop}")

    out_rgb = Path(args.out_rgb)
    out_aop = Path(args.out_aop)
    out_rgb.parent.mkdir(parents=True, exist_ok=True)
    out_aop.parent.mkdir(parents=True, exist_ok=True)

    rgb_writer = cv2.VideoWriter(
        str(out_rgb),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.rgb_fps,
        (pw, ph),
    )

    if args.aop_view == "tdsd":
        aop_video_size = (pw * 2, ph)
    else:
        aop_video_size = (pw, ph)

    aop_writer = cv2.VideoWriter(
        str(out_aop),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.aop_fps,
        aop_video_size,
    )

    if not rgb_writer.isOpened():
        raise RuntimeError(f"failed to open RGB video writer: {out_rgb}")
    if not aop_writer.isOpened():
        raise RuntimeError(f"failed to open AOP video writer: {out_aop}")

    rgb_count = 0
    aop_count = 0

    print("=" * 80)
    print(f"[video] seq={npz_path.stem}, obj={obj_id}")
    print(f"[video] COP frames: {len(cop)}")
    print(f"[video] AOP frames: {len(td)}")
    print(f"[video] using COP interval [{start_cop}, {end_cop}]")
    print(f"[video] out_rgb={out_rgb}")
    print(f"[video] out_aop={out_aop}")
    print("=" * 80)

    for ref_t in range(start_cop, end_cop):
        target_t = ref_t + 1

        if not _valid_box(boxes[obj_id, ref_t], min_box_size):
            continue
        if not _valid_box(boxes[obj_id, target_t], min_box_size):
            continue

        a0 = int(cop_indices[ref_t]) + 1
        a1 = int(cop_indices[target_t]) + 1

        if a1 <= a0:
            continue
        if a1 > len(td) or a1 > len(sd) or a1 > boxes_all.shape[1]:
            continue

        # 检查这个 AOP window 的每个高频 GT 都有效
        high_boxes = boxes_all[obj_id, a0:a1]
        if not all(_valid_box(b, min_box_size) for b in high_boxes):
            continue

        ref = _resize_pad_img_chw(cop[ref_t], content_size, padded_size)
        target = _resize_pad_img_chw(cop[target_t], content_size, padded_size)
        aop = make_aop_tensor(td, sd, a0, a1, content_size, padded_size)

        boxes_seq, scores_seq = forward_interval(
            model=model,
            template=template,
            template_box=template_box,
            ref=ref,
            aop=aop,
            target=target,
            image_hw=image_hw,
            device=device,
        )

        # 低频 RGB/COP 视频：每个 COP target_t 输出一帧，用 interval 最后一个 AOP step 的预测
        rgb_img = tensor_to_rgb_uint8(target)

        rgb_gt = scale_box_np(
            boxes[obj_id, target_t],
            raw_cop_size,
            content_size,
            padded_size,
        )
        rgb_pred = boxes_seq[-1]
        rgb_score = float(scores_seq[-1])

        draw_box(rgb_img, rgb_gt, (0, 255, 0), "GT")
        draw_box(rgb_img, rgb_pred, (255, 0, 0), f"Pred {rgb_score:.3f}")
        put_text(rgb_img, f"RGB/COP rate | seq={npz_path.stem} obj={obj_id}", 22)
        put_text(rgb_img, f"template_t={template_t} ref_t={ref_t} target_t={target_t}", 44)
        put_text(rgb_img, f"AOP window=[{a0},{a1}) K={a1-a0}", 66)

        rgb_writer.write(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        rgb_count += 1

        # 高频 AOP 视频：每一个 TD/SD step 都输出一帧
        for local_k, aop_idx in enumerate(range(a0, a1)):
            aop_img = make_aop_visual_frame(
                td=td,
                sd=sd,
                aop_idx=aop_idx,
                content_size=content_size,
                padded_size=padded_size,
                view=args.aop_view,
            )

            gt_aop = scale_box_np(
                boxes_all[obj_id, aop_idx],
                raw_cop_size,
                content_size,
                padded_size,
            )
            pred_aop = boxes_seq[local_k]
            score_aop = float(scores_seq[local_k])

            # 如果是 TD|SD 双图，需要把框画到两个 panel 上
            if args.aop_view == "tdsd":
                left = aop_img[:, :pw].copy()
                right = aop_img[:, pw:].copy()

                for panel, name in [(left, "TD"), (right, "SD")]:
                    draw_box(panel, gt_aop, (0, 255, 0), "GT")
                    draw_box(panel, pred_aop, (255, 0, 0), f"Pred {score_aop:.3f}")
                    put_text(panel, name, 22)
                    put_text(panel, f"aop_idx={aop_idx}", 44)
                    put_text(panel, f"cop {ref_t}->{target_t} k={local_k+1}/{a1-a0}", 66)

                aop_img = np.concatenate([left, right], axis=1)
            else:
                draw_box(aop_img, gt_aop, (0, 255, 0), "GT")
                draw_box(aop_img, pred_aop, (255, 0, 0), f"Pred {score_aop:.3f}")
                put_text(aop_img, f"{args.aop_view.upper()} AOP rate", 22)
                put_text(aop_img, f"aop_idx={aop_idx}", 44)
                put_text(aop_img, f"cop {ref_t}->{target_t} k={local_k+1}/{a1-a0}", 66)

            aop_writer.write(cv2.cvtColor(aop_img, cv2.COLOR_RGB2BGR))
            aop_count += 1

        if rgb_count % 20 == 0:
            print(f"[progress] rgb_frames={rgb_count}, aop_frames={aop_count}")

    rgb_writer.release()
    aop_writer.release()

    print("=" * 80)
    print(f"[done] RGB/COP video saved: {out_rgb}")
    print(f"[done] AOP TD/SD video saved: {out_aop}")
    print(f"[done] RGB video frames = {rgb_count}")
    print(f"[done] AOP video frames = {aop_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()