from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsn.model import TianmoucHSN  # noqa: E402


def load_yaml(path: str | Path) -> Dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model_checkpoint(model: torch.nn.Module, checkpoint: str | Path, device: torch.device) -> None:
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

    if len(unexpected) > 0:
        print("[checkpoint] unexpected examples:", unexpected[:10])
    if len(missing) > 0:
        print("[checkpoint] missing examples:", missing[:10])


def to_hwc_cop(cop: np.ndarray) -> np.ndarray:
    if cop.ndim == 4 and cop.shape[-1] in (1, 3):
        return cop
    if cop.ndim == 4 and cop.shape[1] in (1, 3):
        return np.transpose(cop, (0, 2, 3, 1))
    raise ValueError(f"Unsupported cop shape: {cop.shape}")


def to_td_t_hw(td: np.ndarray) -> np.ndarray:
    if td.ndim == 3:
        return td
    if td.ndim == 4 and td.shape[1] == 1:
        return td[:, 0]
    if td.ndim == 4 and td.shape[-1] == 1:
        return td[..., 0]
    raise ValueError(f"Unsupported td shape: {td.shape}")


def to_sd_t_hw2(sd: np.ndarray) -> np.ndarray:
    if sd.ndim == 4 and sd.shape[-1] == 2:
        return sd
    if sd.ndim == 4 and sd.shape[1] == 2:
        return np.transpose(sd, (0, 2, 3, 1))
    raise ValueError(f"Unsupported sd shape: {sd.shape}")


def to_boxes_m_t_4(boxes: np.ndarray) -> np.ndarray:
    if boxes.ndim == 2 and boxes.shape[-1] == 4:
        return boxes[None, ...]
    if boxes.ndim == 3 and boxes.shape[-1] == 4:
        return boxes
    raise ValueError(f"Unsupported boxes shape: {boxes.shape}")


def scale_box_to_padded_np(
    box: np.ndarray,
    raw_size: Tuple[int, int],
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> np.ndarray:
    raw_h, raw_w = raw_size
    ch, cw = content_size
    ph, pw = padded_size

    sx = cw / float(raw_w)
    sy = ch / float(raw_h)

    top = (ph - ch) // 2
    left = (pw - cw) // 2

    b = box.astype(np.float32).copy()
    b[[0, 2]] = b[[0, 2]] * sx + left
    b[[1, 3]] = b[[1, 3]] * sy + top

    b[[0, 2]] = np.clip(b[[0, 2]], 0, pw - 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, ph - 1)

    return b


def scale_box_for_output(box: np.ndarray, scale: int, x_offset: int = 0) -> np.ndarray:
    b = box.astype(np.float32).copy()
    b *= float(scale)
    b[[0, 2]] += float(x_offset)
    return b


def image_to_tensor(
    img_hwc: np.ndarray,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> torch.Tensor:
    ch, cw = content_size
    ph, pw = padded_size

    img = cv2.resize(img_hwc, (cw, ch), interpolation=cv2.INTER_AREA)

    if img.ndim == 2:
        img = img[..., None]

    img = img.astype(np.float32)
    if img.max() > 2.0:
        img = img / 255.0

    canvas = np.zeros((ph, pw, img.shape[-1]), dtype=np.float32)
    top = (ph - ch) // 2
    left = (pw - cw) // 2
    canvas[top:top + ch, left:left + cw] = img

    chw = np.transpose(canvas, (2, 0, 1))
    return torch.from_numpy(chw).float()


def image_to_vis_uint8(
    img_hwc: np.ndarray,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> np.ndarray:
    ch, cw = content_size
    ph, pw = padded_size

    img = cv2.resize(img_hwc, (cw, ch), interpolation=cv2.INTER_AREA)

    if img.ndim == 2:
        img = img[..., None]

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = img.astype(np.float32)
    if img.max() <= 2.0:
        img = img * 255.0

    img = np.clip(img, 0, 255).astype(np.uint8)

    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    top = (ph - ch) // 2
    left = (pw - cw) // 2
    canvas[top:top + ch, left:left + cw] = img[..., :3]

    return canvas


def map_to_padded_float(
    m: np.ndarray,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> np.ndarray:
    ch, cw = content_size
    ph, pw = padded_size

    m = cv2.resize(m.astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((ph, pw), dtype=np.float32)
    top = (ph - ch) // 2
    left = (pw - cw) // 2
    canvas[top:top + ch, left:left + cw] = m
    return canvas


def robust_norm(x: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    m = np.percentile(np.abs(x), percentile)
    if m < 1e-6:
        return np.zeros_like(x, dtype=np.float32)

    y = x / m
    y = np.clip(y, -1.0, 1.0)
    return y


def aop_to_tensor(
    td_win: np.ndarray,
    sd_win: np.ndarray,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> torch.Tensor:
    frames = []

    for k in range(len(td_win)):
        td_k = map_to_padded_float(td_win[k], content_size, padded_size)
        sd0 = map_to_padded_float(sd_win[k, :, :, 0], content_size, padded_size)
        sd1 = map_to_padded_float(sd_win[k, :, :, 1], content_size, padded_size)
        frames.append(np.stack([td_k, sd0, sd1], axis=0))

    return torch.from_numpy(np.stack(frames, axis=0)).float()


def aop_frame_to_vis(
    td_k: np.ndarray,
    sd_k: np.ndarray,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
    mode: str = "soft",
) -> np.ndarray:
    td = map_to_padded_float(td_k, content_size, padded_size)
    sd0 = map_to_padded_float(sd_k[:, :, 0], content_size, padded_size)
    sd1 = map_to_padded_float(sd_k[:, :, 1], content_size, padded_size)

    if mode == "gray":
        mag = np.abs(td) + 0.35 * np.sqrt(sd0 ** 2 + sd1 ** 2)
        mag = robust_norm(mag)
        gray = (np.clip(mag, 0.0, 1.0) * 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=-1)

    td_n = robust_norm(td)
    sd_mag = np.sqrt(sd0.astype(np.float32) ** 2 + sd1.astype(np.float32) ** 2)
    sd_n = robust_norm(sd_mag)

    r = np.clip(td_n, 0.0, 1.0)
    b = np.clip(-td_n, 0.0, 1.0)
    g = np.clip(np.abs(sd_n), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)

    if mode == "soft":
        gray = np.clip(np.abs(sd_n) * 0.8 + np.abs(td_n) * 0.2, 0.0, 1.0)
        rgb = 0.65 * np.repeat(gray[..., None], 3, axis=-1) + 0.35 * rgb

    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(np.float32).tolist()
    bx1, by1, bx2, by2 = b.astype(np.float32).tolist()

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0

    return float(inter / union)


def valid_pred_box(box: np.ndarray, image_hw: Tuple[int, int]) -> bool:
    h, w = image_hw
    if not np.isfinite(box).all():
        return False

    x1, y1, x2, y2 = box.astype(np.float32).tolist()
    bw = x2 - x1
    bh = y2 - y1

    if bw < 2 or bh < 2:
        return False
    if bw > w * 1.2 or bh > h * 1.2:
        return False
    if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
        return False

    return True


def smooth_box(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None or alpha <= 0:
        return cur
    return alpha * prev + (1.0 - alpha) * cur


def draw_box(
    img: np.ndarray,
    box: np.ndarray,
    color: Tuple[int, int, int],
    label: str = "",
    thickness: int = 2,
    font_scale: float = 0.45,
) -> None:
    x1, y1, x2, y2 = box.astype(np.int32).tolist()
    h, w = img.shape[:2]

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_thickness = max(1, thickness - 1)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        ty = max(0, y1 - th - baseline - 3)

        cv2.rectangle(
            img,
            (x1, ty),
            (min(w - 1, x1 + tw + 6), min(h - 1, ty + th + baseline + 5)),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1 + 3, ty + th + 1),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )


def draw_info_panel(
    img: np.ndarray,
    lines: list[str],
    font_scale: float = 0.42,
    thickness: int = 1,
    pos: str = "top_left",
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 8
    line_h = int(18 * max(font_scale / 0.42, 0.8))

    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, tw)

    panel_w = max_w + margin * 2
    panel_h = line_h * len(lines) + margin

    h, w = img.shape[:2]

    if pos == "bottom_left":
        x0, y0 = 0, h - panel_h
    elif pos == "top_right":
        x0, y0 = w - panel_w, 0
    else:
        x0, y0 = 0, 0

    x1, y1 = min(w - 1, x0 + panel_w), min(h - 1, y0 + panel_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x0 + margin, y0 + margin + 12 + i * line_h),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def resize_for_video(frame_rgb: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return frame_rgb
    h, w = frame_rgb.shape[:2]
    return cv2.resize(frame_rgb, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)


def add_inset(
    frame_rgb: np.ndarray,
    inset_rgb: np.ndarray,
    title: str = "TD/SD",
    width_ratio: float = 0.28,
) -> None:
    h, w = frame_rgb.shape[:2]

    inset_w = max(80, int(w * width_ratio))
    inset_h = int(inset_w * inset_rgb.shape[0] / max(1, inset_rgb.shape[1]))

    inset = cv2.resize(inset_rgb, (inset_w, inset_h), interpolation=cv2.INTER_AREA)

    pad = 10
    x0 = w - inset_w - pad
    y0 = h - inset_h - pad

    if x0 < 0 or y0 < 0:
        return

    roi = frame_rgb[y0:y0 + inset_h, x0:x0 + inset_w]

    frame_rgb[y0:y0 + inset_h, x0:x0 + inset_w] = cv2.addWeighted(
        roi,
        0.15,
        inset,
        0.85,
        0,
    )

    cv2.rectangle(
        frame_rgb,
        (x0, y0),
        (x0 + inset_w, y0 + inset_h),
        (255, 255, 255),
        1,
    )

    cv2.putText(
        frame_rgb,
        title,
        (x0 + 4, max(12, y0 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def find_npz(data_root: Path, split: str, seq: Optional[str]) -> Path:
    root = data_root / split

    if seq is None:
        files = sorted(root.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files under {root}")
        return files[0]

    seq_path = Path(seq)
    if seq_path.exists():
        return seq_path

    direct = root / seq
    if direct.exists():
        return direct

    if not seq.endswith(".npz"):
        direct = root / f"{seq}.npz"
        if direct.exists():
            return direct

    matches = sorted(root.glob(f"*{seq}*.npz"))
    if not matches:
        raise FileNotFoundError(f"Cannot find seq={seq} under {root}")

    return matches[0]


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)

    required = ["cop", "td", "sd", "boxes", "boxes_all", "cop_indices"]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")

    cop = to_hwc_cop(d["cop"])
    td = to_td_t_hw(d["td"])
    sd = to_sd_t_hw2(d["sd"])
    boxes = to_boxes_m_t_4(d["boxes"]).astype(np.float32)
    boxes_all = to_boxes_m_t_4(d["boxes_all"]).astype(np.float32)
    cop_indices = d["cop_indices"].astype(np.int64)

    if len(cop) != len(cop_indices):
        raise ValueError(f"len(cop)={len(cop)} != len(cop_indices)={len(cop_indices)}")
    if boxes.shape[1] != len(cop):
        raise ValueError(f"boxes length {boxes.shape[1]} != cop length {len(cop)}")
    if boxes_all.shape[1] != len(td):
        raise ValueError(f"boxes_all length {boxes_all.shape[1]} != td length {len(td)}")
    if len(td) != len(sd):
        raise ValueError(f"td length {len(td)} != sd length {len(sd)}")

    return cop, td, sd, boxes, boxes_all, cop_indices


def compute_real_fps(cop_indices: np.ndarray, aop_fps: float = 1000.0) -> float:
    if len(cop_indices) < 2:
        return 20.0
    avg_interval = np.mean(np.diff(cop_indices))
    if avg_interval <= 0:
        return 20.0
    return aop_fps / avg_interval


def previous_cop_id(cop_indices: np.ndarray, frame_idx: int) -> int:
    i = int(np.searchsorted(cop_indices, frame_idx, side="right") - 1)
    return int(np.clip(i, 0, len(cop_indices) - 1))


def forward_hsn_sequence_safe(
    model: torch.nn.Module,
    template: torch.Tensor,
    template_box: torch.Tensor,
    ref: torch.Tensor,
    aop: torch.Tensor,
):
    try:
        return model(
            mode="hsn_sequence",
            template=template,
            template_box=template_box,
            ref=ref,
            aop=aop,
            target=None,
        )
    except TypeError:
        return model.forward_hsn_sequence(
            template=template,
            template_box=template_box,
            ref=ref,
            aop=aop,
            target=None,
        )


def build_base_background(
    view: str,
    frame_idx: int,
    cop_id: int,
    cop: np.ndarray,
    td: np.ndarray,
    sd: np.ndarray,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
    cop_indices: np.ndarray,
    aop_color_mode: str,
) -> tuple[np.ndarray, str]:
    is_cop = int(cop_indices[cop_id]) == int(frame_idx)

    if view == "aop":
        if is_cop:
            return image_to_vis_uint8(cop[cop_id], content_size, padded_size), "COP/RGB"
        return aop_frame_to_vis(td[frame_idx], sd[frame_idx], content_size, padded_size, mode=aop_color_mode), "AOP TD/SD"

    if view == "rgb_hold":
        hold_id = previous_cop_id(cop_indices, frame_idx)
        label = "COP/RGB" if is_cop else f"AOP on RGB-hold(cop={hold_id})"
        return image_to_vis_uint8(cop[hold_id], content_size, padded_size), label

    if view == "side_by_side":
        hold_id = previous_cop_id(cop_indices, frame_idx)
        left = image_to_vis_uint8(cop[hold_id], content_size, padded_size)
        right = aop_frame_to_vis(td[frame_idx], sd[frame_idx], content_size, padded_size, mode=aop_color_mode)
        canvas = np.concatenate([left, right], axis=1)
        label = "RGB-hold + TD/SD"
        return canvas, label

    raise ValueError(f"Unsupported view: {view}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hsn_tianmouc.yaml")
    parser.add_argument("--data-root", default="data/nfs")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--seq", default=None, help="Sequence name, partial name, or .npz path.")
    parser.add_argument("--checkpoint", default="runs/tianmouc_hsn_reproduce/hsn_best.pt")
    parser.add_argument("--out", default="runs/vis/tracking.mp4")
    parser.add_argument("--obj-id", type=int, default=0)
    parser.add_argument("--template-cop", type=int, default=0)
    parser.add_argument("--start-cop", type=int, default=0)
    parser.add_argument("--end-cop", type=int, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-frames-dir", default=None)
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--view",
        default="rgb_hold",
        choices=["rgb_hold", "aop", "side_by_side", "cop_only"],
        help="rgb_hold is recommended for human viewing. aop is for debugging.",
    )
    parser.add_argument("--aop-color-mode", default="soft", choices=["soft", "color", "gray"])
    parser.add_argument("--show-aop-inset", action="store_true", default=True)
    parser.add_argument("--no-aop-inset", dest="show_aop_inset", action="store_false")
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--draw-low-score", action="store_true")
    parser.add_argument("--box-labels", action="store_true")
    parser.add_argument("--font-scale", type=float, default=0.42)
    parser.add_argument("--box-thickness", type=int, default=2)
    parser.add_argument("--info-mode", default="minimal", choices=["none", "minimal", "full"])
    parser.add_argument("--info-pos", default="top_left", choices=["top_left", "top_right", "bottom_left"])
    parser.add_argument("--smooth", type=float, default=0.0, help="0 disables smoothing; 0.7 gives strong smoothing.")

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    raw_cop_size = tuple(data_cfg.get("raw_cop_size", [320, 640]))
    content_size = tuple(data_cfg.get("content_size", [128, 256]))
    padded_size = tuple(data_cfg.get("padded_size", [232, 296]))

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    npz_path = find_npz(Path(args.data_root), args.split, args.seq)
    print(f"[data] npz: {npz_path}")

    cop, td, sd, boxes, boxes_all, cop_indices = load_npz(npz_path)

    if args.obj_id < 0 or args.obj_id >= boxes.shape[0]:
        raise ValueError(f"obj_id={args.obj_id} out of range, M={boxes.shape[0]}")

    model = TianmoucHSN(cfg).to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    model.eval()

    base_h, base_w = padded_size
    if args.view == "side_by_side":
        video_w = base_w * 2 * max(1, args.scale)
    else:
        video_w = base_w * max(1, args.scale)
    video_h = base_h * max(1, args.scale)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_save_dir = None
    if args.save_frames_dir is not None:
        frame_save_dir = Path(args.save_frames_dir)
        frame_save_dir.mkdir(parents=True, exist_ok=True)

    real_fps = compute_real_fps(cop_indices)
    fps = args.fps if args.fps > 0 else real_fps
    print(f"[video] Real COP fps={real_fps:.2f}, Using fps={fps:.2f}")
    print(f"[video] view={args.view}, score_thr={args.score_thr}, draw_low_score={args.draw_low_score}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (video_w, video_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    template_cop = int(args.template_cop)
    start_cop = int(args.start_cop)

    if template_cop < 0 or template_cop >= len(cop):
        raise ValueError(f"template_cop={template_cop} out of range, len(cop)={len(cop)}")

    if start_cop < 0 or start_cop >= len(cop) - 1:
        raise ValueError(f"start_cop={start_cop} out of range, len(cop)={len(cop)}")

    if args.end_cop is None:
        end_cop = len(cop) - 1
    else:
        end_cop = min(int(args.end_cop), len(cop) - 1)

    if end_cop <= start_cop:
        raise ValueError(f"end_cop={end_cop} must be > start_cop={start_cop}")

    obj_id = int(args.obj_id)

    template_img_t = image_to_tensor(cop[template_cop], content_size, padded_size).unsqueeze(0).to(device)
    template_box_np = scale_box_to_padded_np(
        boxes[obj_id, template_cop],
        raw_cop_size,
        content_size,
        padded_size,
    )
    template_box_t = torch.from_numpy(template_box_np).float().unsqueeze(0).to(device)

    written = 0
    prev_smooth_pred = None

    with torch.no_grad():
        for ref_t in range(start_cop, end_cop):
            target_t = ref_t + 1
            a0 = int(cop_indices[ref_t]) + 1
            a1 = int(cop_indices[target_t]) + 1

            if a1 <= a0:
                continue

            td_win = td[a0:a1]
            sd_win = sd[a0:a1]

            ref_ten = image_to_tensor(cop[ref_t], content_size, padded_size).unsqueeze(0).to(device)
            aop_ten = aop_to_tensor(td_win, sd_win, content_size, padded_size).unsqueeze(0).to(device)

            out = forward_hsn_sequence_safe(
                model=model,
                template=template_img_t,
                template_box=template_box_t,
                ref=ref_ten,
                aop=aop_ten,
            )

            cls_seq = out["cls_seq"]
            reg_seq = out["reg_seq"]

            pred_boxes_seq, scores_seq = model.decode_sequence(
                cls_seq,
                reg_seq,
                image_hw=padded_size,
            )

            pred_boxes_seq_np = pred_boxes_seq[0].detach().cpu().numpy()
            scores_seq_np = scores_seq[0].detach().cpu().numpy()

            K = pred_boxes_seq_np.shape[0]

            for k in range(K):
                frame_idx = a0 + k

                if args.view == "cop_only" and frame_idx not in set(cop_indices.tolist()):
                    continue

                current_cop_id = previous_cop_id(cop_indices, frame_idx)

                base_bg, frame_type = build_base_background(
                    view="rgb_hold" if args.view == "cop_only" else args.view,
                    frame_idx=frame_idx,
                    cop_id=current_cop_id,
                    cop=cop,
                    td=td,
                    sd=sd,
                    content_size=content_size,
                    padded_size=padded_size,
                    cop_indices=cop_indices,
                    aop_color_mode=args.aop_color_mode,
                )

                gt_box = scale_box_to_padded_np(
                    boxes_all[obj_id, frame_idx],
                    raw_cop_size,
                    content_size,
                    padded_size,
                )

                pred_box_raw = pred_boxes_seq_np[k].astype(np.float32)
                score = float(scores_seq_np[k])
                iou_raw = iou_xyxy(pred_box_raw, gt_box)

                pred_box = smooth_box(prev_smooth_pred, pred_box_raw, args.smooth)
                prev_smooth_pred = pred_box.copy()

                pred_ok = valid_pred_box(pred_box, padded_size) and (
                    score >= args.score_thr or args.draw_low_score
                )

                out_frame = resize_for_video(base_bg, args.scale)

                if args.view == "side_by_side":
                    x_offsets = [0, base_w * args.scale]
                else:
                    x_offsets = [0]

                for x_offset in x_offsets:
                    gt_scaled = scale_box_for_output(gt_box, args.scale, x_offset=x_offset)
                    draw_box(
                        out_frame,
                        gt_scaled,
                        (0, 255, 0),
                        "GT" if args.box_labels else "",
                        thickness=args.box_thickness,
                        font_scale=args.font_scale,
                    )

                    if pred_ok:
                        pred_scaled = scale_box_for_output(pred_box, args.scale, x_offset=x_offset)
                        draw_box(
                            out_frame,
                            pred_scaled,
                            (255, 0, 0),
                            "Pred" if args.box_labels else "",
                            thickness=args.box_thickness,
                            font_scale=args.font_scale,
                        )

                if args.view == "rgb_hold" and args.show_aop_inset:
                    aop_vis = aop_frame_to_vis(
                        td[frame_idx],
                        sd[frame_idx],
                        content_size,
                        padded_size,
                        mode=args.aop_color_mode,
                    )
                    aop_vis = resize_for_video(aop_vis, max(1, args.scale))
                    add_inset(out_frame, aop_vis, title="TD/SD", width_ratio=0.28)

                if args.info_mode == "none":
                    lines = []
                elif args.info_mode == "minimal":
                    state = "hidden-low-score" if not pred_ok else "shown"
                    lines = [
                        f"{npz_path.stem} | f={frame_idx} | {frame_type}",
                        f"IoU(raw)={iou_raw:.3f} | score={score:.3f} | pred={state}",
                    ]
                else:
                    state = "hidden-low-score" if not pred_ok else "shown"
                    lines = [
                        f"seq: {npz_path.stem}",
                        f"frame: {frame_idx} | {frame_type}",
                        f"interval: {ref_t}->{target_t}",
                        f"IoU(raw): {iou_raw:.3f} | score: {score:.3f} | pred: {state}",
                    ]

                draw_info_panel(
                    out_frame,
                    lines,
                    font_scale=args.font_scale,
                    thickness=1,
                    pos=args.info_pos,
                )

                writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

                if frame_save_dir is not None:
                    cv2.imwrite(
                        str(frame_save_dir / f"{written:06d}.jpg"),
                        cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR),
                    )

                written += 1

                if args.max_frames is not None and written >= args.max_frames:
                    break

            if args.max_frames is not None and written >= args.max_frames:
                break

    writer.release()

    print(f"[done] wrote {written} frames")
    print(f"[done] video: {out_path}")


if __name__ == "__main__":
    main()
