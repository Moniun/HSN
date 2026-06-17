from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def import_official_simulator():
    try:
        from tianmoucv.sim import run_sim_singleimg
    except Exception as e:
        raise ImportError(
            "Cannot import official TianmouCV simulator:\n"
            "  from tianmoucv.sim import run_sim_singleimg\n\n"
            "Install/update TianmouCV first:\n"
            "  pip install -U tianmoucv\n\n"
            f"Original error: {repr(e)}"
        )
    return run_sim_singleimg


def import_official_denoise():
    """
    使用 TianmouCV 官方 denoise 模块里的 LVATF 相关底层滤波函数。
    官方 denoise_defualt_args 默认 denoise_function=LVAFT；
    这里为了适配 simulator 直接输出的 ndarray，直接调用 td_adaptive_filter/sd_adaptive_filter。
    """
    try:
        from tianmoucv.proc.denoise.lvatf import td_adaptive_filter, sd_adaptive_filter
        from tianmoucv.proc.denoise import denoise_defualt_args
    except Exception as e:
        raise ImportError(
            "Cannot import TianmouCV official denoise functions:\n"
            "  from tianmoucv.proc.denoise.lvatf import td_adaptive_filter, sd_adaptive_filter\n\n"
            "Install/update TianmouCV first:\n"
            "  pip install -U tianmoucv\n\n"
            f"Original error: {repr(e)}"
        )

    return td_adaptive_filter, sd_adaptive_filter, denoise_defualt_args


def list_image_files(frames_dir: Path) -> List[Path]:
    files = sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    files = [p for p in files if cv2.imread(str(p), cv2.IMREAD_COLOR) is not None]
    if not files:
        raise RuntimeError(f"No readable image frames found in {frames_dir}")
    return files


def read_frames_from_dir(frames_dir: Path, max_frames: Optional[int]) -> Tuple[List[np.ndarray], List[str]]:
    files = list_image_files(frames_dir)
    if max_frames is not None:
        files = files[:max_frames]

    frames, names = [], []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
            names.append(p.name)

    return frames, names


def read_video_frames(video_path: Path, max_frames: Optional[int]) -> Tuple[List[np.ndarray], List[str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames, names = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        names.append(f"{idx:06d}")
        idx += 1
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    return frames, names


def resize_frame_to_sensor(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    frame = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_td_single(td) -> np.ndarray:
    arr = to_numpy(td).astype(np.float32)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[0] == 2:
            return arr[0] - arr[1]
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 2:
            return arr[..., 0] - arr[..., 1]
        if arr.shape[-1] == 1:
            return arr[..., 0]

    raise ValueError(f"Unsupported TD shape: {arr.shape}")


def to_sd_two(sd0, sd1, prefer: str = "sd1") -> np.ndarray:
    a0 = to_numpy(sd0).astype(np.float32)
    a1 = to_numpy(sd1).astype(np.float32)

    def as_hw2(a):
        if a.ndim == 3:
            if a.shape[0] == 2:
                return np.transpose(a, (1, 2, 0))
            if a.shape[-1] == 2:
                return a
        return None

    first = as_hw2(a1) if prefer == "sd1" else as_hw2(a0)
    if first is not None:
        return first.astype(np.float32)

    second = as_hw2(a0) if prefer == "sd1" else as_hw2(a1)
    if second is not None:
        return second.astype(np.float32)

    def single(a):
        if a.ndim == 2:
            return a
        if a.ndim == 3 and a.shape[0] == 1:
            return a[0]
        if a.ndim == 3 and a.shape[-1] == 1:
            return a[..., 0]
        return None

    s0, s1 = single(a0), single(a1)
    if s0 is not None and s1 is not None:
        return np.stack([s0, s1], axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported SD shapes: sd0={a0.shape}, sd1={a1.shape}")


def resize_aop(td: np.ndarray, sd: np.ndarray, width: int, height: int):
    if td.shape != (height, width):
        td = cv2.resize(td, (width, height), interpolation=cv2.INTER_AREA)

    if sd.shape[:2] != (height, width):
        s0 = cv2.resize(sd[..., 0], (width, height), interpolation=cv2.INTER_AREA)
        s1 = cv2.resize(sd[..., 1], (width, height), interpolation=cv2.INTER_AREA)
        sd = np.stack([s0, s1], axis=-1)

    return td.astype(np.float32), sd.astype(np.float32)


def denoise_one_channel(
    x: np.ndarray,
    fn,
    device: torch.device,
    input_scale: float,
    var_fil_ksize: int,
    adapt_th_min: float,
    adapt_th_max: float,
) -> np.ndarray:
    """
    官方滤波函数内部阈值更接近整数尺度。
    如果 simulator 输出很小，可以试 --denoise-input-scale 255。
    """
    x_in = torch.from_numpy(x.astype(np.float32) * input_scale).to(device)

    with torch.no_grad():
        y = fn(
            x_in,
            min_thr=adapt_th_min,
            max_thr=adapt_th_max,
            kernel_size=var_fil_ksize,
        )

    y = y.detach().cpu().numpy().astype(np.float32)
    y = y / float(input_scale)
    return y


def denoise_tsd(
    td: np.ndarray,
    sd: np.ndarray,
    td_filter,
    sd_filter,
    device: torch.device,
    input_scale: float = 1.0,
    var_fil_ksize: int = 3,
    adapt_th_min: float = 3.0,
    adapt_th_max: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    td_dn = denoise_one_channel(
        td,
        td_filter,
        device,
        input_scale,
        var_fil_ksize,
        adapt_th_min,
        adapt_th_max,
    )

    sd0_dn = denoise_one_channel(
        sd[..., 0],
        sd_filter,
        device,
        input_scale,
        var_fil_ksize,
        adapt_th_min,
        adapt_th_max,
    )

    sd1_dn = denoise_one_channel(
        sd[..., 1],
        sd_filter,
        device,
        input_scale,
        var_fil_ksize,
        adapt_th_min,
        adapt_th_max,
    )

    sd_dn = np.stack([sd0_dn, sd1_dn], axis=-1)
    return td_dn.astype(np.float32), sd_dn.astype(np.float32)


def normalize01_pair(a: np.ndarray, b: np.ndarray, eps: float = 1e-6):
    """
    raw 和 denoised 用同一个可视化范围，方便肉眼比较。
    """
    vals = np.concatenate([a.reshape(-1), b.reshape(-1)]).astype(np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(a), np.zeros_like(b)

    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)

    if hi - lo < eps:
        lo = float(vals.min())
        hi = float(vals.max())

    if hi - lo < eps:
        return np.zeros_like(a), np.zeros_like(b)

    return (
        np.clip((a - lo) / (hi - lo), 0, 1),
        np.clip((b - lo) / (hi - lo), 0, 1),
    )


def heatmap_from_01(x01: np.ndarray, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    gray = (np.clip(x01, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def td_pair_vis(td_raw: np.ndarray, td_dn: np.ndarray):
    """
    TD 可视化：正负混合情况下使用绝对值强度图，保证 raw/dn 同尺度可比。
    """
    raw_abs = np.abs(td_raw).astype(np.float32)
    dn_abs = np.abs(td_dn).astype(np.float32)
    raw01, dn01 = normalize01_pair(raw_abs, dn_abs)
    return heatmap_from_01(raw01), heatmap_from_01(dn01)


def sd_pair_vis(sd_raw: np.ndarray, sd_dn: np.ndarray):
    raw_mag = np.sqrt(sd_raw[..., 0] ** 2 + sd_raw[..., 1] ** 2).astype(np.float32)
    dn_mag = np.sqrt(sd_dn[..., 0] ** 2 + sd_dn[..., 1] ** 2).astype(np.float32)
    raw01, dn01 = normalize01_pair(raw_mag, dn_mag)
    return heatmap_from_01(raw01), heatmap_from_01(dn01)


def put_text(img: np.ndarray, text: str, y: int):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def density(x: np.ndarray, eps: float = 1e-6) -> float:
    return float((np.abs(x) > eps).mean())


def main():
    parser = argparse.ArgumentParser("Test TianmouCV simulator + official denoise, then visualize raw/denoised TD/SD.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", type=str, help="原始 RGB 帧目录")
    src.add_argument("--video", type=str, help="原始视频文件")

    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=300)

    parser.add_argument("--sensor-width", type=int, default=640)
    parser.add_argument("--sensor-height", type=int, default=320)
    parser.add_argument("--aop-width", type=int, default=160)
    parser.add_argument("--aop-height", type=int, default=160)

    parser.add_argument("--xy", action="store_true")
    parser.add_argument("--interp", action="store_true")
    parser.add_argument("--sd-prefer", default="sd1", choices=["sd0", "sd1"])

    parser.add_argument("--device", default="cpu")

    # 官方去噪相关参数
    parser.add_argument("--denoise-input-scale", type=float, default=1.0,
                        help="如果去噪后全黑，试 255；如果保留太多噪声，保持 1 或调大阈值。")
    parser.add_argument("--var-fil-ksize", type=int, default=3)
    parser.add_argument("--adapt-th-min", type=float, default=6.0)
    parser.add_argument("--adapt-th-max", type=float, default=12.0)

    parser.add_argument("--fps", type=float, default=30.0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    run_sim_singleimg = import_official_simulator()
    td_filter, sd_filter, denoise_defualt_args = import_official_denoise()

    print("[official] simulator: tianmoucv.sim.run_sim_singleimg")
    print("[official] denoise: tianmoucv.proc.denoise.lvatf td_adaptive_filter / sd_adaptive_filter")
    print("[config]")
    print(f"  device={device}")
    print(f"  denoise_input_scale={args.denoise_input_scale}")
    print(f"  var_fil_ksize={args.var_fil_ksize}")
    print(f"  adapt_th_min={args.adapt_th_min}")
    print(f"  adapt_th_max={args.adapt_th_max}")

    if args.frames_dir:
        frames_bgr, names = read_frames_from_dir(Path(args.frames_dir), args.max_frames)
        source_name = Path(args.frames_dir).name
    else:
        frames_bgr, names = read_video_frames(Path(args.video), args.max_frames)
        source_name = Path(args.video).stem

    if len(frames_bgr) < 2:
        raise RuntimeError("Need at least 2 frames.")

    h, w = args.aop_height, args.aop_width

    # 每帧输出：2列 TD(raw/dn) + 2列 SD(raw/dn)
    video_size = (w * 4, h)

    out_video = out_dir / f"{source_name}_raw_vs_denoised_TD_SD.mp4"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        video_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {out_video}")

    raw_td_density = []
    dn_td_density = []
    raw_sd_density = []
    dn_sd_density = []

    prev_rgb = None

    for i, frame_bgr in enumerate(frames_bgr):
        cur_rgb = resize_frame_to_sensor(frame_bgr, args.sensor_width, args.sensor_height)

        img_pre, img_cur, td, sd0, sd1 = run_sim_singleimg(
            img_target=cur_rgb,
            img_ref=prev_rgb,
            sensor_width=args.sensor_width,
            sensor_height=args.sensor_height,
            xy=args.xy,
            interp=args.interp,
            device=device,
        )

        td = to_td_single(td)
        sd = to_sd_two(sd0, sd1, prefer=args.sd_prefer)
        td, sd = resize_aop(td, sd, args.aop_width, args.aop_height)

        td_dn, sd_dn = denoise_tsd(
            td=td,
            sd=sd,
            td_filter=td_filter,
            sd_filter=sd_filter,
            device=device,
            input_scale=args.denoise_input_scale,
            var_fil_ksize=args.var_fil_ksize,
            adapt_th_min=args.adapt_th_min,
            adapt_th_max=args.adapt_th_max,
        )

        td_raw_img, td_dn_img = td_pair_vis(td, td_dn)
        sd_raw_img, sd_dn_img = sd_pair_vis(sd, sd_dn)

        put_text(td_raw_img, "TD raw", 22)
        put_text(td_dn_img, "TD denoised", 22)
        put_text(sd_raw_img, "SD raw", 22)
        put_text(sd_dn_img, "SD denoised", 22)

        put_text(td_raw_img, f"idx={i}", 44)
        put_text(td_dn_img, f"idx={i}", 44)
        put_text(sd_raw_img, f"idx={i}", 44)
        put_text(sd_dn_img, f"idx={i}", 44)

        frame = np.concatenate([td_raw_img, td_dn_img, sd_raw_img, sd_dn_img], axis=1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        raw_td_density.append(density(td))
        dn_td_density.append(density(td_dn))
        raw_sd_density.append(density(sd))
        dn_sd_density.append(density(sd_dn))

        prev_rgb = cur_rgb

        if (i + 1) % 50 == 0 or i == 0 or i == len(frames_bgr) - 1:
            print(
                f"[{i + 1}/{len(frames_bgr)}] "
                f"TD density raw/dn={raw_td_density[-1]:.4f}/{dn_td_density[-1]:.4f}, "
                f"SD density raw/dn={raw_sd_density[-1]:.4f}/{dn_sd_density[-1]:.4f}"
            )

    writer.release()

    stats_path = out_dir / f"{source_name}_denoise_stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write(f"source={source_name}\n")
        f.write(f"frames={len(frames_bgr)}\n")
        f.write(f"denoise_input_scale={args.denoise_input_scale}\n")
        f.write(f"var_fil_ksize={args.var_fil_ksize}\n")
        f.write(f"adapt_th_min={args.adapt_th_min}\n")
        f.write(f"adapt_th_max={args.adapt_th_max}\n")
        f.write("\n")
        f.write(f"mean_td_density_raw={np.mean(raw_td_density):.6f}\n")
        f.write(f"mean_td_density_denoised={np.mean(dn_td_density):.6f}\n")
        f.write(f"mean_sd_density_raw={np.mean(raw_sd_density):.6f}\n")
        f.write(f"mean_sd_density_denoised={np.mean(dn_sd_density):.6f}\n")

    print("[done]")
    print(f"  video: {out_video}")
    print(f"  stats: {stats_path}")
    print(f"  mean TD density raw/dn: {np.mean(raw_td_density):.6f} / {np.mean(dn_td_density):.6f}")
    print(f"  mean SD density raw/dn: {np.mean(raw_sd_density):.6f} / {np.mean(dn_sd_density):.6f}")


if __name__ == "__main__":
    main()