from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def find_npz(data_root: Path, split: str, seq: Optional[str]) -> Path:
    root = data_root / split
    if not root.exists():
        raise FileNotFoundError(f"split dir not found: {root}")
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz files under: {root}")
    if seq is None:
        print(f"[info] --seq not specified, use first file: {files[0].name}")
        return files[0]
    exact = root / f"{seq}.npz"
    if exact.exists():
        return exact
    matches = [p for p in files if seq in p.stem]
    if matches:
        print(f"[info] exact seq not found, use matched file: {matches[0].name}")
        return matches[0]
    raise FileNotFoundError(f"cannot find seq={seq} under {root}")


def as_str(x) -> str:
    try:
        arr = np.asarray(x)
        if arr.shape == ():
            return str(arr.item())
        return str(arr.tolist())
    except Exception:
        return str(x)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def array_stats(x: np.ndarray, sample_limit: int = 2_000_000) -> Dict[str, object]:
    arr = np.asarray(x)
    info: Dict[str, object] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.size == 0:
        return info
    if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
        flat = arr.reshape(-1)
        if flat.size > sample_limit:
            idx = np.linspace(0, flat.size - 1, sample_limit).astype(np.int64)
            flat = flat[idx]
        flat = flat.astype(np.float32)
        finite = flat[np.isfinite(flat)]
        if finite.size:
            info.update({
                "min": safe_float(np.min(finite)),
                "max": safe_float(np.max(finite)),
                "mean": safe_float(np.mean(finite)),
                "std": safe_float(np.std(finite)),
                "p01": safe_float(np.percentile(finite, 1)),
                "p50": safe_float(np.percentile(finite, 50)),
                "p99": safe_float(np.percentile(finite, 99)),
                "nonzero_ratio": safe_float(np.mean(np.abs(finite) > 1e-6)),
            })
    else:
        info["example"] = as_str(arr.reshape(-1)[0])
    return info


def normalize01(x: np.ndarray, percentile: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(finite, 100.0 - percentile)
    hi = np.percentile(finite, percentile)
    if hi - lo < eps:
        lo = finite.min()
        hi = finite.max()
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def heatmap_rgb(x: np.ndarray, percentile: float = 99.0, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    x01 = normalize01(x, percentile=percentile)
    gray = (x01 * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def td_to_rgb(td: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    td = td.astype(np.float32)
    if td.min() >= 0:
        return heatmap_rgb(td, percentile=percentile)
    pos = np.clip(td, 0, None)
    neg = np.clip(-td, 0, None)
    mag = np.abs(td)
    pos01 = normalize01(pos, percentile=percentile)
    neg01 = normalize01(neg, percentile=percentile)
    mag01 = normalize01(mag, percentile=percentile)
    rgb = np.zeros((*td.shape, 3), dtype=np.float32)
    rgb[..., 0] = pos01
    rgb[..., 1] = 0.35 * mag01
    rgb[..., 2] = neg01
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def sd_to_rgb(sd: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    sd = np.asarray(sd)
    if sd.ndim == 2:
        mag = np.abs(sd.astype(np.float32))
    elif sd.ndim == 3 and sd.shape[-1] >= 2:
        s0 = sd[..., 0].astype(np.float32)
        s1 = sd[..., 1].astype(np.float32)
        mag = np.sqrt(s0 * s0 + s1 * s1)
    else:
        raise ValueError(f"unsupported sd frame shape: {sd.shape}")
    return heatmap_rgb(mag, percentile=percentile)


def cop_to_rgb(cop: np.ndarray) -> np.ndarray:
    img = np.asarray(cop)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img[..., :3].copy()


def put_text(img: np.ndarray, text: str, y: int, scale: float = 0.55) -> None:
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_box(img: np.ndarray, box: np.ndarray, color: Tuple[int, int, int], label: str = "") -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = int(np.clip(round(x1), 0, w - 1))
    y1 = int(np.clip(round(y1), 0, h - 1))
    x2 = int(np.clip(round(x2), 0, w - 1))
    y2 = int(np.clip(round(y2), 0, h - 1))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def scale_box(box: np.ndarray, from_hw: Tuple[int, int], to_hw: Tuple[int, int]) -> np.ndarray:
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    b = box.astype(np.float32).copy()
    b[[0, 2]] *= to_w / float(from_w)
    b[[1, 3]] *= to_h / float(from_h)
    return b


def valid_box(box: np.ndarray, min_size: float = 1.0) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.isfinite(box).all() and (x2 - x1) >= min_size and (y2 - y1) >= min_size


def get_sensor_hw(data: Dict[str, np.ndarray]) -> Tuple[int, int]:
    if "sensor_size" in data:
        arr = np.asarray(data["sensor_size"]).astype(int).reshape(-1)
        if len(arr) >= 2:
            return int(arr[0]), int(arr[1])
    if "cop" in data:
        return int(data["cop"].shape[1]), int(data["cop"].shape[2])
    return 320, 640


def get_aop_hw(data: Dict[str, np.ndarray]) -> Tuple[int, int]:
    if "aop_size" in data:
        arr = np.asarray(data["aop_size"]).astype(int).reshape(-1)
        if len(arr) >= 2:
            return int(arr[0]), int(arr[1])
    if "td" in data:
        return int(data["td"].shape[1]), int(data["td"].shape[2])
    return 160, 160


def create_contact_sheet(
    data: Dict[str, np.ndarray],
    out_path: Path,
    obj_id: int,
    num_samples: int,
    percentile: float,
) -> None:
    cop = data.get("cop")
    td = data.get("td")
    sd = data.get("sd")
    boxes = data.get("boxes")
    boxes_all = data.get("boxes_all")
    cop_indices = data.get("cop_indices")
    if cop is None or td is None or sd is None:
        print("[warn] missing cop/td/sd; skip contact sheet")
        return

    tc = len(cop)
    ids = np.linspace(0, max(0, tc - 1), min(num_samples, tc)).astype(int).tolist()
    sensor_hw = get_sensor_hw(data)
    aop_hw = get_aop_hw(data)

    rows = []
    for cop_i in ids:
        aop_i = int(cop_indices[cop_i]) if cop_indices is not None and cop_i < len(cop_indices) else min(cop_i, len(td) - 1)
        aop_i = int(np.clip(aop_i, 0, len(td) - 1))

        rgb = cop_to_rgb(cop[cop_i])
        td_img = td_to_rgb(td[aop_i], percentile=percentile)
        sd_img = sd_to_rgb(sd[aop_i], percentile=percentile)

        if boxes is not None and obj_id < boxes.shape[0] and cop_i < boxes.shape[1] and valid_box(boxes[obj_id, cop_i]):
            draw_box(rgb, boxes[obj_id, cop_i], (0, 255, 0), "GT")
        if boxes_all is not None and obj_id < boxes_all.shape[0] and aop_i < boxes_all.shape[1] and valid_box(boxes_all[obj_id, aop_i]):
            b = scale_box(boxes_all[obj_id, aop_i], sensor_hw, aop_hw)
            draw_box(td_img, b, (0, 255, 0), "GT")
            draw_box(sd_img, b, (0, 255, 0), "GT")

        put_text(rgb, f"COP idx={cop_i}", 22)
        put_text(td_img, f"TD aop={aop_i}", 22)
        put_text(sd_img, f"SD aop={aop_i}", 22)

        # resize RGB to AOP height for easy side-by-side if needed
        if rgb.shape[:2] != td_img.shape[:2]:
            rgb_small = cv2.resize(rgb, (td_img.shape[1], td_img.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            rgb_small = rgb
        rows.append(np.concatenate([rgb_small, td_img, sd_img], axis=1))

    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[write] contact sheet: {out_path}")


def write_video(
    frames: List[np.ndarray],
    out_path: Path,
    fps: float,
) -> None:
    if not frames:
        print(f"[warn] no frames for {out_path}")
        return
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {out_path}")
    for img in frames:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[write] video: {out_path} ({len(frames)} frames, {fps} fps)")


def create_cop_video(
    data: Dict[str, np.ndarray],
    out_path: Path,
    obj_id: int,
    fps: float,
    max_frames: Optional[int],
    start: int,
) -> None:
    cop = data.get("cop")
    boxes = data.get("boxes")
    if cop is None:
        print("[warn] no cop in npz; skip COP video")
        return
    end = len(cop) if max_frames is None else min(len(cop), start + max_frames)
    frames = []
    for i in range(start, end):
        img = cop_to_rgb(cop[i])
        if boxes is not None and obj_id < boxes.shape[0] and i < boxes.shape[1] and valid_box(boxes[obj_id, i]):
            draw_box(img, boxes[obj_id, i], (0, 255, 0), "GT")
        put_text(img, f"COP idx={i}", 22)
        frames.append(img)
    write_video(frames, out_path, fps)


def create_aop_video(
    data: Dict[str, np.ndarray],
    out_path: Path,
    obj_id: int,
    fps: float,
    max_frames: Optional[int],
    start: int,
    view: str,
    percentile: float,
) -> None:
    td = data.get("td")
    sd = data.get("sd")
    boxes_all = data.get("boxes_all")
    if td is None or sd is None:
        print("[warn] no td/sd in npz; skip AOP video")
        return
    sensor_hw = get_sensor_hw(data)
    aop_hw = get_aop_hw(data)
    end = len(td) if max_frames is None else min(len(td), start + max_frames)
    frames = []
    for i in range(start, end):
        td_img = td_to_rgb(td[i], percentile=percentile)
        sd_img = sd_to_rgb(sd[i], percentile=percentile)
        if boxes_all is not None and obj_id < boxes_all.shape[0] and i < boxes_all.shape[1] and valid_box(boxes_all[obj_id, i]):
            b = scale_box(boxes_all[obj_id, i], sensor_hw, aop_hw)
            draw_box(td_img, b, (0, 255, 0), "GT")
            draw_box(sd_img, b, (0, 255, 0), "GT")
        put_text(td_img, f"TD aop_idx={i}", 22)
        put_text(sd_img, f"SD aop_idx={i}", 22)
        if view == "td":
            frame = td_img
        elif view == "sd":
            frame = sd_img
        else:
            frame = np.concatenate([td_img, sd_img], axis=1)
        frames.append(frame)
    write_video(frames, out_path, fps)


def create_compare_video(
    raw: Dict[str, np.ndarray],
    denoised: Dict[str, np.ndarray],
    out_path: Path,
    fps: float,
    max_frames: Optional[int],
    start: int,
    percentile: float,
) -> None:
    raw_td, raw_sd = raw.get("td"), raw.get("sd")
    dn_td, dn_sd = denoised.get("td"), denoised.get("sd")
    if raw_td is None or raw_sd is None or dn_td is None or dn_sd is None:
        print("[warn] missing td/sd in raw or denoised; skip compare video")
        return
    n = min(len(raw_td), len(dn_td))
    end = n if max_frames is None else min(n, start + max_frames)
    frames = []
    for i in range(start, end):
        tr = td_to_rgb(raw_td[i], percentile=percentile)
        td = td_to_rgb(dn_td[i], percentile=percentile)
        sr = sd_to_rgb(raw_sd[i], percentile=percentile)
        sd = sd_to_rgb(dn_sd[i], percentile=percentile)
        put_text(tr, f"TD raw idx={i}", 22)
        put_text(td, f"TD denoised idx={i}", 22)
        put_text(sr, f"SD raw idx={i}", 22)
        put_text(sd, f"SD denoised idx={i}", 22)
        frames.append(np.concatenate([tr, td, sr, sd], axis=1))
    write_video(frames, out_path, fps)


def summarize_alignment(data: Dict[str, np.ndarray]) -> Dict[str, object]:
    info: Dict[str, object] = {}
    cop = data.get("cop")
    td = data.get("td")
    cop_indices = data.get("cop_indices")
    boxes = data.get("boxes")
    boxes_all = data.get("boxes_all")
    if cop is not None:
        info["num_cop_frames"] = int(len(cop))
    if td is not None:
        info["num_aop_frames"] = int(len(td))
    if cop_indices is not None:
        ci = np.asarray(cop_indices).astype(int)
        info["cop_indices_first10"] = ci[:10].tolist()
        info["cop_indices_last10"] = ci[-10:].tolist()
        if len(ci) > 1:
            info["cop_step_mean"] = float(np.mean(np.diff(ci)))
            info["cop_step_unique_first20"] = sorted(set(np.diff(ci).astype(int).tolist()))[:20]
    if boxes is not None and cop is not None:
        info["boxes_match_cop_len"] = bool(boxes.shape[1] == len(cop))
    if boxes_all is not None and td is not None:
        info["boxes_all_match_aop_len"] = bool(boxes_all.shape[1] == len(td))
    return info


def main() -> None:
    parser = argparse.ArgumentParser("Inspect generated Tianmouc simulation npz data.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", type=str, help="Path to one generated .npz file")
    src.add_argument("--data-root", type=str, help="Root containing train/ and val/")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--seq", default=None, help="Sequence name or substring when using --data-root")

    parser.add_argument("--compare-npz", type=str, default=None, help="Optional second npz, e.g. denoised version, for raw-vs-denoised video")
    parser.add_argument("--out-dir", default="runs/inspect_tianmouc")
    parser.add_argument("--obj-id", type=int, default=0)

    parser.add_argument("--max-cop-frames", type=int, default=300)
    parser.add_argument("--max-aop-frames", type=int, default=1000)
    parser.add_argument("--start-cop", type=int, default=0)
    parser.add_argument("--start-aop", type=int, default=0)
    parser.add_argument("--cop-fps", type=float, default=10.0)
    parser.add_argument("--aop-fps", type=float, default=100.0)
    parser.add_argument("--aop-view", default="tdsd", choices=["td", "sd", "tdsd"])
    parser.add_argument("--vis-percentile", type=float, default=99.0)
    parser.add_argument("--contact-samples", type=int, default=12)
    parser.add_argument("--no-video", action="store_true")

    args = parser.parse_args()

    if args.npz:
        npz_path = Path(args.npz)
    else:
        npz_path = find_npz(Path(args.data_root), args.split, args.seq)

    data = load_npz(npz_path)
    out_dir = Path(args.out_dir) / npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[npz] {npz_path}")
    print("[keys]", sorted(data.keys()))
    print("=" * 80)

    summary: Dict[str, object] = {
        "npz": str(npz_path),
        "keys": sorted(data.keys()),
        "arrays": {k: array_stats(v) for k, v in data.items()},
        "alignment": summarize_alignment(data),
    }
    for k in ["source", "simulator", "cop_step", "sensor_size", "aop_size", "xy", "interp", "denoise", "denoise_method", "adapt_th_min", "adapt_th_max"]:
        if k in data:
            summary[k] = as_str(data[k])

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[write] summary: {summary_path}")
    print(json.dumps(summary["alignment"], ensure_ascii=False, indent=2))

    create_contact_sheet(
        data=data,
        out_path=out_dir / "contact_sheet_rgb_td_sd.jpg",
        obj_id=args.obj_id,
        num_samples=args.contact_samples,
        percentile=args.vis_percentile,
    )

    if not args.no_video:
        create_cop_video(
            data=data,
            out_path=out_dir / "cop_rate_gt.mp4",
            obj_id=args.obj_id,
            fps=args.cop_fps,
            max_frames=args.max_cop_frames,
            start=args.start_cop,
        )
        create_aop_video(
            data=data,
            out_path=out_dir / "aop_rate_td_sd_gt.mp4",
            obj_id=args.obj_id,
            fps=args.aop_fps,
            max_frames=args.max_aop_frames,
            start=args.start_aop,
            view=args.aop_view,
            percentile=args.vis_percentile,
        )
        if args.compare_npz:
            compare_data = load_npz(Path(args.compare_npz))
            create_compare_video(
                raw=data,
                denoised=compare_data,
                out_path=out_dir / "compare_raw_vs_second_td_sd.mp4",
                fps=args.aop_fps,
                max_frames=args.max_aop_frames,
                start=args.start_aop,
                percentile=args.vis_percentile,
            )

    print("[done]")
    print(f"  output dir: {out_dir}")


if __name__ == "__main__":
    main()
