from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import List, Tuple

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
            "    from tianmoucv.sim import run_sim_singleimg\n\n"
            "Install it first:\n"
            "    pip install -U tianmoucv\n\n"
            f"Original error: {repr(e)}"
        )
    return run_sim_singleimg


def import_official_denoise():
    try:
        from tianmoucv.proc.denoise.lvatf import td_adaptive_filter, sd_adaptive_filter
    except Exception as e:
        raise ImportError(
            "Cannot import official TianmouCV LVATF denoise functions:\n"
            "    from tianmoucv.proc.denoise.lvatf import td_adaptive_filter, sd_adaptive_filter\n\n"
            "Install/update TianmouCV first:\n"
            "    pip install -U tianmoucv\n\n"
            f"Original error: {repr(e)}"
        )
    return td_adaptive_filter, sd_adaptive_filter


def _denoise_one_channel(
    x: np.ndarray,
    filter_fn,
    device: torch.device,
    input_scale: float,
    adapt_th_min: float,
    adapt_th_max: float,
    var_fil_ksize: int,
) -> np.ndarray:
    # TianmouCV 官方 LVATF 函数处理 torch.Tensor。
    # 如果 simulator 输出幅值很小，可用 --denoise-input-scale 255 放大后再滤波。
    x_t = torch.from_numpy(x.astype(np.float32) * float(input_scale)).to(device)
    with torch.no_grad():
        y = filter_fn(
            x_t,
            min_thr=float(adapt_th_min),
            max_thr=float(adapt_th_max),
            kernel_size=int(var_fil_ksize),
        )
    y = y.detach().cpu().numpy().astype(np.float32) / float(input_scale)
    return y


def denoise_tsd_lvatf(
    td: np.ndarray,
    sd: np.ndarray,
    td_filter,
    sd_filter,
    device: torch.device,
    input_scale: float = 1.0,
    adapt_th_min: float = 6.0,
    adapt_th_max: float = 12.0,
    var_fil_ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    td_dn = _denoise_one_channel(
        td, td_filter, device, input_scale, adapt_th_min, adapt_th_max, var_fil_ksize
    )
    sd0_dn = _denoise_one_channel(
        sd[..., 0], sd_filter, device, input_scale, adapt_th_min, adapt_th_max, var_fil_ksize
    )
    sd1_dn = _denoise_one_channel(
        sd[..., 1], sd_filter, device, input_scale, adapt_th_min, adapt_th_max, var_fil_ksize
    )
    sd_dn = np.stack([sd0_dn, sd1_dn], axis=-1)
    return td_dn.astype(np.float16), sd_dn.astype(np.float16)


def list_image_files(frames_dir: Path) -> List[Path]:
    files = sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    valid = []
    for p in files:
        if cv2.imread(str(p), cv2.IMREAD_COLOR) is not None:
            valid.append(p)
    if not valid:
        raise RuntimeError(f"No readable image frames found in {frames_dir}")
    return valid


def read_frames_from_dir(frames_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
    frame_files = list_image_files(frames_dir)
    frames, names = [], []
    for p in frame_files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
            names.append(p.name)
    return frames, names


def read_video_frames(video_path: Path) -> Tuple[List[np.ndarray], List[str]]:
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
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, names


def parse_nfs_txt(txt_path: Path, num_frames: int) -> np.ndarray:
    """
    NfS typical line:
        0 478 443 506 467 1 0 0 1 "aircraft"

    Use:
        nums[1:5] -> x1 y1 x2 y2
        nums[5]   -> frame id, 1-based
    """
    boxes = np.zeros((num_frames, 4), dtype=np.float16)
    valid = np.zeros((num_frames,), dtype=bool)
    rows = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = shlex.split(line)
            nums = []
            for token in parts:
                try:
                    nums.append(float(token))
                except ValueError:
                    break

            if len(nums) >= 6:
                x1, y1, x2, y2 = nums[1:5]
                frame_id = int(nums[5])
            elif len(nums) >= 4:
                x1, y1, x2, y2 = nums[:4]
                frame_id = len(rows) + 1
            else:
                continue

            rows.append((frame_id, [x1, y1, x2, y2]))

    if not rows:
        raise RuntimeError(f"No valid boxes parsed from {txt_path}")

    use_frame_id = all(1 <= fid <= num_frames for fid, _ in rows)

    if use_frame_id:
        for fid, box in rows:
            idx = fid - 1
            boxes[idx] = np.asarray(box, dtype=np.float16)
            valid[idx] = True
    else:
        for i, (_, box) in enumerate(rows[:num_frames]):
            boxes[i] = np.asarray(box, dtype=np.float16)
            valid[i] = True

    last = None
    for i in range(num_frames):
        if valid[i]:
            last = boxes[i].copy()
        elif last is not None:
            boxes[i] = last
            valid[i] = True

    if not valid.all():
        ids = np.where(valid)[0]
        if len(ids) == 0:
            raise RuntimeError(f"No valid boxes after parsing {txt_path}")
        first = boxes[ids[0]].copy()
        for i in range(ids[0]):
            boxes[i] = first
            valid[i] = True

    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float16)

    return boxes[None, ...]  # [1,T,4]


def load_boxes(path: Path, num_frames: int, box_format: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        boxes = np.load(path).astype(np.float16)
        if boxes.ndim == 2 and boxes.shape[-1] == 4:
            boxes = boxes[None, ...]
        elif boxes.ndim == 3 and boxes.shape[-1] == 4:
            pass
        else:
            raise ValueError(f"Unsupported boxes shape: {boxes.shape}")
    else:
        boxes = parse_nfs_txt(path, num_frames)

    if boxes.shape[1] != num_frames:
        n = min(boxes.shape[1], num_frames)
        print(f"[warn] boxes length {boxes.shape[1]} != frames {num_frames}; use {n}")
        boxes = boxes[:, :n]

    if box_format == "xywh":
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        boxes = np.stack([x, y, x + w, y + h], axis=-1)
    elif box_format != "xyxy":
        raise ValueError(f"Unsupported box_format: {box_format}")

    return boxes.astype(np.float16)


def resize_frame_to_sensor(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    frame = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def scale_boxes_to_sensor(boxes: np.ndarray, orig_w: int, orig_h: int, width: int, height: int) -> np.ndarray:
    boxes = boxes.copy().astype(np.float16)
    sx = width / float(orig_w)
    sy = height / float(orig_h)
    boxes[..., [0, 2]] *= sx
    boxes[..., [1, 3]] *= sy
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, width - 1)
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, height - 1)
    return boxes


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_td_single(td) -> np.ndarray:
    arr = to_numpy(td).astype(np.float16)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[0] == 2:
            return (arr[0] - arr[1]).astype(np.float16)
        if arr.shape[0] == 1:
            return arr[0].astype(np.float16)
        if arr.shape[-1] == 2:
            return (arr[..., 0] - arr[..., 1]).astype(np.float16)
        if arr.shape[-1] == 1:
            return arr[..., 0].astype(np.float16)

    raise ValueError(f"Unsupported TD shape: {arr.shape}")


def to_sd_two(sd0, sd1, prefer: str = "sd1") -> np.ndarray:
    a0 = to_numpy(sd0).astype(np.float16)
    a1 = to_numpy(sd1).astype(np.float16)

    def as_hw2(a):
        if a.ndim == 3:
            if a.shape[0] == 2:
                return np.transpose(a, (1, 2, 0))
            if a.shape[-1] == 2:
                return a
        return None

    first = as_hw2(a1) if prefer == "sd1" else as_hw2(a0)
    if first is not None:
        return first.astype(np.float16)

    second = as_hw2(a0) if prefer == "sd1" else as_hw2(a1)
    if second is not None:
        return second.astype(np.float16)

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
        return np.stack([s0, s1], axis=-1).astype(np.float16)

    raise ValueError(f"Unsupported SD shapes: sd0={a0.shape}, sd1={a1.shape}")


def resize_aop(td: np.ndarray, sd: np.ndarray, width: int, height: int):
    if td.shape != (height, width):
        td = cv2.resize(td, (width, height), interpolation=cv2.INTER_AREA)

    if sd.shape[:2] != (height, width):
        s0 = cv2.resize(sd[..., 0], (width, height), interpolation=cv2.INTER_AREA)
        s1 = cv2.resize(sd[..., 1], (width, height), interpolation=cv2.INTER_AREA)
        sd = np.stack([s0, s1], axis=-1)

    return td.astype(np.float16), sd.astype(np.float16)


def main():
    parser = argparse.ArgumentParser("Official TianmouCV simulator with COP downsampling.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", type=str)
    src.add_argument("--video", type=str)

    parser.add_argument("--boxes", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--cop-step", type=int, default=10,
                        help="Save one COP every K original frames. All frames still generate TD/SD.")
    parser.add_argument("--sensor-width", type=int, default=640)
    parser.add_argument("--sensor-height", type=int, default=320)
    parser.add_argument("--aop-width", type=int, default=160)
    parser.add_argument("--aop-height", type=int, default=160)

    parser.add_argument("--box-format", default="xyxy", choices=["xyxy", "xywh"])
    parser.add_argument("--xy", action="store_true")
    parser.add_argument("--interp", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sd-prefer", default="sd1", choices=["sd0", "sd1"])
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--denoise", action="store_true",
                        help="Apply official TianmouCV LVATF denoise to TD/SD before saving npz.")
    parser.add_argument("--denoise-input-scale", type=float, default=1.0,
                        help="Scale TD/SD before LVATF. If denoised output is too sparse/black, try 255.")
    parser.add_argument("--adapt-th-min", type=float, default=6.0)
    parser.add_argument("--adapt-th-max", type=float, default=12.0)
    parser.add_argument("--var-fil-ksize", type=int, default=3)

    args = parser.parse_args()

    if args.cop_step < 1:
        raise ValueError("--cop-step must be >= 1")

    run_sim_singleimg = import_official_simulator()
    td_filter = None
    sd_filter = None
    if args.denoise:
        td_filter, sd_filter = import_official_denoise()

    print("[official] using tianmoucv.sim.run_sim_singleimg")
    print(f"[config] cop_step={args.cop_step}")
    print(f"[config] denoise={args.denoise}")
    if args.denoise:
        print("[official] denoise=tianmoucv.proc.denoise.lvatf")
        print(
            f"[config] LVATF adapt_th_min={args.adapt_th_min}, "
            f"adapt_th_max={args.adapt_th_max}, "
            f"var_fil_ksize={args.var_fil_ksize}, "
            f"input_scale={args.denoise_input_scale}"
        )

    if args.frames_dir:
        frames_bgr, frame_names = read_frames_from_dir(Path(args.frames_dir))
        source = args.frames_dir
    else:
        frames_bgr, frame_names = read_video_frames(Path(args.video))
        source = args.video

    if args.max_frames is not None:
        frames_bgr = frames_bgr[:args.max_frames]
        frame_names = frame_names[:args.max_frames]

    if len(frames_bgr) < args.cop_step + 1:
        raise RuntimeError("Not enough frames for the selected --cop-step")

    orig_h, orig_w = frames_bgr[0].shape[:2]
    n = len(frames_bgr)

    boxes_all = load_boxes(Path(args.boxes), n, args.box_format)
    if boxes_all.shape[1] != n:
        n = min(n, boxes_all.shape[1])
        frames_bgr = frames_bgr[:n]
        frame_names = frame_names[:n]
        boxes_all = boxes_all[:, :n]

    boxes_all = scale_boxes_to_sensor(
        boxes_all, orig_w, orig_h, args.sensor_width, args.sensor_height
    )

    device = torch.device(args.device)

    cop_list = []
    cop_indices = []
    cop_frame_names = []

    td_list = []
    sd_list = []

    prev_rgb = None

    for i, frame_bgr in enumerate(frames_bgr):
        cur_rgb = resize_frame_to_sensor(
            frame_bgr, args.sensor_width, args.sensor_height
        )

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

        if args.denoise:
            td, sd = denoise_tsd_lvatf(
                td=td,
                sd=sd,
                td_filter=td_filter,
                sd_filter=sd_filter,
                device=device,
                input_scale=args.denoise_input_scale,
                adapt_th_min=args.adapt_th_min,
                adapt_th_max=args.adapt_th_max,
                var_fil_ksize=args.var_fil_ksize,
            )

        td_list.append(td)
        sd_list.append(sd)

        if i % args.cop_step == 0:
            cop_list.append(cur_rgb)
            cop_indices.append(i)
            cop_frame_names.append(frame_names[i])

        prev_rgb = cur_rgb

        if (i + 1) % 2000 == 0 or i == 0 or i == n - 1:
            print(f"[sim] {i + 1}/{n}")

    if len(cop_indices) < 2:
        raise RuntimeError("Less than 2 COP frames generated. Reduce --cop-step or use longer sequence.")

    cop_indices = np.asarray(cop_indices, dtype=np.int32)

    cop = np.stack(cop_list, axis=0).astype(np.uint8)               # [Tc,320,640,3]
    td = np.stack(td_list, axis=0).astype(np.float16)               # [Ta,160,160]
    sd = np.stack(sd_list, axis=0).astype(np.float16)               # [Ta,160,160,2]
    boxes_all = boxes_all.astype(np.float16)
    boxes = boxes_all[:, cop_indices, :].astype(np.float16)         # [M,Tc,4]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        cop=cop,                           # [Tc, 320, 640, 3]
        td=td,                             # [Ta, 160, 160]
        sd=sd,                             # [Ta, 160, 160, 2]

        boxes=boxes,                       # [M, Tc, 4]，COP 时间轴框
        boxes_all=boxes_all,               # [M, Ta, 4]，AOP / 原始帧时间轴框

        cop_indices=cop_indices,           # [Tc]，每个 COP 对应原始帧编号
        frame_names=np.asarray(frame_names),
        cop_frame_names=np.asarray(cop_frame_names),

        source=np.asarray(str(source)),
        simulator=np.asarray("tianmoucv.sim.run_sim_singleimg"),
        denoise=np.asarray(bool(args.denoise)),
        denoise_method=np.asarray("tianmoucv.proc.denoise.lvatf" if args.denoise else "none"),
        denoise_input_scale=np.asarray(args.denoise_input_scale, dtype=np.float32),
        adapt_th_min=np.asarray(args.adapt_th_min, dtype=np.float32),
        adapt_th_max=np.asarray(args.adapt_th_max, dtype=np.float32),
        var_fil_ksize=np.asarray(args.var_fil_ksize, dtype=np.int32),
        cop_step=np.asarray(args.cop_step, dtype=np.int32),
        sensor_size=np.asarray([args.sensor_height, args.sensor_width], dtype=np.int32),
        aop_size=np.asarray([args.aop_height, args.aop_width], dtype=np.int32),
        xy=np.asarray(bool(args.xy)),
        interp=np.asarray(bool(args.interp)),
    )

    print("[done]")
    print(f"  output      : {out_path}")
    print(f"  cop         : {cop.shape}, {cop.dtype}")
    print(f"  td          : {td.shape}, {td.dtype}")
    print(f"  sd          : {sd.shape}, {sd.dtype}")
    print(f"  boxes       : {boxes.shape}, {boxes.dtype}")
    print(f"  cop_indices : {cop_indices.shape}, first={cop_indices[:5]}, step={args.cop_step}")
    print(f"  denoise     : {args.denoise}")
    if args.denoise:
        print(f"  lvatf       : adapt_th_min={args.adapt_th_min}, adapt_th_max={args.adapt_th_max}")


if __name__ == "__main__":
    main()
