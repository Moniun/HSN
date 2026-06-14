from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


# Make project root importable when running: python tools/check_data.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsn.data import TianmoucHSNDataset  # noqa: E402


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fmt_shape(x: Any) -> str:
    if hasattr(x, "shape"):
        return str(tuple(x.shape))
    return "-"


def fmt_dtype(x: Any) -> str:
    if hasattr(x, "dtype"):
        return str(x.dtype)
    return type(x).__name__


def find_npz_files(data_root: Path, split: str | None = None) -> List[Path]:
    if split is not None:
        root = data_root / split
        return sorted(root.glob("*.npz"))
    return sorted(data_root.rglob("*.npz"))


def is_xyxy_valid(boxes: np.ndarray, min_size: float = 1.0) -> np.ndarray:
    """
    boxes: [..., 4], xyxy
    return boolean mask [...]
    """
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    finite = np.isfinite(boxes).all(axis=-1)
    return finite & (w >= min_size) & (h >= min_size)


def check_npz_file(path: Path, expected_cop_step: int | None = None, verbose: bool = True) -> bool:
    """
    Check one converted .npz file.

    Expected keys:
        cop         [Tc, H, W, 3]
        td          [Ta, h, w]
        sd          [Ta, h, w, 2]
        boxes       [M, Tc, 4]
        boxes_all   [M, Ta, 4]
        cop_indices [Tc]
    """
    ok = True

    print("\n" + "=" * 100)
    print(f"[NPZ] {path}")

    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return False

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[file size] {size_mb:.2f} MB")

    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[ERROR] np.load failed: {repr(e)}")
        return False

    # NumPy np.load on .npz returns an NpzFile-like object with a .files list.
    print(f"[keys] {d.files}")

    required = ["cop", "td", "sd", "boxes", "boxes_all", "cop_indices"]
    missing = [k for k in required if k not in d.files]
    if missing:
        print(f"[ERROR] missing required keys: {missing}")
        ok = False

    for k in d.files:
        try:
            arr = d[k]
            print(f"{k:18s} shape={fmt_shape(arr):22s} dtype={fmt_dtype(arr)}")
        except Exception as e:
            print(f"{k:18s} [ERROR reading key] {repr(e)}")
            ok = False

    if missing:
        return False

    cop = d["cop"]
    td = d["td"]
    sd = d["sd"]
    boxes = d["boxes"]
    boxes_all = d["boxes_all"]
    cop_indices = d["cop_indices"]

    # Basic ndim / channel checks.
    if not (cop.ndim == 4 and cop.shape[-1] == 3):
        print(f"[ERROR] cop should be [Tc,H,W,3], got {cop.shape}")
        ok = False

    if not (td.ndim == 3):
        print(f"[ERROR] td should be [Ta,h,w], got {td.shape}")
        ok = False

    if not (sd.ndim == 4 and sd.shape[-1] == 2):
        print(f"[ERROR] sd should be [Ta,h,w,2], got {sd.shape}")
        ok = False

    if not (boxes.ndim == 3 and boxes.shape[-1] == 4):
        print(f"[ERROR] boxes should be [M,Tc,4], got {boxes.shape}")
        ok = False

    if not (boxes_all.ndim == 3 and boxes_all.shape[-1] == 4):
        print(f"[ERROR] boxes_all should be [M,Ta,4], got {boxes_all.shape}")
        ok = False

    if not (cop_indices.ndim == 1):
        print(f"[ERROR] cop_indices should be [Tc], got {cop_indices.shape}")
        ok = False

    # Length consistency.
    if len(cop_indices) != len(cop):
        print(f"[ERROR] len(cop_indices)={len(cop_indices)} != len(cop)={len(cop)}")
        ok = False

    if boxes.ndim == 3 and len(cop) != boxes.shape[1]:
        print(f"[ERROR] boxes.shape[1]={boxes.shape[1]} != len(cop)={len(cop)}")
        ok = False

    if boxes_all.ndim == 3 and len(td) != boxes_all.shape[1]:
        print(f"[ERROR] boxes_all.shape[1]={boxes_all.shape[1]} != len(td)={len(td)}")
        ok = False

    if len(sd) != len(td):
        print(f"[ERROR] len(sd)={len(sd)} != len(td)={len(td)}")
        ok = False

    if boxes.ndim == 3 and boxes_all.ndim == 3 and boxes.shape[0] != boxes_all.shape[0]:
        print(f"[ERROR] boxes M={boxes.shape[0]} != boxes_all M={boxes_all.shape[0]}")
        ok = False

    # COP index checks.
    if cop_indices.size > 0:
        if cop_indices[0] != 0:
            print(f"[WARN] cop_indices[0] is {cop_indices[0]}, expected usually 0")

        if not np.all(np.diff(cop_indices) > 0):
            print("[ERROR] cop_indices must be strictly increasing")
            ok = False

        if cop_indices[-1] >= len(td):
            print(f"[ERROR] cop_indices[-1]={cop_indices[-1]} out of td range len={len(td)}")
            ok = False

        diffs = np.diff(cop_indices)
        if diffs.size > 0:
            uniq, cnt = np.unique(diffs, return_counts=True)
            hist = {int(u): int(c) for u, c in zip(uniq, cnt)}
            print(f"[cop_indices diff histogram] {hist}")

            if expected_cop_step is not None:
                bad = diffs[diffs != expected_cop_step]
                if len(bad) > 0:
                    print(
                        f"[WARN] some cop index gaps != expected_cop_step={expected_cop_step}. "
                        f"num_bad={len(bad)}, examples={bad[:10].tolist()}"
                    )

    # Check boxes relation: boxes should equal boxes_all[:, cop_indices].
    try:
        expected_boxes = boxes_all[:, cop_indices, :]
        max_abs = np.max(np.abs(expected_boxes.astype(np.float32) - boxes.astype(np.float32)))
        print(f"[boxes consistency] max |boxes - boxes_all[:,cop_indices]| = {max_abs:.6f}")
        if max_abs > 1e-3:
            print("[ERROR] boxes is not consistent with boxes_all[:, cop_indices]")
            ok = False
    except Exception as e:
        print(f"[ERROR] failed boxes consistency check: {repr(e)}")
        ok = False

    # Valid boxes.
    try:
        valid_cop = is_xyxy_valid(boxes.astype(np.float32), min_size=1.0)
        valid_all = is_xyxy_valid(boxes_all.astype(np.float32), min_size=1.0)
        print(f"[valid boxes] boxes valid ratio     = {valid_cop.mean():.6f}")
        print(f"[valid boxes] boxes_all valid ratio = {valid_all.mean():.6f}")

        if valid_cop.mean() < 0.99:
            print("[WARN] many invalid boxes in boxes")
        if valid_all.mean() < 0.99:
            print("[WARN] many invalid boxes in boxes_all")
    except Exception as e:
        print(f"[WARN] failed valid-box check: {repr(e)}")

    # Basic value ranges.
    try:
        print(f"[cop] min={cop.min()} max={cop.max()} mean={float(cop.mean()):.4f}")
        print(f"[td]  min={float(td.min()):.4f} max={float(td.max()):.4f} mean={float(td.mean()):.4f}")
        print(f"[sd]  min={float(sd.min()):.4f} max={float(sd.max()):.4f} mean={float(sd.mean()):.4f}")

        if float(np.abs(td).mean()) == 0.0:
            print("[WARN] td mean absolute value is 0; TD may be empty")
        if float(np.abs(sd).mean()) == 0.0:
            print("[WARN] sd mean absolute value is 0; SD may be empty")
    except Exception as e:
        print(f"[WARN] failed value-range check: {repr(e)}")

    # Metadata if present.
    for meta_key in ["cop_step", "sensor_size", "aop_size", "simulator", "source"]:
        if meta_key in d.files:
            try:
                print(f"[meta] {meta_key}: {d[meta_key]}")
            except Exception:
                pass

    print("[NPZ RESULT]", "OK" if ok else "FAILED")
    return ok


def print_tensor_info(name: str, x: Any):
    if torch.is_tensor(x):
        x_detached = x.detach().cpu()
        if x_detached.numel() > 0 and x_detached.dtype.is_floating_point:
            print(
                f"{name:22s} shape={tuple(x.shape)!s:24s} "
                f"dtype={str(x.dtype):14s} min={float(x_detached.min()):9.4f} "
                f"max={float(x_detached.max()):9.4f} mean={float(x_detached.mean()):9.4f}"
            )
        else:
            print(f"{name:22s} shape={tuple(x.shape)!s:24s} dtype={str(x.dtype)} value={x_detached}")
    else:
        print(f"{name:22s} type={type(x).__name__} value={x}")


def check_dataset(cfg: Dict[str, Any], split: str, data_root: str | None, batch_size: int, num_workers: int) -> bool:
    ok = True

    print("\n" + "=" * 100)
    print(f"[DATASET] split={split}, data_root={data_root}")

    try:
        ds = TianmoucHSNDataset(cfg, split=split, data_root=data_root)
    except Exception as e:
        print(f"[ERROR] dataset init failed: {repr(e)}")
        return False

    print(f"[dataset] len={len(ds)}")
    if len(ds) == 0:
        print("[ERROR] dataset length is 0")
        return False

    # Check first sample.
    try:
        item = ds[0]
    except Exception as e:
        print(f"[ERROR] ds[0] failed: {repr(e)}")
        return False

    print("\n[SAMPLE 0]")
    expected_keys = [
        "template",
        "ref",
        "target",
        "aop",
        "template_box",
        "ref_box",
        "target_box",
        "target_boxes_seq",
        "aop_frame_indices",
        "seq_name",
        "obj_id",
        "template_t",
        "ref_t",
        "target_t",
        "aop_start",
        "aop_end",
    ]

    missing = [k for k in expected_keys if k not in item]
    if missing:
        print(f"[ERROR] sample missing keys: {missing}")
        ok = False

    for k in expected_keys:
        if k in item:
            print_tensor_info(k, item[k])

    if "aop" in item and "target_boxes_seq" in item:
        K_aop = item["aop"].shape[0]
        K_box = item["target_boxes_seq"].shape[0]
        if K_aop != K_box:
            print(f"[ERROR] aop K={K_aop} != target_boxes_seq K={K_box}")
            ok = False
        else:
            print(f"[OK] AOP steps match target_boxes_seq: K={K_aop}")

    if "aop_frame_indices" in item and "aop" in item:
        if item["aop_frame_indices"].numel() != item["aop"].shape[0]:
            print(
                f"[ERROR] len(aop_frame_indices)={item['aop_frame_indices'].numel()} "
                f"!= aop K={item['aop'].shape[0]}"
            )
            ok = False

    # Check boxes in padded coordinates.
    padded_h, padded_w = tuple(cfg.get("data", cfg).get("padded_size", [232, 296]))
    for key in ["template_box", "ref_box", "target_box"]:
        if key in item and torch.is_tensor(item[key]):
            b = item[key]
            if not (0 <= float(b[0]) <= padded_w - 1 and 0 <= float(b[2]) <= padded_w - 1):
                print(f"[WARN] {key} x out of padded image range: {b.tolist()}")
            if not (0 <= float(b[1]) <= padded_h - 1 and 0 <= float(b[3]) <= padded_h - 1):
                print(f"[WARN] {key} y out of padded image range: {b.tolist()}")

    if "target_boxes_seq" in item:
        bseq = item["target_boxes_seq"]
        if not torch.isfinite(bseq).all():
            print("[ERROR] target_boxes_seq contains nan/inf")
            ok = False
        if (bseq[:, 2] <= bseq[:, 0]).any() or (bseq[:, 3] <= bseq[:, 1]).any():
            print("[ERROR] target_boxes_seq contains invalid xyxy boxes")
            ok = False

    # Check DataLoader collation.
    print("\n[DATALOADER]")
    try:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )
        batch = next(iter(loader))
        for k in ["template", "ref", "target", "aop", "template_box", "target_box", "target_boxes_seq", "aop_frame_indices"]:
            if k in batch:
                print_tensor_info("batch." + k, batch[k])

        if batch["aop"].shape[1] != batch["target_boxes_seq"].shape[1]:
            print(
                f"[ERROR] batch aop K={batch['aop'].shape[1]} "
                f"!= batch target_boxes_seq K={batch['target_boxes_seq'].shape[1]}"
            )
            ok = False
        else:
            print(f"[OK] batch K={batch['aop'].shape[1]}")
    except Exception as e:
        print(f"[ERROR] DataLoader failed: {repr(e)}")
        print(
            "[HINT] If this is a stack-size error, some samples may have different K. "
            "Check cop_indices gaps or use a custom collate_fn / fixed cop_step."
        )
        ok = False

    print("[DATASET RESULT]", "OK" if ok else "FAILED")
    return ok


def check_model_forward(
    cfg: Dict[str, Any],
    split: str,
    data_root: str | None,
    checkpoint: str | None,
    batch_size: int,
    num_workers: int,
) -> bool:
    """
    Optional model forward check.

    This checks:
        forward_hsn_sequence()
        decode_sequence()

    It does not require a trained checkpoint. If no checkpoint is provided,
    it tests random initialized model shapes only.
    """
    print("\n" + "=" * 100)
    print("[MODEL FORWARD CHECK]")

    ok = True

    try:
        from hsn.model import TianmoucHSN
    except Exception as e:
        print(f"[ERROR] import model failed: {repr(e)}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    try:
        ds = TianmoucHSNDataset(cfg, split=split, data_root=data_root)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        batch = next(iter(loader))
    except Exception as e:
        print(f"[ERROR] failed loading data for model check: {repr(e)}")
        return False

    try:
        model = TianmoucHSN(cfg).to(device)

        if checkpoint:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.exists():
                print(f"[WARN] checkpoint not found, using random weights: {ckpt_path}")
            else:
                ckpt = torch.load(ckpt_path, map_location=device)
                state = ckpt.get("model", ckpt.get("state_dict", ckpt))
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"[checkpoint] loaded {ckpt_path}")
                print(f"[checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")

        model.eval()

        batch_t = {}
        for k, v in batch.items():
            batch_t[k] = v.to(device) if torch.is_tensor(v) else v

        with torch.no_grad():
            if not hasattr(model, "forward_hsn_sequence"):
                print("[ERROR] model has no forward_hsn_sequence()")
                return False

            out = model.forward_hsn_sequence(
                template=batch_t["template"],
                template_box=batch_t["template_box"],
                ref=batch_t["ref"],
                aop=batch_t["aop"],
                target=batch_t.get("target", None),
            )

            cls_seq = out["cls_seq"]
            reg_seq = out["reg_seq"]

            print(f"[forward] len(cls_seq)={len(cls_seq)}, len(reg_seq)={len(reg_seq)}")
            print_tensor_info("cls_seq[0]", cls_seq[0])
            print_tensor_info("reg_seq[0]", reg_seq[0])

            if len(cls_seq) != batch_t["aop"].shape[1]:
                print(f"[ERROR] output K={len(cls_seq)} != input AOP K={batch_t['aop'].shape[1]}")
                ok = False

            if not hasattr(model, "decode_sequence"):
                print("[ERROR] model has no decode_sequence()")
                return False

            if not hasattr(model, "decode"):
                print("[ERROR] model has no decode(); decode_sequence() will fail")
                return False

            image_hw = tuple(cfg.get("data", cfg).get("padded_size", [232, 296]))
            boxes_seq, scores_seq = model.decode_sequence(cls_seq, reg_seq, image_hw=image_hw)

            print_tensor_info("boxes_seq", boxes_seq)
            print_tensor_info("scores_seq", scores_seq)

            if boxes_seq.shape[:2] != batch_t["target_boxes_seq"].shape[:2]:
                print(
                    f"[ERROR] boxes_seq shape {tuple(boxes_seq.shape)} not aligned with "
                    f"target_boxes_seq {tuple(batch_t['target_boxes_seq'].shape)}"
                )
                ok = False
            else:
                print("[OK] boxes_seq aligns with target_boxes_seq")

    except Exception as e:
        print(f"[ERROR] model forward check failed: {repr(e)}")
        ok = False

    print("[MODEL RESULT]", "OK" if ok else "FAILED")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hsn_tianmouc.yaml")
    parser.add_argument("--data-root", default="/root/autodl-tmp/tianmouc_hsn_reproduce/data/nfs", help="Override cfg['data']['root'], e.g. data/nfs")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--npz", default=None, help="Check a specific npz file first")
    parser.add_argument("--max-files", type=int, default=1, help="How many npz files to inspect")
    parser.add_argument("--expected-cop-step", type=int, default=None, help="Expected cop_indices gap, e.g. 8 or 10")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-check", action="store_true", help="Also run model.forward_hsn_sequence and decode_sequence")
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint for model-check")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", cfg)

    data_root = Path(args.data_root or data_cfg.get("root", "./data/nfs"))
    print("=" * 100)
    print("[CONFIG]")
    print(f"config: {args.config}")
    print(f"data_root: {data_root}")
    print(f"split: {args.split}")
    print(f"expected_cop_step: {args.expected_cop_step}")
    print(f"batch_size: {args.batch_size}")

    # 1. Raw npz check.
    if args.npz:
        npz_files = [Path(args.npz)]
    else:
        split_files = find_npz_files(data_root, args.split)
        if split_files:
            npz_files = split_files[: args.max_files]
        else:
            npz_files = find_npz_files(data_root, None)[: args.max_files]

    if not npz_files:
        print(f"[ERROR] No npz files found under {data_root}")
        raise SystemExit(1)

    all_ok = True
    for p in npz_files:
        all_ok = check_npz_file(p, expected_cop_step=args.expected_cop_step) and all_ok

    # 2. Dataset and dataloader check.
    all_ok = check_dataset(
        cfg=cfg,
        split=args.split,
        data_root=str(data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    ) and all_ok

    # 3. Optional model forward check.
    if args.model_check:
        all_ok = check_model_forward(
            cfg=cfg,
            split=args.split,
            data_root=str(data_root),
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ) and all_ok

    print("\n" + "=" * 100)
    print("[FINAL RESULT]", "OK" if all_ok else "FAILED")

    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
