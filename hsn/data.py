from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _to_hwc_cop(cop: np.ndarray) -> np.ndarray:
    if cop.ndim == 4 and cop.shape[-1] in (1, 3):
        return cop
    if cop.ndim == 4 and cop.shape[1] in (1, 3):
        return np.transpose(cop, (0, 2, 3, 1))
    raise ValueError(f"Unsupported cop shape: {cop.shape}")


def _to_td_t_hw(td: np.ndarray) -> np.ndarray:
    if td.ndim == 3:
        return td
    if td.ndim == 4 and td.shape[1] == 1:
        return td[:, 0]
    if td.ndim == 4 and td.shape[-1] == 1:
        return td[..., 0]
    raise ValueError(f"Unsupported td shape: {td.shape}")


def _to_sd_t_hw2(sd: np.ndarray) -> np.ndarray:
    if sd.ndim == 4 and sd.shape[-1] == 2:
        return sd
    if sd.ndim == 4 and sd.shape[1] == 2:
        return np.transpose(sd, (0, 2, 3, 1))
    raise ValueError(f"Unsupported sd shape: {sd.shape}")


def _to_boxes_m_t_4(boxes: np.ndarray) -> np.ndarray:
    if boxes.ndim == 2 and boxes.shape[-1] == 4:
        return boxes[None, ...]
    if boxes.ndim == 3 and boxes.shape[-1] == 4:
        return boxes
    raise ValueError(f"Unsupported boxes shape: {boxes.shape}")


def _valid_box(box: np.ndarray, min_size: float) -> bool:
    box = box.astype(np.float32)

    if not np.isfinite(box).all():
        return False

    x1, y1, x2, y2 = box.tolist()

    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def _resize_pad_img_chw(
    img_hwc: np.ndarray,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> torch.Tensor:
    ch, cw = content_size
    ph, pw = padded_size

    img = cv2.resize(
        img_hwc,
        (cw, ch),
        interpolation=cv2.INTER_AREA,
    )

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


def _resize_pad_map(
    m: np.ndarray,
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> np.ndarray:
    ch, cw = content_size
    ph, pw = padded_size

    m = cv2.resize(
        m.astype(np.float32),
        (cw, ch),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((ph, pw), dtype=np.float32)

    top = (ph - ch) // 2
    left = (pw - cw) // 2

    canvas[top:top + ch, left:left + cw] = m

    return canvas


def _scale_box_to_padded(
    box: np.ndarray,
    raw_size: Tuple[int, int],
    content_size: Tuple[int, int],
    padded_size: Tuple[int, int],
) -> torch.Tensor:
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

    return torch.from_numpy(b).float()


class TianmoucHSNDataset(Dataset):
    """
    mode='ann':
        ANN/COP-only training.
        Only reads cop + boxes.

    mode='hsn':
        HSN high-frequency training.
        Reads cop + td + sd + boxes + boxes_all + cop_indices.
    """

    def __init__(
        self,
        cfg: Union[Dict[str, Any], str, Path],
        split: str = "train",
        data_root: Optional[Union[str, Path, Dict[str, Any]]] = None,
        mode: str = "hsn",
    ):
        if mode not in ("ann", "hsn"):
            raise ValueError(f"mode must be 'ann' or 'hsn', got {mode}")

        self.mode = mode
        self.samples = []
        self._cache: OrderedDict[str, Any] = OrderedDict()

        if isinstance(cfg, (str, Path)):
            legacy_root = Path(cfg)
            legacy_split_name = split

            if not isinstance(data_root, dict):
                raise TypeError(
                    "Legacy usage requires TianmoucHSNDataset(root, split_name, cfg)"
                )

            self.cfg = data_root
            data_cfg = self.cfg.get("data", self.cfg)
            self.root = legacy_root / legacy_split_name

        else:
            self.cfg = cfg
            data_cfg = self.cfg.get("data", self.cfg)

            root = Path(data_root or data_cfg.get("root", "./data/nfs"))

            if split in ("train", "training"):
                split_name = data_cfg.get("train_split", "train")
            elif split in ("val", "valid", "validation", "test"):
                split_name = data_cfg.get("val_split", "val")
            else:
                split_name = split

            self.root = root / split_name

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")

        self.files = sorted(self.root.glob("*.npz"))

        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.root}")

        self.raw_cop_size = tuple(data_cfg.get("raw_cop_size", [320, 640]))
        self.raw_aop_size = tuple(data_cfg.get("raw_aop_size", [160, 160]))
        self.content_size = tuple(data_cfg.get("content_size", [128, 256]))
        self.padded_size = tuple(data_cfg.get("padded_size", [232, 296]))
        self.min_box_size = float(data_cfg.get("min_box_size", 4))

        self.fixed_aop_steps = data_cfg.get("fixed_aop_steps", None)

        if self.fixed_aop_steps is not None:
            self.fixed_aop_steps = int(self.fixed_aop_steps)

        if self.mode == "ann":
            default_cache_files = int(data_cfg.get("ann_cache_files", 2))
        else:
            default_cache_files = int(data_cfg.get("hsn_cache_files", 0))

        self.cache_files = int(data_cfg.get("cache_files", default_cache_files))

        self._build_index()

    def _cache_get(self, path: Path, load_fn):
        if self.cache_files <= 0:
            return load_fn(path)

        key = str(path)

        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

        value = load_fn(path)
        self._cache[key] = value

        while len(self._cache) > self.cache_files:
            self._cache.popitem(last=False)

        return value

    def _load_ann_npz_uncached(self, path: Path):
        d = np.load(path, allow_pickle=True)

        cop = _to_hwc_cop(d["cop"])
        boxes = _to_boxes_m_t_4(d["boxes"]).astype(np.float32)

        if boxes.shape[1] != len(cop):
            raise ValueError(
                f"boxes shape {boxes.shape} does not match cop length {len(cop)} in {path}"
            )

        return cop, boxes

    def _load_ann_npz(self, path: Path):
        return self._cache_get(path, self._load_ann_npz_uncached)

    def _load_hsn_npz_uncached(self, path: Path):
        d = np.load(path, allow_pickle=True)

        cop = _to_hwc_cop(d["cop"])
        td = _to_td_t_hw(d["td"])
        sd = _to_sd_t_hw2(d["sd"])
        boxes = _to_boxes_m_t_4(d["boxes"]).astype(np.float32)

        if "boxes_all" not in d.files:
            raise KeyError(
                f"{path} does not contain boxes_all. "
                "Please regenerate data with updated simulate_tianmouc.py."
            )

        boxes_all = _to_boxes_m_t_4(d["boxes_all"]).astype(np.float32)

        if "cop_indices" not in d.files:
            raise KeyError(
                f"{path} does not contain cop_indices. "
                "Please regenerate data with --cop-step."
            )

        cop_indices = d["cop_indices"].astype(np.int64)

        if len(cop) != len(cop_indices):
            raise ValueError(f"len(cop) != len(cop_indices) in {path}")

        if boxes.shape[1] != len(cop):
            raise ValueError(
                f"boxes shape {boxes.shape} does not match cop length {len(cop)} in {path}"
            )

        if boxes_all.shape[1] != len(td):
            raise ValueError(
                f"boxes_all length {boxes_all.shape[1]} does not match td length {len(td)} in {path}"
            )

        if len(sd) != len(td):
            raise ValueError(
                f"sd length {len(sd)} does not match td length {len(td)} in {path}"
            )

        if cop_indices.size > 1 and not np.all(np.diff(cop_indices) > 0):
            raise ValueError(f"cop_indices must be strictly increasing in {path}")

        return cop, td, sd, boxes, boxes_all, cop_indices

    def _load_hsn_npz(self, path: Path):
        return self._cache_get(path, self._load_hsn_npz_uncached)

    def _build_index(self):
        for file_id, path in enumerate(self.files):
            if self.mode == "ann":
                self._build_ann_index_for_file(file_id, path)
            else:
                self._build_hsn_index_for_file(file_id, path)

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in {self.root} with mode={self.mode}"
            )

        print(
            f"[dataset] mode={self.mode}, files={len(self.files)}, "
            f"samples={len(self.samples)}, root={self.root}"
        )

    def _build_ann_index_for_file(self, file_id: int, path: Path):
        try:
            cop, boxes = self._load_ann_npz(path)
        except Exception as e:
            print(f"[skip][ann] {path.name}: {e}")
            return

        num_objects, num_cop, _ = boxes.shape

        for obj_id in range(num_objects):
            valid_times = [
                t for t in range(num_cop)
                if _valid_box(boxes[obj_id, t], self.min_box_size)
            ]

            if len(valid_times) < 2:
                continue

            template_t = valid_times[0]

            for target_t in valid_times[1:]:
                self.samples.append(
                    {
                        "file_id": file_id,
                        "obj_id": obj_id,
                        "template_t": template_t,
                        "target_t": target_t,
                    }
                )

    def _build_hsn_index_for_file(self, file_id: int, path: Path):
        try:
            cop, td, sd, boxes, boxes_all, cop_indices = self._load_hsn_npz(path)
        except Exception as e:
            print(f"[skip][hsn] {path.name}: {e}")
            return

        num_objects, num_cop, _ = boxes.shape

        for obj_id in range(num_objects):
            valid_times = [
                t for t in range(num_cop)
                if _valid_box(boxes[obj_id, t], self.min_box_size)
            ]

            if len(valid_times) < 2:
                continue

            template_t = valid_times[0]

            for ref_t in range(template_t, num_cop - 1):
                target_t = ref_t + 1

                if not _valid_box(boxes[obj_id, ref_t], self.min_box_size):
                    continue

                if not _valid_box(boxes[obj_id, target_t], self.min_box_size):
                    continue

                a0 = int(cop_indices[ref_t]) + 1
                a1 = int(cop_indices[target_t]) + 1

                if a1 <= a0:
                    continue

                if a1 > len(td) or a1 > len(sd) or a1 > boxes_all.shape[1]:
                    continue

                K = a1 - a0

                if self.fixed_aop_steps is not None and K != self.fixed_aop_steps:
                    continue

                high_boxes = boxes_all[obj_id, a0:a1]

                if not all(_valid_box(b, self.min_box_size) for b in high_boxes):
                    continue

                self.samples.append(
                    {
                        "file_id": file_id,
                        "obj_id": obj_id,
                        "template_t": template_t,
                        "ref_t": ref_t,
                        "target_t": target_t,
                        "aop_start": a0,
                        "aop_end": a1,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.mode == "ann":
            return self._getitem_ann(idx)

        return self._getitem_hsn(idx)

    def _getitem_ann(self, idx: int):
        s = self.samples[idx]
        path = self.files[s["file_id"]]

        cop, boxes = self._load_ann_npz(path)

        obj_id = s["obj_id"]
        template_t = s["template_t"]
        target_t = s["target_t"]

        template = _resize_pad_img_chw(
            cop[template_t],
            self.content_size,
            self.padded_size,
        )

        target = _resize_pad_img_chw(
            cop[target_t],
            self.content_size,
            self.padded_size,
        )

        template_box = _scale_box_to_padded(
            boxes[obj_id, template_t],
            self.raw_cop_size,
            self.content_size,
            self.padded_size,
        )

        target_box = _scale_box_to_padded(
            boxes[obj_id, target_t],
            self.raw_cop_size,
            self.content_size,
            self.padded_size,
        )

        return {
            "template": template,
            "target": target,
            "search": target,
            "template_box": template_box,
            "target_box": target_box,
            "search_box": target_box,
            "seq_name": path.stem,
            "obj_id": torch.tensor(obj_id, dtype=torch.long),
            "template_t": torch.tensor(template_t, dtype=torch.long),
            "target_t": torch.tensor(target_t, dtype=torch.long),
            "search_t": torch.tensor(target_t, dtype=torch.long),
        }

    def _getitem_hsn(self, idx: int):
        s = self.samples[idx]
        path = self.files[s["file_id"]]

        cop, td, sd, boxes, boxes_all, cop_indices = self._load_hsn_npz(path)

        obj_id = s["obj_id"]
        template_t = s["template_t"]
        ref_t = s["ref_t"]
        target_t = s["target_t"]
        a0 = s["aop_start"]
        a1 = s["aop_end"]

        template = _resize_pad_img_chw(
            cop[template_t],
            self.content_size,
            self.padded_size,
        )

        ref = _resize_pad_img_chw(
            cop[ref_t],
            self.content_size,
            self.padded_size,
        )

        target = _resize_pad_img_chw(
            cop[target_t],
            self.content_size,
            self.padded_size,
        )

        td_win = td[a0:a1]
        sd_win = sd[a0:a1]

        aop_frames = []

        for k in range(len(td_win)):
            td_k = _resize_pad_map(
                td_win[k],
                self.content_size,
                self.padded_size,
            )

            sd0 = _resize_pad_map(
                sd_win[k, :, :, 0],
                self.content_size,
                self.padded_size,
            )

            sd1 = _resize_pad_map(
                sd_win[k, :, :, 1],
                self.content_size,
                self.padded_size,
            )

            aop_frames.append(np.stack([td_k, sd0, sd1], axis=0))

        aop = torch.from_numpy(np.stack(aop_frames, axis=0)).float()

        template_box = _scale_box_to_padded(
            boxes[obj_id, template_t],
            self.raw_cop_size,
            self.content_size,
            self.padded_size,
        )

        ref_box = _scale_box_to_padded(
            boxes[obj_id, ref_t],
            self.raw_cop_size,
            self.content_size,
            self.padded_size,
        )

        target_box = _scale_box_to_padded(
            boxes[obj_id, target_t],
            self.raw_cop_size,
            self.content_size,
            self.padded_size,
        )

        target_boxes_seq_raw = boxes_all[obj_id, a0:a1]

        target_boxes_seq = torch.stack(
            [
                _scale_box_to_padded(
                    b,
                    self.raw_cop_size,
                    self.content_size,
                    self.padded_size,
                )
                for b in target_boxes_seq_raw
            ],
            dim=0,
        )

        return {
            "template": template,
            "ref": ref,
            "target": target,
            "aop": aop,
            "template_box": template_box,
            "ref_box": ref_box,
            "target_box": target_box,
            "target_boxes_seq": target_boxes_seq,
            "aop_frame_indices": torch.arange(a0, a1, dtype=torch.long),
            "seq_name": path.stem,
            "obj_id": torch.tensor(obj_id, dtype=torch.long),
            "template_t": torch.tensor(template_t, dtype=torch.long),
            "ref_t": torch.tensor(ref_t, dtype=torch.long),
            "target_t": torch.tensor(target_t, dtype=torch.long),
            "aop_start": torch.tensor(a0, dtype=torch.long),
            "aop_end": torch.tensor(a1, dtype=torch.long),
        }