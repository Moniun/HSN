from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple, Optional

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
    raise ValueError(f"Unsupported sd shapes: {sd.shape}")


def _to_boxes_m_t_4(boxes: np.ndarray) -> np.ndarray:
    if boxes.ndim == 2 and boxes.shape[-1] == 4:
        return boxes[None, ...]
    if boxes.ndim == 3 and boxes.shape[-1] == 4:
        return boxes
    raise ValueError(f"Unsupported boxes shape: {boxes.shape}")


def _valid_box(box: np.ndarray, min_size: float) -> bool:
    x1, y1, x2, y2 = box.tolist()
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def _resize_pad_img_chw(
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


def _resize_pad_map(
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


class SharedMemoryCache:
    """使用共享内存的缓存，所有worker共享同一缓存"""
    _shared_cache = {}
    _lock = None

    @classmethod
    def get(cls, key):
        return cls._shared_cache.get(key)

    @classmethod
    def put(cls, key, value, max_size=4):
        if len(cls._shared_cache) >= max_size:
            oldest = next(iter(cls._shared_cache.keys()))
            del cls._shared_cache[oldest]
        cls._shared_cache[key] = value

    @classmethod
    def __len__(cls):
        return len(cls._shared_cache)

    @classmethod
    def clear(cls):
        cls._shared_cache.clear()


class TianmoucHSNDataset(Dataset):
    def __init__(self, cfg: Dict, split: str = "train", data_root: str | None = None):
        data_cfg = cfg.get("data", cfg)

        root = Path(data_root or data_cfg.get("root", "./data/nfs"))
        split_name = data_cfg.get("train_split", "train") if split == "train" else data_cfg.get("val_split", "val")
        self.root = root / split_name

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")

        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.root}")

        self.raw_cop_size = tuple(data_cfg.get("raw_cop_size", [320, 640]))
        self.content_size = tuple(data_cfg.get("content_size", [128, 256]))
        self.padded_size = tuple(data_cfg.get("padded_size", [232, 296]))
        self.min_box_size = float(data_cfg.get("min_box_size", 4))

        self.cache_size = int(data_cfg.get("dataset_cache_size", 4))

        self.samples = []

        self._build_index()

    def _load_npz(self, path: Path) -> dict:
        d = np.load(path, allow_pickle=True)

        cop = _to_hwc_cop(d["cop"])
        td = _to_td_t_hw(d["td"])
        sd = _to_sd_t_hw2(d["sd"])
        boxes = _to_boxes_m_t_4(d["boxes"]).astype(np.float32)
        boxes_all = _to_boxes_m_t_4(d["boxes_all"]).astype(np.float32)
        cop_indices = d["cop_indices"].astype(np.int64)

        return {
            "cop": cop,
            "td": td,
            "sd": sd,
            "boxes": boxes,
            "boxes_all": boxes_all,
            "cop_indices": cop_indices,
        }

    def _get_data(self, file_id: int, path: Path) -> dict:
        key = str(path)
        data = SharedMemoryCache.get(key)
        if data is None:
            data = self._load_npz(path)
            SharedMemoryCache.put(key, data, max_size=self.cache_size)
        return data

    def _build_index(self):
        for file_id, path in enumerate(self.files):
            try:
                d = np.load(path, allow_pickle=True)

                cop = _to_hwc_cop(d["cop"])
                td = _to_td_t_hw(d["td"])
                sd = _to_sd_t_hw2(d["sd"])
                boxes = _to_boxes_m_t_4(d["boxes"]).astype(np.float32)
                boxes_all = _to_boxes_m_t_4(d["boxes_all"]).astype(np.float32)
                cop_indices = d["cop_indices"].astype(np.int64)

            except Exception as e:
                print(f"[skip] {path.name}: {e}")
                continue

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

                    high_boxes = boxes_all[obj_id, a0:a1]
                    if not all(_valid_box(b, self.min_box_size) for b in high_boxes):
                        continue

                    self.samples.append({
                        "file_id": file_id,
                        "obj_id": obj_id,
                        "template_t": template_t,
                        "ref_t": ref_t,
                        "target_t": target_t,
                        "aop_start": a0,
                        "aop_end": a1,
                    })

        if not self.samples:
            raise RuntimeError("No valid tracking samples found.")

        print(f"[dataset] files={len(self.files)}, samples={len(self.samples)}, cache_size={self.cache_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        file_id = s["file_id"]
        path = self.files[file_id]

        data = self._get_data(file_id, path)

        cop = data["cop"]
        td = data["td"]
        sd = data["sd"]
        boxes = data["boxes"]
        boxes_all = data["boxes_all"]

        obj_id = s["obj_id"]
        template_t = s["template_t"]
        ref_t = s["ref_t"]
        target_t = s["target_t"]
        a0 = s["aop_start"]
        a1 = s["aop_end"]

        template = _resize_pad_img_chw(cop[template_t], self.content_size, self.padded_size)
        ref = _resize_pad_img_chw(cop[ref_t], self.content_size, self.padded_size)
        target = _resize_pad_img_chw(cop[target_t], self.content_size, self.padded_size)

        td_win = td[a0:a1]
        sd_win = sd[a0:a1]

        aop_frames = []
        for k in range(len(td_win)):
            td_k = _resize_pad_map(td_win[k], self.content_size, self.padded_size)
            sd0 = _resize_pad_map(sd_win[k, :, :, 0], self.content_size, self.padded_size)
            sd1 = _resize_pad_map(sd_win[k, :, :, 1], self.content_size, self.padded_size)
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
        target_boxes_seq = torch.stack([
            _scale_box_to_padded(
                b,
                self.raw_cop_size,
                self.content_size,
                self.padded_size,
            )
            for b in target_boxes_seq_raw
        ], dim=0)

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
