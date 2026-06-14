from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch


def _resize_chw_or_hw(arr: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if arr.ndim == 2:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3):
        chans = [cv2.resize(arr[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(arr.shape[0])]
        return np.stack(chans, axis=0)
    if arr.ndim == 3 and arr.shape[-1] in (1, 2, 3):
        chans = [cv2.resize(arr[..., c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(arr.shape[-1])]
        return np.stack(chans, axis=-1)
    raise ValueError(f'Unsupported resize shape: {arr.shape}')


def pad_chw(x: np.ndarray, padded_hw: Tuple[int, int]) -> np.ndarray:
    """Pad CHW to padded_hw centered with zeros."""
    c, h, w = x.shape
    ph, pw = padded_hw
    if h > ph or w > pw:
        raise ValueError(f'Cannot pad shape {(h,w)} to {(ph,pw)}')
    out = np.zeros((c, ph, pw), dtype=x.dtype)
    top = (ph - h) // 2
    left = (pw - w) // 2
    out[:, top:top + h, left:left + w] = x
    return out


def preprocess_cop(frame: np.ndarray, content_hw: Tuple[int, int], padded_hw: Tuple[int, int]) -> torch.Tensor:
    """COP frame -> float tensor [3,Hp,Wp] in [0,1]."""
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = np.transpose(frame, (1, 2, 0))
    frame = frame.astype(np.float32)
    if frame.max() > 1.5:
        frame = frame / 255.0
    resized = cv2.resize(frame, (content_hw[1], content_hw[0]), interpolation=cv2.INTER_LINEAR)
    chw = np.transpose(resized, (2, 0, 1))
    return torch.from_numpy(pad_chw(chw, padded_hw)).float()


def preprocess_aop(td: np.ndarray, sd: np.ndarray, content_hw: Tuple[int, int], padded_hw: Tuple[int, int]) -> torch.Tensor:
    """Single AOP timestep TD+SD -> tensor [3,Hp,Wp]."""
    td = np.asarray(td)
    sd = np.asarray(sd)
    if td.ndim == 3:
        if td.shape[0] == 1:
            td = td[0]
        elif td.shape[-1] == 1:
            td = td[..., 0]
    if sd.ndim == 3 and sd.shape[0] == 2:
        sd_chw = sd
    elif sd.ndim == 3 and sd.shape[-1] == 2:
        sd_chw = np.transpose(sd, (2, 0, 1))
    else:
        raise ValueError(f'Unsupported SD shape: {sd.shape}')
    td_r = cv2.resize(td.astype(np.float32), (content_hw[1], content_hw[0]), interpolation=cv2.INTER_LINEAR)[None]
    sd_r = _resize_chw_or_hw(sd_chw.astype(np.float32), content_hw)
    aop = np.concatenate([td_r, sd_r], axis=0)
    # robust per-sample scale, keeps polarity/sign
    scale = np.percentile(np.abs(aop), 99.0)
    if scale > 1e-6:
        aop = np.clip(aop / scale, -1.0, 1.0)
    return torch.from_numpy(pad_chw(aop.astype(np.float32), padded_hw)).float()


def scale_box_from_raw(box: np.ndarray, raw_hw: Tuple[int, int], content_hw: Tuple[int, int], padded_hw: Tuple[int, int]) -> torch.Tensor:
    """Scale raw COP xyxy box to padded preprocessed coordinates."""
    raw_h, raw_w = raw_hw
    ch, cw = content_hw
    ph, pw = padded_hw
    sx = cw / raw_w
    sy = ch / raw_h
    top = (ph - ch) // 2
    left = (pw - cw) // 2
    box = box.astype(np.float32).copy()
    box[[0, 2]] = box[[0, 2]] * sx + left
    box[[1, 3]] = box[[1, 3]] * sy + top
    box[[0, 2]] = np.clip(box[[0, 2]], 0, pw - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, ph - 1)
    return torch.from_numpy(box).float()
