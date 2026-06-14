from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hsn.data import TianmoucHSNDataset
from hsn.losses import HSNLoss
from hsn.model import TianmoucHSN
from hsn.utils import batch_to_device, ensure_dir, load_config, seed_everything, select_device


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--config', default='configs/hsn_tianmouc.yaml')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', default=None)


def prepare(args):
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg['data']['root'] = args.data_root
    if args.epochs is not None:
        cfg['train']['override_epochs'] = args.epochs
    if args.device is not None:
        cfg['train']['device'] = args.device
    seed_everything(int(cfg.get('seed', 0)))
    device = select_device(cfg['train'].get('device', 'cuda'))
    ensure_dir(cfg['train']['out_dir'])
    return cfg, device


def make_loader(cfg, split: str, batch_size: int, shuffle: bool):
    ds = TianmoucHSNDataset(cfg, split=split, data_root=cfg['data']['root'])

    num_workers = int(cfg['train'].get('num_workers', 0))
    persistent_workers = bool(cfg['train'].get('persistent_workers', False)) and num_workers > 0
    prefetch_factor = int(cfg['train'].get('prefetch_factor', 2)) if num_workers > 0 else None

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': bool(cfg['train'].get('pin_memory', True)),
        'drop_last': shuffle,
        'persistent_workers': persistent_workers,
    }

    if prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(ds, **loader_kwargs)


def save_checkpoint(path: str, model: nn.Module, optimizer=None, epoch: int = 0, metrics=None):
    payload = {
        'model': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics or {},
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str, strict: bool = True, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=strict)
    return ckpt


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
