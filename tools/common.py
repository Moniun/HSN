from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from hsn.data import TianmoucHSNDataset
from hsn.utils import ensure_dir, load_config, seed_everything, select_device


@dataclass
class DistInfo:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/hsn_tianmouc.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-auto-ddp", action="store_true")


def _auto_launch_ddp_if_needed(args) -> None:
    if getattr(args, "no_auto_ddp", False):
        return

    if os.environ.get("LOCAL_RANK") is not None:
        return

    if os.environ.get("HSN_DDP_LAUNCHED") == "1":
        return

    if not torch.cuda.is_available():
        return

    n_gpu = torch.cuda.device_count()

    if n_gpu <= 1:
        return

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={n_gpu}",
        sys.argv[0],
        *sys.argv[1:],
    ]

    env = os.environ.copy()
    env["HSN_DDP_LAUNCHED"] = "1"

    print(f"[auto-ddp] Detected {n_gpu} GPUs. Relaunching with torchrun:")
    print("[auto-ddp]", " ".join(cmd), flush=True)

    code = subprocess.call(cmd, env=env)
    raise SystemExit(code)


def _setup_distributed() -> DistInfo:
    if "LOCAL_RANK" not in os.environ:
        return DistInfo(False, rank=0, local_rank=0, world_size=1)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
        )

    return DistInfo(
        distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def prepare_distributed(args):
    _auto_launch_ddp_if_needed(args)

    cfg = load_config(args.config)

    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root

    if args.epochs is not None:
        cfg["train"]["override_epochs"] = args.epochs

    if args.device is not None:
        cfg["train"]["device"] = args.device

    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers

    seed_everything(int(cfg.get("seed", 0)))

    dist_info = _setup_distributed()

    if dist_info.distributed:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{dist_info.local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = select_device(cfg["train"].get("device", "cuda"))

    ensure_dir(cfg["train"]["out_dir"])

    return cfg, device, dist_info


def prepare(args):
    cfg = load_config(args.config)

    if getattr(args, "data_root", None) is not None:
        cfg["data"]["root"] = args.data_root

    if getattr(args, "epochs", None) is not None:
        cfg["train"]["override_epochs"] = args.epochs

    if getattr(args, "device", None) is not None:
        cfg["train"]["device"] = args.device

    seed_everything(int(cfg.get("seed", 0)))

    device = select_device(cfg["train"].get("device", "cuda"))
    ensure_dir(cfg["train"]["out_dir"])

    return cfg, device


def is_main_process(dist_info: Optional[DistInfo] = None) -> bool:
    if dist_info is None:
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    return dist_info.rank == 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def wrap_ddp(
    model: nn.Module,
    device: torch.device,
    dist_info: DistInfo,
    find_unused_parameters: bool = False,
) -> nn.Module:
    if not dist_info.distributed:
        return model

    if device.type == "cuda":
        return DDP(
            model,
            device_ids=[dist_info.local_rank],
            output_device=dist_info.local_rank,
            find_unused_parameters=find_unused_parameters,
        )

    return DDP(
        model,
        find_unused_parameters=find_unused_parameters,
    )


def make_loader(
    cfg: Dict[str, Any],
    split: str,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    data_root: Optional[str] = None,
    mode: str = "hsn",
    distributed: bool = False,
):
    ds = TianmoucHSNDataset(
        cfg,
        split=split,
        data_root=data_root or cfg["data"]["root"],
        mode=mode,
    )

    if batch_size is None:
        if mode == "ann":
            batch_size = int(cfg["train"]["ann_batch_size"])
        else:
            batch_size = int(cfg["train"]["hsn_batch_size"])

    if mode == "hsn" and bool(cfg["train"].get("hsn_sequential", True)):
        shuffle = False

    sampler = None

    if distributed:
        sampler = DistributedSampler(
            ds,
            shuffle=shuffle,
            drop_last=shuffle,
        )
        shuffle = False

    num_workers = int(cfg["train"].get("num_workers", 0))
    pin_memory = bool(cfg["train"].get("pin_memory", True))
    persistent_workers = bool(cfg["train"].get("persistent_workers", False)) and num_workers > 0

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": shuffle,
        "persistent_workers": persistent_workers,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))

    return DataLoader(ds, **loader_kwargs)


def set_epoch(loader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)

    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def move_batch_to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def all_reduce_mean(value: float, device: torch.device, dist_info: DistInfo) -> float:
    t = torch.tensor(value, dtype=torch.float32, device=device)

    if dist_info.distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist_info.world_size

    return float(t.detach().cpu())


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict

    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}

    return state_dict


def save_checkpoint(
    path,
    model: nn.Module,
    optimizer=None,
    epoch: int = 0,
    metrics=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    path,
    strict: bool = True,
    map_location="cpu",
):
    ckpt = torch.load(path, map_location=map_location)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    state = _strip_module_prefix(state)

    unwrap_model(model).load_state_dict(
        state,
        strict=strict,
    )

    return ckpt