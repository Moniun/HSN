from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from common import (
    add_common_args,
    all_reduce_mean,
    cleanup_distributed,
    is_main_process,
    load_checkpoint,
    make_loader,
    move_batch_to_device,
    prepare_distributed,
    save_checkpoint,
    set_epoch,
    set_requires_grad,
    unwrap_model,
    wrap_ddp,
)
from hsn.losses import HSNLoss
from hsn.model import TianmoucHSN


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary, returning a new dictionary."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_stage_config(cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """
    Backward-compatible stage config.

    Existing code has one global cfg['loss'] and cfg['train']. This function lets you add:
      ann_loss / ann_train
      hsn_loss / hsn_train
    to the same yaml, then applies only the requested stage overrides.
    """
    cfg = copy.deepcopy(cfg)
    loss_key = f"{stage}_loss"
    train_key = f"{stage}_train"

    if loss_key in cfg:
        cfg["loss"] = deep_update(cfg.get("loss", {}), cfg[loss_key])
    if train_key in cfg:
        cfg["train"] = deep_update(cfg.get("train", {}), cfg[train_key])

    cfg["active_stage"] = stage
    return cfg


def set_hsn_trainable(core, cfg, epoch: int) -> bool:
    """
    HSN phase: optionally train ANN along with SNN/HU.
    """
    freeze_head_epochs = int(cfg["train"].get("freeze_head_epochs", 3))
    train_head = epoch > freeze_head_epochs
    
    train_ann = bool(cfg["train"].get("train_ann", False))
    freeze_ann_epochs = int(cfg["train"].get("freeze_ann_epochs", 0))
    train_ann = train_ann and (epoch > freeze_ann_epochs)
    
    if train_ann:
        set_requires_grad(core.ann, True)
        core.ann.train()
    else:
        set_requires_grad(core.ann, False)
        core.ann.eval()
    
    set_requires_grad(core.snn, True)
    set_requires_grad(core.feature_hu, True)
    set_requires_grad(core.head, train_head)
    
    core.snn.train()
    core.feature_hu.train()
    if train_head:
        core.head.train()
    else:
        core.head.eval()
    
    return train_head, train_ann


def compute_hsn_losses(
    out,
    target_boxes_seq: torch.Tensor,
    criterion: HSNLoss,
    cfg,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Supervise every AOP step using boxes_all-derived target_boxes_seq.

    out keys from model(mode='hsn_sequence'):
      cls_seq, reg_seq, pred_feat_seq, target_feat

    target_boxes_seq: [B, K, 4]
    """
    cls_seq = out["cls_seq"]
    reg_seq = out["reg_seq"]
    pred_feat_seq = out["pred_feat_seq"]
    target_feat = out["target_feat"]

    K = len(cls_seq)
    if K != target_boxes_seq.shape[1]:
        raise RuntimeError(
            f"K mismatch: output K={K}, target K={target_boxes_seq.shape[1]}"
        )

    cls_weight = float(cfg["loss"].get("cls_weight", 1.0))
    reg_weight = float(cfg["loss"].get("reg_weight", 1.0))
    feat_weight = float(cfg["loss"].get("feat_weight", 1.0))

    loss_cls_total = 0.0
    loss_reg_total = 0.0
    stat_pos = 0.0
    stat_neg = 0.0
    stat_max_iou_sum = 0.0

    for k in range(K):
        loss_cls_k, loss_reg_k, stat = criterion.cls_reg_loss(
            cls_seq[k],
            reg_seq[k],
            target_boxes_seq[:, k],
        )
        loss_cls_total = loss_cls_total + loss_cls_k
        loss_reg_total = loss_reg_total + loss_reg_k
        stat_pos += float(stat.get("num_pos", 0.0))
        stat_neg += float(stat.get("num_neg", 0.0))
        stat_max_iou_sum += float(stat.get("max_iou_mean", 0.0))

    loss_cls = loss_cls_total / K
    loss_reg = loss_reg_total / K

    # Only the last AOP step corresponds to the target COP teacher feature.
    loss_feat = criterion.feature_loss(pred_feat_seq[-1], target_feat)

    loss = cls_weight * loss_cls + reg_weight * loss_reg + feat_weight * loss_feat
    metrics = {
        "loss": loss,
        "loss_cls": loss_cls,
        "loss_reg": loss_reg,
        "loss_feat": loss_feat,
        "avg_pos": stat_pos / K,
        "avg_neg": stat_neg / K,
        "avg_max_iou": stat_max_iou_sum / K,
        "K": float(K),
    }
    return loss, metrics


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    cfg,
    dist_info,
    epoch: int,
    freeze_head_epochs: int,
):
    model.train()
    core = unwrap_model(model)
    train_head, train_ann = set_hsn_trainable(core, cfg, epoch)

    use_amp = bool(cfg["train"].get("amp", False))

    total_loss_sum = 0.0
    total_cls_sum = 0.0
    total_reg_sum = 0.0
    total_feat_sum = 0.0
    total_pos_sum = 0.0
    total_iou_sum = 0.0
    total_k_sum = 0.0
    total_batches = 0

    phase = "train_hsn+head" if train_head else "train_hsn_freeze_head"
    pbar = tqdm(
        loader,
        desc=phase,
        ncols=120,
        disable=not is_main_process(dist_info),
        dynamic_ncols=True,
    )

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            out = model(
                mode="hsn_sequence",
                template=batch["template"],
                template_box=batch["template_box"],
                ref=batch["ref"],
                aop=batch["aop"],
                target=batch["target"],
            )
            loss, metrics = compute_hsn_losses(
                out=out,
                target_boxes_seq=batch["target_boxes_seq"],
                criterion=criterion,
                cfg=cfg,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=10.0)
            optimizer.step()

        total_loss_sum += float(metrics["loss"].detach().cpu())
        total_cls_sum += float(metrics["loss_cls"].detach().cpu())
        total_reg_sum += float(metrics["loss_reg"].detach().cpu())
        total_feat_sum += float(metrics["loss_feat"].detach().cpu())
        total_pos_sum += float(metrics["avg_pos"])
        total_iou_sum += float(metrics["avg_max_iou"])
        total_k_sum += float(metrics["K"])
        total_batches += 1

        if is_main_process(dist_info):
            n = max(1, total_batches)
            pbar.set_postfix(
                {
                    "loss": f"{total_loss_sum / n:.4f}",
                    "cls": f"{total_cls_sum / n:.4f}",
                    "reg": f"{total_reg_sum / n:.4f}",
                    "feat": f"{total_feat_sum / n:.4f}",
                    "pos": f"{total_pos_sum / n:.1f}",
                    "maxIoU": f"{total_iou_sum / n:.3f}",
                    "K": f"{total_k_sum / n:.1f}",
                    "head": "on" if train_head else "off",
                }
            )

    num_batches = max(1, total_batches)
    metrics = {
        "loss": total_loss_sum / num_batches,
        "loss_cls": total_cls_sum / num_batches,
        "loss_reg": total_reg_sum / num_batches,
        "loss_feat": total_feat_sum / num_batches,
        "avg_pos": total_pos_sum / num_batches,
        "avg_max_iou": total_iou_sum / num_batches,
        "avg_K": total_k_sum / num_batches,
        "head_trainable": float(train_head),
    }

    for k in list(metrics.keys()):
        metrics[k] = all_reduce_mean(metrics[k], device, dist_info)

    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, cfg, dist_info):
    model.eval()

    total_loss_sum = 0.0
    total_cls_sum = 0.0
    total_reg_sum = 0.0
    total_feat_sum = 0.0
    total_pos_sum = 0.0
    total_iou_sum = 0.0
    total_k_sum = 0.0
    total_batches = 0

    pbar = tqdm(
        loader,
        desc="val_hsn",
        ncols=120,
        disable=not is_main_process(dist_info),
        dynamic_ncols=True,
    )

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        out = model(
            mode="hsn_sequence",
            template=batch["template"],
            template_box=batch["template_box"],
            ref=batch["ref"],
            aop=batch["aop"],
            target=batch["target"],
        )
        loss, metrics = compute_hsn_losses(
            out=out,
            target_boxes_seq=batch["target_boxes_seq"],
            criterion=criterion,
            cfg=cfg,
        )

        total_loss_sum += float(metrics["loss"].detach().cpu())
        total_cls_sum += float(metrics["loss_cls"].detach().cpu())
        total_reg_sum += float(metrics["loss_reg"].detach().cpu())
        total_feat_sum += float(metrics["loss_feat"].detach().cpu())
        total_pos_sum += float(metrics["avg_pos"])
        total_iou_sum += float(metrics["avg_max_iou"])
        total_k_sum += float(metrics["K"])
        total_batches += 1

    num_batches = max(1, total_batches)
    metrics = {
        "loss": total_loss_sum / num_batches,
        "loss_cls": total_cls_sum / num_batches,
        "loss_reg": total_reg_sum / num_batches,
        "loss_feat": total_feat_sum / num_batches,
        "avg_pos": total_pos_sum / num_batches,
        "avg_max_iou": total_iou_sum / num_batches,
        "avg_K": total_k_sum / num_batches,
    }

    for k in list(metrics.keys()):
        metrics[k] = all_reduce_mean(metrics[k], device, dist_info)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--ann-checkpoint", default=None)
    parser.add_argument("--freeze-head-epochs", type=int, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    args = parser.parse_args()

    cfg, device, dist_info = prepare_distributed(args)
    cfg = apply_stage_config(cfg, "hsn")

    if args.batch_size is not None:
        cfg["train"]["hsn_batch_size"] = args.batch_size

    ann_checkpoint = (
        args.ann_checkpoint
        or cfg.get("checkpoint", {}).get("ann_pretrained")
        or os.path.join(cfg["train"]["out_dir"], "ann_best.pt")
    )

    freeze_head_epochs = int(
        args.freeze_head_epochs
        if args.freeze_head_epochs is not None
        else cfg["train"].get("freeze_head_epochs", 3)
    )
    base_lr = float(cfg["train"].get("hsn_lr", cfg["train"].get("lr", 5e-5)))
    head_lr = float(
        args.head_lr
        if args.head_lr is not None
        else cfg["train"].get("head_lr", base_lr * 0.2)
    )
    ann_lr = float(cfg["train"].get("ann_lr", base_lr * 0.1))

    if is_main_process(dist_info):
        print(f"[HSN] distributed={dist_info.distributed}, world_size={dist_info.world_size}")
        print(f"[HSN] device={device}")
        print(f"[HSN] batch_size_per_gpu={cfg['train']['hsn_batch_size']}")
        print(f"[HSN] num_workers_per_process={cfg['train'].get('num_workers', 0)}")
        print(f"[HSN] out_dir={cfg['train']['out_dir']}")
        print(f"[HSN] ann_checkpoint={ann_checkpoint}")
        print(f"[HSN] freeze_head_epochs={freeze_head_epochs}")
        print(f"[HSN] base_lr={base_lr}, head_lr={head_lr}, ann_lr={ann_lr}")
        print(f"[HSN] train_ann={cfg['train'].get('train_ann', False)}")
        print("[HSN] active loss:", cfg["loss"])

    model = TianmoucHSN(cfg).to(device)
    load_checkpoint(model, ann_checkpoint, strict=False, map_location=device)

    # Start with head frozen. Each epoch will re-apply trainability.
    set_hsn_trainable(model, cfg, epoch=0)

    model = wrap_ddp(model, device, dist_info, find_unused_parameters=False)
    core = unwrap_model(model)

    train_loader = make_loader(
        cfg,
        "train",
        cfg["train"]["hsn_batch_size"],
        True,
        data_root=cfg["data"]["root"],
        mode="hsn",
        distributed=dist_info.distributed,
    )
    val_loader = make_loader(
        cfg,
        "val",
        cfg["train"]["hsn_batch_size"],
        False,
        data_root=cfg["data"]["root"],
        mode="hsn",
        distributed=dist_info.distributed,
    )

    criterion = HSNLoss(cfg, core.anchor_gen).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": core.ann.parameters(), "lr": ann_lr},
            {"params": core.snn.parameters(), "lr": base_lr},
            {"params": core.feature_hu.parameters(), "lr": base_lr},
            {"params": core.head.parameters(), "lr": head_lr},
        ],
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", False)))

    epochs = int(
        cfg["train"].get(
            "override_epochs",
            cfg["train"].get("epochs_hsn", cfg["train"].get("epochs", 10)),
        )
    )

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        set_epoch(train_loader, epoch)
        start_time = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            dist_info=dist_info,
            epoch=epoch,
            freeze_head_epochs=freeze_head_epochs,
        )
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
            dist_info=dist_info,
        )

        if is_main_process(dist_info):
            elapsed = time.time() - start_time
            print(
                f"[HSN] epoch={epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"train_pos={train_metrics['avg_pos']:.1f} | "
                f"val_pos={val_metrics['avg_pos']:.1f} | "
                f"train_maxIoU={train_metrics['avg_max_iou']:.3f} | "
                f"val_maxIoU={val_metrics['avg_max_iou']:.3f} | "
                f"train_K={train_metrics['avg_K']:.1f} | "
                f"val_K={val_metrics['avg_K']:.1f} | "
                f"head={'on' if train_metrics['head_trainable'] > 0.5 else 'off'} | "
                f"time={elapsed:.2f}s"
            )

            out_dir = cfg["train"]["out_dir"]
            os.makedirs(out_dir, exist_ok=True)
            save_checkpoint(
                os.path.join(out_dir, "hsn_last.pt"),
                model,
                optimizer,
                epoch,
                {"train": train_metrics, "val": val_metrics},
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    os.path.join(out_dir, "hsn_best.pt"),
                    model,
                    optimizer,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                )
                print(f"[best] val_loss={best_val:.6f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
