from __future__ import annotations

import argparse
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from common import (
    add_common_args,
    all_reduce_mean,
    cleanup_distributed,
    is_main_process,
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


def run_epoch(
    model,
    loss_fn,
    loader,
    optimizer,
    scaler,
    device,
    cfg,
    dist_info,
    train: bool,
):
    model.train(train)

    use_amp = bool(cfg["train"].get("amp", False))

    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_batches = 0

    pbar = tqdm(
        loader,
        desc="train_ann" if train else "val_ann",
        leave=False,
        ncols=120,
        disable=not is_main_process(dist_info),
    )

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(train):
            with autocast(enabled=use_amp):
                out = model(
                    mode="ann",
                    template=batch["template"],
                    template_box=batch["template_box"],
                    search=batch["target"],
                )

                loss_cls, loss_reg, _ = loss_fn.cls_reg_loss(
                    out["cls"],
                    out["reg"],
                    batch["target_box"],
                )

                loss_feat_reg = out["search_feat"].pow(2).mean()

                loss = (
                    loss_fn.loss_cfg["cls_weight"] * loss_cls
                    + loss_fn.loss_cfg["reg_weight"] * loss_reg
                    + loss_fn.loss_cfg.get("reg_feature_weight", 0.0) * loss_feat_reg
                )

            if train:
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_cls_loss += float(loss_cls.detach().cpu())
        total_reg_loss += float(loss_reg.detach().cpu())
        total_batches += 1

        if is_main_process(dist_info):
            n = max(1, total_batches)

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / n:.4f}",
                    "cls": f"{total_cls_loss / n:.4f}",
                    "reg": f"{total_reg_loss / n:.4f}",
                }
            )

    num_batches = max(1, total_batches)

    metrics = {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "reg_loss": total_reg_loss / num_batches,
    }

    for k in list(metrics.keys()):
        metrics[k] = all_reduce_mean(
            metrics[k],
            device,
            dist_info,
        )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    cfg, device, dist_info = prepare_distributed(args)

    if args.batch_size is not None:
        cfg["train"]["ann_batch_size"] = args.batch_size

    if is_main_process(dist_info):
        print(f"[ANN] distributed={dist_info.distributed}, world_size={dist_info.world_size}")
        print(f"[ANN] device={device}")
        print(f"[ANN] batch_size_per_gpu={cfg['train']['ann_batch_size']}")
        print(f"[ANN] num_workers_per_process={cfg['train'].get('num_workers', 0)}")

    model = TianmoucHSN(cfg).to(device)

    set_requires_grad(model.snn, False)
    set_requires_grad(model.feature_hu, False)
    set_requires_grad(model.ann, True)
    set_requires_grad(model.head, True)

    model = wrap_ddp(
        model,
        device,
        dist_info,
        find_unused_parameters=False,
    )

    core = unwrap_model(model)

    train_loader = make_loader(
        cfg,
        "train",
        cfg["train"]["ann_batch_size"],
        True,
        data_root=cfg["data"]["root"],
        mode="ann",
        distributed=dist_info.distributed,
    )

    val_loader = make_loader(
        cfg,
        "val",
        cfg["train"]["ann_batch_size"],
        False,
        data_root=cfg["data"]["root"],
        mode="ann",
        distributed=dist_info.distributed,
    )

    loss_fn = HSNLoss(
        cfg,
        core.anchor_gen,
    ).to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", False)))

    epochs = int(
        cfg["train"].get(
            "override_epochs",
            cfg["train"].get("epochs_ann", cfg["train"].get("epochs", 10)),
        )
    )

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        set_epoch(train_loader, epoch)

        start_time = time.time()

        train_metrics = run_epoch(
            model,
            loss_fn,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            dist_info,
            train=True,
        )

        val_metrics = run_epoch(
            model,
            loss_fn,
            val_loader,
            optimizer,
            scaler,
            device,
            cfg,
            dist_info,
            train=False,
        )

        if is_main_process(dist_info):
            elapsed = time.time() - start_time

            print(
                f"[ANN] epoch={epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"time={elapsed:.2f}s"
            )

            out_dir = cfg["train"]["out_dir"]

            save_checkpoint(
                os.path.join(out_dir, "ann_last.pt"),
                model,
                optimizer,
                epoch,
                {"train": train_metrics, "val": val_metrics},
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]

                save_checkpoint(
                    os.path.join(out_dir, "ann_best.pt"),
                    model,
                    optimizer,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                )

                print(f"[best] val_loss={best_val:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
