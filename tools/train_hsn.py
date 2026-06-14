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


def _freeze_for_hsn(core):
    set_requires_grad(core.ann, False)
    set_requires_grad(core.snn, True)
    set_requires_grad(core.feature_hu, True)
    set_requires_grad(core.head, True)

    core.ann.eval()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    cfg,
    dist_info,
):
    model.train()

    core = unwrap_model(model)
    _freeze_for_hsn(core)

    use_amp = bool(cfg["train"].get("amp", False))
    feat_weight = float(cfg["loss"].get("feat_weight", 1.0))

    total_loss_sum = 0.0
    total_cls_sum = 0.0
    total_reg_sum = 0.0
    total_feat_sum = 0.0
    total_batches = 0

    pbar = tqdm(
        loader,
        desc="train_hsn",
        ncols=120,
        disable=not is_main_process(dist_info),
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

            cls_seq = out["cls_seq"]
            reg_seq = out["reg_seq"]
            pred_feat_seq = out["pred_feat_seq"]
            target_feat = out["target_feat"]
            target_boxes_seq = batch["target_boxes_seq"]

            K = len(cls_seq)

            if K != target_boxes_seq.shape[1]:
                raise RuntimeError(
                    f"K mismatch: output K={K}, target K={target_boxes_seq.shape[1]}"
                )

            loss_cls_total = 0.0
            loss_reg_total = 0.0

            for k in range(K):
                loss_cls_k, loss_reg_k, _ = criterion.cls_reg_loss(
                    cls_seq[k],
                    reg_seq[k],
                    target_boxes_seq[:, k],
                )

                loss_cls_total = loss_cls_total + loss_cls_k
                loss_reg_total = loss_reg_total + loss_reg_k

            loss_cls = loss_cls_total / K
            loss_reg = loss_reg_total / K

            loss_feat = criterion.feature_loss(
                pred_feat_seq[-1],
                target_feat,
            )

            loss = loss_cls + loss_reg + feat_weight * loss_feat

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss_sum += float(loss.detach().cpu())
        total_cls_sum += float(loss_cls.detach().cpu())
        total_reg_sum += float(loss_reg.detach().cpu())
        total_feat_sum += float(loss_feat.detach().cpu())
        total_batches += 1

        if is_main_process(dist_info):
            n = max(1, total_batches)

            pbar.set_postfix({
                "loss": f"{total_loss_sum / n:.4f}",
                "cls": f"{total_cls_sum / n:.4f}",
                "reg": f"{total_reg_sum / n:.4f}",
                "feat": f"{total_feat_sum / n:.4f}",
                "K": K,
            })

    num_batches = max(1, total_batches)

    metrics = {
        "loss": total_loss_sum / num_batches,
        "loss_cls": total_cls_sum / num_batches,
        "loss_reg": total_reg_sum / num_batches,
        "loss_feat": total_feat_sum / num_batches,
    }

    for k in list(metrics.keys()):
        metrics[k] = all_reduce_mean(
            metrics[k],
            device,
            dist_info,
        )

    return metrics


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    cfg,
    dist_info,
):
    model.eval()

    feat_weight = float(cfg["loss"].get("feat_weight", 1.0))

    total_loss_sum = 0.0
    total_cls_sum = 0.0
    total_reg_sum = 0.0
    total_feat_sum = 0.0
    total_batches = 0

    pbar = tqdm(
        loader,
        desc="val_hsn",
        ncols=120,
        disable=not is_main_process(dist_info),
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

        cls_seq = out["cls_seq"]
        reg_seq = out["reg_seq"]
        pred_feat_seq = out["pred_feat_seq"]
        target_feat = out["target_feat"]
        target_boxes_seq = batch["target_boxes_seq"]

        K = len(cls_seq)

        loss_cls_total = 0.0
        loss_reg_total = 0.0

        for k in range(K):
            loss_cls_k, loss_reg_k, _ = criterion.cls_reg_loss(
                cls_seq[k],
                reg_seq[k],
                target_boxes_seq[:, k],
            )

            loss_cls_total = loss_cls_total + loss_cls_k
            loss_reg_total = loss_reg_total + loss_reg_k

        loss_cls = loss_cls_total / K
        loss_reg = loss_reg_total / K

        loss_feat = criterion.feature_loss(
            pred_feat_seq[-1],
            target_feat,
        )

        loss = loss_cls + loss_reg + feat_weight * loss_feat

        total_loss_sum += float(loss.detach().cpu())
        total_cls_sum += float(loss_cls.detach().cpu())
        total_reg_sum += float(loss_reg.detach().cpu())
        total_feat_sum += float(loss_feat.detach().cpu())
        total_batches += 1

    num_batches = max(1, total_batches)

    metrics = {
        "loss": total_loss_sum / num_batches,
        "loss_cls": total_cls_sum / num_batches,
        "loss_reg": total_reg_sum / num_batches,
        "loss_feat": total_feat_sum / num_batches,
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
    parser.add_argument("--ann-checkpoint", required=True)
    args = parser.parse_args()

    cfg, device, dist_info = prepare_distributed(args)

    if args.batch_size is not None:
        cfg["train"]["hsn_batch_size"] = args.batch_size

    if is_main_process(dist_info):
        print(f"[HSN] distributed={dist_info.distributed}, world_size={dist_info.world_size}")
        print(f"[HSN] device={device}")
        print(f"[HSN] batch_size_per_gpu={cfg['train']['hsn_batch_size']}")
        print(f"[HSN] num_workers_per_process={cfg['train'].get('num_workers', 0)}")

    model = TianmoucHSN(cfg).to(device)

    load_checkpoint(
        model,
        args.ann_checkpoint,
        strict=False,
        map_location=device,
    )

    _freeze_for_hsn(model)

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

    criterion = HSNLoss(
        cfg,
        core.anchor_gen,
    ).to(device)

    params = []
    params += list(core.snn.parameters())
    params += list(core.feature_hu.parameters())
    params += list(core.head.parameters())

    optimizer = torch.optim.Adam(
        [p for p in params if p.requires_grad],
        lr=float(cfg["train"].get("lr", 1e-4)),
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
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            cfg,
            dist_info,
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            cfg,
            dist_info,
        )

        if is_main_process(dist_info):
            elapsed = time.time() - start_time

            print(
                f"[HSN] epoch={epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"time={elapsed:.2f}s"
            )

            out_dir = cfg["train"]["out_dir"]

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