from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from tools.common import (
    prepare,
    make_loader,
    save_checkpoint,
    load_checkpoint,
    set_requires_grad,
)
from hsn.losses import HSNLoss


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()

    # HSN phase: ANN acts as stable teacher by default.
    model.ann.eval()
    set_requires_grad(model.ann, False)
    set_requires_grad(model.snn, True)
    set_requires_grad(model.feature_hu, True)
    set_requires_grad(model.head, True)

    use_amp = bool(cfg["train"].get("amp", False))
    feat_weight = float(cfg["loss"].get("feat_weight", 1.0))

    total_loss_sum = 0.0
    total_cls_sum = 0.0
    total_reg_sum = 0.0
    total_feat_sum = 0.0

    pbar = tqdm(loader, desc="train_hsn", ncols=120)

    for batch in pbar:
        template = batch["template"].to(device, non_blocking=True)
        ref = batch["ref"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        aop = batch["aop"].to(device, non_blocking=True)

        # [B, K, 4]
        target_boxes_seq = batch["target_boxes_seq"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            out = model.forward_hsn_sequence(
                template=template,
                template_box=batch["template_box"].to(device, non_blocking=True),
                ref=ref,
                aop=aop,
                target=target,
            )

            cls_seq = out["cls_seq"]
            reg_seq = out["reg_seq"]
            pred_feat_seq = out["pred_feat_seq"]
            target_feat = out["target_feat"]

            K = len(cls_seq)
            if K != target_boxes_seq.shape[1]:
                raise RuntimeError(
                    f"K mismatch: model outputs {K} steps, "
                    f"target_boxes_seq has {target_boxes_seq.shape[1]} steps"
                )

            loss_cls_total = 0.0
            loss_reg_total = 0.0

            for k in range(K):
                loss_cls_k, loss_reg_k = criterion.cls_reg(
                    cls_seq[k],
                    reg_seq[k],
                    target_boxes_seq[:, k],
                    model.anchor_gen,
                )
                loss_cls_total = loss_cls_total + loss_cls_k
                loss_reg_total = loss_reg_total + loss_reg_k

            loss_cls = loss_cls_total / K
            loss_reg = loss_reg_total / K

            # Feature alignment is only applied to the final step, because target is next COP.
            # If you later save high-frequency RGB teacher frames, this can be extended per step.
            loss_feat = criterion.feature_align(pred_feat_seq[-1], target_feat)

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

        n = max(1, pbar.n + 1)
        pbar.set_postfix({
            "loss": total_loss_sum / n,
            "cls": total_cls_sum / n,
            "reg": total_reg_sum / n,
            "feat": total_feat_sum / n,
            "K": K,
        })

    num_batches = max(1, len(loader))

    return {
        "loss": total_loss_sum / num_batches,
        "loss_cls": total_cls_sum / num_batches,
        "loss_reg": total_reg_sum / num_batches,
        "loss_feat": total_feat_sum / num_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()

    feat_weight = float(cfg["loss"].get("feat_weight", 1.0))

    total_loss_sum = 0.0
    total_cls_sum = 0.0
    total_reg_sum = 0.0
    total_feat_sum = 0.0

    pbar = tqdm(loader, desc="val_hsn", ncols=120)

    for batch in pbar:
        template = batch["template"].to(device, non_blocking=True)
        ref = batch["ref"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        aop = batch["aop"].to(device, non_blocking=True)
        target_boxes_seq = batch["target_boxes_seq"].to(device, non_blocking=True)

        out = model.forward_hsn_sequence(
            template=template,
            template_box=batch["template_box"].to(device, non_blocking=True),
            ref=ref,
            aop=aop,
            target=target,
        )

        cls_seq = out["cls_seq"]
        reg_seq = out["reg_seq"]
        pred_feat_seq = out["pred_feat_seq"]
        target_feat = out["target_feat"]

        K = len(cls_seq)

        loss_cls_total = 0.0
        loss_reg_total = 0.0

        for k in range(K):
            loss_cls_k, loss_reg_k = criterion.cls_reg(
                cls_seq[k],
                reg_seq[k],
                target_boxes_seq[:, k],
                model.anchor_gen,
            )
            loss_cls_total = loss_cls_total + loss_cls_k
            loss_reg_total = loss_reg_total + loss_reg_k

        loss_cls = loss_cls_total / K
        loss_reg = loss_reg_total / K
        loss_feat = criterion.feature_align(pred_feat_seq[-1], target_feat)

        loss = loss_cls + loss_reg + feat_weight * loss_feat

        total_loss_sum += float(loss.detach().cpu())
        total_cls_sum += float(loss_cls.detach().cpu())
        total_reg_sum += float(loss_reg.detach().cpu())
        total_feat_sum += float(loss_feat.detach().cpu())

        n = max(1, pbar.n + 1)
        pbar.set_postfix({
            "loss": total_loss_sum / n,
            "cls": total_cls_sum / n,
            "reg": total_reg_sum / n,
            "feat": total_feat_sum / n,
            "K": K,
        })

    num_batches = max(1, len(loader))

    return {
        "loss": total_loss_sum / num_batches,
        "loss_cls": total_cls_sum / num_batches,
        "loss_reg": total_reg_sum / num_batches,
        "loss_feat": total_feat_sum / num_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hsn_tianmouc.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ann-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg, model, device = prepare(args.config)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    train_loader = make_loader(cfg, "train", data_root=args.data_root)
    val_loader = make_loader(cfg, "val", data_root=args.data_root)

    load_checkpoint(model, args.ann_checkpoint, strict=False, map_location=device)

    # Freeze ANN teacher by default.
    model.ann.eval()
    set_requires_grad(model.ann, False)
    set_requires_grad(model.snn, True)
    set_requires_grad(model.feature_hu, True)
    set_requires_grad(model.head, True)

    criterion = HSNLoss(cfg).to(device)

    params = []
    params += list(model.snn.parameters())
    params += list(model.feature_hu.parameters())
    params += list(model.head.parameters())

    optimizer = torch.optim.Adam(
        [p for p in params if p.requires_grad],
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", False)))

    out_dir = Path(cfg["train"].get("out_dir", "./runs/tianmouc_hsn_reproduce"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(int(cfg["train"].get("epochs", 10))):
        print(f"\n[Epoch {epoch + 1}]")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            cfg,
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            cfg,
        )

        print("[train]", train_metrics)
        print("[val]  ", val_metrics)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            out_dir / "hsn_last.pt",
            metrics={"train": train_metrics, "val": val_metrics},
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                out_dir / "hsn_best.pt",
                metrics={"train": train_metrics, "val": val_metrics},
            )
            print(f"[best] val loss = {best_val:.6f}")


if __name__ == "__main__":
    main()
