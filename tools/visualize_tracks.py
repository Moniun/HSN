from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.common import make_loader, move_batch_to_device, load_checkpoint
from hsn.utils import load_config, select_device
from hsn.model import TianmoucHSN
from hsn.losses import HSNLoss


@torch.no_grad()
def run_loss(model, loss_fn, loader, device, cfg, mode):
    assert mode in ["eval", "train_bn"]

    if mode == "eval":
        model.eval()
    else:
        model.train()
        # 使用 batch stats，但不更新 running_mean/running_var
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.0

    total = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_pos = 0.0
    total_iou = 0.0
    n = 0

    use_amp = bool(cfg["train"].get("amp", False))

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with autocast(enabled=use_amp):
            out = model(
                mode="ann",
                template=batch["template"],
                template_box=batch["template_box"],
                search=batch["target"],
            )

            loss_cls, loss_reg, stat = loss_fn.cls_reg_loss(
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

        total += float(loss.cpu())
        total_cls += float(loss_cls.cpu())
        total_reg += float(loss_reg.cpu())
        total_pos += float(stat.get("num_pos", 0.0))
        total_iou += float(stat.get("max_iou_mean", 0.0))
        n += 1

    n = max(n, 1)
    return {
        "loss": total / n,
        "cls": total_cls / n,
        "reg": total_reg / n,
        "pos": total_pos / n,
        "maxIoU": total_iou / n,
    }


def main():
    cfg = load_config("configs/hsn_tianmouc.yaml")
    device = select_device(cfg["train"].get("device", "cuda"))

    model = TianmoucHSN(cfg).to(device)
    load_checkpoint(model, "runs/tianmouc_hsn_reproduce/ann_last.pt", map_location=device)
    loss_fn = HSNLoss(cfg, model.anchor_gen).to(device)

    train_loader = make_loader(
        cfg, "train", cfg["train"]["ann_batch_size"],
        shuffle=False, data_root=cfg["data"]["root"], mode="ann"
    )
    val_loader = make_loader(
        cfg, "val", cfg["train"]["ann_batch_size"],
        shuffle=False, data_root=cfg["data"]["root"], mode="ann"
    )

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        for mode in ["eval", "train_bn"]:
            # 每次重新加载 checkpoint，避免 train_bn 模式污染 BN 状态
            model = TianmoucHSN(cfg).to(device)
            load_checkpoint(model, "runs/tianmouc_hsn_reproduce/ann_last.pt", map_location=device)
            loss_fn = HSNLoss(cfg, model.anchor_gen).to(device)

            m = run_loss(model, loss_fn, loader, device, cfg, mode)
            print(
                f"{split:5s} | {mode:8s} | "
                f"loss={m['loss']:.4f} cls={m['cls']:.4f} reg={m['reg']:.4f} "
                f"pos={m['pos']:.1f} maxIoU={m['maxIoU']:.3f}"
            )


if __name__ == "__main__":
    main()