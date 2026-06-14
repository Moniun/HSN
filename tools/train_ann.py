from __future__ import annotations

import argparse
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from common import add_common_args, prepare, make_loader, save_checkpoint, set_requires_grad
from hsn.losses import HSNLoss
from hsn.model import TianmoucHSN
from hsn.utils import batch_to_device


def run_epoch(model, loss_fn, loader, optimizer, scaler, device, cfg, train=True):
    model.train(train)

    use_amp = bool(cfg["train"].get("amp", False))
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    pbar = tqdm(loader, desc="train" if train else "val", leave=False, ncols=100)

    for batch in pbar:
        batch = batch_to_device(batch, device)

        with torch.set_grad_enabled(train):
            with autocast(enabled=use_amp):
                out = model.module.forward_ann(batch['template'], batch['template_box'], batch['target']) if isinstance(model, nn.DataParallel) else model.forward_ann(batch['template'], batch['template_box'], batch['target'])
                loss_cls, loss_reg, _ = loss_fn.cls_reg_loss(out['cls'], out['reg'], batch['target_box'])
                loss_feat_reg = out['search_feat'].pow(2).mean()
                loss = (loss_fn.loss_cfg['cls_weight'] * loss_cls +
                        loss_fn.loss_cfg['reg_weight'] * loss_reg +
                        loss_fn.loss_cfg['reg_feature_weight'] * loss_feat_reg)

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

        n = max(1, pbar.n + 1)
        pbar.set_postfix({
            "loss": f"{total_loss/n:.4f}",
            "cls": f"{total_cls_loss/n:.4f}",
            "reg": f"{total_reg_loss/n:.4f}",
        })

    num_batches = max(1, len(loader))
    return {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "reg_loss": total_reg_loss / num_batches,
    }


def warmup_data(model, loader, device, num_batches=3):
    """预热数据加载，避免第一次迭代卡顿"""
    print("[warmup] Preloading data...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="warmup", total=num_batches, leave=False)):
            if i >= num_batches:
                break
            batch = batch_to_device(batch, device)
            _ = model.module.forward_ann(batch['template'], batch['template_box'], batch['target']) if isinstance(model, nn.DataParallel) else model.forward_ann(batch['template'], batch['template_box'], batch['target'])
            if device.type == "cuda":
                torch.cuda.synchronize()
    print("[warmup] Done")


def get_model_module(model):
    """获取原始模型，兼容DataParallel和普通模型"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    cfg, device = prepare(args)

    use_multigpu = cfg['train'].get('use_multigpu', False) and torch.cuda.device_count() > 1

    print(f"[config] epochs={cfg['train']['epochs_ann']}")
    print(f"[config] batch_size={cfg['train']['ann_batch_size']}")
    print(f"[config] num_workers={cfg['train'].get('num_workers', 0)}")
    print(f"[config] persistent_workers={cfg['train'].get('persistent_workers', False)}")
    print(f"[config] prefetch_factor={cfg['train'].get('prefetch_factor', 0)}")
    print(f"[config] amp={cfg['train'].get('amp', False)}")
    print(f"[config] use_multigpu={use_multigpu}")
    print(f"[config] dataset_cache_size={cfg['data'].get('dataset_cache_size', 4)}")
    print(f"[device] Using {device} (count: {torch.cuda.device_count()})")

    base_model = TianmoucHSN(cfg).to(device)

    if use_multigpu:
        print(f"[MultiGPU] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    core_model = get_model_module(model)

    set_requires_grad(core_model.snn, False)
    set_requires_grad(core_model.feature_hu, False)
    set_requires_grad(core_model.ann, True)
    set_requires_grad(core_model.head, True)

    train_loader = make_loader(cfg, cfg['data']['train_split'], cfg['train']['ann_batch_size'], True)
    val_loader = make_loader(cfg, cfg['data']['val_split'], cfg['train']['ann_batch_size'], False)

    warmup_data(model, train_loader, device)

    loss_fn = HSNLoss(cfg, core_model.anchor_gen)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg['train']['lr'],
                                weight_decay=cfg['train'].get('weight_decay', 0.0))

    scaler = GradScaler(enabled=bool(cfg['train'].get('amp', False)))

    epochs = int(cfg['train'].get('override_epochs', cfg['train']['epochs_ann']))
    best_val = float('inf')

    print(f"\n[ANN Training] Starting {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = run_epoch(model, loss_fn, train_loader, optimizer, scaler, device, cfg, True)
        val_metrics = run_epoch(model, loss_fn, val_loader, optimizer, scaler, device, cfg, False)

        elapsed = time.time() - start_time

        print(f"[ANN] epoch={epoch:3d} | "
              f"train_loss={train_metrics['loss']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"time={elapsed:.2f}s")

        save_checkpoint(os.path.join(cfg['train']['out_dir'], 'ann_last.pt'),
                       model, optimizer, epoch, {'train': train_metrics, 'val': val_metrics})

        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            save_checkpoint(os.path.join(cfg['train']['out_dir'], 'ann_best.pt'),
                           model, optimizer, epoch, {'train': train_metrics, 'val': val_metrics})
            print(f"[best] val_loss={best_val:.4f}")


if __name__ == '__main__':
    main()