from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common import add_common_args, prepare, make_loader, load_checkpoint
from hsn.model import TianmoucHSN
from hsn.utils import batch_to_device


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, label=None):
    """在图像上绘制边界框"""
    image = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if label is not None:
            cv2.putText(image, f"{label}_{i}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="visualization")
    parser.add_argument("--max-sequences", type=int, default=5, 
                        help="最多可视化多少个序列")
    parser.add_argument("--fps", type=int, default=10, 
                        help="输出视频帧率")
    args = parser.parse_args()
    
    cfg, device = prepare(args)
    
    model = TianmoucHSN(cfg).to(device)
    load_checkpoint(model, args.checkpoint, strict=True, map_location=device)
    model.eval()
    
    batch_size = 1
    loader = make_loader(
        cfg,
        cfg["data"]["val_split"],
        batch_size,
        shuffle=False,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_hw = tuple(cfg["data"]["padded_size"])
    raw_size = tuple(cfg["data"]["raw_cop_size"])  # (320, 640)
    
    seq_count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="visualizing", ncols=120):
            if seq_count >= args.max_sequences:
                break
            
            batch = batch_to_device(batch, device)
            
            out = model.forward_hsn_sequence(
                template=batch["template"],
                template_box=batch["template_box"],
                ref=batch["ref"],
                aop=batch["aop"],
                target=None,
            )
            
            pred_boxes_seq, _ = model.decode_sequence(
                out["cls_seq"],
                out["reg_seq"],
                image_hw=image_hw,
            )
            
            gt_boxes_seq = batch["target_boxes_seq"]
            ref_images = batch["ref"].cpu().numpy()
            
            B, K, _ = gt_boxes_seq.shape
            assert B == 1, "Batch size must be 1 for visualization"
            
            seq_name = f"sequence_{seq_count:03d}"
            video_path = output_dir / f"{seq_name}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, 
                                          (raw_size[1], raw_size[0]))
            
            for k in range(K):
                frame = ref_images[0, k].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                frame = (frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                pred_box = pred_boxes_seq[0, k].cpu().numpy()
                gt_box = gt_boxes_seq[0, k].cpu().numpy()
                
                frame = draw_boxes(frame, [gt_box], color=(0, 255, 0), 
                                   thickness=2, label="GT")
                frame = draw_boxes(frame, [pred_box], color=(0, 0, 255), 
                                   thickness=2, label="Pred")
                
                frame = cv2.resize(frame, (raw_size[1], raw_size[0]))
                video_writer.write(frame)
            
            video_writer.release()
            print(f"[saved] {video_path}")
            
            seq_count += 1
    
    print(f"\nVisualization completed! Saved {seq_count} videos to {output_dir}")


if __name__ == "__main__":
    main()
