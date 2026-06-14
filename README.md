# Tianmouc-HSN Reproduce

A paper-oriented reimplementation of the **Hybrid Sensing Network (HSN)** from
"A Framework for the General Design and Computation of Hybrid Neural Networks",
with only the data source replaced by Tianmouc data.

This project targets the HSN setting:

- APS/DVS in the paper -> Tianmouc **COP / AOP**
- APS/frame static pathway -> **COP RGB** `[T,320,640,3]`
- DVS/transient pathway -> **AOP TD+SD**: `TD [Ta,160,160]`, `SD [Ta,160,160,2]`
- ANN what pathway -> ResNet-22-style ANN
- SNN where pathway -> ResNet-22-style SNN with LIF neurons
- Hybrid Unit -> maps SNN dynamic feature to ANN feature-space increment
- Head -> HU/RPN-like 4-conv anchor classification/regression head, 6 anchors/location

The central formula is:

```text
F_pred(t+dt) = F_ANN(t) + HU(F_SNN(AOP[t:t+dt]))
```

Then the target/template feature and `F_pred` are merged and used for anchor cls/reg tracking.

---

## 1. Data layout

Training scripts read `.npz` files from:

```text
data/tianmouc_hsn_npz/
  train/
    seq_000.npz
    seq_001.npz
  val/
    seq_100.npz
```

Each `.npz` file should contain:

```text
cop   : [T,320,640,3] or [T,3,320,640]
td    : [Ta,160,160] or [Ta,1,160,160]
sd    : [Ta,160,160,2] or [Ta,2,160,160]
boxes : [M,T,4] or [T,M,4] or [T,4]
```

`boxes[m,t]` must be the same object trajectory over time. Default box format is `xyxy` in the original COP coordinate system `640x320`.

If you only have per-frame detection boxes without track ids, first associate boxes into tracks before saving `boxes`.

---

## 2. Tianmouc spatial preprocessing

The paper resized CLEVRER frames to `128x192` and padded them to `232x296`. Since Tianmouc COP is `320x640` with a 2:1 aspect ratio, this project keeps the same final padded size but uses a content area of:

```text
COP 320x640 -> resize to 128x256 -> pad to 232x296
```

AOP `160x160` TD/SD is resized to the same content area `128x256` and padded to `232x296` so that ANN and SNN features align spatially.

Default config:

```yaml
data:
  raw_cop_size: [320, 640]
  raw_aop_size: [160, 160]
  content_size: [128, 256]
  padded_size: [232, 296]
```

---

## 3. Install

```bash
pip install -r requirements.txt
```

Optional, only if you want to simulate Tianmouc data from ordinary videos:

```bash
pip install tianmoucv
```

---

## 4. Sanity check with synthetic data

```bash
python tools/make_synthetic.py --out data/synthetic_hsn
python tools/check_data.py --config configs/hsn_tianmouc.yaml --data-root data/synthetic_hsn
```

---

## 5. Training

### Phase 1: train ANN what pathway

```bash
python tools/train_ann.py --config configs/hsn_tianmouc.yaml
```

This trains ANN ResNet-22 + HU/RPN head with COP frames only:

```text
L_ann = L_cls + L_reg + lambda_reg_feature * L_feature_regularization
```

### Phase 2: train HSN / SNN where pathway

```bash
python tools/train_hsn.py \
  --config configs/hsn_tianmouc.yaml \
  --ann-checkpoint runs/tianmouc_hsn_reproduce/ann_best.pt
```

This freezes the ANN teacher by default and trains SNN ResNet-22 + feature HU + task HU/head:

```text
F_pred(t+dt) = F_ANN(t) + HU(F_SNN(AOP[t:t+dt]))
L_hsn = L_cls + L_reg + lambda_feat * MSE(F_pred, F_ANN(t+dt))
```

### Optional: joint task fine-tuning

```bash
python tools/finetune_task.py \
  --config configs/hsn_tianmouc.yaml \
  --checkpoint runs/tianmouc_hsn_reproduce/hsn_best.pt
```

This is not the main paper phase; use it as an ablation or Tianmouc-domain adaptation step.

---

## 6. Evaluation

Offline tracking metrics:

```bash
python tools/eval_offline.py \
  --config configs/hsn_tianmouc.yaml \
  --checkpoint runs/tianmouc_hsn_reproduce/hsn_best.pt
```

Streaming metrics with latency compensation:

```bash
python tools/eval_streaming.py \
  --config configs/hsn_tianmouc.yaml \
  --checkpoint runs/tianmouc_hsn_reproduce/hsn_best.pt \
  --latency-steps 1
```

Streaming evaluation compares prediction at time `t` with ground truth at `t+latency`.

---

## 7. Simulating Tianmouc-like data from ordinary video

```bash
python tools/simulate_tianmouc.py \
  --video raw_data/videos/seq_000.mp4 \
  --boxes raw_data/boxes/seq_000.npy \
  --out data/tianmouc_hsn_npz/train/seq_000.npz
```

The script tries to call TianmouCV simulator APIs. If the exact API differs in your installed version, it falls back to a simple RGB/TD/SD approximation and prints a warning. For paper experiments, prefer the official TianmouCV conversion pipeline.

---

## 8. Important notes

1. This is a reproduction-oriented project, not the public toy disk demo.
2. The main model uses ResNet-22-style ANN/SNN backbones and 6 anchors/location.
3. The project assumes Tianmouc COP and AOP are temporally aligned through `aop_per_cop`.
4. For real experiments, verify track ids carefully. Wrong track association will make single-object tracking training invalid.
