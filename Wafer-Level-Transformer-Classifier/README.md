# Wafer-Level Transformer Classifier  
_A Swin-Large pipeline for WM-811K wafer-map defect detection_

This project trains a **vision transformer** (Swin-Large, ImageNet‑21k pre‑trained) to predict whether an entire wafer is **defective (1)** or **normal (0)** from its wafer‑map bitmap.

Everything happens inside one notebook (`wafer_map_training.ipynb`) – data download, preprocessing, training, checkpoints and final export.

---

## Table of Contents
1. Key Features  
2. Quick Start  
3. Data Pipeline  
4. Model & Training  
5. Checkpoints & Logs  
6. Inference Demo  
7. Customisation Tips  
8. Results  
9. License & Citation  

---

## 1  Key Features

| ✔︎ | Description |
|----|-------------|
| **KaggleHub fetch** | One‑liner download of the public **WM‑811K** wafer‑map dataset. |
| **Robust preprocessing** | Handles 1‑D, 2‑D, 3‑D maps, crops non‑square, resizes to **52 × 52**. |
| **Stratified split** | 80 / 20 train/val with `train_test_split(..., stratify=labels)`. |
| **Transformer backbone** | `swin_large_patch4_window7_224` fine‑tuned for binary defect detection. |
| **Modern training loop** | Mixed‑precision (`torch.cuda.amp`), `torch.compile`, cosine LR, per‑epoch checkpoints. |
| **Scalable loaders** | `num_workers=16`, pinned memory, persistent workers, prefetching. |
| **Artefacts** | `checkpoints/epoch_*.pth` + `swin_large_final.pth` ready for deployment. |

---

## 2  Quick Start

```bash
git clone https://github.com/<your-org>/Wafer-Level-Transformer-Classifier.git
cd Wafer-Level-Transformer-Classifier

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch, timm, kagglehub, kornia…

# put Kaggle API token (kaggle.json) in ~/.kaggle
jupyter lab wafer_map_training.ipynb
```

_No GPU?_ Lower `batch_size` to 16 and expect ~3× longer epochs.

---

## 3  Data Pipeline

| Stage | Details |
|-------|---------|
| **Download** | `kagglehub.dataset_download("qingyi/wm811k-wafer-map")` |
| **Load** | First `.pkl` in `/kaggle/input/wm811k-wafer-map/`. |
| **Normalize shapes** | Squeeze singleton channels, reshape 1‑D, centre‑crop rectangles. |
| **Resize** | Any non‑52×52 map is resized (nearest). |
| **Labels** | `failureType == 'none'` → 0, else 1. |
| **Split** | Stratified 80 / 20, `random_state=42`. |

---

## 4  Model & Training

| Hyper‑param | Value |
|-------------|-------|
| Backbone | Swin‑Large Patch‑4 Window‑7 224 |
| Epochs | 10 |
| Batch size | 128 (AMP) |
| Optimizer | AdamW (lr = 5e‑5, wd = 1e‑2) |
| Scheduler | CosineAnnealingLR (T_max = 10) |
| Loss | CrossEntropy |
| Augment | RandomResizedCrop, H‑flip, ColorJitter |

---

## 5  Checkpoints & Logs

```
checkpoints/
├── epoch_1.pth
├── …
└── epoch_10.pth
swin_large_final.pth
```

Each `.pth` contains model, optimizer, scheduler, scaler states and metrics.

---

## 6  Inference Demo

```python
import timm, torch, torchvision.transforms as T
from PIL import Image
model = timm.create_model("swin_large_patch4_window7_224", num_classes=2)
model.load_state_dict(torch.load("swin_large_final.pth", map_location="cpu"))
model.eval()
```

---

## 7  Customisation Tips

* Swap backbone for `swin_tiny_patch4_window7_224` for edge GPUs.  
* Change `num_classes` and labels for multi‑class failure detection.  
* Add **Kornia** augmentations for rotation / erasing.  

---

## 8  Results (epoch 10)

| Metric | Value    |
|--------|----------|
| Train accuracy | ≈ 99.72 % |
| Val accuracy | ≈ 99.56 % |
| Epoch time (RTX A6000) | ~45 m    |
