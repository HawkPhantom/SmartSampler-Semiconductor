# Semiconductor QA – Two‑Pronged Toolkit

This repository hosts **two complementary pipelines for wafer‑level quality assurance**:

| folder | purpose | tech stack |
|--------|---------|-----------|
| **`Highest-Risk-Defect-Selection/`** | RL agent that chooses the _k_ chips most likely to be defective so you probe **fewer dies** for the same yield insight. | Gymnasium, Stable‑Baselines3 (PPO), PyTorch |
| **`Wafer-Level-Transformer-Classifier/`** | Vision Transformer that classifies a **full wafer map** as *defective* or *normal* using the WM‑811K dataset. | PyTorch, Timm (Swin‑Large), Mixed‑Precision |

---

## Why two approaches?

| scenario | tool to use |
|----------|-------------|
| **Inline probing (real‑time)** – you can only destructively test a handful of dies, but want to catch as many bad ones as possible. | **Highest‑Risk‑Defect‑Selection** (reinforcement learning) |
| **Post‑process analysis / incoming lot screening** – you have the complete wafer map bitmap and need a quick pass/fail decision. | **Wafer‑Level‑Transformer‑Classifier** (vision model) |

Combined, they offer both **depth‑first** (smart sampling) and **breadth‑first** (whole‑wafer) QA strategies.

---

## Quick Start (TL;DR)

```bash

# create shared venv (Python 3.10+)
python -m venv .venv && source .venv/bin/activate

# install base requirements
pip install -r Highest-Risk-Defect-Selection/requirements.txt
pip install -r Wafer-Level-Transformer-Classifier/requirements.txt
```

### 1. Highest‑Risk‑Defect‑Selection

```
cd Highest-Risk-Defect-Selection
python train.py --timesteps 300000             # ~15 min on RTX A6000
python choose_chips.py    --model-path logs/ppo_full_features/final_full.zip    --csv-out probe_plan.csv
```

Produces `probe_plan.csv` listing the k chips with the highest defect probability.

### 2. Wafer‑Level‑Transformer‑Classifier

```
cd Wafer-Level-Transformer-Classifier
jupyter lab  # run wafer_map_training.ipynb OR convert to script:

jupyter nbconvert --to python wafer_map_training.ipynb --output train.py
python train.py
```

Final model weights saved as `swin_large_final.pth` and checkpoints per epoch.

---

## Repository Layout

```
SmartSampler-Semiconductor/
├── Highest-Risk-Defect-Selection/
│   ├── env.py, train.py, choose_chips.py, config.py, eny.py, train.py, utils.py, requirements.py
│   └── README.md
├── Wafer-Level-Transformer-Classifier/
│   ├── wafer_map_training.ipynb
│   └── README.md
└── README.md  ← (this file)
```

---

## Prerequisites

* **CUDA‑capable GPU** (optional but recommended)
* Python ≥ 3.10  
* See each sub‑folder’s README for detailed dep versions.

---

