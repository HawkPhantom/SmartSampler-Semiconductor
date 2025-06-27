# Smart Sampling RL Pipeline

Reinforcement-learning framework that learns **where to probe a wafer** so you find as many defective chips as possible with only `k` tests.  
A synthetic‐data generator, **Gymnasium** environment, **Stable-Baselines3 (PPO)** training script, evaluation helpers and a “top-k recommendation” utility are all included.

> **TL;DR**  
> 1 ️⃣ `pip install -r requirements.txt`  
> 2 ️⃣ `python train.py --timesteps 300000`  
> 3 ️⃣ `python choose_chips.py --model-path logs/ppo_full_features/final_full.zip`  
> Get a CSV listing the k chips most likely to be defective ✨

---

## Table of Contents
1. [Project Structure](#project-structure)  
2. [Installation](#installation)  
3. [Quick Start](#quick-start)  
4. [Environment & Data](#environment--data)  
5. [Training](#training)  
6. [Monitoring & Checkpoints](#monitoring--checkpoints)  
7. [Inference / Choosing Chips](#inference--choosing-chips)  
8. [Hyper-parameter Tuning](#hyper-parameter-tuning)  
9. [Troubleshooting](#troubleshooting)  
10. [Roadmap](#roadmap)  
11. [Citation](#citation)  
12. [License](#license)

---

## Project Structure
```
smart-sampling-rl/
│
├── env.py               ← Gymnasium environment (SmartSamplingEnv)
├── utils.py             ← Synthetic wafer generator + make_env()
├── train.py             ← PPO training pipeline
├── choose_chips.py      ← Offline chip-selection utility
├── callbacks.py         ← Eval / checkpoint callbacks for SB3
├── config.py            ← CLI arg helper
├── requirements.txt
└── README.md            ← you are here
```

### Key Files
| file | role |
|------|------|
| **env.py** | Reward logic & state encoding. Action = chip index, reward = {+1 (defect first-hit), 0 (good first-hit), -0.1 (re-test)}; episode ends after `max_steps`. |
| **utils.py** | Generates a *new* synthetic wafer every episode (100 chips × 23 features) with realistic process/sensor noise. |
| **train.py** | Wraps multiple `make_env` copies into `VecEnv`, trains a SB3 PPO agent, auto-normalises observations, logs to TensorBoard. |
| **choose_chips.py** | Loads a checkpoint, samples the policy once (`s₀ → π(a)`), obtains **top-k probabilities** and exports full feature rows to CSV. |

---

## Installation

### 2. Python 3.10+ & virtual env (optional)
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
```

### 3. Dependencies
```bash
pip install -r requirements.txt
```
`requirements.txt` includes **gymnasium>=0.29**, **stable-baselines3>=2.3.0**, **torch>=2.2**, **pandas**, **numpy**, **tensorboard**, etc.

---

## Quick Start

Train a policy for 300k timesteps, evaluate every 20k, save checkpoints:
```bash
python train.py     --timesteps 300000     --max-steps 20          # probes per wafer
```

Export the k chips with highest defect probability (top-5 by default):
```bash
python choose_chips.py     --model-path logs/ppo_full_features/final_full.zip     --csv-out    chosen_seed42.csv     --seed       42     --max-steps  5
```
Open `chosen_seed42.csv` to see:

| chip_index | temp_mean | … | cnn_defect_score | prob_defect | gt_label |
|------------|-----------|---|------------------|-------------|----------|
| 77 | 883.2 | … | 0.76 | **0.93** | DEFECT |
| 12 | 892.1 | … | 0.48 | **0.84** | OK |
| … | … | … | … | … | … |

*(`gt_label` is only present for the synthetic generator; remove for real data.)*

---

## Environment & Data

### Feature Vector (23 dims)
| Category | Features |
|----------|----------|
| **Process** | `temp_mean`, `temp_std`, `gas_flow`, `deposition`, `doping`, `etching`, `cooling_rate` |
| **Geometry** | `wafer_x`, `wafer_y`, `layer_var`, `die_rot` |
| **Sensor** | `vibration`, `pressure_var`, `tool_wear` |
| **Operational** | `shift_id`, `operator_id`, `time_since_maint` |
| **Image-based** | `img_lat1-3`, `defect_count`, `texture_score`, `cnn_defect_score` |

A chip is labelled **defective** if *any* of:
```python
temp_mean  < 890  or
doping     > 0.85 or
texture_score < 0.2 or
defect_count  > 0
```

### Observation Space
```
[f11 … f1M , tested1 , f21 … f2M , tested2 , … ]  →  shape = 100 × (23+1)
```
`testedᵢ ∈ {0,1}` flips to 1 the first time a chip is probed.

---

## Training

| flag | default | description |
|------|---------|-------------|
| `--timesteps` | `200000` | SB3 total environment steps |
| `--max-steps` | `5` | How many probes per episode (k) |
| `--eval-episodes` | `10` | Episodes during each periodic evaluation |
| `--log-dir` | `logs/ppo_full_features` | All TensorBoard & checkpoints here |

Example **8 probes / wafer**:
```bash
python train.py --timesteps 500000 --max-steps 8
```

### Hyper-parameters (PPO)
```python
n_steps      = 2048
batch_size   = 64
gamma        = 0.99
gae_lambda   = 0.95
clip_range   = 0.2
learning_rate= 3e-4
ent_coef     = 0.01
```

---

## Monitoring & Checkpoints

* **TensorBoard:**  
  ```bash
  tensorboard --logdir logs/ppo_full_features/tensorboard
  ```
* **EvalCallback:** Saves `best_model.zip` when mean reward improves.  
* **CheckpointCallback:** Snapshots every 50 k steps (`checkpoints/ppo_full_<step>.zip`).

---

## Inference / Choosing Chips

`choose_chips.py` performs **single-pass ranking**:

1. Builds a synthetic wafer with your seed  
2. Normalises obs via `VecNormalize`  
3. Computes π(a \| s₀) over 100 chips  
4. Picks top‑k distinct indices  
5. Dumps chip rows + prob to CSV

---

## Hyper‑parameter Tuning
```bash
for k in 5 8 12; do
  for lr in 3e-4 1e-4; do
    python train.py         --timesteps 200000         --max-steps $k         --log-dir logs/ppo_k${k}_lr${lr}
  done
done
```

---

## Roadmap
- [ ] Training with Real Wafer Data
- [ ] Curriculum (k=5 → k=1)  
- [ ] Severity‑graded rewards  
- [ ] REST deployment (`fastapi`)  
- [ ] ONNX export


