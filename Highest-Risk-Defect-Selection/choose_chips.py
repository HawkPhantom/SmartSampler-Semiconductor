#!/usr/bin/env python3
# choose_chips.py  ─────────────────────────────────────────────────────────
"""
Pick k chips (k = max_steps) with the highest defect probability according to
the PPO policy and dump every feature for those chips to a CSV file.

Example
-------
python choose_chips.py \
       --model-path logs/ppo_full_features/final_full.zip \
       --csv-out    chosen_seed42.csv \
       --seed       42
"""

import argparse, pathlib
import numpy as np, pandas as pd, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# → Make sure this import points to *exactly* the utils.py you showed
from utils import make_env


# ───────────────────────── CLI ────────────────────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Recommend chips + feature CSV export")
    p.add_argument("--model-path", required=True,
                   help="Path to your PPO checkpoint (e.g. final_full.zip)")
    p.add_argument("--csv-out", default="chosen_chips.csv",
                   help="Output CSV filename (default: chosen_chips.csv)")
    p.add_argument("--max-steps", type=int, default=5,
                   help="How many chips to recommend (k)")
    p.add_argument("--seed", type=int, default=123,
                   help="RNG seed for wafer generation")
    return p.parse_args()


# ────────────────────────── MAIN ──────────────────────────────────────────
def main() -> None:
    args = cli()

    # 1️⃣  Rebuild the single-env evaluation setup
    base_env = make_env(args.seed, max_steps=args.max_steps)
    eval_env = DummyVecEnv([lambda: base_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    # 2️⃣  Load PPO checkpoint *with* env attached
    model = PPO.load(args.model_path, env=eval_env, device="cpu")

    # 3️⃣  Observation before any testing
    obs = eval_env.reset()                          # shape (1, obs_dim)

    # 4️⃣  Full probability distribution π(a | s₀)
    with torch.no_grad():
        tensor = torch.as_tensor(obs, dtype=torch.float32,
                                 device=model.device)
        dist  = model.policy.get_distribution(tensor)
        probs = dist.distribution.probs.squeeze(0).cpu().numpy()

    # 5️⃣  Pick top-k distinct indices
    k      = args.max_steps
    chosen = probs.argsort()[-k:][::-1].tolist()    # highest → lowest

    core = base_env.unwrapped                       # strip Monitor

    # 6️⃣  Build DataFrame with the *named* 23 features
    feature_names = [
        "temp_mean", "temp_std", "gas_flow", "deposition", "doping",
        "etching", "cooling_rate",
        "wafer_x", "wafer_y", "layer_var", "die_rot",
        "vibration", "pressure_var", "tool_wear",
        "shift_id", "operator_id", "time_since_maint",
        "img_lat1", "img_lat2", "img_lat3",
        "defect_count", "texture_score",
        "cnn_defect_score"
    ]

    chip_data = core.data[chosen]                   # shape (k, 23)
    df = pd.DataFrame(chip_data, columns=feature_names)
    df.insert(0, "chip_index", chosen)
    df["prob_defect"] = probs[chosen]
    # ground-truth label is simulation-only; drop if using real wafers
    df["gt_label"]    = ["DEFECT" if l else "OK" for l in core.labels[chosen]]

    # 7️⃣  Write CSV
    out_path = pathlib.Path(args.csv_out).expanduser().resolve()
    df.to_csv(out_path, index=False)

    # 8️⃣  Console summary
    print("\nRecommended inspection plan")
    print("-" * 34)
    print(f"Wafer size            : {core.N} chips")
    print(f"Features per chip     : {core.M}")
    print(f"Chosen chip indices   : {sorted(chosen)}")
    print(f"CSV written to        : {out_path}\n")


if __name__ == "__main__":
    main()