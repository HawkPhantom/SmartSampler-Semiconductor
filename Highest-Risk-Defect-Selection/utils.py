# utils.py
import numpy as np
from env import SmartSamplingEnv
from stable_baselines3.common.monitor import Monitor


def make_env(seed: int, *, max_steps: int = 5):
    """
    Factory that returns a seeded, monitored `SmartSamplingEnv`.
    Generates a fresh synthetic wafer each time for diversification.
    """
    rng = np.random.RandomState(seed)
    N   = 100          # chips per wafer

    # -------- 1) PROCESS PARAMETERS --------------------------------------
    temp_mean    = rng.normal(900,  10, N)
    temp_std     = rng.normal(5,     1, N)
    gas_flow     = rng.normal(20,    2, N)
    deposition   = rng.normal(12,    1, N)
    doping       = rng.normal(0.8, 0.05, N)
    etching      = rng.normal(30,    5, N)
    cooling_rate = rng.normal(2,   0.5, N)

    # -------- 2) POSITION & GEOMETRY -------------------------------------
    wafer_x   = rng.uniform(-50, 50, N)
    wafer_y   = rng.uniform(-50, 50, N)
    layer_var = rng.normal(0.2, 0.05, N)
    die_rot   = rng.uniform(0, 360, N)

    # -------- 3) SENSOR VARIATIONS ---------------------------------------
    vibration    = rng.normal(0.1, 0.02, N)
    pressure_var = rng.normal(5.0, 0.5, N)
    tool_wear    = rng.normal(1.0, 0.2, N)

    # -------- 4) OPERATIONAL / HUMAN FACTORS -----------------------------
    shift_id         = rng.randint(1, 4,  N)
    operator_id      = rng.randint(1, 10, N)
    time_since_maint = rng.uniform(0, 100, N)

    # -------- 5) IMAGE-BASED / QUALITY SCORES ----------------------------
    img_lat1      = rng.normal(0, 1, N)
    img_lat2      = rng.normal(0, 1, N)
    img_lat3      = rng.normal(0, 1, N)
    defect_count  = rng.poisson(0.1, N)
    texture_score = rng.uniform(0, 1, N)
    cnn_defect_score = rng.uniform(0, 1, N)

    # ------- assemble feature matrix (N × 22) ----------------------------
    data = np.column_stack([
        temp_mean, temp_std, gas_flow, deposition, doping, etching, cooling_rate,
        wafer_x, wafer_y, layer_var, die_rot,
        vibration, pressure_var, tool_wear,
        shift_id, operator_id, time_since_maint,
        img_lat1, img_lat2, img_lat3, defect_count, texture_score, cnn_defect_score
    ])

    # ------- binary defect label ----------------------------------------
    labels = (
        (temp_mean < 890) |
        (doping    > 0.85) |
        (texture_score < 0.2) |
        (defect_count  > 0)
    ).astype(int)

    env = SmartSamplingEnv(data, labels, max_steps=max_steps)
    env = Monitor(env)
    env.reset(seed=seed)        # reproducible RNG in Gymnasium
    return env