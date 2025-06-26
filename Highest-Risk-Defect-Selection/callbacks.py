# callbacks.py
import os
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


def make_callbacks(log_dir: str, eval_env, eval_episodes: int = 10):
    """
    * EvalCallback   – evaluates & saves the best model every `eval_freq` steps
    * CheckpointCall – periodic snapshots, handy for long runs / resuming
    """
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=20_000,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_full",
    )
    return [eval_cb, ckpt_cb]