# config.py
import argparse
import os


def parse_args():
    """
    CLI helper.  Example:
        python train.py --timesteps 300000 --max-steps 8
    """
    p = argparse.ArgumentParser("PPO on SmartSamplingEnv")
    p.add_argument("--timesteps",     type=int, default=200_000,  help="total training steps")
    p.add_argument("--eval-episodes", type=int, default=10,       help="# episodes during each eval")
    p.add_argument("--log-dir",       type=str, default="logs/ppo_full_features")
    p.add_argument("--max-steps",     type=int, default=5,        help="tests per episode")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    return args