# train.py
import numpy as np

from stable_baselines3          import PPO
from stable_baselines3.common.utils   import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils      import make_env
from callbacks  import make_callbacks
from config     import parse_args


def evaluate(model: PPO, env, n_episodes: int) -> float:
    """Helper: deterministic policy rollouts on a *vector* env of size 1."""
    rewards = []

    for _ in range(n_episodes):
        obs = env.reset()
        done   = False
        total  = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, d, _ = env.step(action)
            total += float(r[0])  # unwrap vectorised reward
            done = bool(d[0])  # unwrap vectorised done flag

        rewards.append(total)

    return float(np.mean(rewards))


def main():
    args = parse_args()
    set_random_seed(0)

    # 1) parallel training envs  -----------------------------------------
    train_env = DummyVecEnv(
        [lambda i=i: make_env(i, max_steps=args.max_steps) for i in range(4)]
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # 2) single eval env (no reward norm) --------------------------------
    eval_env = DummyVecEnv([lambda: make_env(999, max_steps=args.max_steps)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # 3) callbacks -------------------------------------------------------
    callbacks = make_callbacks(args.log_dir, eval_env, args.eval_episodes)

    # 4) PPO hyper-params ------------------------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"{args.log_dir}/tensorboard",
    )

    # 5) train -----------------------------------------------------------
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    # 6) save & reload (sanity check) -----------------------------------
    final_path = f"{args.log_dir}/final_full"
    model.save(final_path)

    loaded = PPO.load(final_path, env=eval_env)

    # 7) offline evaluation --------------------------------------------
    mean_rew = evaluate(loaded, eval_env, args.eval_episodes)
    print(f"Avg reward over {args.eval_episodes} eval episodes = {mean_rew:.2f}")


if __name__ == "__main__":
    main()