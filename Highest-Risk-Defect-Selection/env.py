# env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SmartSamplingEnv(gym.Env):
    """
    “Smart sampling” environment for chip-production QA.

    Observation  (shape = N × (M+1) → flattened):
        [f11 … f1M , tested1 , f21 … f2M , tested2 , … ]

    Action
        Discrete index in [0, N)  →  choose a chip to test.

    Reward
        +1   first time a defective chip is tested
         0   first time a good chip is tested
        −0.1 re-testing any already-tested chip

    Episode ends after `max_steps` selections (time-limit truncation).
    """

    metadata = {"render.modes": []}

    def __init__(self, data: np.ndarray, labels: np.ndarray, max_steps: int = 5):
        super().__init__()
        assert data.shape[0] == labels.shape[0], "data / labels length mismatch"

        self.data   = data.astype(np.float32)
        self.labels = labels.astype(np.int32)

        self.N, self.M = self.data.shape
        self.max_steps = max_steps

        # Gymnasium spaces
        self.action_space      = spaces.Discrete(self.N)
        obs_len                = self.N * (self.M + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
        )

        # internal state
        self.tested     = None      # will be initialised in reset()
        self.step_count = 0

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        self.tested     = np.zeros(self.N, dtype=np.float32)
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = int(action)  # cast from ndarray if coming from VecEnv

        if self.tested[action] == 0:
            reward = float(self.labels[action])          # 1 or 0
            self.tested[action] = 1.0
        else:
            reward = -0.1                                # re-test penalty

        self.step_count += 1
        terminated = False                               # never “natural” terminate
        truncated  = self.step_count >= self.max_steps   # time-limit stop

        obs  = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _get_obs(self):
        full = np.hstack([self.data, self.tested.reshape(-1, 1)])
        return full.flatten()

    def render(self):
        tested_ids = np.where(self.tested == 1)[0].tolist()
        print(f"Step {self.step_count}  |  tested chips: {tested_ids}")