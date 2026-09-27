"""Optional Gymnasium adapter for the frozen nominal tracking task.

Import this module after installing the train extra. The numerical core itself
requires only NumPy and SciPy.
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("Install the training dependencies: pip install 'swim4track[train]'") from exc

from .tracking import REPRESENTATIONS, Swim4TrackCore


class Swim4TrackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, representation="swim4track"):
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(23,), dtype=np.float32)
        self.core = Swim4TrackCore(representation=representation)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        observation = self.core.reset(
            seed=seed,
            state=options.get("state"),
            trajectory=options.get("trajectory"),
        )
        info = {
            "trajectory": self.core.trajectory.name,
            "trajectory_manifest": self.core.trajectory.manifest(),
            "reference_position": self.core.reference.position_ned.copy(),
            "reference_rpy": self.core.reference.rpy.copy(),
            "representation": self.core.representation,
        }
        return observation, info

    def step(self, action):
        return self.core.step(action)


def make_swim4track_env():
    return Swim4TrackEnv()
