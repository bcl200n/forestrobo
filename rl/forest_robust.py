"""Version 2 baseline: isolated assets, domain randomization and sensor faults.

Observations still contain simulator pose/height: NOT a deployable estimator.
"""
from collections import deque
import tempfile
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from mujoco_forest_env import ForestWheelEnv


class RobustForestEnv(gym.Wrapper):
    def __init__(self, terrain_seed=100, profile='smooth', faults=True):
        cache = Path(__file__).resolve().parent / '.asset_cache'
        cache.mkdir(exist_ok=True)
        self.assets = tempfile.TemporaryDirectory(prefix='forest_', dir=cache)
        super().__init__(ForestWheelEnv(seed=terrain_seed, profile=profile,
                                       asset_dir=self.assets.name))
        self.faults = faults
        self.mass = self.unwrapped.model.body_mass.copy()
        self.inertia = self.unwrapped.model.body_inertia.copy()
        self.friction = self.unwrapped.model.geom_friction.copy()
        self.queue = deque()

    def _sensor(self, obs):
        obs = obs.copy()
        if self.faults:
            obs[:7] += self.np_random.normal(0, 0.015, 7)
            if self.np_random.random() < 0.10:
                obs[9:12] = self.last_sensor[9:12]  # stale, not perfect range
            else:
                obs[9:12] = np.clip(obs[9:12] + self.np_random.normal(0, .04, 3), 0, 6)
        self.last_sensor = obs.copy()
        return obs

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        m = self.unwrapped.model
        scale = self.np_random.uniform(.85, 1.15) if self.faults else 1.
        m.body_mass[:] = self.mass * scale
        m.body_inertia[:] = self.inertia * scale
        m.geom_friction[:] = self.friction
        if self.faults:
            m.geom_friction[:, 0] *= self.np_random.uniform(.35, 1.2)
        # mj_setConst evaluates qpos0; use scratch data to preserve reset pose.
        mujoco.mj_setConst(m, mujoco.MjData(m))
        mujoco.mj_forward(m, self.unwrapped.data)
        self.delay = int(self.np_random.integers(0, 4)) if self.faults else 0
        self.queue.clear()
        self.queue.extend(np.zeros(16, np.float32) for _ in range(self.delay))
        self.last_sensor = obs.copy()
        return self._sensor(obs), info

    def step(self, action):
        self.queue.append(np.asarray(action).copy())
        obs, reward, terminated, truncated, info = self.env.step(self.queue.popleft())
        return self._sensor(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()
        self.assets.cleanup()
