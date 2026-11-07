"""Sensor-fault history baseline for two embodiments (not a real state estimator).

Actor reads simulated odometry/IMU, encoder state and ray-cast ranges, not an
obstacle-coordinate oracle. No ROS drivers, localization or learned estimator
are claimed. Privileged geometry remains in reward/termination only.
"""
from collections import deque
from pathlib import Path
import tempfile
import gymnasium as gym
import mujoco
import numpy as np
from mujoco_forest_env import ForestWheelEnv

SCHEMA = 'forest-history-v3.1'
SCENARIOS = ('nominal', 'degraded', 'blackout')


class ForestHistoryEnv(gym.Wrapper):
    def __init__(self, terrain_seed=100, profile='smooth', embodiment='wheel',
                 scenario='degraded', history=4, randomize=True, max_steps=750):
        if scenario not in SCENARIOS or history < 1:
            raise ValueError('Invalid scenario/history')
        cache = Path(__file__).resolve().parent / '.asset_cache'
        cache.mkdir(exist_ok=True)
        self.assets = tempfile.TemporaryDirectory(prefix='v3_', dir=cache)
        try:
            env = ForestWheelEnv(seed=terrain_seed, profile=profile,
                                 asset_dir=self.assets.name, embodiment=embodiment,
                                 max_steps=max_steps)
        except Exception:
            self.assets.cleanup()
            raise
        super().__init__(env)
        self.scenario, self.history, self.randomize = scenario, history, randomize
        self.frames = deque(maxlen=history)
        self.commands = deque()
        self.frame_dim = env.observation_space.shape[0] + 12 + 12 + 3 + 1
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                               (history*self.frame_dim,), np.float32)
        m = env.model
        self.original = {key: getattr(m, key).copy() for key in
                         ('body_mass', 'body_inertia', 'body_ipos', 'geom_friction',
                          'actuator_gainprm', 'actuator_biasprm', 'dof_damping')}
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        self.ray_mask = np.array([1, 0, 0, 0, 0, 0], np.uint8)

    def _rays(self):
        e = self.unwrapped
        p = e.data.qpos[:3].copy()
        yaw = e._yaw()
        ranges = []
        geom = np.array([-1], np.int32)
        for angle in np.linspace(-np.pi, np.pi, 12, endpoint=False):
            direction = np.array([np.cos(angle+yaw), np.sin(angle+yaw), 0.])
            distance = mujoco.mj_ray(e.model, e.data, p + [0, 0, .03], direction,
                                     self.ray_mask, 1, -1, geom)
            ranges.append(min(6., distance) if distance >= 0 else 6.)
        height = mujoco.mj_ray(e.model, e.data, p, np.array([0., 0., -1.]),
                               self.ray_mask, 1, -1, geom)
        return np.asarray(ranges, np.float32), float(np.clip(height, 0, 2) if height >= 0 else 0)

    def _frame(self, obs, reset=False):
        ranges, ground_range = self._rays()
        rng = self.np_random
        if reset:
            self.range_hold = ranges.copy()
            self.nav_hold = np.r_[obs[7:9], self.unwrapped._distance]
            self.ages = np.zeros(3)
            self.blackout_remaining = 0
        if not reset and self.scenario != 'nominal' and self.blackout_remaining == 0:
            probability = .025 if self.scenario == 'blackout' else .006
            if rng.random() < probability:
                self.blackout_remaining = int(rng.integers(20, 101))  # .4--2 s
        lost = not reset and self.blackout_remaining > 0
        if lost:
            self.blackout_remaining -= 1
        valid = np.ones(12, np.float32)
        sensor_noise = self.scenario != 'nominal'
        if sensor_noise:
            obs[:7] += rng.normal(0, .018, 7)
            obs[13:37] += rng.normal(0, .003, 24)
            ranges = np.clip(ranges + rng.normal(0, .035, 12), 0, 6)
        if lost:
            ranges = self.range_hold.copy()
            obs[7:9] = self.nav_hold[:2]
            goal_range = self.nav_hold[2]
            self.ages[:2] += .02
            valid[:] = 0
        else:
            if sensor_noise:
                valid = (rng.random(12) > .08).astype(np.float32)
                ranges = np.where(valid > 0, ranges, self.range_hold)
            self.range_hold = ranges.copy()
            goal_range = self.unwrapped._distance
            self.nav_hold = np.r_[obs[7:9], goal_range]
            self.ages[:2] = 0
        # Replace the tree-coordinate oracle with actual world-geometry ray bins.
        obs[9:12] = [min(ranges[5:8]), min(ranges[8:11]), min(ranges[2:5])]
        obs[12] = ground_range
        # Communication is not used by locomotion: this channel only indicates age.
        self.ages[2] = self.ages[2] + .02 if lost else 0
        return np.concatenate((obs, ranges/6, valid, np.minimum(self.ages, 5)/5,
                               [min(goal_range, 30)/30])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        m = self.unwrapped.model
        for key, value in self.original.items():
            getattr(m, key)[:] = value
        rng = self.np_random
        self.domain = dict(payload_kg=0., friction_scale=1., strength=1.)
        if self.randomize:
            mass_scale = rng.uniform(.9, 1.1)
            m.body_mass[:] *= mass_scale
            m.body_inertia[:] *= mass_scale
            # Torso lumped payload: parallel-axis inertia about combined CoM.
            payload = rng.uniform(0, 7.5)
            offset = rng.uniform([-.10, -.04, .08], [.10, .04, .20])
            mass = m.body_mass[self.torso]
            shift = payload*offset/(mass+payload)
            m.body_mass[self.torso] += payload
            m.body_ipos[self.torso] += shift
            m.body_inertia[self.torso] += mass*payload/(mass+payload)*(np.sum(offset**2)-offset**2)
            friction, strength = rng.uniform(.35, 1.2), rng.uniform(.75, 1.05)
            m.geom_friction[:, 0] *= friction
            m.actuator_gainprm[:] *= strength
            m.actuator_biasprm[:] *= strength
            m.dof_damping[:] *= rng.uniform(.8, 1.2)
            self.domain = dict(payload_kg=float(payload), friction_scale=float(friction), strength=float(strength))
        mujoco.mj_setConst(m, mujoco.MjData(m))
        mujoco.mj_forward(m, self.unwrapped.data)
        # Re-read sensors after constants change, rather than returning stale reset state.
        obs, _ = self.unwrapped._obs(self.unwrapped.last)
        delay = int(rng.integers(0, 4)) if self.randomize else 0
        self.commands.clear()
        self.commands.extend(np.zeros(self.action_space.shape, np.float32) for _ in range(delay))
        self.frames.clear()
        frame = self._frame(obs, reset=True)
        self.frames.extend(frame.copy() for _ in range(self.history))
        self.domain['action_delay_ms'] = delay*20
        return np.concatenate(self.frames), dict(obs_schema=SCHEMA, domain=self.domain.copy())

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Invalid action')
        self.commands.append(action.copy())
        obs, reward, terminated, truncated, info = self.env.step(self.commands.popleft())
        self.frames.append(self._frame(obs))
        info['range_age_s'] = float(self.ages[0])
        info['nav_age_s'] = float(self.ages[1])
        return np.concatenate(self.frames), reward, terminated, truncated, info

    def close(self):
        self.env.close()
        self.assets.cleanup()
