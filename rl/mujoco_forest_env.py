# -*- coding: utf-8 -*-
"""MuJoCo 原生林下轮足导航环境（训练=回放同物理引擎）
动作: 16 维 = 12 腿姿态残差(±0.35) + 4 轮速(±30 rad/s)
课程: terrain smoothing via profile smooth/std
"""
import json
import math
import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import terrain as terr
from joint_layout import addresses

BASE = os.path.dirname(os.path.abspath(__file__))
WQ0 = np.tile(np.array([0.0, 0.55, -1.15], np.float32), 4)
WHEEL_R = 0.075

# RSI 可调奖励参数 (env_cfg.json 缺省值 = v14 硬编码值, 不配置时行为不变)
CFG_DEFAULT = dict(prog_w=2.5, alive_w=0.2, tilt_rel_w=2.0, tilt_abs_w=0.4,
                   tree_w=0.05, act_w=0.01, smooth_w=0.03, fall_tilt_deg=60.0)


def _load_cfg():
    try:
        with open(os.path.join(BASE, 'env_cfg.json'), encoding='utf-8') as f:
            c = json.load(f)
        return {k: c.get(k, v) for k, v in CFG_DEFAULT.items()}
    except Exception:
        return dict(CFG_DEFAULT)


class ForestWheelEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, profile='smooth', seed=7, generate=True, asset_dir=None, embodiment='wheel', max_steps=500):
        if embodiment not in ('wheel', 'foot'):
            raise ValueError('embodiment must be wheel or foot')
        self.wheel = embodiment == 'wheel'
        self.q0 = WQ0.copy() if self.wheel else np.tile([0., .9, -1.72], 4).astype(np.float32)
        self.max_steps = max_steps
        asset_dir = os.path.abspath(asset_dir or BASE)
        self._rng = np.random.default_rng(seed)
        if generate:
            import generate_model as gm
            gm.generate(seed=seed, wheel=self.wheel, profile=profile, out_dir=asset_dir)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(asset_dir, 'forest_dog_wheel.xml' if self.wheel else 'forest_dog.xml'))
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        self.leg_q, self.leg_v = addresses(self.model, ('abd', 'hip', 'knee'))
        self.wheel_q, self.wheel_v = addresses(self.model, ('wheel',)) if self.wheel else (np.array([], int), np.array([], int))
        self.model.geom_group[self.model.geom_bodyid > 0] = 1
        self.robot_geoms = self.model.geom_bodyid > 0
        self.tree_geoms = {i for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith('tree_')}
        self.h = terr.build_from_params(json.loads(
            (Path(asset_dir) / 'terrain_meta.json').read_text(encoding='utf-8')))
        lay = json.loads((Path(asset_dir) / 'forest_layout.json').read_text(encoding='utf-8'))
        self.trees = np.array(lay['trees'], np.float32).reshape(-1, 4)[:, :3]  # x,y,r (空表安全)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                (57 if self.wheel else 49,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (16 if self.wheel else 12,), np.float32)
        self._prev_pos = None
        self._prev_yaw = 0.0
        self.cfg = _load_cfg()

    def _yaw(self):
        w, x, y, z = self.data.qpos[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _obs(self, last):
        d = self.data
        pos = d.qpos[:3]
        w, x, y, z = d.qpos[3:7]
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        Rt = R.T
        yaw = self._yaw()
        vel = (pos - self._prev_pos) / 0.02 if self._prev_pos is not None \
            else np.zeros(3)
        yawr = ((yaw - self._prev_yaw + math.pi) % (2 * math.pi) - math.pi) / 0.02 \
            if self._prev_pos is not None else 0.0
        v_body = Rt @ vel
        grav = Rt @ np.array([0, 0, -1.0])
        gv = self.goal - pos[:2]
        gd = float(np.hypot(*gv))
        gun = gv / max(gd, 1e-6)
        goal_b = (Rt @ np.array([gun[0], gun[1], 0.0]))[:2]
        laser = np.full(3, 6.0)
        rx = self.trees[:, 0] - pos[0]
        ry = self.trees[:, 1] - pos[1]
        dh = np.hypot(rx, ry) - self.trees[:, 2]
        near = dh < 6.0
        if near.any():
            ang = np.arctan2(ry[near], rx[near])
            rr = (ang - yaw + math.pi) % (2 * math.pi) - math.pi
            dn = dh[near]
            for m, lo, hi in ((0, -0.52, 0.52), (1, 0.52, 2.09),
                              (2, -2.09, -0.52)):
                s = (rr >= lo) & (rr < hi)
                if s.any():
                    laser[m] = max(0.0, dn[s].min())
        z_rel = pos[2] - float(self.h(pos[0], pos[1]))
        wheel_w = d.qvel[self.wheel_v]
        return np.concatenate([
            v_body, grav, [yawr], goal_b, laser, [z_rel],
            d.qpos[self.leg_q] - self.q0, d.qvel[self.leg_v], last, wheel_w * 0.05
        ]).astype(np.float32), gd

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        hd = float(self._rng.uniform(-math.pi, math.pi))
        x0, y0 = 1.0, float(self._rng.uniform(-0.8, 0.8))
        self.data.qpos[:3] = [x0, y0, float(self.h(x0, y0)) + 0.60]
        self.data.qpos[3:7] = [math.cos(hd / 2), 0, 0, math.sin(hd / 2)]
        self.data.qpos[self.leg_q] = self.q0
        mujoco.mj_forward(self.model, self.data)
        ga = hd + self._rng.uniform(-1.2, 1.2)
        gd = self._rng.uniform(4.0, 7.0)
        self.goal = np.array([x0 + math.cos(ga) * gd,
                              y0 + math.sin(ga) * gd])
        self._prev_pos = self.data.qpos[:3].copy()
        self._prev_yaw = hd
        self.last = np.zeros(self.action_space.shape, np.float32)
        self._k = 0
        self._distance = float(np.linalg.norm(self.goal - self.data.qpos[:2]))
        return self._obs(self.last)[0], {}

    def step(self, action):
        a = np.asarray(action, np.float32)
        if a.shape != self.action_space.shape or not np.isfinite(a).all():
            raise ValueError('Action shape or finite-value check failed')
        a = np.clip(a, -1, 1)
        self.data.ctrl[:12] = self.q0 + a[:12] * 0.35
        if self.wheel:
            self.data.ctrl[12:16] = a[12:] * 30.0
        energy = 0.0
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
            energy += float(np.sum(np.abs(self.data.actuator_force * self.data.actuator_velocity))) * self.model.opt.timestep
        self._k += 1
        obs, gd = self._obs(a)

        C = self.cfg
        r = C['prog_w'] * (self._distance - gd) / 0.02 + C['alive_w']        # 前进 + 存活
        # 倾斜惩罚 v2 (2026-11 真实地形版): 惩罚机体上轴偏离"地形法线"的角距,
        # 而非偏离重力 — 在陡坡上贴合坡面 (随坡倾侧) 才是正确姿态, 旧版强迫水平会被地形掀翻
        pos = self.data.qpos
        eps = 0.30
        nx = (float(self.h(pos[0] - eps, pos[1])) - float(self.h(pos[0] + eps, pos[1]))) / (2 * eps)
        ny = (float(self.h(pos[0], pos[1] - eps)) - float(self.h(pos[0], pos[1] + eps))) / (2 * eps)
        nz = 1.0 / math.sqrt(nx * nx + ny * ny + 1.0)
        n = np.array([nx * nz, ny * nz, nz])
        w, x_, y_, z_ = pos[3:7]
        u_b = np.array([2 * (x_ * z_ - w * y_), 2 * (y_ * z_ + w * x_),
                        1 - 2 * (x_ * x_ + y_ * y_)])       # 机体上轴 (世界系)
        tilt_excess = 1.0 - float(np.dot(u_b, n))
        r -= C['tilt_rel_w'] * tilt_excess
        r -= C['tilt_abs_w'] * float(np.linalg.norm(obs[3:5]))  # 绝对倾斜轻惩罚 (保留少量水平偏好)
        r -= C['tree_w'] * float(np.sum(np.maximum(0.0, 0.45 - obs[9:12])))  # 树近
        r -= C['act_w'] * float(np.sum(a ** 2)) + C['smooth_w'] * float(np.sum(np.abs(a - self.last)))
        z_rel = float(obs[12])
        reach = gd < 1.2
        fallen = float(obs[5]) > -math.cos(math.radians(C['fall_tilt_deg'])) or z_rel < 0.18 or z_rel > 0.85
        clearance = np.hypot(self.trees[:, 0] - self.data.qpos[0],
                             self.trees[:, 1] - self.data.qpos[1]) - self.trees[:, 2]
        collision = bool(np.any(clearance < 0.30))
        # Check actual contacts too: a rectangular torso is not a point robot.
        for contact in self.data.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in self.tree_geoms and self.robot_geoms[g2]) or (g2 in self.tree_geoms and self.robot_geoms[g1]):
                collision = True
        reach = bool(reach and not fallen and not collision)
        if reach:
            r += 15.0
        term = bool(fallen or collision or reach)
        truncated = bool(self._k >= self.max_steps and not term)
        if term and not reach:
            r -= 10.0
        self._distance = gd
        path_step = float(np.linalg.norm(self.data.qpos[:2] - self._prev_pos[:2]))
        self._prev_pos = self.data.qpos[:3].copy()
        self._prev_yaw = self._yaw()
        self.last = a.copy()
        return obs, float(r), term, truncated, {'is_success': bool(reach and not fallen and not collision),
                'goal_distance': gd, 'energy_j': energy, 'path_m': path_step, 'failure': 'fall' if fallen else 'collision' if collision else ''}
