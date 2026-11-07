# -*- coding: utf-8 -*-
"""轮足版废墟搜救环境: 差速转向(汽车级接口) + 腿部固定站姿
动作: [左轮速度, 右轮速度] ∈ [-1,1] → ±30 rad/s
观测: 53 维同腿式布局(关节部分为轮足腿角)
"""
import json
import math
import os

import gymnasium as gym
import mujoco
import numpy as np

import terrain as terr

BASE = os.path.dirname(os.path.abspath(__file__))
WQ0 = np.array([0.0, 0.55, -1.15] * 4, np.float32)


class RubbleWheelEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, profile='flat', seed=11, generate=True):
        os.chdir(BASE)
        self._rng = np.random.default_rng(seed)
        if generate:
            import generate_rubble as gr
            gr.generate(seed=11, profile=profile, wheel=True)
        self.model = mujoco.MjModel.from_xml_path('rubble_dog.xml')
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        self.h = terr.build_from_params(json.load(
            open('terrain_rubble_meta.json', encoding='utf-8')))
        lay = json.load(open('rubble_layout.json', encoding='utf-8'))
        self.surv = np.array(lay['survivors'], np.float32)
        self.k = len(self.surv)
        haz = np.array(lay['hazards'], np.float32)
        self.haz = haz[np.hypot(haz[:, 0] - 1.0, haz[:, 1]) > 2.2]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (53,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._prev_pos = None
        self._prev_yaw = 0.0
        self._prev_tdist = None
        self._prev_tgt = -1

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
        rel = self.surv[:, :2] - pos[:2]
        dist = np.linalg.norm(rel, axis=1)
        dist = np.where(self.found, 1e9, dist)
        ki = int(dist.argmin())
        tdist = float(dist[ki])
        tv = rel[ki] / max(tdist, 1e-6)
        tdir = (Rt @ np.array([tv[0], tv[1], 0.0]))[:2]
        self.tgt_idx, self.tgt_dist = ki, tdist
        laser = np.full(3, 6.0)
        rx = self.haz[:, 0] - pos[0]
        ry = self.haz[:, 1] - pos[1]
        dh = np.hypot(rx, ry) - self.haz[:, 2]
        near = dh < 6.0
        if near.any():
            ang = np.arctan2(ry[near], rx[near])
            rr = (ang - yaw + math.pi) % (2 * math.pi) - math.pi
            dn = dh[near]
            for m, lo, hi in ((0, -0.52, 0.52), (1, 0.52, 2.09), (2, -2.09, -0.52)):
                s = (rr >= lo) & (rr < hi)
                if s.any():
                    laser[m] = max(0.0, dn[s].min())
        z_rel = pos[2] - float(self.h(pos[0], pos[1]))
        return np.concatenate([
            v_body, grav, [yawr], tdir, laser, [z_rel],
            d.qpos[7:19] - WQ0, d.qvel[6:18], np.zeros(12, np.float32),
            [self.found.sum() / self.k, min(tdist, 25.0) / 25.0, 0.0, 1.0]
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        hd = float(self._rng.uniform(-math.pi, math.pi))
        x0, y0 = 1.0, float(self._rng.uniform(-0.8, 0.8))
        self.data.qpos[:3] = [x0, y0, float(self.h(x0, y0)) + 0.60]
        self.data.qpos[3:7] = [math.cos(hd / 2), 0, 0, math.sin(hd / 2)]
        mujoco.mj_forward(self.model, self.data)
        self._t0 = self.data.time
        self.found = np.zeros(self.k, bool)
        self._prev_pos = self.data.qpos[:3].copy()
        self._prev_yaw = hd
        self._prev_tdist = None
        self._prev_tgt = -1
        self.last = np.zeros(2, np.float32)
        return self._obs(self.last), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        self.data.ctrl[:12] = WQ0                 # 腿部固定站姿
        # a[0]=纵向速度, a[1]=转向 → 左右轮差速 (FR,FL,RR,RL)
        v = a[0] * 30.0
        w = a[1] * 20.0
        self.data.ctrl[12] = v + w                # FR (右)
        self.data.ctrl[13] = v - w                # FL (左)
        self.data.ctrl[14] = v + w                # RR (右)
        self.data.ctrl[15] = v - w                # RL (左)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        obs = self._obs(self.last)

        pos = self.data.qpos[:3]
        tdist = self.tgt_dist
        r = 0.8 - 1.0 * float(np.linalg.norm(obs[3:5])) \
            - 0.8 * float(np.sum(np.maximum(0.0, 0.45 - obs[9:12]))) \
            - 0.05 * float(np.sum(a ** 2)) \
            - 0.1 * float(np.sum(np.abs(a - self.last)))
        if self._prev_tdist is not None and self._prev_tgt == self.tgt_idx:
            r += float(np.clip(3.0 * (self._prev_tdist - tdist), -2.0, 2.0))
        self._prev_tdist, self._prev_tgt = tdist, self.tgt_idx
        if tdist < 1.0 and not self.found[self.tgt_idx]:
            self.found[self.tgt_idx] = True
            r += 60.0
        allf = bool(self.found.all())
        if allf:
            r += 20.0

        pitch = math.degrees(math.asin(max(-1, min(1,
            2 * (self.data.qpos[3] * self.data.qpos[5]
                 - self.data.qpos[6] * self.data.qpos[4])))))
        z_rel = float(obs[12])
        term = abs(pitch) > 62 or z_rel < 0.18 or z_rel > 0.8 or allf \
            or (self.data.time - self._t0) >= 18.0
        self._prev_pos = pos.copy()
        self._prev_yaw = self._yaw()
        self.last = a.copy()
        return obs, float(r), term, allf, {}
