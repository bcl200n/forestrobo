# -*- coding: utf-8 -*-
"""MuJoCo 版废墟搜救环境（单环境 Gymnasium, 训练评测同物理引擎）
动作: [转向, 速度, 10×微残差] → CPG 结构化映射
观测: 53 维, 与 eval_rubble.py 完全一致
"""
import json
import math
import os

import gymnasium as gym
import mujoco
import numpy as np

import gait
import terrain as terr

BASE = os.path.dirname(os.path.abspath(__file__))


class RubbleSearchEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, profile='flat', seed=11, generate=True):
        os.chdir(BASE)
        self.profile = profile
        self._rng = np.random.default_rng(seed)
        if generate:                        # 子进程必须 False, 避免并发写文件
            import generate_rubble as gr
            gr.generate(seed=11, profile=profile)
        self.model = mujoco.MjModel.from_xml_path('rubble_dog.xml')
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        self.h = terr.build_from_params(json.load(
            open('terrain_rubble_meta.json', encoding='utf-8')))
        lay = json.load(open('rubble_layout.json', encoding='utf-8'))
        self.surv = np.array(lay['survivors'], np.float32)
        self.k = len(self.surv)
        haz = np.array(lay['hazards'], np.float32)
        self.haz = haz[np.hypot(haz[:, 0] - 1.0, haz[:, 1]) > 2.2]  # 出生区外
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (53,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self._t = 0.0
        self._k = 0
        self._prev_pos = None
        self._prev_yaw = 0.0
        self._prev_tdist = None
        self._prev_tgt = -1

    # ---------- 内部 ----------
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
        ph = self._t * gait.FREQ * 2 * math.pi
        return np.concatenate([
            v_body, grav, [yawr], tdir, laser, [z_rel],
            d.qpos[7:19] - gait.Q0, d.qvel[6:18], last,
            [self.found.sum() / self.k, min(tdist, 25.0) / 25.0,
             math.sin(ph), math.cos(ph)]]).astype(np.float32)

    # ---------- API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        hd = float(self._rng.uniform(-math.pi, math.pi))
        x0, y0 = 1.0, float(self._rng.uniform(-0.8, 0.8))
        self.data.qpos[:3] = [x0, y0, float(self.h(x0, y0)) + 0.46]
        self.data.qpos[3:7] = [math.cos(hd / 2), 0, 0, math.sin(hd / 2)]
        self.data.qpos[7:] = gait.Q0
        mujoco.mj_forward(self.model, self.data)
        self.found = np.zeros(self.k, bool)
        self._t = 0.0
        self._k = 0
        self._prev_pos = self.data.qpos[:3].copy()
        self._prev_yaw = hd
        self._prev_tdist = None
        self._prev_tgt = -1
        self.last = np.zeros(12, np.float32)
        return self._obs(self.last), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        c = gait.cpg(self._t).copy()
        c[0::3] += a[0] * 0.22
        c[1::3] += a[1] * 0.18
        c[2:12] += a[2:] * 0.08
        self.data.ctrl[:] = c
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self._t += 0.02
        self._k += 1
        obs = self._obs(self.last)

        # ---- 奖励 ----
        pos = self.data.qpos[:3]
        tdist = self.tgt_dist
        r = 0.6 - 1.2 * float(np.linalg.norm(obs[3:5])) \
            - 0.8 * float(np.sum(np.maximum(0.0, 0.45 - obs[9:12]))) \
            - 0.01 * float(np.sum(a ** 2)) \
            - 0.03 * float(np.sum(np.abs(a - self.last)))
        if self._prev_tdist is not None and self._prev_tgt == self.tgt_idx:
            r += float(np.clip(2.5 * (self._prev_tdist - tdist), -2.0, 2.0))
        self._prev_tdist, self._prev_tgt = tdist, self.tgt_idx
        newly = tdist < 1.0 and not self.found[self.tgt_idx]
        if newly:
            self.found[self.tgt_idx] = True
            r += 60.0
        allf = bool(self.found.all())
        if allf:
            r += 20.0

        # ---- 终止 ----
        pitch = math.degrees(math.asin(max(-1, min(1,
            2 * (self.data.qpos[3] * self.data.qpos[5]
                 - self.data.qpos[6] * self.data.qpos[4])))))
        z_rel = float(obs[12])
        term = abs(pitch) > 60 or z_rel < 0.16 or z_rel > 0.75 or allf \
            or self._k >= 900

        self._prev_pos = pos.copy()
        self._prev_yaw = self._yaw()
        self.last = a.copy()
        return obs, float(r), term, allf, {}

