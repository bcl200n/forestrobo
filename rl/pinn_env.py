# -*- coding: utf-8 -*-
"""PINN 虚拟环境 — 用学到的动力学模型替代 MuJoCo 物理仿真 (快 100×)

与 ForestWheelEnv 接口完全兼容 (gym.Env), 但 step() 走神经网络而非 MuJoCo。
用于: ① PPO 快速预训练 (分钟级) → ② MuJoCo 精调 (少步数)
"""
import json
import math
import os

import gymnasium as gym
import numpy as np
import torch

from pinn_dynamics import DynResNet

BASE = os.path.dirname(os.path.abspath(__file__))


class PINNEnv(gym.Env):
    """基于 PINN 动力学模型的虚拟环境 (与 ForestWheelEnv 同 obs/action 空间)"""

    metadata = {'render_modes': []}

    def __init__(self, pinn_path='models/pinn/dyn_model.pt', seed=7,
                 horizon=500, device='cpu'):
        ckpt = torch.load(os.path.join(BASE, pinn_path), map_location=device)
        self.model = DynResNet(obs_dim=57, act_dim=16).to(device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.obs_mean = ckpt['obs_mean']
        self.obs_std = ckpt['obs_std']
        self.device = device
        self._rng = np.random.default_rng(seed)

        # 与 ForestWheelEnv 一致的空间定义
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (57,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (16,), np.float32)
        self.max_steps = horizon
        self._k = 0

        # 简化奖励 (与 ForestWheelEnv 同结构, 用 obs 中的 goal_b 做进度)
        self.q0 = np.tile(np.array([0.0, 0.55, -1.15], np.float32), 4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # 从训练数据分布采样初始 obs
        self._obs = self._rng.normal(0, 1, 57).astype(np.float32)
        # 确保合理初值: v_body≈0, grav≈(0,0,-1), 关节在标称位
        self._obs[:3] = 0.0
        self._obs[3:6] = (0.0, 0.0, -1.0)
        self._obs[12] = 0.55  # z_rel
        self._obs[13:25] = self.q0 - self.q0  # 腿关节偏差=0
        self._obs[25:37] = 0.0  # 关节速度=0
        self._k = 0
        self._prev_goal_dist = float(np.hypot(self._obs[7], self._obs[8]))
        return self._obs.copy(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        obs_t = torch.from_numpy(self._obs).unsqueeze(0).to(self.device)
        act_t = torch.from_numpy(a).unsqueeze(0).to(self.device)

        obs_n = (obs_t - self.obs_mean) / self.obs_std
        with torch.no_grad():
            d_n = self.model(obs_n, act_t)
        d_raw = (d_n * self.obs_std).squeeze(0).cpu().numpy()

        new_obs = self._obs + d_raw.astype(np.float32)
        # clip 到物理合理范围
        new_obs = np.clip(new_obs, -10, 10)
        new_obs[9:12] = np.clip(new_obs[9:12], 0.0, 6.0)  # laser ≥0
        new_obs[12] = np.clip(new_obs[12], 0.1, 1.0)       # z_rel
        self._obs = new_obs
        self._k += 1

        # 奖励 (与 ForestWheelEnv v20 简化版)
        goal_dist = float(np.hypot(new_obs[7], new_obs[8]))
        r = 2.2 * (self._prev_goal_dist - goal_dist) / 0.02 + 0.4
        r += 0.6 * new_obs[7]        # 航向
        r += 0.8 * new_obs[0]        # 前向速度
        r -= 0.15 * abs(new_obs[1])  # 侧滑
        r -= 0.012 * max(0, abs(new_obs[3]) + abs(new_obs[4]) - 0.3) ** 2 * 100  # 陡坡

        # 终止: 到达 或 姿态崩
        grav_z = new_obs[5]
        reach = goal_dist < 0.12
        fallen = grav_z > -0.5 or new_obs[12] < 0.15 or new_obs[12] > 0.9
        if reach:
            r += 15.0
        if fallen and not reach:
            r -= 10.0

        term = bool(reach or fallen)
        trunc = bool(self._k >= self.max_steps and not term)
        self._prev_goal_dist = goal_dist

        return new_obs.copy(), float(r), term, trunc, {
            'is_success': bool(reach and not fallen)}
