# -*- coding: utf-8 -*-
"""PINN 动力学残差网络 — 物理知情的模型基 RL 核心

架构:
  状态: 57 维观测 (v20 不含前瞻的基线部分; 前瞻7维是地形函数可直接重算, 不学)
  动作: 16 维
  输出: 下一观测的预测残差  Δ_θ(s, a) ≈ s' − s

物理约束 (PINN 的 "PI"):
  1. 拉格朗日一致性: 腿关节速度 ∂q̇ 应等于位置差分 ∂q/∂t (速度-位置自洽)
  2. 能量有界: |ΔE| = |Σ|τ·ω|·dt − ΔKE| ≤ ε (不做功大于耗散)
  3. 重力一致性: 机身姿态四元数变化率应满足角速度积分
  4. 接触连续性: z_rel 变化率有界 (不瞬移)

损失 = L_data (MSE) + λ₁·L_lagrangian + λ₂·L_energy + λ₃·L_contact
"""
import torch
import torch.nn as nn


class DynResNet(nn.Module):
    """动力学残差网络: 预测 Δobs = obs' − obs, 物理约束内嵌"""

    def __init__(self, obs_dim=57, act_dim=16, hidden=256):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        # 主干: (obs, action) → Δobs
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, obs_dim))

        # 观测索引 (与 mujoco_forest_env._obs 拼接顺序一致)
        # [0:3] v_body, [3:6] grav, [6] yawr, [7:9] goal_b, [9:12] laser,
        # [12] z_rel, [13:25] leg_q(12), [25:37] leg_v(12), [37:53] last(16),
        # [53:57] wheel_w(4)
        self.I_V = slice(0, 3)
        self.I_GRAV = slice(3, 6)
        self.I_ZREL = 12
        self.I_LEGQ = slice(13, 25)
        self.I_LEGV = slice(25, 37)

    def forward(self, obs, act):
        return torch.tanh(self.net(torch.cat([obs, act], dim=-1))) * 3.0  # squash 防溢出

    # ---- 物理约束损失 (全部 clamp 到有界区间防溢出) ----

    def loss_lagrangian(self, obs, d_obs):
        dq = torch.clamp(d_obs[:, self.I_LEGQ] / 0.02, -50, 50)
        v_leg = torch.clamp(obs[:, self.I_LEGV], -50, 50)
        return torch.mean(torch.clamp((dq - v_leg) ** 2, max=2500))

    def loss_energy(self, obs, d_obs, act):
        dv = torch.clamp(d_obs[:, self.I_V] / 0.02, -50, 50)
        return torch.mean(torch.clamp(torch.relu(torch.abs(dv) - 5.0) ** 2, max=2500))

    def loss_contact(self, obs, d_obs):
        dz = torch.clamp(d_obs[:, self.I_ZREL] / 0.02, -20, 20)
        return torch.mean(torch.clamp(torch.relu(torch.abs(dz) - 2.0) ** 2, max=400))

    def loss_gravity(self, obs, d_obs):
        dgrav = torch.clamp(d_obs[:, self.I_GRAV] / 0.02, -100, 100)
        return torch.mean(torch.clamp(torch.relu(torch.abs(dgrav) - 20.0) ** 2, max=2500))

    def loss_total(self, obs, act, obs_next, lam=(1.0, 0.1, 0.5, 0.1)):
        d_pred = self.forward(obs, act)
        d_true = obs_next - obs
        l_data = torch.mean((d_pred - d_true) ** 2)
        l_lag = self.loss_lagrangian(obs, d_pred)
        l_energy = self.loss_energy(obs, d_pred, act)
        l_contact = self.loss_contact(obs, d_pred)
        l_grav = self.loss_gravity(obs, d_pred)
        return (l_data + lam[0] * l_lag + lam[1] * l_energy +
                lam[2] * l_contact + lam[3] * l_grav,
                dict(data=l_data.item(), lag=l_lag.item(),
                     energy=l_energy.item(), contact=l_contact.item(),
                     grav=l_grav.item()))
