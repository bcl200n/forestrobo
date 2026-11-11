# -*- coding: utf-8 -*-
"""PINN 动力学网络离线训练 — 物理约束 + 数据拟合联合优化"""
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pinn_dynamics import DynResNet

# ---- 数据 ----
data = np.load('models/pinn/transitions.npz')
obs = torch.from_numpy(data['obs'])
act = torch.from_numpy(data['act'])
nxt = torch.from_numpy(data['next'])

# 归一化 (逐维 z-score)
obs_mean, obs_std = obs.mean(0), obs.clamp(min=1e-4).std(0).clamp(min=1e-4)

obs_n = (obs - obs_mean) / obs_std
delta_raw = nxt - obs
delta_n = delta_raw / obs_std

ds = TensorDataset(obs_n, act, delta_n)
n_train = int(len(ds) * 0.9)
n_val = len(ds) - n_train
train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=2048, shuffle=False)

device = torch.device('cpu')
model = DynResNet(obs_dim=57, act_dim=16).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

EPOCHS = 100
best_val = float('inf')
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for obs_b, act_b, delta_b in train_dl:
        opt.zero_grad()
        d_pred_n = model(obs_b, act_b)
        l_data = F.mse_loss(d_pred_n, delta_b)
        # 物理约束 (反归一化到原始空间)
        obs_raw = obs_b * obs_std + obs_mean
        d_pred_raw = d_pred_n * obs_std
        l_lag = model.loss_lagrangian(obs_raw, d_pred_raw)
        l_energy = model.loss_energy(obs_raw, d_pred_raw, act_b)
        l_contact = model.loss_contact(obs_raw, d_pred_raw)
        l_grav = model.loss_gravity(obs_raw, d_pred_raw)
        loss = l_data + 1.0 * l_lag + 0.1 * l_energy + 0.5 * l_contact + 0.1 * l_grav
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += l_data.item()
    sched.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for obs_b, act_b, delta_b in val_dl:
                d_pred = model(obs_b, act_b)
                val_loss += F.mse_loss(d_pred, delta_b, reduction='sum').item()
                val_count += len(obs_b)
        val_mse = val_loss / val_count
        if val_mse < best_val:
            best_val = val_mse
            torch.save({'model': model.statedict() if hasattr(model, 'statedict') else model.state_dict(),
                        'obs_mean': obs_mean, 'obs_std': obs_std},
                       'models/pinn/dyn_model.pt')
        print('epoch {:3d}: train {:.6f} | val {:.6f} | {:.1f}s'.format(
            epoch + 1, total_loss / len(train_dl), val_mse, time.time() - t0), flush=True)

# ---- 单步预测精度 (原始空间) ----
model.eval()
with torch.no_grad():
    idx = np.random.choice(len(val_ds), min(1000, len(val_ds)), replace=False)
    obs_s = obs_n[idx]
    act_s = act[idx]
    d_pred = model(obs_s, act_s) * obs_std
    d_true = delta_raw[idx]
    per_dim_mae = (d_pred - d_true).abs().mean(0)

print('\n单步预测 MAE (前 12 维):', [round(float(v), 4) for v in per_dim_mae[:12]])
print('总体 MAE: {:.4f}'.format(float(per_dim_mae.mean())))
print('PINN_TRAIN_DONE best_val={:.6f} 用时 {:.1f} 分钟'.format(
    best_val, (time.time() - t0) / 60))
