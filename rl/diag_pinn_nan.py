# -*- coding: utf-8 -*-
"""诊断 PINN 训练 NaN: 检查数据分布"""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

d = np.load('models/pinn/transitions.npz')
obs, act, nxt = d['obs'], d['act'], d['next']
print('obs: shape {}, nan {}, inf {}'.format(obs.shape, np.isnan(obs).sum(), np.isinf(obs).sum()))
print('act: nan {}, inf {}'.format(np.isnan(act).sum(), np.isinf(act).sum()))
print('nxt: nan {}, inf {}'.format(np.isnan(nxt).sum(), np.isinf(nxt).sum()))

std = obs.std(0)
print('\nobs 各维 std (零方差维会致 NaN):')
for i, s in enumerate(std):
    flag = ' ← 零!' if s < 1e-8 else ''
    if flag or i < 16:
        print('  [{:2d}] std={:.6f} min={:.3f} max={:.3f}{}'.format(
            i, s, obs[:, i].min(), obs[:, i].max(), flag))

delta = nxt - obs
dstd = delta.std(0)
print('\ndelta 各维 std:')
for i, s in enumerate(dstd):
    if s < 1e-8:
        print('  [{:2d}] std={:.2e} ← 零!'.format(i, s))
print('\nDELTA_DIAG_DONE')
