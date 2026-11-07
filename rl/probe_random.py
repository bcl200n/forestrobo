# -*- coding: utf-8 -*-
"""终极诊断: 真实训练环境 + 随机策略的发现率/位移/观测正确性"""
import math

import numpy as np

from newton_vecenv import ForestDogNewtonEnv

env = ForestDogNewtonEnv(num_envs=8, seed=11, task='search', device='cpu')
obs = env.reset()

print('幸存者距离出生点:', np.round(
    np.linalg.norm(env.survivors[:, :2] - np.mean([[0.4, 0], [1.6, 0]], 0),
                   axis=1), 2))

rng = np.random.default_rng(0)
disc = 0
min_dist = np.full(8, 99.0)
disp = np.zeros(8)
for ep in range(30):
    obs = env.reset()
    start = env._jq[:, :2].copy()
    for k in range(900):
        a = rng.normal(0, 0.5, (8, 12)).astype(np.float32)
        obs, r, done, info = env.step(a)
        d_all = np.linalg.norm(
            env.survivors[None, :, :2] - env._jq[:, None, :2], axis=2)
        min_dist = np.minimum(min_dist, d_all.min(axis=1))
        for i in np.where(done)[0]:
            info_i = info[i]
            if info_i.get('success'):
                disc += int(env._found[i].sum()) if False else 4
            disp[i] = np.hypot(*(env._jq[i, :2] - start[i]))
    # 最后一段的观测方向 vs 几何真值
    if ep == 29:
        for w in (0, 1):
            i = env._tgt_idx[w]
            dx = env.survivors[i, 0] - env._jq[w, 0]
            dz = env.survivors[i, 1] - env._jq[w, 1]
            print(f'世界{w}: 观测方向 {np.round(obs[w, 7:9], 2)} | '
                  f'几何真值 ({dx:.2f},{dz:.2f}) 距离 {env._tgt_dist[w]:.2f}')

print(f'\n30 回合随机策略: 全找齐 {disc} 次 | 全程最近距幸存者 '
      f'mean {min_dist.mean():.2f} / min {min_dist.min():.2f} m')
print('PROBE_DONE')
