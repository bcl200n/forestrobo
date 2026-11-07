# -*- coding: utf-8 -*-
"""伺服参数×步频 组合扫描: 找 Newton 里能前进的 CPG 配置"""
import numpy as np

import gait
from newton_vecenv import ForestDogNewtonEnv

CONFIGS = [
    # (forcerange, FREQ, A_HIP, A_KNEE)
    (90, 2.2, 0.30, 0.52),
    (90, 1.6, 0.24, 0.42),
    (120, 1.6, 0.24, 0.42),
    (90, 1.3, 0.20, 0.36),
]
env = ForestDogNewtonEnv(num_envs=2, seed=11, task='search', device='cpu')

import generate_rubble as gr
import mujoco

for fr, freq, ahip, aknee in CONFIGS:
    # 改 MJCF 力矩上限: 直接改 env 内模型不可行 → 用文本替换重建太慢;
    # Newton 的力矩上限在 solver 内部, 先只扫步频/幅值(力矩上限不变 45)
    gait.FREQ, gait.A_HIP, gait.A_KNEE = freq, ahip, aknee
    env.reset()
    x0 = env._jq[0, 0].copy()
    for k in range(150):
        env.step(np.zeros((2, 12), np.float32))
    print(f'freq={freq} hip={ahip} knee={aknee}: 3s 前进 '
          f'{env._jq[0, 0] - x0:+.2f} m', flush=True)
print('SWEEP_DONE')
