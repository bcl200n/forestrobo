# -*- coding: utf-8 -*-
"""Newton vs MuJoCo 步态方向对比: 同一 CPG 控制, 各跑 3 秒看前进符号"""
import numpy as np

from newton_vecenv import ForestDogNewtonEnv

env = ForestDogNewtonEnv(num_envs=2, seed=11, task='search', device='cpu')
env.reset()
x0 = env._jq[0, 0].copy()
for k in range(150):
    env.step(np.zeros((2, 12), np.float32))
print('Newton 环境 CPG 3s 前进: %+.2f m' % (env._jq[0, 0] - x0))
print('NEWTON_DIR_DONE')
