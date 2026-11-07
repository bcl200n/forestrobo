# -*- coding: utf-8 -*-
"""Newton 位置伺服跟踪测试: 目标摆动时关节跟不跟"""
import numpy as np

import gait
from newton_vecenv import ForestDogNewtonEnv

env = ForestDogNewtonEnv(num_envs=2, seed=11, task='search', device='cpu')
env.reset()

err_max = 0.0
q_amp = 0.0
for k in range(150):
    env.step(np.zeros((2, 12), np.float32))
    tgt = gait.cpg(env._t)                      # 本步实际目标
    err = np.abs(env._jq[0, 7:19] - tgt)
    err_max = max(err_max, float(err.max()))
    q_amp = max(q_amp, float(np.abs(env._jq[0, 7:19] - gait.Q0).max()))

print(f'3 秒内: 目标跟踪最大误差 {err_max:.3f} rad | '
      f'关节相对站姿摆幅 {q_amp:.3f} rad')
print('TRACK_TEST_DONE')
