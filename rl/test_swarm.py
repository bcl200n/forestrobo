# -*- coding: utf-8 -*-
"""编队验证：16 只狗 6 秒小跑后统计"""
import mujoco
import numpy as np

import swarm

m = swarm.make_model()
d = mujoco.MjData(m)
swarm.reset(m, d)
for _ in range(int(6 / 0.005)):
    swarm.step(m, d)
zs = np.array([d.qpos[i * 19 + 2] for i in range(swarm.N)])
xs = np.array([d.qpos[i * 19] for i in range(swarm.N)])
print('16 狗 6s 后: 高度 min %.3f / mean %.3f m | 前进 mean %.2f m | 站立数 %d/16'
      % (zs.min(), zs.mean(), xs.mean() - swarm.X0 - 1.5 * swarm.ROW_DX,
         int((zs > 0.30).sum())))
print('SWARM_TEST_DONE')
