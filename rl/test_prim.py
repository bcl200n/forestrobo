# -*- coding: utf-8 -*-
"""基元测试: 转向指令/速度指令在训练环境里的实际效果"""
import math
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from mujoco_search_env import RubbleSearchEnv

env = RubbleSearchEnv(profile='flat', generate=False)


def yaw():
    w, x, y, z = env.data.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


for label, act in (('转向 a0=+1', np.array([1, 0] + [0] * 10, np.float32)),
                   ('转向 a0=-1', np.array([-1, 0] + [0] * 10, np.float32)),
                   ('速度 a1=+1', np.array([0, 1] + [0] * 10, np.float32)),
                   ('全零(纯CPG)', np.zeros(12, np.float32))):
    o, _ = env.reset()
    y0, x0 = yaw(), env.data.qpos[0]
    for _ in range(150):
        o, r, term, tr, _ = env.step(act)
        if term:
            break
    dy = math.degrees((yaw() - y0 + math.pi) % (2 * math.pi) - math.pi)
    print(f'  {label}: 3s 转向 {dy:+7.1f}° | 前进 {env.data.qpos[0] - x0:+.2f} m'
          f' | 终止={term}')
print('PRIM_TEST_DONE')
