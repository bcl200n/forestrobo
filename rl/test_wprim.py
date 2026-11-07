# -*- coding: utf-8 -*-
"""轮足差速基元: 前进/左转/右转符号验证"""
import math
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from mujoco_wheel_env import RubbleWheelEnv

env = RubbleWheelEnv(profile='flat', generate=False)


def yaw():
    w, x, y, z = env.data.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


for label, a in (('前进 a=[.75,0]', [.75, 0]),
                 ('左转? a=[.75,-1]', [.75, -1]),
                 ('右转? a=[.75,+1]', [.75, +1])):
    o, _ = env.reset()
    y0, x0 = yaw(), env.data.qpos[0]
    for _ in range(200):
        o, r, term, tr, _ = env.step(np.array(a, np.float32))
        if term:
            break
    dy = math.degrees((yaw() - y0 + math.pi) % (2 * math.pi) - math.pi)
    print(f'  {label}: 4s 转向 {dy:+8.1f}° | 位移x {env.data.qpos[0] - x0:+.2f} '
          f'| 位移y {env.data.qpos[1]:+.2f} | 终止={term}')
print('WHEEL_PRIM_DONE')
