# -*- coding: utf-8 -*-
"""新步态下: 双向转向基元 + 专家单回合轨迹诊断"""
import math
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from mujoco_search_env import RubbleSearchEnv

env = RubbleSearchEnv(profile='flat', generate=False)


def yaw():
    w, x, y, z = env.data.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


for s in (+1, -1):
    o, _ = env.reset()
    y0 = yaw()
    a = np.zeros(12, np.float32)
    a[0] = s
    fell = False
    for _ in range(150):
        o, r, term, tr, _ = env.step(a)
        if term:
            fell = True
            break
    dy = math.degrees((yaw() - y0 + math.pi) % (2 * math.pi) - math.pi)
    print(f'  a0={s:+d}: 转向 {dy:+7.1f}° / 3s | 提前终止={fell}')

# 专家轨迹: 低增益, 打印 tdir 与朝向误差
o, _ = env.reset()
print('专家轨迹(低增益1.0):')
for k in range(300):
    tx, ty = float(o[7]), float(o[8])
    a = np.zeros(12, np.float32)
    a[0] = float(np.clip(-1.0 * ty, -1, 1))
    a[1] = float(np.clip(0.2 + 0.8 * tx, 0, 1))
    o, r, term, tr, _ = env.step(a)
    if k % 60 == 0:
        print(f'  t={k*0.02:4.1f}s tdir=({o[7]:+.2f},{o[8]:+.2f}) '
              f'目标距离={o[50]*25:.1f}m found={env.found.sum()}')
    if term:
        print(f'  t={k*0.02:.1f}s 终止! found={env.found.sum()}')
        break
print('DIAG_DONE')
