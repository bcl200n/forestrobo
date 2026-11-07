# -*- coding: utf-8 -*-
"""基元组合扫描: FREQ×A_HIP符号×速度符号 → 选前进+能转的正解"""
import math
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import gait
from mujoco_search_env import RubbleSearchEnv

env = RubbleSearchEnv(profile='flat', generate=False)


def yaw():
    w, x, y, z = env.data.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def probe(freq, ahip, speed_sign, steer_gain=0.35):
    gait.FREQ, gait.A_HIP = freq, ahip
    o, _ = env.reset()
    y0, x0 = yaw(), env.data.qpos[0]
    a = np.zeros(12, np.float32)
    a[1] = 1.0 * speed_sign
    for _ in range(150):
        o, r, term, tr, _ = env.step(a)
        if term:
            break
    fwd = env.data.qpos[0] - x0
    # 转向能力
    o, _ = env.reset()
    y0 = yaw()
    a2 = np.zeros(12, np.float32)
    a2[0] = 1.0
    for _ in range(150):
        o, r, term, tr, _ = env.step(a2)
        if term:
            break
    dy = math.degrees((yaw() - y0 + math.pi) % (2 * math.pi) - math.pi)
    print(f'  f={freq} hip={ahip:+.2f} spd={speed_sign:+d}: '
          f'前进 {fwd:+.2f} m | a0转向 {dy:+6.1f}°')


for freq in (1.8, 2.2):
    for ahip in (0.26, -0.26):
        probe(freq, ahip, +1)
probe(1.8, 0.26, -1)
probe(1.8, -0.26, -1)
print('PRIM_SCAN_DONE')
