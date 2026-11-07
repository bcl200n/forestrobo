# -*- coding: utf-8 -*-
"""v3 摔倒率统计: 按终止原因分类 20 回合"""
import math
import os
import sys

import numpy as np
from stable_baselines3 import PPO

import enjoy_wheel as ej

MODEL = ('models/' + (sys.argv[1] if len(sys.argv) > 1 else 'wheel_v3')
         + '/final.zip')
ej.policy = PPO.load(MODEL, device='cpu')
print('策略:', MODEL)

rng = np.random.default_rng(0)
stats = {'到达': 0, '摔倒(俯仰)': 0, '塌陷/弹飞': 0, '超时(活满10s)': 0}
durs = []
for ep in range(20):
    hd = rng.uniform(-math.pi, math.pi)
    ej.reset(1.0, rng.uniform(-0.8, 0.8), hd)
    ga = rng.uniform(-math.pi, math.pi)
    gd0 = rng.uniform(3, 6)                      # v3/v4b 可达目标距
    goal = np.array([ej.data.qpos[0] + math.cos(ga) * gd0,
                     ej.data.qpos[1] + math.sin(ga) * gd0])
    ej.PREV = [ej.data.qpos[:3].copy(), hd]
    last = np.zeros(16, np.float32)
    outcome = '超时(活满10s)'
    for k in range(500):
        o, gd = ej.obs_vec(goal, last)
        act, _ = ej.policy.predict(o, deterministic=True)
        ej.step_env(act, goal)
        last = act
        ej.PREV = [ej.data.qpos[:3].copy(),
                   math.atan2(2 * (ej.data.qpos[3] * ej.data.qpos[6]
                                   + ej.data.qpos[4] * ej.data.qpos[5]),
                              1 - 2 * (ej.data.qpos[5] ** 2
                                       + ej.data.qpos[6] ** 2))]
        p, _ = ej.pitch_roll()
        z_rel = ej.data.qpos[2] - float(ej.h(ej.data.qpos[0], ej.data.qpos[1]))
        if gd < 1.0:
            outcome = '到达'
            break
        if abs(p) > 53:
            outcome = '摔倒(俯仰)'
            break
        if z_rel < 0.16 or z_rel > 0.75:
            outcome = '塌陷/弹飞'
            break
    stats[outcome] += 1
    durs.append((k + 1) * 0.02)
print('20 回合结局:', stats)
print('平均存活 %.1fs (最长 %.1f)' % (np.mean(durs), max(durs)))
print('FALL_STATS_DONE')
