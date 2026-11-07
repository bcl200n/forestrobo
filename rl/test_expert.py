# -*- coding: utf-8 -*-
"""专家寻的控制器 + 验证: 反馈律 (转向对准目标 → 前进)"""
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import generate_rubble as gr

gr.generate(seed=11, profile='flat')
from mujoco_search_env import RubbleSearchEnv


def expert_action(obs):
    """obs[7:9]=机体坐标系目标方向(tx 前, ty 左)"""
    tx, ty = float(obs[7]), float(obs[8])
    steer = float(np.clip(-2.2 * ty, -1.0, 1.0))     # 目标在左(ty>0)→左转
    speed = float(np.clip(0.2 + 0.8 * tx, 0.0, 1.0))  # 对准后全速
    a = np.zeros(12, np.float32)
    a[0], a[1] = steer, speed
    return a


env = RubbleSearchEnv(profile='flat', generate=False)
tot = []
for ep in range(15):
    o, _ = env.reset()
    for k in range(900):
        o, r, term, tr, _ = env.step(expert_action(o))
        if term:
            break
    tot.append(int(env.found.sum()))
print('专家控制器: 平均发现 %.1f/%d | 分布 %s | 最后回合用时 %.1fs'
      % (np.mean(tot), env.k, sorted(tot), env.data.time))
print('EXPERT_TEST_DONE')
