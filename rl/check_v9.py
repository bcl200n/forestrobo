# -*- coding: utf-8 -*-
"""MuJoCo 环境快速体检: 随机基线 + phase1 模型, 平地场景"""
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import generate_rubble as gr

gr.generate(seed=11, profile='flat')
from mujoco_search_env import RubbleSearchEnv

env = RubbleSearchEnv(profile='flat', generate=False)

# 1) 随机策略 20 回合
rng = np.random.default_rng(0)
tot = []
for _ in range(20):
    o, _ = env.reset()
    for k in range(900):
        o, r, term, tr, _ = env.step(rng.normal(0, 0.4, 12))
        if term:
            break
    tot.append(int(env.found.sum()))
print('随机策略: 平均发现 %.1f/%d | 分布 %s'
      % (np.mean(tot), env.k, sorted(tot)))

# 2) phase1 模型 10 回合
from stable_baselines3 import PPO

pol = PPO.load('models/rubble_v9/final', device='cpu')
tot = []
for _ in range(10):
    o, _ = env.reset()
    for k in range(900):
        a, _ = pol.predict(o, deterministic=True)
        o, r, term, tr, _ = env.step(a)
        if term:
            break
    tot.append(int(env.found.sum()))
print('v9 策略: 平均发现 %.1f/%d | 分布 %s'
      % (np.mean(tot), env.k, sorted(tot)))
print('CHECK_DONE')

