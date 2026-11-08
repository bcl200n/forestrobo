# -*- coding: utf-8 -*-
"""冒烟: DEM 地形块实现验证 + v13 策略在真实坡度上的基线"""
import io
import json
import math
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import terrain as terr
import generate_model as gm

bank = json.load(open('dem_patches.json', encoding='utf-8'))

# 1. 坡度实现验证 (C3 最陡块)
p = bank['C3'][0]
h = terr.make_dem_fn(p['lat'], p['lon'], p['rot'], seed=1)
ax = np.linspace(-12, 12, 41)
X, Y = np.meshgrid(ax, ax)
V = h(X, Y)
# 剔除出生整平区后拟合平面
mask = (np.abs(X - 1.0) > 5) & (np.abs(Y) > 4)
A = np.stack([X[mask], Y[mask], np.ones(mask.sum())], 1)
coef, *_ = np.linalg.lstsq(A, V[mask].ravel(), rcond=None)
realized = math.degrees(math.atan(math.hypot(coef[0], coef[1])))
print('C3 块 DEM 坡度 {}° → 地形场实现 {:.1f}° (高差 {:.1f}m)'.format(
    p['slope'], realized, float(V.max() - V.min())))

# 2. 生成模型 + 环境随机步进
gm.generate(seed=7, wheel=True, profile='std', h_fn=h)
from mujoco_forest_env import ForestWheelEnv
env = ForestWheelEnv(profile='std', generate=False)
obs, _ = env.reset(seed=3)
for k in range(60):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, _ = env.reset()
print('随机步进 60 步 OK, obs 维 {}'.format(obs.shape))

# 3. v13 基线: 各级坡度 10 回合
from stable_baselines3 import PPO
model = PPO.load('models/forest_v13/final', device='cpu')
for cls in ('C1', 'C2', 'C3'):
    q = bank[cls][0]
    hh = terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=2)
    gm.generate(seed=7, wheel=True, profile='std', h_fn=hh)
    e = ForestWheelEnv(profile='std', generate=False)
    surv = reach = 0
    for ep in range(10):
        o, _ = e.reset(seed=100 + ep)
        done = False
        while not done:
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, info = e.step(a)
            done = term or trunc
            if term:
                break
        else:
            pass
        if not term:
            surv += 1
        if info.get('success'):
            reach += 1
    print('v13 基线 {}: 存活 {}/10, 到达 {}/10 (目标 {}°)'.format(
        cls, surv, reach, q['slope']))
print('SMOKE_DONE')
