# -*- coding: utf-8 -*-
"""确认 v15 是否倒走: 机头朝向 vs 实际运动方向的夹角统计"""
import io
import json
import math
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from mujoco_forest_env import ForestWheelEnv
import generate_model as gm
import terrain as terr

bank = json.load(open('dem_patches.json', encoding='utf-8'))
model = PPO.load('models/forest_v15/final', device='cpu')

q = bank['C1'][7]
gm.generate(seed=999, wheel=True, profile='std',
            h_fn=terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=999))
env = ForestWheelEnv(profile='std', generate=False)

for seed in (9100, 9101):
    o, _ = env.reset(seed=seed)
    angs = []
    prev = env.data.qpos[:2].copy()
    term = False
    for k in range(500):
        a, _ = model.predict(o, deterministic=True)
        o, r, term, trunc, info = env.step(a)
        p = env.data.qpos[:2]
        v = p - prev
        prev = p
        if np.hypot(*v) > 1e-4:
            w, x_, y_, z_ = env.data.qpos[3:7]
            yaw = math.atan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
            va = math.atan2(v[1], v[0])
            d = math.degrees((va - yaw + math.pi) % (2 * math.pi) - math.pi)
            angs.append(d)
        if term or trunc:
            break
    angs = np.array(angs)
    fwd = (np.abs(angs) < 90).mean()
    print('seed {}: 机身-运动夹角 中位 {:.0f}°, 正向行驶比例 {:.0%}, 到达={}'.format(
        seed, np.median(np.abs(angs)) * np.sign(np.median(angs)), fwd,
        info.get('is_success')))
