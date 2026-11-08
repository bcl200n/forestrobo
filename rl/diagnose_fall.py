# -*- coding: utf-8 -*-
"""死因诊断: 在 C1 (20°) 上跑若干回合, 记录终止原因与状态轨迹"""
import io
import json
import math
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import generate_model as gm
import terrain as terr
from mujoco_forest_env import ForestWheelEnv
from stable_baselines3 import PPO

bank = json.load(open('dem_patches.json', encoding='utf-8'))
q = bank['C1'][0]
hh = terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=2)
gm.generate(seed=7, wheel=True, profile='std', h_fn=hh)
env = ForestWheelEnv(profile='std', generate=False)

model = PPO.load(sys.argv[1] if len(sys.argv) > 1 else 'models/forest_v13/final', device='cpu')

for ep in range(6):
    o, _ = env.reset(seed=100 + ep)
    grav = []          # obs[3:5] = 重力水平分量 (倾斜指示)
    z_rel = []
    min_tree = []
    cause = 'timeout'
    for k in range(500):
        a, _ = model.predict(o, deterministic=True)
        o, r, term, trunc, info = env.step(a)
        grav.append(float(np.linalg.norm(o[3:5])))
        z_rel.append(float(o[12]))
        if term:
            gz = grav[-1]
            z = z_rel[-1]
            if gz > math.sin(math.radians(55)):
                cause = '倾覆(>55°) tilt={:.0f}°'.format(math.degrees(math.asin(min(gz, 1))))
            elif z < 0.18:
                cause = '塌陷 z_rel={:.2f}'.format(z)
            elif z > 0.8:
                cause = '弹起 z_rel={:.2f}'.format(z)
            else:
                cause = '树碰撞'
            break
    print('ep{}: {} @step{}, 最大倾角 {:.0f}°, z_rel 范围 {:.2f}-{:.2f}, 走过 {:.1f}m'.format(
        ep, cause, k + 1, math.degrees(math.asin(min(max(grav), 1))),
        min(z_rel), max(z_rel),
        float(np.hypot(env.data.qpos[0] - 1.0, env.data.qpos[1]))))
print('DIAG_DONE')
