# -*- coding: utf-8 -*-
"""快速抽查存档: 指定 ckpt 在三档坡度的存活/到达 (10 回合/档)"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import generate_model as gm
import terrain as terr
from mujoco_forest_env import ForestWheelEnv
from stable_baselines3 import PPO

ckpt = sys.argv[1] if len(sys.argv) > 1 else 'models/forest_v14/ckpt_00_C1'
model = PPO.load(ckpt, device='cpu')
bank = json.load(open('dem_patches.json', encoding='utf-8'))

for cls in ('C1', 'C2', 'C3'):
    q = bank[cls][0]
    hh = terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=2)
    gm.generate(seed=7, wheel=True, profile='std', h_fn=hh)
    e = ForestWheelEnv(profile='std', generate=False)
    surv = reach = 0
    lens = []
    for ep in range(10):
        o, _ = e.reset(seed=100 + ep)
        term = False
        steps = 0
        while True:
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, info = e.step(a)
            steps += 1
            if term or trunc:
                break
        lens.append(steps)
        if not term:
            surv += 1
        if info.get('success'):
            reach += 1
    print('{} @ {}: 存活 {}/10, 到达 {}/10, 平均步数 {:.0f}'.format(
        os.path.basename(ckpt), cls, surv, reach, sum(lens) / len(lens)))
print('EVAL_DONE')
