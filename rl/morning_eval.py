# -*- coding: utf-8 -*-
"""晨报最终对照评估: v13 vs v14 vs RSI最佳, 留出地形块, 20 回合/档"""
import io
import json
import math
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from mujoco_forest_env import ForestWheelEnv
import generate_model as gm
import terrain as terr

bank = json.load(open('dem_patches.json', encoding='utf-8'))
MODELS = [('v13', 'models/forest_v13/final'),
          ('v14', 'models/forest_v14/final'),
          ('rsi_best', 'models/rsi_best.zip')]

rows = {}
for tag, path in MODELS:
    model = PPO.load(path, device='cpu')
    res = {}
    for kind in ('C1', 'C2', 'C3'):
        q = bank[kind][7]                     # 留出块
        hh = terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=999)
        gm.generate(seed=999, wheel=True, profile='std', h_fn=hh)
        env = ForestWheelEnv(profile='std', generate=False)
        surv = reach = 0
        causes = {}
        for ep in range(20):
            o, _ = env.reset(seed=9000 + ep)
            term = False
            cause = 'timeout'
            for k in range(500):
                a, _ = model.predict(o, deterministic=True)
                o, r, term, trunc, info = env.step(a)
                if term or trunc:
                    cause = info.get('failure') or ('reach' if info.get('is_success') else 'timeout')
                    break
            if not term:
                surv += 1
            if info.get('is_success'):
                reach += 1
            causes[cause] = causes.get(cause, 0) + 1
        res[kind] = dict(surv=surv, reach=reach, causes=causes, slope=q['slope'])
    # 平地泛化 (未见 seed)
    gm.generate(seed=999, wheel=True, profile='smooth')
    env = ForestWheelEnv(profile='smooth', generate=False)
    ss = 0
    for ep in range(10):
        o, _ = env.reset(seed=9500 + ep)
        term = False
        for k in range(500):
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        if not term:
            ss += 1
    res['smooth'] = dict(surv=ss)
    rows[tag] = res
    print('{}: C1 {}/20 C2 {}/20 C3 {}/20 smooth {}/10'.format(
        tag, surv_c1 := res['C1']['surv'], res['C2']['surv'], res['C3']['surv'], ss))

with open('rsi/morning_eval.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print('MORNING_EVAL_DONE')
