# -*- coding: utf-8 -*-
"""v15 终评: v13/v14/v15/零动作 四方对照, 真实 DEM 留出块 20 回合/档 + v15 域内抽查"""
import io
import json
import os
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from mujoco_forest_env import ForestWheelEnv
import generate_model as gm
import terrain as terr

bank = json.load(open('dem_patches.json', encoding='utf-8'))
LOG = open('rsi/log.md', 'a', encoding='utf-8')


def log(s):
    line = '[{}] {}'.format(time.strftime('%H:%M:%S'), s)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def rollout(policy, env, n_ep, seed0):
    surv = reach = 0
    lens = []
    for ep in range(n_ep):
        o, _ = env.reset(seed=seed0 + ep)
        term = False
        steps = 0
        for k in range(500):
            a = policy(o)
            o, r, term, trunc, info = env.step(a)
            steps += 1
            if term or trunc:
                break
        lens.append(steps)
        if not term:
            surv += 1
        if info.get('is_success'):
            reach += 1
    return surv, reach, sum(lens) / len(lens)


# ---- 1. 真实 DEM 留出块 (C1 20 / C2 28 / C3 36) ----
results = {}
policies = [
    ('v13', lambda o: PPO_MODELS['v13'].predict(o, deterministic=True)[0]),
    ('v14', lambda o: PPO_MODELS['v14'].predict(o, deterministic=True)[0]),
    ('v15', lambda o: PPO_MODELS['v15'].predict(o, deterministic=True)[0]),
    ('zero', lambda o: np.zeros(16, np.float32)),
]
PPO_MODELS = {'v13': PPO.load('models/forest_v13/final', device='cpu'),
              'v14': PPO.load('models/forest_v14/final', device='cpu'),
              'v15': PPO.load('models/forest_v15/final', device='cpu')}

for kind in ('C1', 'C2', 'C3'):
    q = bank[kind][7]
    hh = terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=999)
    gm.generate(seed=999, wheel=True, profile='std', h_fn=hh)
    env = ForestWheelEnv(profile='std', generate=False)
    for name, pol in policies:
        s, r_, l = rollout(pol, env, 20, 9000)
        results.setdefault(name, {})[kind] = (s, r_, l)
    env.close()

# 平地泛化
gm.generate(seed=999, wheel=True, profile='smooth')
env = ForestWheelEnv(profile='smooth', generate=False)
for name, pol in policies:
    s, r_, l = rollout(pol, env, 10, 9500)
    results.setdefault(name, {})['smooth'] = (s, r_, l)
env.close()

log('== v15 终评: 真实 DEM 留出块 (20 回合) ==')
for name in ('v13', 'v14', 'v15', 'zero'):
    r3 = results[name]
    log('{}: C1 {}/20 C2 {}/20 C3 {}/20 smooth {}/10'.format(
        name, *r3['C1'][:2], *r3['C2'][:2], *r3['C3'][:2], *r3['smooth'][:2]))

# ---- 2. v15 域内抽查: 全新随机地形 30° 上限, 未见种子 ----
base_fn = terr.make_h_fn(seed=13, span=24.0, amp1=0.30, amp2=0.15, amp3=0.05,
                         rough_amp=0.02, tilt_deg=30.0)
gm.generate(seed=13, wheel=True, profile='std', h_fn=base_fn)
meta = json.load(open('terrain_meta.json', encoding='utf-8'))
meta['kind'] = 'rand'
json.dump(meta, open('terrain_meta.json', 'w', encoding='utf-8'))
json.dump({'rand_min_slope_deg': 8.0, 'rand_max_slope_deg': 30.0, 'contour_bias': 1},
          open('env_cfg.json', 'w'))
env = ForestWheelEnv(profile='std', generate=False, seed=4242)
s, r_, l = rollout(lambda o: PPO_MODELS['v15'].predict(o, deterministic=True)[0], env, 20, 7000)
log('v15 域内 (随机 8-30° 全新种子): 存活 {}/20, 到达 {}/20, 平均 {:.0f} 步'.format(s, r_, l))
json.dump({k: {kk: list(vv) for kk, vv in v.items()} for k, v in results.items()},
          open('rsi/v15_final_eval.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('V15_FINAL_EVAL_DONE')
