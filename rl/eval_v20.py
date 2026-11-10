# -*- coding: utf-8 -*-
"""v20 验收: v18 (57维, lookahead关) vs v20 (64维, lookahead开), 留出块 40 回合/档"""
import io
import json
import math
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


def yaw_of(env):
    w, x_, y_, z_ = env.data.qpos[3:7]
    return math.atan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))


def run(model, env, n_ep, seed0):
    reach = fall = timeout = 0
    fwd_all = []
    for ep in range(n_ep):
        o, _ = env.reset(seed=seed0 + ep)
        term = False
        prev = env.data.qpos[:2].copy()
        angs = []
        for k in range(500):
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, info = env.step(a)
            p = env.data.qpos[:2]
            v = p - prev
            prev = p
            if np.hypot(*v) > 1e-4:
                va = math.atan2(v[1], v[0])
                d = math.degrees((va - yaw_of(env) + math.pi) % (2 * math.pi) - math.pi)
                angs.append(abs(d) < 90)
            if term or trunc:
                break
        fwd_all.append(np.mean(angs) if angs else 0.0)
        if info.get('is_success'):
            reach += 1
        elif term:
            fall += 1
        else:
            timeout += 1
    return reach, fall, timeout, float(np.mean(fwd_all))


def set_env_cfg(lookahead):
    cfg = dict(contour_bias=1, prog_w=2.2, alive_w=0.4, head_w=0.6,
               fwd_vel_w=0.8, lat_w=0.15, steep_w=0.012, steep_ref_deg=18.0,
               rand_min_slope_deg=8.0, rand_max_slope_deg=30.0,
               lookahead=lookahead)
    json.dump(cfg, open('env_cfg.json', 'w'))


MODELS = [('v18', 'models/forest_v18/final', False),
          ('v20', 'models/forest_v20/final', True)]
loaded = {n: PPO.load(p, device='cpu') for n, p, _ in MODELS}

results = {}
for kind in ('C1', 'C2', 'C3'):
    q = bank[kind][7]
    gm.generate(seed=999, wheel=True, profile='std',
                h_fn=terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=999))
    for name, path, la in MODELS:
        set_env_cfg(la)
        env = ForestWheelEnv(profile='std', generate=False)
        results.setdefault(name, {})[kind] = run(loaded[name], env, 40, 9800)
        env.close()

gm.generate(seed=999, wheel=True, profile='smooth')
for name, path, la in MODELS:
    set_env_cfg(la)
    env = ForestWheelEnv(profile='smooth', generate=False)
    results.setdefault(name, {})['smooth'] = run(loaded[name], env, 10, 9900)
    env.close()

log('== v20 冠军对照 (40 回合/档): 到达/摔/超时 | 正向 ==')
for name in ('v18', 'v20'):
    r3 = results[name]
    log('{}:'.format(name))
    for k in ('C1', 'C2', 'C3', 'smooth'):
        reach, fall, tmo, fwd = r3[k]
        n = 10 if k == 'smooth' else 40
        log('  {}: 到达 {}/{} 摔 {}/{} 超时 {}/{} | 正向 {:.0%}'.format(
            k, reach, n, fall, n, tmo, n, fwd))
json.dump({k: {kk: list(vv) for kk, vv in v.items()} for k, v in results.items()},
          open('rsi/v20_eval.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('V20_EVAL_DONE')
