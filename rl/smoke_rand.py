# -*- coding: utf-8 -*-
"""v15 随机地形引擎冒烟: 3 次重置 → hfield 变化 / 落点贴地 / 目标等高线偏置 / 随机步进"""
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

base_fn = terr.make_h_fn(seed=7, span=24.0, amp1=0.30, amp2=0.15, amp3=0.05,
                         rough_amp=0.02, tilt_deg=30.0)
gm.generate(seed=7, wheel=True, profile='std', h_fn=base_fn)
meta = json.load(open('terrain_meta.json', encoding='utf-8'))
meta['kind'] = 'rand'
json.dump(meta, open('terrain_meta.json', 'w', encoding='utf-8'))
json.dump({'rand_min_slope_deg': 0.0, 'rand_max_slope_deg': 25.0, 'contour_bias': 1},
          open('env_cfg.json', 'w'))

env = ForestWheelEnv(profile='std', generate=False)
print('rand 模式:', env._rand, ' elev={:.2f}'.format(env._elev))
snaps = []
for ep in range(3):
    o, _ = env.reset(seed=ep)
    z0 = env.data.qpos[2]
    h0 = float(env.h(1.0, env.data.qpos[1]))
    gv = env.goal - env.data.qpos[:2]
    snaps.append(env._hf_data[:5].copy())
    print('ep{}: 出生落差 z-h={:+.3f}m, hfield头5={}', z0 - 0.60 - h0)
    print('   hfield 前 5 值:', np.round(snaps[-1], 4))
    # 30 步随机动作
    term = False
    for k in range(30):
        o, r, term, trunc, info = env.step(env.action_space.sample())
        if term:
            break
    print('   30 步随机: term={} z_rel={:.2f}'.format(term, float(o[12])))
print('hfield 跨重置变化:', not np.allclose(snaps[0], snaps[1]))
print('SMOKE_RAND_DONE')
