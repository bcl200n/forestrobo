# -*- coding: utf-8 -*-
"""从 v20 训练 monitor CSV 提取 (obs, action, obs') 转移数据集

monitor CSV 只存回合奖励/长度, 不存逐步观测——需要回放 v20 策略
在随机地形上收集转移样本。
"""
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

N_EP = 400  # 400 回合 × ~150 步 ≈ 6 万转移

# 建 rand 环境 (lookahead 关, 用 57 维基线 obs)
base_fn = terr.make_h_fn(seed=7, span=24.0, amp1=0.30, amp2=0.15,
                         amp3=0.05, rough_amp=0.02, tilt_deg=30.0)
gm.generate(seed=7, wheel=True, profile='std', h_fn=base_fn)
meta = json.load(open('terrain_meta.json', encoding='utf-8'))
meta['kind'] = 'rand'
json.dump(meta, open('terrain_meta.json', 'w', encoding='utf-8'))

json.dump({'rand_min_slope_deg': 8.0, 'rand_max_slope_deg': 30.0,
           'contour_bias': 1, 'prog_w': 2.2, 'alive_w': 0.4, 'head_w': 0.6,
           'fwd_vel_w': 0.8, 'lat_w': 0.15, 'steep_w': 0.012,
           'steep_ref_deg': 18.0, 'lookahead': False},
          open('env_cfg.json', 'w'))

env = ForestWheelEnv(profile='std', generate=False)
model = PPO.load('models/forest_v20/final', device='cpu')

# v20 策略需要 64 维 obs, 但我们的 PINN 用 57 维 (截掉前瞻 7 维)
# 环境生成 57 维 (lookahead=False), 需要单独跑 v20 的话用 lookahead=True
# 但收集数据用 v18 (57 维, 同奖励配置) 更干净
model = PPO.load('models/forest_v18/final', device='cpu')

obs_buf, act_buf, next_buf = [], [], []
for ep in range(N_EP):
    o, _ = env.reset(seed=20000 + ep)
    done = False
    prev = o.copy()
    while not done:
        a, _ = model.predict(o, deterministic=False)  # 随机策略探索更广
        o, r, term, trunc, info = env.step(a)
        done = term or trunc
        obs_buf.append(prev)
        act_buf.append(a)
        next_buf.append(o.copy())
        prev = o.copy()

obs_s = np.array(obs_buf, np.float32)
act_s = np.array(act_buf, np.float32)
next_s = np.array(next_buf, np.float32)
os.makedirs('models/pinn', exist_ok=True)
np.savez_compressed('models/pinn/transitions.npz',
                    obs=obs_s, act=act_s, next=next_s)
print('转移数据: {} 条 (obs {}, act {})'.format(
    len(obs_s), obs_s.shape[1], act_s.shape[1]))
print('COLLECT_DONE')
