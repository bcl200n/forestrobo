# -*- coding: utf-8 -*-
"""诊断: 两个模型的 obs 空间 + env_cfg lookahead 开关是否生效"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from mujoco_forest_env import ForestWheelEnv
import generate_model as gm
import terrain as terr

for tag, p in (('v18', 'models/forest_v18/final'), ('v20', 'models/forest_v20/final')):
    m = PPO.load(p, device='cpu')
    print('{} policy obs space: {}'.format(tag, m.observation_space.shape))

json.dump({'lookahead': False}, open('env_cfg.json', 'w'))
h_fn = terr.make_h_fn(seed=1, span=24.0, amp1=0.3, amp2=0.15, amp3=0.05,
                      rough_amp=0.02, tilt_deg=30.0)
gm.generate(seed=1, wheel=True, profile='std', h_fn=h_fn)
e = ForestWheelEnv(profile='std', generate=False)
print('lookahead=False env obs space:', e.observation_space.shape, ' flag:', e._use_lookahead)

json.dump({'lookahead': True}, open('env_cfg.json', 'w'))
e2 = ForestWheelEnv(profile='std', generate=False)
print('lookahead=True  env obs space:', e2.observation_space.shape, ' flag:', e2._use_lookahead)
