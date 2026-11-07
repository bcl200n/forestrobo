# -*- coding: utf-8 -*-
"""轮足 v4b: 修正到达半径(1.2m 真实抵达), 目标 3~6m, v3 热启动, 3M 步"""
import os
import time

from stable_baselines3 import PPO

import generate_model as gm
from newton_vecenv import ForestDogNewtonEnv
from train_forest_dog import EpLogCallback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print('=== v4b: 生成标准山地 ===', flush=True)
gm.generate(wheel=True, profile='std')
env = ForestDogNewtonEnv(num_envs=64, seed=7, wheel=True,
                         goal_range=(3.0, 6.0))
model = PPO.load('models/wheel_v3/final', env=env, device='cpu')
t0 = time.time()
model.learn(total_timesteps=3_000_000, callback=EpLogCallback(print_every=30.0))
os.makedirs('models/wheel_v4b', exist_ok=True)
model.save('models/wheel_v4b/final')
env.close()
print('V4B_DONE 用时 %.1f 分钟 → models/wheel_v4b/final.zip'
      % ((time.time() - t0) / 60), flush=True)
