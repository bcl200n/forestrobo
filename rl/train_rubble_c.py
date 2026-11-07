# -*- coding: utf-8 -*-
"""废墟搜救课程训练:
阶段 1  平地废墟(无瓦砾堆, 少障碍) 1.5M → 学会"走路找人"
阶段 2  完整废墟 2.5M, 热启动 → 适应瓦砾地形
产物: models/rubble_c/phase1.zip + final.zip
"""
import os
import time

from stable_baselines3 import PPO

import generate_rubble as gr
from newton_vecenv import ForestDogNewtonEnv
from train_forest_dog import EpLogCallback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
WORLDS = 32


def make_env(profile):
    gr.generate(seed=11, profile=profile)
    return ForestDogNewtonEnv(num_envs=WORLDS, seed=11, task='search')


t0 = time.time()
print('=== 阶段 1/2: 平地废墟 学找人 (1.5M 步) ===', flush=True)
env1 = make_env('flat')
model = PPO('MlpPolicy', env1, n_steps=256,
            batch_size=min(4096, 256 * WORLDS), n_epochs=5,
            learning_rate=1.5e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.015,
            policy_kwargs=dict(net_arch=[128, 128, 64]),
            verbose=0, seed=11, device='cpu')
model.learn(total_timesteps=1_500_000, callback=EpLogCallback(print_every=30.0))
os.makedirs('models/rubble_c', exist_ok=True)
model.save('models/rubble_c/phase1')
print('阶段 1 完成, 用时 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)
env1.close()

print('=== 阶段 2/2: 完整废墟 热启动 (2.5M 步) ===', flush=True)
env2 = make_env('full')
model = PPO.load('models/rubble_c/phase1', env=env2, device='cpu')
model.learn(total_timesteps=2_500_000, callback=EpLogCallback(print_every=30.0))
model.save('models/rubble_c/final')
env2.close()
print('RUBBLE_C_DONE 总用时 %.1f 分钟 → models/rubble_c/final.zip'
      % ((time.time() - t0) / 60), flush=True)
