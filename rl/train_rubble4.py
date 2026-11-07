# -*- coding: utf-8 -*-
"""废墟搜救 v4: 结构化动作(转向/速度指令) + 奖励重平衡(发现+60)
阶段 1 平地 1.2M → 阶段 2 完整废墟 1.8M
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
print('=== 阶段 1/2: 平地 学转向找人 (1.2M) ===', flush=True)
env1 = make_env('flat')
model = PPO('MlpPolicy', env1, n_steps=256,
            batch_size=min(4096, 256 * WORLDS), n_epochs=5,
            learning_rate=1.5e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.02,
            policy_kwargs=dict(net_arch=[128, 128, 64]),
            verbose=0, seed=11, device='cpu')
model.learn(total_timesteps=1_200_000,
            callback=EpLogCallback(print_every=30.0))
os.makedirs('models/rubble_v4', exist_ok=True)
model.save('models/rubble_v4/phase1')
print('阶段 1 完成 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)
env1.close()

print('=== 阶段 2/2: 完整废墟 热启动 (1.8M) ===', flush=True)
env2 = make_env('full')
model = PPO.load('models/rubble_v4/phase1', env=env2, device='cpu')
model.learn(total_timesteps=1_800_000,
            callback=EpLogCallback(print_every=30.0))
model.save('models/rubble_v4/final')
env2.close()
print('RUBBLE_V4_DONE 总用时 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)
