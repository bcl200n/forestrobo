# -*- coding: utf-8 -*-
"""轮足 v3 课程学习训练:
阶段 1  平滑地形(缓起伏) 2M 步 → 学满速滚动导航
阶段 2  标准山地地形 3M 步, 从阶段 1 热启动 → 适应粗糙地形
产物: models/wheel_v3/phase1.zip + final.zip
"""
import os
import time

from stable_baselines3 import PPO

import generate_model as gm
from newton_vecenv import ForestDogNewtonEnv
from train_forest_dog import EpLogCallback

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

WORLDS = 64
PH1, PH2 = 2_000_000, 3_000_000


def make_env(profile):
    gm.generate(wheel=True, profile=profile)
    return ForestDogNewtonEnv(num_envs=WORLDS, seed=7, wheel=True)


def ppo_common(env):
    return dict(
        n_steps=256, batch_size=min(4096, 256 * WORLDS), n_epochs=5,
        learning_rate=1.5e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128, 64]),
        verbose=0, seed=7, device='cpu')


t0 = time.time()
print('=== 阶段 1/2: 平滑地形 学滚动 (%d 步) ===' % PH1, flush=True)
env1 = make_env('smooth')
model = PPO('MlpPolicy', env1, **ppo_common(env1))
model.learn(total_timesteps=PH1, callback=EpLogCallback(print_every=30.0))
os.makedirs('models/wheel_v3', exist_ok=True)
model.save('models/wheel_v3/phase1')
print('阶段 1 完成, 用时 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)
env1.close()

print('=== 阶段 2/2: 标准山地 热启动 (%d 步) ===' % PH2, flush=True)
env2 = make_env('std')
model = PPO.load('models/wheel_v3/phase1', env=env2, device='cpu')
model.learn(total_timesteps=PH2, callback=EpLogCallback(print_every=30.0))
model.save('models/wheel_v3/final')
env2.close()
print('V3_DONE 总用时 %.1f 分钟 → models/wheel_v3/final.zip'
      % ((time.time() - t0) / 60), flush=True)
