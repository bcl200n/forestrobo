# -*- coding: utf-8 -*-
"""废墟搜救训练: 足式机器狗在建筑废墟中搜索 4 名幸存者
用法: python train_rubble.py --worlds 32 --steps 2000000
"""
import argparse
import os
import time

from stable_baselines3 import PPO

from generate_rubble import generate
from newton_vecenv import ForestDogNewtonEnv
from train_forest_dog import EpLogCallback

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worlds', type=int, default=32)
    p.add_argument('--steps', type=int, default=2_000_000)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--name', type=str, default='rubble_v1')
    a = p.parse_args()

    print('=== 1/2 生成废墟搜救场景 ===')
    generate(seed=a.seed)

    print(f'=== 2/2 Newton GPU 搜救训练（{a.worlds} 世界 × 找 4 名幸存者）===')
    env = ForestDogNewtonEnv(num_envs=a.worlds, seed=a.seed, task='search')
    model = PPO(
        'MlpPolicy', env,
        n_steps=256, batch_size=min(4096, 256 * a.worlds), n_epochs=5,
        learning_rate=1.5e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.015,                       # 搜救需要探索
        policy_kwargs=dict(net_arch=[128, 128, 64]),
        verbose=0, seed=a.seed, device='cpu')
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=EpLogCallback(print_every=30.0))
    out = os.path.join('models', a.name)
    os.makedirs(out, exist_ok=True)
    model.save(os.path.join(out, 'final'))
    print(f'训练完成: {a.steps:,} 步 / {(time.time() - t0) / 60:.1f} 分钟 '
          f'→ models/{a.name}/final.zip')


if __name__ == '__main__':
    main()
