# -*- coding: utf-8 -*-
"""林下机器狗强化学习训练（PPO, GPU 批量物理仿真）

用法:
  .venv\Scripts\python.exe train_forest_dog.py --worlds 64 --steps 5000000
  .venv\Scripts\python.exe train_forest_dog.py --worlds 8 --steps 40000   # 快速验证
可选:
  --seed 7        随机种子
  --name exp1     实验名（模型保存到 models/exp1/）
"""
import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from generate_model import generate
from newton_vecenv import ForestDogNewtonEnv


class EpLogCallback(BaseCallback):
    """打印回合回报与成功率"""

    def __init__(self, print_every=20.0):
        super().__init__()
        self.print_every = print_every
        self.rets, self.lens, self.succ = [], [], []
        self.last_t = time.time()

    def _on_step(self):
        infos = self.locals['infos']
        for info in infos:
            if 'episode' in info:
                self.rets.append(info['episode']['r'])
                self.lens.append(info['episode']['l'])
                self.succ.append(1.0 if info.get('success') else 0.0)
        now = time.time()
        if now - self.last_t > self.print_every and self.rets:
            n = len(self.rets)
            sps = self.num_timesteps / (now - self.model._total_t_start
                                        if hasattr(self.model, '_total_t_start')
                                        else max(now - self.start_t, 1e-9))
            print(f"steps {self.num_timesteps:>10,} | "
                  f"回合回报 {np.mean(self.rets[-min(n, 100):]):>8.1f} | "
                  f"成功率 {np.mean(self.succ[-min(n, 100):]) * 100:>5.1f}% | "
                  f"平均长度 {np.mean(self.lens[-min(n, 100):]):>6.0f}")
            self.last_t = now
        return True

    def _on_training_start(self):
        self.start_t = time.time()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worlds', type=int, default=64, help='并行世界数(GPU)')
    p.add_argument('--steps', type=int, default=5_000_000, help='总环境步数')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--name', type=str, default='forest_dog_ppo')
    p.add_argument('--terrain-seed', type=int, default=7)
    p.add_argument('--wheel', action='store_true', help='轮足构型 (16 维动作)')
    args = p.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    print('=== 1/3 生成林下场景模型 ===')
    generate(seed=args.terrain_seed, wheel=args.wheel)

    print(f'=== 2/3 创建 Newton GPU 批量环境（{args.worlds} 世界, '
          f'{"轮足" if args.wheel else "足式"}）===')
    env = ForestDogNewtonEnv(num_envs=args.worlds, seed=args.seed, wheel=args.wheel)

    print('=== 3/3 PPO 训练 ===')
    model = PPO(
        'MlpPolicy', env,
        n_steps=256,
        batch_size=min(4096, 256 * args.worlds),
        n_epochs=5,
        learning_rate=1.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128, 64]),
        verbose=0, seed=args.seed, device='cpu',
    )

    out_dir = os.path.join(base, 'models', args.name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    model.learn(total_timesteps=args.steps,
                callback=EpLogCallback(print_every=30.0))
    dt = time.time() - t0
    print(f'训练完成: {args.steps:,} 步 / {dt/60:.1f} 分钟 '
          f'({args.steps/max(dt,1e-9):,.0f} steps/s)')

    path = os.path.join(out_dir, 'final.zip')
    model.save(path)
    print(f'模型已保存: {path}')


if __name__ == '__main__':
    main()
