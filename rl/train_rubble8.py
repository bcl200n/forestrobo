# -*- coding: utf-8 -*-
"""废墟搜救 v8: MuJoCo 物理训练(消除引擎失配), 12 并行环境
阶段 1 平地+近距 1.2M → 阶段 2 完整废墟 1.8M
"""
import os
import time


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from mujoco_search_env import RubbleSearchEnv
    from train_forest_dog import EpLogCallback

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    NENV = 12

    def make(profile):
        return SubprocVecEnv(
            [(lambda p=profile: (lambda: RubbleSearchEnv(profile=p)))()
             for _ in range(NENV)], start_method='spawn')

    t0 = time.time()
    print(f'=== 阶段 1/2: MuJoCo 平地+近距 (1.2M × {NENV} 并行) ===', flush=True)
    env1 = make('flat')
    model = PPO('MlpPolicy', env1, n_steps=256, batch_size=3072, n_epochs=5,
                learning_rate=1.5e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.02,
                policy_kwargs=dict(net_arch=[128, 128, 64]),
                verbose=0, seed=11, device='cpu')
    model.learn(total_timesteps=1_200_000,
                callback=EpLogCallback(print_every=30.0))
    os.makedirs('models/rubble_v8', exist_ok=True)
    model.save('models/rubble_v8/phase1')
    print('阶段 1 完成 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)
    env1.close()

    print('=== 阶段 2/2: 完整废墟 热启动 (1.8M) ===', flush=True)
    env2 = make('full')
    model = PPO.load('models/rubble_v8/phase1', env=env2, device='cpu')
    model.learn(total_timesteps=1_800_000,
                callback=EpLogCallback(print_every=30.0))
    model.save('models/rubble_v8/final')
    env2.close()
    print('RUBBLE_V8_DONE 总用时 %.1f 分钟' % ((time.time() - t0) / 60),
          flush=True)


if __name__ == '__main__':
    main()
