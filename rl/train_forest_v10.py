# -*- coding: utf-8 -*-
"""林下轮足导航 v10: MuJoCo 原生训练(同物理回放) + 平滑→标准地形课程 + 摔倒重罚
阶段 1 smooth 8M → 阶段 2 std 12M, 12 CPU 并行
"""


def main():
    import os
    import time

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from mujoco_forest_env import ForestWheelEnv
    from train_forest_dog import EpLogCallback

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    NENV = 12

    def make(profile):
        return SubprocVecEnv(
            [lambda p=profile, i=i: Monitor(ForestWheelEnv(profile=p,
                                                           generate=False),
                                            f'logs/forest_v10_{p}_{i}')
             for i in range(NENV)], start_method='spawn')

    import generate_model as gm
    t0 = time.time()
    print('=== 阶段 1/2: smooth 8M (学站稳+滚动) ===', flush=True)
    gm.generate(seed=7, wheel=True, profile='smooth')
    env1 = make('smooth')
    model = PPO('MlpPolicy', env1, n_steps=256, batch_size=3072, n_epochs=5,
                learning_rate=lambda p: 2e-4 * (1 - p), gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128, 64]),
                verbose=0, seed=7, device='cpu')
    model.learn(total_timesteps=8_000_000,
                callback=EpLogCallback(print_every=60.0))
    os.makedirs('models/forest_v10', exist_ok=True)
    model.save('models/forest_v10/phase1')
    print('阶段 1 完成 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)
    env1.close()

    print('=== 阶段 2/2: std 山地 12M (抗摔迁移) ===', flush=True)
    gm.generate(seed=7, wheel=True, profile='std')
    env2 = make('std')
    model = PPO.load('models/forest_v10/phase1', env=env2, device='cpu')
    model.learn(total_timesteps=12_000_000,
                callback=EpLogCallback(print_every=60.0))
    model.save('models/forest_v10/final')
    env2.close()
    print('FOREST_V10_DONE 总用时 %.1f 分钟' % ((time.time() - t0) / 60),
          flush=True)


if __name__ == '__main__':
    main()
