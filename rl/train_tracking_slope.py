# -*- coding: utf-8 -*-
"""过夜训练: 速度跟踪残差在粗糙斜坡上 (推广跟踪能力到非平面地形)"""


def main():
    import os
    import time
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from forest_tracking_residual import ForestTrackingEnv
    from train_forest_dog import EpLogCallback

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    NENV = 8

    # 更难的课程: 更陡坡 + 更多速度指令组合
    # 阶段1: 坡度 -8~+8 度, 标准指令
    # 阶段2: 坡度 -12~+12 度, 混合指令 + 传感器噪声
    t0 = time.time()
    print('=== 阶段 1/2: 斜坡 ±8° 跟踪 (2M) ===', flush=True)
    env1 = SubprocVecEnv(
        [lambda s=s, i=i: ForestTrackingEnv(slope=s, level='mixed', horizon=400)
         for i, s in enumerate([-8, -4, 0, 0, 4, 8, 2, -2])],
        start_method='spawn')
    model = PPO('MlpPolicy', env1, n_steps=256, batch_size=2048, n_epochs=5,
                learning_rate=lambda p: 2e-4 * (1 - p), gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128, 64]),
                verbose=0, seed=42, device='cpu')
    model.learn(total_timesteps=2_000_000,
                callback=EpLogCallback(print_every=120.0))
    os.makedirs('models/tracking_slope', exist_ok=True)
    model.save('models/tracking_slope/phase1')
    print('阶段 1 完成 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)
    env1.close()

    print('=== 阶段 2/2: 斜坡 ±12° + 更多噪声 (3M) ===', flush=True)
    env2 = SubprocVecEnv(
        [lambda s=s, i=i: ForestTrackingEnv(slope=s, level='mixed', horizon=500)
         for i, s in enumerate([-12, -6, 0, 0, 6, 12, 3, -3])],
        start_method='spawn')
    model = PPO.load('models/tracking_slope/phase1', env=env2, device='cpu')
    model.learn(total_timesteps=3_000_000,
                callback=EpLogCallback(print_every=120.0))
    model.save('models/tracking_slope/final')
    env2.close()
    print('TRACKING_SLOPE_DONE 总用时 %.1f 分钟' % ((time.time() - t0) / 60),
          flush=True)


if __name__ == '__main__':
    main()
