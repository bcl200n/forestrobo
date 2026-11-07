# -*- coding: utf-8 -*-
"""rubble_v9: 平地 20M 步决战(足够样本量), Monitor 曲线可见"""


def main():
    import os
    import time

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    import generate_rubble as gr
    from mujoco_search_env import RubbleSearchEnv
    from train_forest_dog import EpLogCallback

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    NENV = 12
    gr.generate(seed=11, profile='flat')
    os.makedirs('logs', exist_ok=True)

    env = SubprocVecEnv(
        [lambda i=i: Monitor(RubbleSearchEnv(profile='flat', generate=False),
                             f'logs/rubble_v9_{i}') for i in range(NENV)],
        start_method='spawn')

    t0 = time.time()
    model = PPO('MlpPolicy', env, n_steps=256, batch_size=3072, n_epochs=5,
                learning_rate=lambda p: 2e-4 * (1 - p), gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
                policy_kwargs=dict(net_arch=[128, 128, 64]),
                verbose=0, seed=11, device='cpu')
    model.learn(total_timesteps=20_000_000,
                callback=EpLogCallback(print_every=60.0))
    os.makedirs('models/rubble_v9', exist_ok=True)
    model.save('models/rubble_v9/final')
    env.close()
    print('V9_DONE 用时 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
