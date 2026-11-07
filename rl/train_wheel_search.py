# -*- coding: utf-8 -*-
"""轮足差速搜救: 专家验证 + PPO 训练(2维动作, 干净接口)"""


def expert(obs):
    a = np.zeros(2, np.float32)
    a[0] = 0.75                                   # 前进
    a[1] = float(np.clip(-1.5 * obs[8], -1, 1))   # 目标偏左(ty>0)→左转
    return a


def main():
    import os
    import time

    import numpy as np

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import generate_rubble as gr
    from mujoco_wheel_env import RubbleWheelEnv

    # 专家验证
    gr.generate(seed=11, profile='flat', wheel=True)
    env = RubbleWheelEnv(profile='flat', generate=False)
    tot = []
    for _ in range(10):
        o, _ = env.reset()
        for _ in range(900):
            o, r, term, tr, _ = env.step(expert(o))
            if term:
                break
        tot.append(int(env.found.sum()))
    print(f'专家: 平均发现 {np.mean(tot):.1f}/{env.k} | 分布 {sorted(tot)}',
          flush=True)

    # PPO 训练
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from train_forest_dog import EpLogCallback

    NENV = 12
    venv = SubprocVecEnv(
        [lambda: Monitor(RubbleWheelEnv(profile='flat', generate=False),
                         f'logs/wheel_search_{i}')
         for i in range(NENV)], start_method='spawn')
    t0 = time.time()
    model = PPO('MlpPolicy', venv, n_steps=256, batch_size=3072, n_epochs=5,
                learning_rate=lambda p: 2e-4 * (1 - p), gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
                policy_kwargs=dict(net_arch=[128, 128, 64]),
                verbose=0, seed=11, device='cpu')
    model.learn(total_timesteps=6_000_000,
                callback=EpLogCallback(print_every=60.0))
    os.makedirs('models/wheel_search', exist_ok=True)
    model.save('models/wheel_search/final')
    venv.close()
    print('WHEEL_SEARCH_DONE 用时 %.1f 分钟' % ((time.time() - t0) / 60),
          flush=True)


if __name__ == '__main__':
    import numpy as np
    main()
