# -*- coding: utf-8 -*-
"""rubble_v8 阶段 2: 完整废墟热启动(场景由父进程生成一次, 子进程只读)"""


def main():
    import os
    import time

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    import generate_rubble as gr
    from mujoco_search_env import RubbleSearchEnv
    from train_forest_dog import EpLogCallback

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    NENV = 12
    gr.generate(seed=11, profile='full')     # 父进程生成一次

    t0 = time.time()
    env2 = SubprocVecEnv(
        [lambda: RubbleSearchEnv(profile='full', generate=False)
         for _ in range(NENV)], start_method='spawn')
    model = PPO.load('models/rubble_v8/phase1', env=env2, device='cpu')
    model.learn(total_timesteps=1_800_000,
                callback=EpLogCallback(print_every=30.0))
    model.save('models/rubble_v8/final')
    env2.close()
    print('V8_PHASE2_DONE 用时 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
