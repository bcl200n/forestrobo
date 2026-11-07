# -*- coding: utf-8 -*-
"""rubble→forest v12: v11 热启动续训标准山地 12M"""


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

    import generate_model as gm
    gm.generate(seed=7, wheel=True, profile='std')
    env = SubprocVecEnv(
        [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False),
                             f'logs/forest_v13_{i}') for i in range(NENV)],
        start_method='spawn')
    model = PPO.load('models/forest_v12/final', env=env, device='cpu')
    t0 = time.time()
    model.learn(total_timesteps=12_000_000,
                callback=EpLogCallback(print_every=60.0))
    os.makedirs('models/forest_v13', exist_ok=True)
    model.save('models/forest_v13/final')
    env.close()
    print('forest_v13_DONE 用时 %.1f 分钟' % ((time.time() - t0) / 60),
          flush=True)


if __name__ == '__main__':
    main()


