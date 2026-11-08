# -*- coding: utf-8 -*-
"""v14 断点续训: 从 ckpt_08_C3 续跑 chunk 09-11 (空树表崩溃已修复)"""


def main():
    import json
    import os
    import time

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from mujoco_forest_env import ForestWheelEnv
    from train_forest_dog import EpLogCallback
    import generate_model as gm
    import terrain as terr

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    bank = json.load(open('dem_patches.json', encoding='utf-8'))
    # 原计划 chunk 9,10,11
    schedule = [('C3', 1), ('C2', 3), ('C3', 3)]
    NENV, CHUNK = 12, 1_000_000

    model = PPO.load('models/forest_v14/ckpt_08_C3', device='cpu')
    t0 = time.time()
    for ci, (cls, pi) in enumerate(schedule):
        p = bank[cls][pi]
        h_fn = terr.make_dem_fn(p['lat'], p['lon'], p['rot'], seed=100 + ci + 9)
        gm.generate(seed=200 + ci + 9, wheel=True, profile='std', h_fn=h_fn)
        env = SubprocVecEnv(
            [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False),
                                 'logs/forest_v14r_{}_{}'.format(ci, i)) for i in range(NENV)],
            start_method='spawn')
        model.set_env(env)
        model.learn(total_timesteps=CHUNK, callback=EpLogCallback(print_every=120.0),
                    reset_num_timesteps=False)
        model.save('models/forest_v14/ckpt_{:02d}_{}'.format(ci + 9, cls))
        env.close()
        print('chunk {:02d} [{} {:.0f}°] DONE 累计 {:.1f}h'.format(
            ci + 9, cls, p['slope'], (time.time() - t0) / 3600), flush=True)

    model.save('models/forest_v14/final')
    print('forest_v14_RESUME_DONE', flush=True)


if __name__ == '__main__':
    main()
