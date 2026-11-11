# -*- coding: utf-8 -*-
"""forest v22: 非对称轮速限幅 — 倒车物理低效化, 最后一击治倒车

原理: 反转轮速只给 30% 力度 (rev_scale 0.3), 前转全速。
倒车在物理上变得低效 → PPO 自会发现正向更优 → 不需要奖励强迫。
v18 热启动 (57 维 obs), 8-30° 直接续训 12M (精调, 非从零)。
"""
import json
import os
import time


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    from mujoco_forest_env import ForestWheelEnv
    from train_forest_dog import EpLogCallback
    import generate_model as gm
    import terrain as terr

    NENV, TOTAL = 12, 12_000_000

    base_fn = terr.make_h_fn(seed=7, span=24.0, amp1=0.30, amp2=0.15,
                             amp3=0.05, rough_amp=0.02, tilt_deg=30.0)
    gm.generate(seed=7, wheel=True, profile='std', h_fn=base_fn)
    meta = json.load(open('terrain_meta.json', encoding='utf-8'))
    meta['kind'] = 'rand'
    json.dump(meta, open('terrain_meta.json', 'w', encoding='utf-8'))

    json.dump({'rand_min_slope_deg': 8.0, 'rand_max_slope_deg': 30.0,
               'contour_bias': 1,
               'prog_w': 2.2, 'alive_w': 0.4, 'head_w': 0.6,
               'fwd_vel_w': 0.8, 'lat_w': 0.15,
               'steep_w': 0.012, 'steep_ref_deg': 18.0,
               'lookahead': False, 'rev_scale': 0.3},
              open('env_cfg.json', 'w'))
    print('cfg v22: rev_scale 0.3 (倒车限到 30% 力度)', flush=True)

    env = SubprocVecEnv(
        [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False, seed=1100 + i),
                             'logs/forest_v22_{}'.format(i)) for i in range(NENV)],
        start_method='spawn')
    model = PPO.load('models/forest_v18/final', env=env, device='cpu')
    model.learning_rate = 1e-4

    class SaveEvery(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_save = 4_000_000
        def _on_step(self):
            if self.model.num_timesteps >= self.next_save:
                self.model.save('models/forest_v22/ckpt_{:02.0f}M'.format(
                    self.model.num_timesteps / 1e6))
                self.next_save += 4_000_000
            return True

    os.makedirs('models/forest_v22', exist_ok=True)
    t0 = time.time()
    model.learn(total_timesteps=TOTAL,
                callback=CallbackList([SaveEvery(), EpLogCallback(print_every=300.0)]),
                reset_num_timesteps=False)
    model.save('models/forest_v22/final')
    env.close()
    print('forest_v22_DONE 用时 %.1f 小时' % ((time.time() - t0) / 3600), flush=True)


if __name__ == '__main__':
    main()
