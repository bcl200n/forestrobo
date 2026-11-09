# -*- coding: utf-8 -*-
"""forest v17: v16 续训 — 正向行驶 + 缓坡绕行 (用户建议)

新增三项奖励:
  fwd_vel_w 0.8: 奖励速度的机头方向分量 (治倒车)
  lat_w 0.15:    侧滑轻罚 (治蟹行)
  steep_w 0.03:  当地坡度超过 18° 二次惩罚 (同样到目标, 绕缓路得分高)
课程: 随机 8-30° 续跑 24M, v16 热启动
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

    NENV = 12
    TOTAL = 24_000_000

    base_fn = terr.make_h_fn(seed=7, span=24.0, amp1=0.30, amp2=0.15,
                             amp3=0.05, rough_amp=0.02, tilt_deg=30.0)
    gm.generate(seed=7, wheel=True, profile='std', h_fn=base_fn)
    meta = json.load(open('terrain_meta.json', encoding='utf-8'))
    meta['kind'] = 'rand'
    json.dump(meta, open('terrain_meta.json', 'w', encoding='utf-8'))

    json.dump({'rand_min_slope_deg': 8.0, 'rand_max_slope_deg': 30.0,
               'contour_bias': 1,
               'prog_w': 1.8, 'alive_w': 0.4, 'head_w': 0.6,
               'fwd_vel_w': 0.8, 'lat_w': 0.15,
               'steep_w': 0.03, 'steep_ref_deg': 18.0},
              open('env_cfg.json', 'w'))
    print('cfg v17: head 0.6 fwd_vel 0.8 lat 0.15 steep 0.03@18°', flush=True)

    env = SubprocVecEnv(
        [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False, seed=600 + i),
                             'logs/forest_v17_{}'.format(i)) for i in range(NENV)],
        start_method='spawn')
    model = PPO.load('models/forest_v16/final', env=env, device='cpu')
    model.learning_rate = 1e-4

    class SaveEvery(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_save = 4_000_000

        def _on_step(self):
            if self.model.num_timesteps >= self.next_save:
                self.model.save('models/forest_v17/ckpt_{:02.0f}M'.format(
                    self.model.num_timesteps / 1e6))
                self.next_save += 4_000_000
            return True

    os.makedirs('models/forest_v17', exist_ok=True)
    t0 = time.time()
    model.learn(total_timesteps=TOTAL,
                callback=CallbackList([SaveEvery(), EpLogCallback(print_every=300.0)]),
                reset_num_timesteps=False)
    model.save('models/forest_v17/final')
    env.close()
    print('forest_v17_DONE 用时 %.1f 小时' % ((time.time() - t0) / 3600), flush=True)


if __name__ == '__main__':
    main()
