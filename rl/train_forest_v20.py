# -*- coding: utf-8 -*-
"""forest v20: 地形前瞻观测版 — 从零训练 (治"盲走": 倒车/摔率/绕而不达的共同根因)

obs 57→64 (+7: 前方5点 1.0-3.4m + 侧方2点 ±40°@2.2m 高程差)
奖励沿用 v18 冠军配置; 课程同 v15 三阶段; 24M
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

    NENV, TOTAL, PHASE = 12, 24_000_000, 8_000_000
    BASE_CFG = dict(contour_bias=1, prog_w=2.2, alive_w=0.4, head_w=0.6,
                    fwd_vel_w=0.8, lat_w=0.15, steep_w=0.012, steep_ref_deg=18.0,
                    lookahead=True)

    base_fn = terr.make_h_fn(seed=7, span=24.0, amp1=0.30, amp2=0.15,
                             amp3=0.05, rough_amp=0.02, tilt_deg=30.0)
    gm.generate(seed=7, wheel=True, profile='std', h_fn=base_fn)
    meta = json.load(open('terrain_meta.json', encoding='utf-8'))
    meta['kind'] = 'rand'
    json.dump(meta, open('terrain_meta.json', 'w', encoding='utf-8'))

    def set_cfg(lo, hi):
        json.dump(dict(rand_min_slope_deg=lo, rand_max_slope_deg=hi, **BASE_CFG),
                  open('env_cfg.json', 'w'))

    set_cfg(0.0, 12.0)

    class Curriculum(BaseCallback):
        cur = 'A'

        def _on_step(self):
            t = self.model.num_timesteps
            if PHASE <= t < 2 * PHASE and self.cur != 'B':
                self.cur = 'B'
                set_cfg(4.0, 22.0)
                print('[课程] B: 4-22°', flush=True)
            elif t >= 2 * PHASE and self.cur != 'C':
                self.cur = 'C'
                set_cfg(8.0, 30.0)
                print('[课程] C: 8-30°', flush=True)
            return True

    class SaveEvery(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_save = 6_000_000

        def _on_step(self):
            if self.model.num_timesteps >= self.next_save:
                self.model.save('models/forest_v20/ckpt_{:02.0f}M'.format(
                    self.model.num_timesteps / 1e6))
                self.next_save += 6_000_000
            return True

    env = SubprocVecEnv(
        [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False, seed=900 + i),
                             'logs/forest_v20_{}'.format(i)) for i in range(NENV)],
        start_method='spawn')
    model = PPO('MlpPolicy', env, verbose=0, seed=7,
                learning_rate=3e-4, n_steps=2048, batch_size=192,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.003, device='cpu')

    os.makedirs('models/forest_v20', exist_ok=True)
    t0 = time.time()
    model.learn(total_timesteps=TOTAL,
                callback=CallbackList([Curriculum(), SaveEvery(),
                                       EpLogCallback(print_every=300.0)]))
    model.save('models/forest_v20/final')
    env.close()
    print('forest_v20_DONE 用时 %.1f 小时' % ((time.time() - t0) / 3600), flush=True)


if __name__ == '__main__':
    main()
