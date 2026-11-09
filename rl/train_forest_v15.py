# -*- coding: utf-8 -*-
"""forest v15: 逐回合地形随机化训练 (修复单地形过拟合的根本管线)

与 v13/v14 的本质区别:
  - 每个回合随机采样地形 (坡度课程 12°→22°→30°, 坡向/微地貌/树位全随机) → 策略必须泛化
  - 从零训练 (v13 单地形先验有毒: 在未见地形上 1/10, 且续训自举失败)
  - 目标方向 70% 偏置等高线 (横坡是机械稳定方向, mech_probe 实证)
课程经 env_cfg.json 热更新 (worker 每回合检查 mtime, 无需重启):
  0-8M: max_slope 12° | 8-16M: 22° | 16-24M: 30°
"""
import json
import os
import time


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    from mujoco_forest_env import ForestWheelEnv
    from train_forest_dog import EpLogCallback
    import generate_model as gm
    import terrain as terr

    NENV = 12
    TOTAL = 24_000_000

    # 生成 rand 底模: elevation 按 30° 最坏情况定型 (后续每回合只改数据不改尺寸)
    base_fn = terr.make_h_fn(seed=7, span=24.0, amp1=0.30, amp2=0.15,
                             amp3=0.05, rough_amp=0.02, tilt_deg=30.0)
    gm.generate(seed=7, wheel=True, profile='std', h_fn=base_fn)
    meta_p = 'terrain_meta.json'
    meta = json.load(open(meta_p, encoding='utf-8'))
    meta['kind'] = 'rand'
    json.dump(meta, open(meta_p, 'w', encoding='utf-8'))
    print('rand 底模: elevation={:.2f} span={}'.format(meta['elevation'], meta['span']), flush=True)

    # 初始课程 (Phase A)
    json.dump({'rand_min_slope_deg': 0.0, 'rand_max_slope_deg': 12.0, 'contour_bias': 1},
              open('env_cfg.json', 'w'))

    class Curriculum(BaseCallback):
        def _on_step(self):
            t = self.model.num_timesteps
            if 8_000_000 <= t < 16_000_000 and self.cur != 'B':
                self.cur = 'B'
                json.dump({'rand_min_slope_deg': 4.0, 'rand_max_slope_deg': 22.0,
                           'contour_bias': 1}, open('env_cfg.json', 'w'))
                print('[课程] Phase B: 坡度 4-22°', flush=True)
            elif t >= 16_000_000 and self.cur != 'C':
                self.cur = 'C'
                json.dump({'rand_min_slope_deg': 8.0, 'rand_max_slope_deg': 30.0,
                           'contour_bias': 1}, open('env_cfg.json', 'w'))
                print('[课程] Phase C: 坡度 8-30°', flush=True)
            return True

        cur = 'A'

    env = SubprocVecEnv(
        [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False, seed=100 + i),
                             'logs/forest_v15_{}'.format(i)) for i in range(NENV)],
        start_method='spawn')

    model = PPO('MlpPolicy', env, verbose=0, seed=7,
                learning_rate=3e-4, n_steps=2048, batch_size=192,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.003,
                device='cpu')

    os.makedirs('models/forest_v15', exist_ok=True)
    t0 = time.time()

    class SaveEvery(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_save = 2_000_000

        def _on_step(self):
            if self.model.num_timesteps >= self.next_save:
                self.model.save('models/forest_v15/ckpt_{:02.0f}M'.format(
                    self.model.num_timesteps / 1e6))
                self.next_save += 2_000_000
            return True

    from stable_baselines3.common.callbacks import CallbackList
    model.learn(total_timesteps=TOTAL,
                callback=CallbackList([Curriculum(), SaveEvery(),
                                       EpLogCallback(print_every=300.0)]))
    model.save('models/forest_v15/final')
    env.close()
    print('forest_v15_DONE 用时 %.1f 小时' % ((time.time() - t0) / 3600), flush=True)


if __name__ == '__main__':
    main()
