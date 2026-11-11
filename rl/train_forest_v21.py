# -*- coding: utf-8 -*-
"""forest v21: PINN-augmented PPO — 两阶段训练

阶段1: PINN 虚拟环境预训练 (纯神经网络前向, 无 MuJoCo, ~5分钟)
阶段2: MuJoCo 真实仿真精调 (少量步数, 从 PINN 先验热启动)

对照判据: v21 (PINN预训练+精调) vs v20 (纯 MuJoCo)
  - 同等总训练时间下质量更高?
  - 同等质量下 MuJoCo 步数更少?
"""
import json
import os
import time


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from pinn_env import PINNEnv
    from train_forest_dog import EpLogCallback

    # ---- 阶段 1: PINN 虚拟环境预训练 (快) ----
    print('=' * 60, flush=True)
    print('阶段1: PINN 虚拟环境预训练', flush=True)
    print('=' * 60, flush=True)
    PRETRAIN_STEPS = 5_000_000  # 500万步在 PINN 上 ≈ 几分钟
    env = DummyVecEnv([lambda: Monitor(PINNEnv(), 'logs/pinn_pre')])
    model = PPO('MlpPolicy', env, verbose=0, seed=7,
                learning_rate=3e-4, n_steps=2048, batch_size=192,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.003, device='cpu')
    t0 = time.time()
    model.learn(total_timesteps=PRETRAIN_STEPS,
                callback=EpLogCallback(print_every=60.0))
    print('PINN 预训练完成: {} 步 / {:.1f} 分钟'.format(
        PRETRAIN_STEPS, (time.time() - t0) / 60), flush=True)
    os.makedirs('models/forest_v21', exist_ok=True)
    model.save('models/forest_v21/pinn_pretrained')

    # ---- 阶段 2: MuJoCo 精调 (少步数) ----
    print('=' * 60, flush=True)
    print('阶段2: MuJoCo 精调 (12M, 对比 v20 的 24M)', flush=True)
    print('=' * 60, flush=True)
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from mujoco_forest_env import ForestWheelEnv
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    import generate_model as gm
    import terrain as terr

    NENV, TOTAL, PHASE = 12, 12_000_000, 4_000_000
    BASE_CFG = dict(contour_bias=1, prog_w=2.2, alive_w=0.4, head_w=0.6,
                    fwd_vel_w=0.8, lat_w=0.15, steep_w=0.012, steep_ref_deg=18.0,
                    lookahead=False)  # PINN 预训练用 57 维, 精调也 57 维 (对比 v18 而非 v20)

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
                self.cur = 'B'; set_cfg(4.0, 22.0)
                print('[课程] B: 4-22°', flush=True)
            elif t >= 2 * PHASE and self.cur != 'C':
                self.cur = 'C'; set_cfg(8.0, 30.0)
                print('[课程] C: 8-30°', flush=True)
            return True

    class SaveEvery(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_save = 4_000_000
        def _on_step(self):
            if self.model.num_timesteps >= self.next_save:
                self.model.save('models/forest_v21/ckpt_{:02.0f}M'.format(
                    self.model.num_timesteps / 1e6))
                self.next_save += 4_000_000
            return True

    # 从 PINN 预训练模型加载 (57 维 obs 一致)
    env = SubprocVecEnv(
        [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False, seed=1000 + i),
                             'logs/forest_v21_{}'.format(i)) for i in range(NENV)],
        start_method='spawn')
    model = PPO.load('models/forest_v21/pinn_pretrained', env=env, device='cpu')
    model.learning_rate = 1e-4  # 精调用较低 lr

    t1 = time.time()
    model.learn(total_timesteps=TOTAL,
                callback=CallbackList([Curriculum(), SaveEvery(),
                                       EpLogCallback(print_every=300.0)]),
                reset_num_timesteps=False)
    model.save('models/forest_v21/final')
    env.close()
    total_min = (time.time() - t0) / 60
    print('forest_v21_DONE: PINN预训练 {} 步 + MuJoCo精调 {} 步, 总计 {:.1f} 分钟'.format(
        PRETRAIN_STEPS, TOTAL, total_min), flush=True)
    print('对照: v20 纯 MuJoCo 24M 步 = 144 分钟', flush=True)


if __name__ == '__main__':
    main()
