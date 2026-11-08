# -*- coding: utf-8 -*-
"""晨前最后实验 (v2, 修 __main__ 保护):
  0. v13 在原生训练地形 (seed=7 smooth/std) 评估 — 验证单一地形过拟合假设
  1. v13 在全新地形 (seed=999) 评估 — 泛化对照
  2. 超温和适配: v13 → S12 → S16 (lr 1e-5), 每步在留出块评估
结果追加 rsi/log.md
"""
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

bank = json.load(open('dem_patches.json', encoding='utf-8'))
log_f = open('rsi/log.md', 'a', encoding='utf-8')


def log(s):
    line = '[{}] {}'.format(time.strftime('%H:%M:%S'), s)
    print(line, flush=True)
    log_f.write(line + '\n')
    log_f.flush()


def eval_model(model, tag, smooth_seed=999):
    import numpy as np
    from stable_baselines3 import PPO
    from mujoco_forest_env import ForestWheelEnv
    import generate_model as gm
    import terrain as terr

    out = {}
    for kind in ('C1', 'C2', 'C3'):
        q = bank[kind][7]
        gm.generate(seed=999, wheel=True, profile='std',
                    h_fn=terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=999))
        env = ForestWheelEnv(profile='std', generate=False)
        surv = reach = 0
        for ep in range(15):
            o, _ = env.reset(seed=7000 + ep)
            term = False
            for k in range(500):
                a, _ = model.predict(o, deterministic=True)
                o, r, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            if not term:
                surv += 1
            if info.get('is_success'):
                reach += 1
        out[kind] = (surv, reach)
    gm.generate(seed=smooth_seed, wheel=True, profile='smooth')
    env = ForestWheelEnv(profile='smooth', generate=False)
    ss = 0
    for ep in range(10):
        o, _ = env.reset(seed=8000 + ep)
        term = False
        for k in range(500):
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        if not term:
            ss += 1
    out['smooth'] = (ss, 10)
    log('{}: C1 {}/15 C2 {}/15 C3 {}/15 smooth(seed{}) {}/10'.format(
        tag, *out['C1'], *out['C2'], *out['C3'], smooth_seed, *out['smooth']))
    return out


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from mujoco_forest_env import ForestWheelEnv
    import generate_model as gm
    import terrain as terr

    v13 = PPO.load('models/forest_v13/final', device='cpu')

    log('== 补充实验 A: v13 原生地形 (seed=7, 训练时所见) ==')
    eval_model(v13, 'v13_native_smooth', smooth_seed=7)

    log('== 补充实验 B: 超温和适配 lr 1e-5 (S12 → S16) ==')
    base = v13
    for name, tilt in (('S12', 12.0), ('S16', 16.0)):
        h_fn = terr.make_h_fn(seed=77, span=44.0, amp1=0.55, amp2=0.22,
                              amp3=0.08, rough_amp=0.03, tilt_deg=tilt)
        gm.generate(seed=77, wheel=True, profile='std', h_fn=h_fn)
        env = SubprocVecEnv(
            [lambda i=i: Monitor(ForestWheelEnv(profile='std', generate=False),
                                 'logs/gentle_{}_{}'.format(name, i)) for i in range(12)],
            start_method='spawn')
        model = PPO.load('models/forest_v13/final' if name == 'S12'
                         else 'models/gentle_S12', env=env, device='cpu')
        model.learning_rate = 1e-5
        model.ent_coef = 0.0
        model.learn(total_timesteps=1_000_000, reset_num_timesteps=False)
        model.save('models/gentle_{}'.format(name))
        env.close()
        eval_model(PPO.load('models/gentle_{}'.format(name), device='cpu'),
                   'gentle_{}'.format(name))

    log('SUPPLEMENT_DONE')


if __name__ == '__main__':
    main()
