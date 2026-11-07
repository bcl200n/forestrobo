# -*- coding: utf-8 -*-
"""轮足策略统一评测: 多模型同条件对比表
用法: python eval_all.py [模型名 ...]   默认 v1 v2 v3
"""
import math
import os
import sys

import numpy as np
from stable_baselines3 import PPO

import enjoy_wheel as ej

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CANDIDATES = {
    'v1': 'models/wheel_v1/final.zip',
    'v2': 'models/wheel_v2/final.zip',
    'v3p1': 'models/wheel_v3/phase1.zip',
    'v3': 'models/wheel_v3/final.zip',
    'v4b': 'models/wheel_v4b/final.zip',
}
N_EP = 10


def run_eval(n_ep=N_EP, goal_range=(10.0, 18.0)):
    rng = np.random.default_rng(0)
    succ, spd, dist = 0, [], []
    for _ in range(n_ep):
        hd = rng.uniform(-math.pi, math.pi)
        ej.reset(1.0, rng.uniform(-0.8, 0.8), hd)
        ga = rng.uniform(-math.pi, math.pi)
        gd0 = rng.uniform(*goal_range)
        x0, y0 = ej.data.qpos[0], ej.data.qpos[1]
        goal = np.array([x0 + math.cos(ga) * gd0, y0 + math.sin(ga) * gd0])
        ej.PREV = [ej.data.qpos[:3].copy(), hd]
        last = np.zeros(16, np.float32)
        t0 = ej.data.time
        gd = gd0
        for _ in range(500):
            o, gd = ej.obs_vec(goal, last)
            act, _ = ej.policy.predict(o, deterministic=True)
            ej.step_env(act, goal)
            last = act
            ej.PREV = [ej.data.qpos[:3].copy(),
                       math.atan2(2 * (ej.data.qpos[3] * ej.data.qpos[6]
                                       + ej.data.qpos[4] * ej.data.qpos[5]),
                                  1 - 2 * (ej.data.qpos[5] ** 2
                                           + ej.data.qpos[6] ** 2))]
            p, _ = ej.pitch_roll()
            z_rel = ej.data.qpos[2] - float(ej.h(ej.data.qpos[0], ej.data.qpos[1]))
            if gd < 1.0 or abs(p) > 53 or z_rel < 0.16 or z_rel > 0.75:
                break
        dt = ej.data.time - t0
        trav = math.hypot(ej.data.qpos[0] - x0, ej.data.qpos[1] - y0)
        spd.append(trav / dt)
        dist.append(gd0 - gd)
        succ += 1 if gd < 1.0 else 0
    return succ, float(np.mean(spd)), float(np.mean(dist))


if __name__ == '__main__':
    names = sys.argv[1:] or [n for n in ('v1', 'v2', 'v3')
                             if os.path.exists(CANDIDATES[n])]
    GOAL_RANGE = {'v4b': (3.0, 6.0)}          # 各模型按训练目标距离评测
    print(f'{"模型":<6}{"成功率":>8}{"平均速度":>10}{"平均接近(m)":>12}')
    print('-' * 40)
    for n in names:
        path = CANDIDATES[n]
        if not os.path.exists(path):
            print(f'{n:<8}缺失: {path}')
            continue
        ej.policy = PPO.load(path, device='cpu')
        s, v, d = run_eval(goal_range=GOAL_RANGE.get(n, (10.0, 18.0)))
        print(f'{n:<8}{s}/{N_EP:>4}{v:>9.2f}{d:>11.1f}')
    print('EVAL_ALL_DONE')
