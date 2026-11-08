# -*- coding: utf-8 -*-
"""机械判据实验 (无 RL, 纯姿态保持): 平台在真实坡度上能否站住/直行

零动作 = 12 腿标称姿态 + 4 轮 0 速 → 纯机械+伺服刚度对抗坡面。
若 20° 都站不住 → 机电极限实锤 (RL 无解, 需改构型);
若能站住但学不出来 → 训练管线问题 (单地形过拟合, 需随机化重训)。
"""
import io
import json
import math
import os
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import generate_model as gm
import terrain as terr
from mujoco_forest_env import ForestWheelEnv

bank = json.load(open('dem_patches.json', encoding='utf-8'))
log_f = open('rsi/log.md', 'a', encoding='utf-8')


def log(s):
    line = '[{}] {}'.format(time.strftime('%H:%M:%S'), s)
    print(line, flush=True)
    log_f.write(line + '\n')
    log_f.flush()


def probe(kind, drive=0.0):
    """drive: 轮速指令比例 (0=纯站立, 0.5=半速直行)"""
    q = bank[kind][0]
    hh = terr.make_dem_fn(q['lat'], q['lon'], q['rot'], seed=2)
    gm.generate(seed=7, wheel=True, profile='std', h_fn=hh)
    env = ForestWheelEnv(profile='std', generate=False)
    act = np.zeros(16, np.float32)
    act[12:] = drive
    hold = 0
    slides = []
    for ep in range(10):
        o, _ = env.reset(seed=100 + ep)
        term = False
        steps = 0
        x0 = env.data.qpos[0]
        for k in range(500):
            o, r, term, trunc, info = env.step(act)
            steps += 1
            if term or trunc:
                break
        if not term:
            hold += 1
        slides.append((steps * 0.02, env.data.qpos[0] - x0))
    log('机械判据 {} ({}°, 轮速{}): 站住 {}/10, 平均坚持 {:.1f}s, 平均滑移 {:+.2f}m'.format(
        kind, q['slope'], drive, hold,
        sum(s[0] for s in slides) / 10, sum(s[1] for s in slides) / 10))


log('== 机械判据实验: 零动作姿态保持 (纯伺服, 无 RL) ==')
for kind in ('C1', 'C2', 'C3'):
    probe(kind, drive=0.0)
probe('C1', drive=0.5)
log('MECH_PROBE_DONE')
