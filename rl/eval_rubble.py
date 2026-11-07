# -*- coding: utf-8 -*-
"""废墟搜救策略评测与可视化（CPG 残差版, 匹配 rubble_v2+ 训练语义）
用法:
  python eval_rubble.py --eval    # 无头评测 10 回合
  python eval_rubble.py           # 交互窗口看搜索过程
  python eval_rubble.py --model rubble_v2
"""
import json
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

import gait
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL = ('models/' + (sys.argv[sys.argv.index('--model') + 1]
                      if '--model' in sys.argv else 'rubble_v2') + '/final.zip')
print('策略:', MODEL)

model = mujoco.MjModel.from_xml_path('rubble_dog.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)
h = terr.build_from_params(json.load(open('terrain_rubble_meta.json',
                                          encoding='utf-8')))
lay = json.load(open('rubble_layout.json', encoding='utf-8'))
surv = np.array(lay['survivors'], np.float32)
haz = np.array(lay['hazards'], np.float32)
Q0 = gait.Q0
policy = PPO.load(MODEL, device='cpu')


def obs_vec(found, last, t):
    pos = data.qpos[:3]
    w, x, y, z = data.qpos[3:7]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    Rt = R.T
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    vel = (pos - PREV[0]) / 0.02
    yawr = ((yaw - PREV[1] + math.pi) % (2 * math.pi) - math.pi) / 0.02
    v_body = Rt @ vel
    grav = Rt @ np.array([0, 0, -1.0])
    rel = surv[:, :2] - pos[:2]
    dist = np.linalg.norm(rel, axis=1)
    dist = np.where(found, 1e9, dist)
    ki = dist.argmin()
    tdist = dist[ki]
    tv = rel[ki] / max(tdist, 1e-6)
    gz3 = np.array([tv[0], tv[1], 0.0])
    tdir = (Rt @ gz3)[:2]
    laser = np.full(3, 6.0)
    rx = haz[:, 0] - pos[0]
    ry = haz[:, 1] - pos[1]
    d = np.hypot(rx, ry) - haz[:, 2]
    near = d < 6.0
    if near.any():
        ang = np.arctan2(ry[near], rx[near])
        rr = (ang - yaw + math.pi) % (2 * math.pi) - math.pi
        dn = d[near]
        for m, lo, hi in ((0, -0.52, 0.52), (1, 0.52, 2.09), (2, -2.09, -0.52)):
            s = (rr >= lo) & (rr < hi)
            if s.any():
                laser[m] = max(0.0, dn[s].min())
    z_rel = pos[2] - float(h(pos[0], pos[1]))
    ph = t * gait.FREQ * 2 * math.pi
    return (np.concatenate([v_body, grav, [yawr], tdir, laser, [z_rel],
                            data.qpos[7:19] - Q0, data.qvel[6:18], last,
                            [found.sum() / 4.0, min(tdist, 25.0) / 25.0,
                             math.sin(ph), math.cos(ph)]]).astype(np.float32),
            tdist, math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x))))))


def reset(hd_rng):
    mujoco.mj_resetData(model, data)
    hd = float(hd_rng.uniform(-math.pi, math.pi))
    data.qpos[:3] = [1.0, float(hd_rng.uniform(-0.8, 0.8)),
                     float(h(1.0, 0.0)) + 0.46]
    data.qpos[3:7] = [math.cos(hd / 2), 0, 0, math.sin(hd / 2)]
    data.qpos[7:] = Q0
    mujoco.mj_forward(model, data)
    return hd


def run_episode(hd_rng, sync=None):
    """一个搜索回合; 返回 (发现数, 用时, 是否摔倒)"""
    hd = reset(hd_rng)
    global PREV
    PREV = [data.qpos[:3].copy(), hd]
    found = np.zeros(4, bool)
    last = np.zeros(12, np.float32)
    fell = False
    for k in range(900):
        o, tdist, pitch = obs_vec(found, last, k * 0.02)
        act, _ = policy.predict(o, deterministic=True)
        c = gait.cpg(k * 0.02).copy()                  # v4 结构化动作语义
        c[0::3] += act[0] * 0.22                       # a0=转向→外摆差动
        c[1::3] += act[1] * 0.18                       # a1=速度→髋共模
        c[2:12] += act[2:] * 0.08                      # 微残差
        data.ctrl[:] = c
        for _ in range(4):
            mujoco.mj_step(model, data)
        last = act
        PREV = [data.qpos[:3].copy(),
                math.atan2(2 * (data.qpos[3] * data.qpos[6]
                                + data.qpos[4] * data.qpos[5]),
                           1 - 2 * (data.qpos[5] ** 2 + data.qpos[6] ** 2))]
        if sync:
            sync()
        d_all = np.linalg.norm(surv[:, :2] - data.qpos[:2], axis=1)
        for i in range(4):
            if not found[i] and d_all[i] < 1.0:
                found[i] = True
        z_rel = data.qpos[2] - float(h(data.qpos[0], data.qpos[1]))
        if abs(pitch) > 53 or z_rel < 0.16:
            fell = True
            break
        if found.all():
            break
    return int(found.sum()), data.time, fell


if '--eval' in sys.argv:
    rng = np.random.default_rng(0)
    tot, falls, all4 = [], 0, 0
    for ep in range(10):
        n, t, fell = run_episode(rng)
        tot.append(n)
        falls += fell
        all4 += (n == 4)
        print(f'  回合{ep + 1}: 发现 {n}/4 | 用时 {t:4.1f}s | '
              f'{"摔倒" if fell else ("全部找到" if n == 4 else "超时")}')
    print(f'== 搜救评测: 平均发现 {np.mean(tot):.1f}/4 | 全齐 {all4}/10 | '
          f'摔倒率 {falls * 10}% ==')
else:
    print('已打开搜救策略窗口（绿色胶囊=幸存者）；关闭窗口退出。')
    rng = np.random.default_rng(1)
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            n, t, fell = run_episode(rng, sync=lambda: (v.sync(), time.sleep(0.008)))
            time.sleep(0.6)
    print('查看器已退出')
