# -*- coding: utf-8 -*-
"""轮足策略评测与可视化回放
用法:
  python enjoy_wheel.py --eval                  # 无头评测 10 回合
  python enjoy_wheel.py                         # 交互窗口看策略行驶
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
MODEL = ('models/' + (sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv
                      else 'wheel_v1') + '/final.zip')
print('策略:', MODEL)

model = mujoco.MjModel.from_xml_path('forest_dog_wheel.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)
with open('terrain_meta.json', encoding='utf-8') as f:
    h = terr.build_from_params(json.load(f))
with open('forest_layout.json', encoding='utf-8') as f:
    trees = [t[:3] for t in json.load(f)['trees']]
trees_np = np.array(trees)

policy = PPO.load(MODEL, device='cpu')


def pitch_roll():
    w, x, y, z = data.qpos[3:7]
    return (math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x))))),
            math.degrees(math.asin(max(-1, min(1, 2 * (w * x + y * z))))))


def obs_vec(goal, last_act):
    pos = data.qpos[:3]
    quat = data.qpos[3:7]
    w, x, y, z = quat
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
    gv = goal - pos[:2]
    gd = np.linalg.norm(gv)
    gun = gv / max(gd, 1e-6)
    gz3 = np.array([gun[0], gun[1], 0.0])
    goal_b = (Rt @ gz3)[:2]
    laser = np.full(3, 6.0)
    rx = trees_np[:, 0] - pos[0]
    ry = trees_np[:, 1] - pos[1]
    d = np.hypot(rx, ry) - trees_np[:, 2]
    near = d < 6.0
    if near.any():
        ang = np.arctan2(ry[near], rx[near])
        rel = (ang - yaw + math.pi) % (2 * math.pi) - math.pi
        dn = d[near]
        for m, lo, hi in ((0, -0.52, 0.52), (1, 0.52, 2.09), (2, -2.09, -0.52)):
            s = (rel >= lo) & (rel < hi)
            if s.any():
                laser[m] = max(0.0, dn[s].min())
    z_rel = pos[2] - float(h(pos[0], pos[1]))
    return np.concatenate([v_body, grav, [yawr], goal_b, laser, [z_rel],
                           data.qpos[7:19] - gait.WQ0, data.qvel[6:18],
                           last_act, data.qvel[18:22] * 0.05]).astype(np.float32), gd


def step_env(act, goal):
    """写控制并推进 0.02s (4 子步)"""
    data.ctrl[:12] = gait.WQ0 + act[:12] * 0.35
    data.ctrl[12:16] = act[12:16] * 40.0
    for _ in range(4):
        mujoco.mj_step(model, data)


def reset(x0, y0, hd):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [x0, y0, float(h(x0, y0)) + gait.WSTAND_Z + 0.03]
    data.qpos[3:7] = [math.cos(hd / 2), 0, 0, math.sin(hd / 2)]
    data.qpos[7:19] = gait.WQ0
    mujoco.mj_forward(model, data)


def eval_mode(n_ep=10):
    rng = np.random.default_rng(0)
    succ, dists, speeds = 0, [], []
    for ep in range(n_ep):
        hd = rng.uniform(-math.pi, math.pi)
        reset(1.0, rng.uniform(-0.8, 0.8), hd)
        ga = rng.uniform(-math.pi, math.pi)
        gd0 = rng.uniform(10, 18)
        x0, y0 = data.qpos[0], data.qpos[1]
        goal = np.array([x0 + math.cos(ga) * gd0, y0 + math.sin(ga) * gd0])
        global PREV
        PREV = [data.qpos[:3].copy(), hd]
        last = np.zeros(16, np.float32)
        gone, t0 = 0.0, data.time
        for k in range(500):
            o, gd = obs_vec(goal, last)
            act, _ = policy.predict(o, deterministic=True)
            step_env(act, goal)
            last = act
            PREV = [data.qpos[:3].copy(),
                    math.atan2(2 * (data.qpos[3] * data.qpos[6] + data.qpos[4] * data.qpos[5]),
                               1 - 2 * (data.qpos[5] ** 2 + data.qpos[6] ** 2))]
            p, r = pitch_roll()
            z_rel = data.qpos[2] - float(h(data.qpos[0], data.qpos[1]))
            if gd < 1.0 or abs(p) > 53 or z_rel < 0.16 or z_rel > 0.75:
                break
        dt = data.time - t0
        trav = math.hypot(data.qpos[0] - x0, data.qpos[1] - y0)
        dists.append(gd0 - gd if gd < 1.0 else -1)
        speeds.append(trav / dt)
        succ += 1 if gd < 1.0 else 0
        print(f'  回合{ep + 1}: {"成功" if gd < 1.0 else "失败"} '
              f'位移 {trav:5.1f} m / {dt:4.1f} s (均速 {trav / dt:4.2f} m/s) '
              f'剩余距离 {gd:5.1f} m')
    print(f'== 评测: 成功率 {succ}/{n_ep} | 平均位移速度 '
          f'{np.mean(speeds):.2f} m/s ==')


def view_mode():
    print('已打开轮足策略窗口（wheel_v1）；关闭窗口退出。')
    rng = np.random.default_rng(1)
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            hd = rng.uniform(-math.pi, math.pi)
            reset(1.0, rng.uniform(-0.8, 0.8), hd)
            ga = rng.uniform(-math.pi, math.pi)
            goal = np.array([data.qpos[0] + math.cos(ga) * 14,
                             data.qpos[1] + math.sin(ga) * 14])
            global PREV
            PREV = [data.qpos[:3].copy(), hd]
            last = np.zeros(16, np.float32)
            for k in range(500):
                if not v.is_running():
                    return
                o, gd = obs_vec(goal, last)
                act, _ = policy.predict(o, deterministic=True)
                step_env(act, goal)
                last = act
                PREV = [data.qpos[:3].copy(),
                        math.atan2(2 * (data.qpos[3] * data.qpos[6] + data.qpos[4] * data.qpos[5]),
                                   1 - 2 * (data.qpos[5] ** 2 + data.qpos[6] ** 2))]
                v.sync()
                time.sleep(0.01)
                p, _ = pitch_roll()
                z_rel = data.qpos[2] - float(h(data.qpos[0], data.qpos[1]))
                if gd < 1.0 or abs(p) > 53 or z_rel < 0.16 or z_rel > 0.75:
                    break
            time.sleep(0.5)
    print('查看器已退出')


if __name__ == '__main__':
    eval_mode() if '--eval' in sys.argv else view_mode()

