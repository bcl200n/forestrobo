# -*- coding: utf-8 -*-
"""林下轮足导航收官视频: v3 在标准山地巡航实录"""
import math
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

import enjoy_wheel as ej

MODEL = ('models/' + (sys.argv[1] if len(sys.argv) > 1 else 'wheel_v3')
         + '/final.zip')
print('策略:', MODEL)
ej.policy = PPO.load(MODEL, device='cpu')

FPS = 30
renderer = mujoco.Renderer(ej.model, height=720, width=1280)
cam = mujoco.MjvCamera()
frames = []
rng = np.random.default_rng(5)

for ep in range(3):
    hd = rng.uniform(-math.pi, math.pi)
    ej.reset(1.0, rng.uniform(-0.8, 0.8), hd)
    ga = hd + rng.uniform(-0.6, 0.6)              # 目标大致在前方
    goal = np.array([ej.data.qpos[0] + math.cos(ga) * 8,
                     ej.data.qpos[1] + math.sin(ga) * 8])
    ej.PREV = [ej.data.qpos[:3].copy(), hd]
    last = np.zeros(16, np.float32)
    for k in range(10 * FPS):
        o, gd = ej.obs_vec(goal, last)
        act, _ = ej.policy.predict(o, deterministic=True)
        ej.step_env(act, goal)
        last = act
        ej.PREV = [ej.data.qpos[:3].copy(),
                   math.atan2(2 * (ej.data.qpos[3] * ej.data.qpos[6]
                                   + ej.data.qpos[4] * ej.data.qpos[5]),
                              1 - 2 * (ej.data.qpos[5] ** 2
                                       + ej.data.qpos[6] ** 2))]
        x, y = ej.data.qpos[0], ej.data.qpos[1]
        cam.lookat[:] = [x, y, float(ej.h(x, y)) + 0.8]
        cam.distance = 4.5
        cam.azimuth = 115
        cam.elevation = -14
        renderer.update_scene(ej.data, cam)
        frames.append(renderer.render())
        p, _ = ej.pitch_roll()
        z_rel = ej.data.qpos[2] - float(ej.h(x, y))
        if gd < 1.0 or abs(p) > 53 or z_rel < 0.16:
            break

renderer.close()
imageio.mimsave('forest_cruise_final.mp4', frames, fps=FPS, quality=9,
                macro_block_size=None)
print('VIDEO_OK forest_cruise_final.mp4', len(frames), '帧',
      os.path.getsize('forest_cruise_final.mp4') // 1024, 'KB')
