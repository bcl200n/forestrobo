# -*- coding: utf-8 -*-
"""渲染 v2 轮足策略实况视频"""
import math

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

import enjoy_wheel as ej   # 复用 obs/reset/step 逻辑

FPS = 30
renderer = mujoco.Renderer(ej.model, height=720, width=1280)
cam = mujoco.MjvCamera()
frames = []
rng = np.random.default_rng(3)

for ep in range(3):
    hd = rng.uniform(-math.pi, math.pi)
    ej.reset(1.0, rng.uniform(-0.8, 0.8), hd)
    ga = rng.uniform(-0.9, 0.9) + hd          # 目标大致在前方
    goal = np.array([ej.data.qpos[0] + math.cos(ga) * 14,
                     ej.data.qpos[1] + math.sin(ga) * 14])
    ej.PREV = [ej.data.qpos[:3].copy(), hd]
    last = np.zeros(16, np.float32)
    for k in range(10 * FPS):
        o, gd = ej.obs_vec(goal, last)
        act, _ = ej.policy.predict(o, deterministic=True)
        ej.step_env(act, goal)
        last = act
        ej.PREV = [ej.data.qpos[:3].copy(),
                   math.atan2(2 * (ej.data.qpos[3] * ej.data.qpos[6] + ej.data.qpos[4] * ej.data.qpos[5]),
                              1 - 2 * (ej.data.qpos[5] ** 2 + ej.data.qpos[6] ** 2))]
        x, y = ej.data.qpos[0], ej.data.qpos[1]
        cam.lookat[:] = [x, y, float(ej.h(x, y)) + 0.8]
        cam.distance = 4.2
        cam.azimuth = 115
        cam.elevation = -12
        renderer.update_scene(ej.data, cam)
        frames.append(renderer.render())
        p, _ = ej.pitch_roll()
        z_rel = ej.data.qpos[2] - float(ej.h(x, y))
        if gd < 1.0 or abs(p) > 53 or z_rel < 0.16:
            break

renderer.close()
imageio.mimsave('wheel_v2_demo.mp4', frames, fps=FPS, quality=9,
                macro_block_size=None)
import os
print('VIDEO_OK wheel_v2_demo.mp4', len(frames), '帧',
      os.path.getsize('wheel_v2_demo.mp4') // 1024, 'KB')
