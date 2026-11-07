# -*- coding: utf-8 -*-
"""新悬架站立标定 + 直线行驶基元"""
import json
import math
import os

import mujoco
import numpy as np

import gait
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path('forest_dog_wheel.xml')
m.opt.timestep = 0.005
d = mujoco.MjData(m)
h = terr.build_from_params(json.load(open('terrain_meta.json',
                                          encoding='utf-8')))

# 1) 站立下沉
mujoco.mj_resetDataKeyframe(m, d, 0)
x, y = 1.0, 0.0
d.qpos[:3] = [x, y, float(h(x, y)) + 0.63]
d.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)
for _ in range(400):
    d.ctrl[:12] = gait.WQ0
    d.ctrl[12:16] = 0
    mujoco.mj_step(m, d)
zr = d.qpos[2] - float(h(d.qpos[0], d.qpos[1]))
pitch = math.degrees(math.asin(max(-1, min(1,
    2 * (d.qpos[3] * d.qpos[5] - d.qpos[6] * d.qpos[4])))))
print('站立: z_rel=%.3f (目标≈0.55) 俯仰=%+.1f° | FR hip=%.2f knee=%.2f'
      % (zr, pitch, d.qpos[8] - gait.WLEG0[1], d.qpos[9] - gait.WLEG0[2]))

# 2) 直线行驶 6s @ 轮速 15/22.5/30 rad/s
for vw in (15, 22.5, 30):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[:3] = [1.0, 0.0, float(h(1.0, 0.0)) + 0.63]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    x0 = d.qpos[0]
    fell = False
    for k in range(int(6 / 0.005)):
        d.ctrl[:12] = gait.WQ0
        d.ctrl[12:16] = vw
        mujoco.mj_step(m, d)
        p = math.degrees(math.asin(max(-1, min(1,
            2 * (d.qpos[3] * d.qpos[5] - d.qpos[6] * d.qpos[4])))))
        z_rel = d.qpos[2] - float(h(d.qpos[0], d.qpos[1]))
        if abs(p) > 55 or z_rel < 0.18:
            fell = True
            break
    dist = d.qpos[0] - x0
    dt = d.time
    print('轮速 %4.1f rad/s: 6s 前进 %.2f m (%.2f m/s) %s'
          % (vw, dist, dist / dt, '→摔!' if fell else '→稳定'))
print('MECH_TEST_DONE')
