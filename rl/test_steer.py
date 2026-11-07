# -*- coding: utf-8 -*-
"""转向权限实测: CPG 步态 + 外摆差动偏置, 6 秒能转多少度"""
import math
import os

import mujoco
import numpy as np

import gait
import generate_rubble as gr
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))
gr.generate(seed=11, profile='flat')
model = mujoco.MjModel.from_xml_path('rubble_dog.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)
h = terr.build_from_params(__import__('json').load(
    open('terrain_rubble_meta.json', encoding='utf-8')))


def yaw():
    w, x, y, z = data.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def test(abd_bias, label):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [1.0, 0.0, float(h(1.0, 0.0)) + 0.46]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = gait.Q0
    mujoco.mj_forward(model, data)
    y0, x0 = yaw(), data.qpos[0]
    for k in range(int(6 / 0.005)):
        c = gait.cpg(k * 0.005).copy()
        c[0::3] += abd_bias                 # 外摆差动: FR/FL/RR/RL 同偏置
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
    dy = math.degrees((yaw() - y0 + math.pi) % (2 * math.pi) - math.pi)
    print(f'  {label}: 6s 转向 {dy:+7.1f}° ({dy / 6:+5.1f}°/s) | '
          f'前进 {data.qpos[0] - x0:+.2f} m | 高度 {data.qpos[2] - float(h(data.qpos[0], data.qpos[1])):.2f}')


print('== 外摆差动偏置 → 转向速率 ==')
for b in (0.15, 0.30, 0.45):
    test(b, f'abd 偏置 {b:+.2f}')
test(-0.30, 'abd 偏置 -0.30')
print('STEER_TEST_DONE')
