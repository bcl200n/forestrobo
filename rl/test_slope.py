# -*- coding: utf-8 -*-
"""陡坡爬升测试：纯净坡面（无障碍），CPG+前庭反射爬坡
45% 坡度 = 24.2°；另测 30° / 35° 压极限。测完恢复山地地形模型。
"""
import json
import math
import os

import mujoco
import numpy as np

import gait
import generate_model as gm
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def pitch_of(q):
    w, x, y, z = q[3], q[4], q[5], q[6]
    return math.degrees(math.asin(np.clip(2 * (w * y - z * x), -1, 1)))


def run_slope(slope_deg):
    ramp = terr.make_ramp_fn(slope_deg)
    gm.generate(h_fn=ramp, n_trees=0, n_rocks=0, n_logs=0)
    model = mujoco.MjModel.from_xml_path('forest_dog.xml')
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    with open('terrain_meta.json', encoding='utf-8') as f:
        h = terr.build_from_params(json.load(f))

    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [1.0, 0.0, float(h(1.0, 0.0)) + 0.45]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = gait.Q0
    data.ctrl[:] = gait.Q0
    mujoco.mj_forward(model, data)

    n = int(10 / 0.005)
    zr, ps = [], []
    for _ in range(n):
        data.ctrl[:] = gait.cpg(data.time, pitch=math.radians(pitch_of(data.qpos)))
        mujoco.mj_step(model, data)
        zr.append(data.qpos[2] - float(h(data.qpos[0], data.qpos[1])))
        ps.append(pitch_of(data.qpos))
    zr, ps = np.array(zr), np.array(ps)
    climb = float(h(data.qpos[0], data.qpos[1])) - float(h(1.0, 0.0))
    verdict = '爬升成功' if (zr[-90:].mean() > 0.25 and data.qpos[0] > 4.0) else '失速/打滑'
    print(f'  {slope_deg:4.1f}° ({math.tan(math.radians(slope_deg)) * 100:.0f}% 坡度): '
          f'前进 {data.qpos[0] - 1.0:5.2f} m | 爬升 {climb:5.2f} m | '
          f'相对高度 {zr.mean():.2f}/{zr.min():.2f} | 俯仰 {ps.mean():+5.1f}° '
          f'[{ps.min():+.0f},{ps.max():+.0f}] → {verdict}')
    return data.qpos[0] - 1.0, climb


print('== 陡坡爬升测试（纯净坡面, CPG + 前庭反射, 10 s）==')
for s in (24.2, 30.0, 35.0):
    run_slope(s)

gm.generate()      # 恢复山地地形模型
print('已恢复山地地形模型')
print('SLOPE_TEST_DONE')
