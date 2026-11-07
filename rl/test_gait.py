# -*- coding: utf-8 -*-
"""步态验证：站立 2 s → 小跑 10 s，输出躯干高度统计与前进距离"""
import math
import os

import mujoco
import numpy as np

import gait

os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path('forest_dog.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)


def reset():
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [1.0, 0.0, 0.45]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = gait.Q0
    data.ctrl[:] = gait.Q0
    mujoco.mj_forward(model, data)


def run(seconds, use_cpg):
    n = int(seconds / model.opt.timestep)
    zs = []
    x0 = data.qpos[0]
    for _ in range(n):
        data.ctrl[:] = gait.cpg(data.time) if use_cpg else gait.Q0
        mujoco.mj_step(model, data)
        zs.append(data.qpos[2])
    zs = np.array(zs)
    print(f'{"小跑" if use_cpg else "站立"} {seconds:.0f}s: '
          f'高度 均值 {zs.mean():.3f} / 最低 {zs.min():.3f} / 末段 {zs[-90:].mean():.3f} m | '
          f'前进 {data.qpos[0] - x0:.2f} m | 俯仰 {math.degrees(data.qpos[4]):.1f}°')
    return zs


reset()
run(2.0, use_cpg=False)
reset()
run(10.0, use_cpg=True)
print('GAIT_TEST_DONE')
