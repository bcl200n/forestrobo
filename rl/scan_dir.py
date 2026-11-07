# -*- coding: utf-8 -*-
"""CPG 方向四组合扫描: 髋幅符号 × 膝相位符号"""
import json
import os

import mujoco

import gait
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path('rubble_dog.xml')
m.opt.timestep = 0.005
d = mujoco.MjData(m)
h = terr.build_from_params(json.load(open('terrain_rubble_meta.json',
                                          encoding='utf-8')))

for hip_s, knee_s in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
    gait.A_HIP = 0.30 * hip_s
    gait._KNEE_PHASE = -0.7 * knee_s   # 供 cpg 内引用(若存在)
    d2 = mujoco.MjData(m)
    d2.qpos[:3] = [1.0, 0.0, float(h(1.0, 0.0)) + 0.46]
    d2.qpos[3:7] = [1, 0, 0, 0]
    d2.qpos[7:] = gait.Q0
    mujoco.mj_forward(m, d2)
    x0 = d2.qpos[0]
    for k in range(600):
        c = gait.cpg(k * 0.005).copy()
        if knee_s < 0:                      # 翻转膝相位: 折叠发生在另一半周期
            c[2::3] = 2 * gait.LEG0[2] - c[2::3]
        d2.ctrl[:] = c
        mujoco.mj_step(m, d2)
    print(f'hip{hip_s:+d} knee{knee_s:+d}: 3s 前进 {d2.qpos[0] - x0:+.2f} m')
