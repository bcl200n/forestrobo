# -*- coding: utf-8 -*-
"""对称性数值诊断: 各轮世界坐标/离地/关节达标情况"""
import json
import math
import os

import mujoco
import numpy as np

import gait
import generate_model as gm
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

gm.generate(h_fn=terr.make_ramp_fn(24.2), n_trees=0, n_rocks=0, n_logs=0, wheel=True)
model = mujoco.MjModel.from_xml_path('forest_dog_wheel.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)
with open('terrain_meta.json', encoding='utf-8') as f:
    h = terr.build_from_params(json.load(f))

P4 = ('FR', 'FL', 'RR', 'RL')
wheel_geom = {p: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{p}_wheel')
              for p in P4}
jnt = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
       for j in [f'{p}_{k}' for p in P4 for k in ('abd', 'hip', 'knee', 'wheel')]}

data.qpos[:3] = [0.8, 0.0, float(h(0.8, 0.0)) + gait.WSTAND_Z]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:19] = gait.WQ0
mujoco.mj_forward(model, data)
for _ in range(200):
    data.ctrl[:] = gait.wheel_drive(0.0)
    mujoco.mj_step(model, data)

print('躯干: xyz =', np.round(data.qpos[:3], 3))
w, x, y, z = data.qpos[3:7]
print('roll %+.2f° pitch %+.2f°' % (
    math.degrees(math.asin(max(-1, min(1, 2 * (w * x + y * z))))),
    math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))))
print('\n各轮世界位置与离地(轮心-地面, 贴地应≈+0.075):')
for p in P4:
    gp = data.geom_xpos[wheel_geom[p]]
    print(f'  {p}: x={gp[0]:+.3f} y={gp[1]:+.3f} z={gp[2]:.3f} '
          f'离地={gp[2] - float(h(gp[0], gp[1])):+.3f}')
print('\n关节实际角(指令 abd=0.00 hip=0.55 knee=-1.15):')
for p in P4:
    q = [data.qpos[model.jnt_qposadr[jnt[f'{p}_{k}']]]
         for k in ('abd', 'hip', 'knee')]
    print(f'  {p}: abd={q[0]:+.3f} hip={q[1]:+.3f} knee={q[2]:+.3f}')
print('\n关节力矩(N·m):', np.round(data.actuator_force[:12], 1).tolist())
print('SYM_DIAG_DONE')
