# -*- coding: utf-8 -*-
"""轮足诊断：轮速 / 驱动力 / 接触状态 / 轮子是否离地"""
import math
import os

import mujoco
import numpy as np

import gait
import generate_model as gm
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ramp = terr.make_ramp_fn(24.2)
gm.generate(h_fn=ramp, n_trees=0, n_rocks=0, n_logs=0, wheel=True)
model = mujoco.MjModel.from_xml_path('forest_dog_wheel.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)
with open('terrain_meta.json', encoding='utf-8') as f:
    h = terr.build_from_params(__import__('json').load(f))

data.qpos[:3] = [1.0, 0.0, float(h(1.0, 0.0)) + 0.47]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:19] = gait.Q0
data.ctrl[:12] = gait.Q0
mujoco.mj_forward(model, data)

wheel_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{p}_wheel')
             for p in ('FR', 'FL', 'RR', 'RL')]
wheel_geom = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{p}_wheel')
              for p in ('FR', 'FL', 'RR', 'RL')]
print('nq =', model.nq, '| nu =', model.nu)
print('轮关节 dof 起点:', [int(w) for w in wheel_ids])

for k in range(4):
    data.ctrl[:] = gait.wheel_drive(1.5, pitch=math.radians(
        math.degrees(math.asin(max(-1, min(1, 2 * (data.qpos[3] * data.qpos[5]
                                               - data.qpos[6] * data.qpos[4])))))))
    for _ in range(100):   # 0.5 s
        mujoco.mj_step(model, data)
    print(f't={data.time:4.1f}s pos_x={data.qpos[0]:6.3f} '
          f'轮qvel={[round(float(data.qvel[model.jnt_dofadr[i]]), 1) for i in wheel_ids]} '
          f'轮力矩={np.round(data.actuator_force[12:], 1)} '
          f'接触数={data.ncon}')
    # 各轮接触法向力
    fn = {p: 0.0 for p in ('FR', 'FL', 'RR', 'RL')}
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = con.geom1, con.geom2
        for p, gid in zip(('FR', 'FL', 'RR', 'RL'), wheel_geom):
            if g1 == gid or g2 == gid:
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, f6)
                fn[p] += float(f6[0])
    print('   轮接触法向力(N):', {p: round(v, 1) for p, v in fn.items()})
print('WHEEL_DIAG_DONE')
