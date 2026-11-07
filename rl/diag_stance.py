# -*- coding: utf-8 -*-
"""站立诊断：足端位置 / 关节下垂 / 所需力矩"""
import mujoco
import numpy as np

import gait

model = mujoco.MjModel.from_xml_path('forest_dog.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
data.ctrl[:] = gait.Q0
mujoco.mj_forward(model, data)

foot_ids = []
for i in range(model.ngeom):
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if nm and 'foot' in nm:
        foot_ids.append(i)
hip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{p}_hip')
           for p in ('FR', 'FL', 'RR', 'RL')]
print('关键帧瞬时: 躯干 z =', round(data.qpos[2], 3))
for fid, bid, nm in zip(foot_ids, hip_ids, ('FR', 'FL', 'RR', 'RL')):
    fz = data.geom_xpos[fid][2]
    hz = data.xpos[bid][2]
    fx = data.geom_xpos[fid][0]
    hx = data.xpos[bid][0]
    print(f'  {nm}: 髋 z={hz:.3f} x={hx:.3f} | 足 z={fz:.3f} x={fx:.3f} '
          f'| 足低于髋 {hz - fz:.3f} m')

print('\n保持 0.5 s 后各关节偏差 (目标 - 实际):')
for _ in range(100):
    data.ctrl[:] = gait.Q0
    mujoco.mj_step(model, data)
err = gait.Q0 - data.qpos[7:]
for nm in ('FR', 'FL', 'RR', 'RL'):
    i = ('FR', 'FL', 'RR', 'RL').index(nm)
    print(f'  {nm}: abd {err[3*i]:+.3f} hip {err[3*i+1]:+.3f} knee {err[3*i+2]:+.3f}')
print('躯干 z =', round(data.qpos[2], 3), '| 姿态四元数 =', np.round(data.qpos[3:7], 3))
print('actuator force:', np.round(data.actuator_force, 1))
