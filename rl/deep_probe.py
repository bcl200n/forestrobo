# -*- coding: utf-8 -*-
"""轮足坡上行驶深探: 轮滑差 / 接触载荷分布 / 驱动力矩"""
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
wheel_jnt = {p: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{p}_wheel')
             for p in P4}


def pitch_of():
    w, x, y, z = data.qpos[3:7]
    return math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))


data.qpos[:3] = [0.8, 0.0, float(h(0.8, 0.0)) + gait.WSTAND_Z]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:19] = gait.WQ0
mujoco.mj_forward(model, data)

for t in range(5):
    for _ in range(int(1.0 / 0.005)):
        v = 1.0 * min(1.0, data.time / 2.0)
        data.ctrl[:] = gait.wheel_drive(v, pitch=math.radians(pitch_of()))
        mujoco.mj_step(model, data)
    x = data.qpos[0]
    v_body = (x - (0.8 if t == 0 else prev_x)) / 1.0
    prev_x = x
    wqvel = {p: float(data.qvel[model.jnt_dofadr[wheel_jnt[p]]]) for p in P4}
    slip = {p: wqvel[p] * gait.WHEEL_R - v_body for p in P4}
    fn, ft = {p: 0.0 for p in P4}, {p: 0.0 for p in P4}
    legs_n, other = 0.0, 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        f6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, f6)
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ''
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ''
        hit = None
        for p in P4:
            if f'{p}_wheel' in (n1, n2):
                hit = p
                break
        if hit:
            fn[hit] += float(f6[0])
            ft[hit] += math.hypot(float(f6[1]), float(f6[2]))
        elif 'wheel' in n1 or 'wheel' in n2:
            other += float(f6[0])
        else:
            legs_n += float(f6[0])
    print(f't={t+1}s x={x:5.2f} v={v_body:5.2f}m/s 俯仰{pitch_of():+5.1f}°')
    print('   轮速rad/s', {p: round(wqvel[p], 1) for p in P4},
          '| 滑差m/s', {p: round(slip[p], 2) for p in P4})
    print('   轮法向N', {p: round(fn[p], 1) for p in P4},
          '| 轮切向N', {p: round(ft[p], 1) for p in P4},
          '| 腿壳法向N', round(legs_n, 1),
          '| 力矩', np.round(data.actuator_force[12:], 1).tolist())
print('DEEP_PROBE_DONE')
