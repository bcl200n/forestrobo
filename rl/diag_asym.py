# -*- coding: utf-8 -*-
"""左右不对称诊断: 平地上静置, 打印横滚/各轮位置与地形高差/轮穿透深度"""
import json
import math
import os

import mujoco
import numpy as np

import gait
import generate_model as gm
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

gm.generate(wheel=True)     # 山地地形(出生区平整), 不用坡面排除法
model = mujoco.MjModel.from_xml_path('forest_dog_wheel.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)
with open('terrain_meta.json', encoding='utf-8') as f:
    h = terr.build_from_params(json.load(f))

P4 = ('FR', 'FL', 'RR', 'RL')
wheel_geom = {p: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{p}_wheel')
              for p in P4}

data.qpos[:3] = [0.8, 0.0, float(h(0.8, 0.0)) + gait.WSTAND_Z]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:19] = gait.WQ0
data.ctrl[:12] = gait.WQ0
mujoco.mj_forward(model, data)

for _ in range(300):
    data.ctrl[:] = gait.wheel_drive(0.0)
    mujoco.mj_step(model, data)

w, x, y, z = data.qpos[3:7]
roll = math.degrees(math.asin(max(-1, min(1, 2 * (w * x + y * z)))))
pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))
print(f'平地静置: 俯仰 {pitch:+.2f}° 横滚 {roll:+.2f}°')
for p in P4:
    gp = data.geom_xpos[wheel_geom[p]]
    dh = float(gp[2] - h(gp[0], gp[1]))
    print(f'  {p}_wheel: 底面({gp[0]:+.3f},{gp[1]:+.3f}) 轮心离地 {dh:.3f} m '
          f'(应 ≈ +0.075)')
print('ncon =', data.ncon, '| 轮接触法向力:')
fn = {p: 0.0 for p in P4}
for c in range(data.ncon):
    con = data.contact[c]
    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ''
    n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ''
    f6 = np.zeros(6)
    mujoco.mj_contactForce(model, data, c, f6)
    for p in P4:
        if f'{p}_wheel' in (n1, n2):
            fn[p] += float(f6[0])
print('  ', {p: round(v, 1) for p, v in fn.items()})
print('ASYM_DONE')
