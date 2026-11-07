# -*- coding: utf-8 -*-
"""列出坡上静止/行驶时全部接触对: 谁在蹭地"""
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
    h = terr.build_from_params(__import__('json').load(f))

data.qpos[:3] = [1.0, 0.0, float(h(1.0, 0.0)) + 0.47]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:19] = gait.Q0
mujoco.mj_forward(model, data)


def list_contacts(tag):
    from collections import Counter
    cnt = Counter()
    fnorm = Counter()
    for c in range(data.ncon):
        con = data.contact[c]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or '?'
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or '?'
        other = n2 if 'floor' in n1 or 'terrain' in n1 else n1
        f6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, f6)
        cnt[other] += 1
        fnorm[other] += float(f6[0])
    print(f'--- {tag} (x={data.qpos[0]:.3f}) ---')
    for k in cnt:
        print(f'   {k}: {cnt[k]} 点, 法向力 {fnorm[k]:7.1f} N')


for _ in range(300):    # 1.5 s 静置
    data.ctrl[:] = gait.wheel_drive(0.0)
    mujoco.mj_step(model, data)
list_contacts('静置(轮速0)')

for _ in range(600):    # 3 s 行驶
    data.ctrl[:] = gait.wheel_drive(1.0, pitch=math.radians(-20))
    mujoco.mj_step(model, data)
list_contacts('行驶(轮速1.0)')
print('CONTACTS_DONE')
