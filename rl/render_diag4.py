# -*- coding: utf-8 -*-
"""渲染四视角诊断帧(当前轮足模型, 坡上静置)"""
import json
import math
import os

import imageio.v2 as imageio
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

data.qpos[:3] = [0.8, 0.0, float(h(0.8, 0.0)) + gait.WSTAND_Z + 0.00]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:19] = gait.WQ0
mujoco.mj_forward(model, data)
for _ in range(200):
    data.ctrl[:] = gait.wheel_drive(0.0)
    mujoco.mj_step(model, data)

r = mujoco.Renderer(model, height=720, width=1280)
cam = mujoco.MjvCamera()
cx, cz = data.qpos[0], data.qpos[2]
for name, azim, elev, dist in (('side', 90, -8, 2.2), ('front', 0, -10, 2.2),
                               ('rear', 180, -10, 2.2), ('iso', 45, -18, 2.8)):
    cam.lookat[:] = [cx, 0, cz - 0.1]
    cam.distance = dist
    cam.azimuth = azim
    cam.elevation = elev
    r.update_scene(data, cam)
    imageio.imwrite(f'diag_{name}.png', r.render())
print('FRAMES4_OK x=%.3f z=%.3f' % (cx, cz))
