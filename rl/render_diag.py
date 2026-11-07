# -*- coding: utf-8 -*-
"""渲染左后轮楔入状态的特写帧"""
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

data.qpos[:3] = [1.0, 0.0, float(h(1.0, 0.0)) + 0.47]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:19] = gait.Q0
mujoco.mj_forward(model, data)
for _ in range(150):
    data.ctrl[:] = gait.wheel_drive(0.0)
    mujoco.mj_step(model, data)

r = mujoco.Renderer(model, height=720, width=1280)
cam = mujoco.MjvCamera()
cam.lookat[:] = [data.qpos[0], 0, data.qpos[2]]
cam.distance = 1.6
cam.azimuth = 90
cam.elevation = -15
r.update_scene(data, cam)
imageio.imwrite('diag_frame.png', r.render())
print('FRAME_OK x=%.3f z=%.3f zrel=%.3f' %
      (data.qpos[0], data.qpos[2],
       data.qpos[2] - float(h(data.qpos[0], data.qpos[1]))))
