# -*- coding: utf-8 -*-
"""交互式 3D 查看器：山地林下场景 + 反射步态小跑（地形版）
鼠标左键旋转 / 滚轮缩放 / 右键平移；关窗退出
"""
import json
import math
import os
import time

import mujoco
import mujoco.viewer

import gait
import terrain as terr

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

if not os.path.exists('forest_dog.xml'):
    from generate_model import generate
    generate()

model = mujoco.MjModel.from_xml_path('forest_dog.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)
with open('terrain_meta.json', encoding='utf-8') as f:
    h = terr.build_from_params(json.load(f))


def pitch_of():
    w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))


def reset():
    mujoco.mj_resetData(model, data)
    x0, y0 = 1.0, 0.0
    data.qpos[:3] = [x0, y0, float(h(x0, y0)) + 0.45]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = gait.Q0
    data.ctrl[:] = gait.Q0
    mujoco.mj_forward(model, data)


reset()
print('已打开交互窗口：山地林下场景，机器狗反射步态小跑；关闭窗口退出。')
with mujoco.viewer.launch_passive(model, data) as v:
    last = time.time()
    while v.is_running():
        now = time.time()
        steps = int((now - last) / model.opt.timestep)
        last += steps * model.opt.timestep
        for _ in range(min(steps, 60)):
            data.ctrl[:] = gait.cpg(data.time, pitch=math.radians(pitch_of()))
            mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)
        # 相对地形高度过低（趴伏）或跑出场景则重置
        z_rel = data.qpos[2] - float(h(data.qpos[0], data.qpos[1]))
        if z_rel < 0.10 or data.qpos[0] > 21 or abs(data.qpos[1]) > 13:
            reset()
print('查看器已退出')
