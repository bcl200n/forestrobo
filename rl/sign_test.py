# -*- coding: utf-8 -*-
"""符号实验: 锁轮静置坡上, 施加前/后髋差动偏置, 看俯仰与前/后轮法向力如何变化"""
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

wheel_geom = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{p}_wheel')
              for p in ('FR', 'FL', 'RR', 'RL')]


def settle(front_off, rear_off, seconds=1.5):
    data.qpos[:3] = [1.0, 0.0, float(h(1.0, 0.0)) + 0.47]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = gait.Q0
    mujoco.mj_forward(model, data)
    ctrl = np.concatenate([gait.Q0, np.zeros(4)])
    for i, s in enumerate((1, 1, -1, -1)):
        ctrl[3 * i + 1] = gait.LEG0[1] + (front_off if s == 1 else rear_off)
    for _ in range(int(seconds / 0.005)):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
    w, x, y, z = data.qpos[3:7]
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))
    fn = dict(F=0.0, R=0.0)
    for c in range(data.ncon):
        g1, g2 = data.contact[c].geom1, data.contact[c].geom2
        for p, gid, key in zip(('FR', 'FL', 'RR', 'RL'), wheel_geom, 'FFRR'):
            if g1 == gid or g2 == gid:
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, f6)
                fn['F' if key == 'F' else 'R'] += float(f6[0])
    print(f'  前髋{front_off:+.2f} 后髋{rear_off:+.2f} → 俯仰 {pitch:+6.1f}° | '
          f'前轮 {fn["F"]:7.1f} N | 后轮 {fn["R"]:7.1f} N | x={data.qpos[0]:.3f}')


print('== 髋偏置 → 俯仰/载荷 符号实验 (24.2° 坡, 锁轮) ==')
settle(0.0, 0.0)
settle(+0.20, 0.0)
settle(-0.20, 0.0)
settle(0.0, +0.20)
settle(0.0, -0.20)
print('SIGN_TEST_DONE')
