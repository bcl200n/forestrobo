# -*- coding: utf-8 -*-
"""轮足爬坡测试：纯净坡面 24.2°/30°/35° + 山地地形行驶
对照组: 上一轮足式(迈步)在 24.2° 即失速
"""
import json
import math
import os

import mujoco
import numpy as np

import gait
import generate_model as gm
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def pitch_of(d):
    w, x, y, z = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
    return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))


def run(xml, h, v_ref=1.5, seconds=10.0, label=''):
    model = mujoco.MjModel.from_xml_path(xml)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    x0, y0 = 0.8, 0.0                       # 出生在整平区内
    data.qpos[:3] = [x0, y0, float(h(x0, y0)) + gait.WSTAND_Z + 0.03]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = gait.WQ0
    data.ctrl[:12] = gait.WQ0
    mujoco.mj_forward(model, data)
    n = int(seconds / 0.005)
    zr, ps = [], []
    t_ramp = 2.0                            # 速度软起动, 防扭矩反冲
    e = 0.3
    for k in range(n):
        v = v_ref * min(1.0, data.time / t_ramp)
        # 调平目标 = 当地坡面角(沿航向梯度), 机身贴合坡面保持四轮均载
        gx = (float(h(data.qpos[0] + e, data.qpos[1]))
              - float(h(data.qpos[0] - e, data.qpos[1]))) / (2 * e)
        pitch_tgt = -math.atan(gx)
        data.ctrl[:] = gait.wheel_drive(v, pitch=math.radians(pitch_of(data)),
                                        pitch_target=pitch_tgt)
        mujoco.mj_step(model, data)
        zr.append(data.qpos[2] - float(h(data.qpos[0], data.qpos[1])))
        ps.append(pitch_of(data))
    zr, ps = np.array(zr), np.array(ps)
    climb = float(h(data.qpos[0], data.qpos[1])) - float(h(x0, y0))
    ok = data.qpos[0] - x0 > 2.0 and zr[-90:].mean() > 0.30
    print(f'  {label}: 前进 {data.qpos[0] - x0:5.2f} m | 爬升 {climb:5.2f} m | '
          f'均速 {(data.qpos[0] - x0) / seconds:4.2f} m/s | '
          f'相对高度 {zr.mean():.2f}/{zr.min():.2f} | 俯仰 {ps.mean():+5.1f}° '
          f'→ {"爬升成功" if ok else "失速"}')
    return ok


with open('terrain_meta.json', encoding='utf-8') as f:
    h_hill = terr.build_from_params(json.load(f))

print('== 轮足爬坡测试（纯净坡面, 站姿调平 + 轮毂速度伺服, 10 s）==')
for s in (24.2, 30.0, 35.0):
    ramp = terr.make_ramp_fn(s)
    gm.generate(h_fn=ramp, n_trees=0, n_rocks=0, n_logs=0, wheel=True)
    with open('terrain_meta.json', encoding='utf-8') as f:
        h = terr.build_from_params(json.load(f))
    run('forest_dog_wheel.xml', h, label=f'{s:4.1f}° ({math.tan(math.radians(s)) * 100:.0f}% 坡度)')

print('== 山地地形行驶（起伏 + 6° 均值坡, 12 s）==')
gm.generate(wheel=True)
with open('terrain_meta.json', encoding='utf-8') as f:
    h_hill = terr.build_from_params(json.load(f))
run('forest_dog_wheel.xml', h_hill, v_ref=2.0, seconds=12.0, label='山地地形')
gm.generate()   # 恢复足式山地模型
print('SLOPE_WHEEL_DONE')
