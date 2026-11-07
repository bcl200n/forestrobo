# -*- coding: utf-8 -*-
"""地形标定 + 山地行走验证
1. 五点投放: 狗落到已知 (x,y), 静置后躯干高度应 ≈ H(x,y)+0.446 → 校验高度图映射
2. 山地小跑 8 s: 相对高度稳定性和俯仰统计
"""
import json
import math
import os

import mujoco
import numpy as np

import gait
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path('forest_dog.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)
with open('terrain_meta.json', encoding='utf-8') as f:
    h_fn = terr.build_from_params(json.load(f))


def pitch_of(q):
    w, x, y, z = q[3], q[4], q[5], q[6]
    return math.degrees(math.asin(np.clip(2 * (w * y - z * x), -1, 1)))


def place(x, y):
    mujoco.mj_resetData(model, data)
    z = float(h_fn(x, y))
    data.qpos[:3] = [x, y, z + 0.55]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = gait.Q0
    data.ctrl[:] = gait.Q0
    mujoco.mj_forward(model, data)


print('== 五点投放标定 (静置 1s, 期望躯干z - 地形H ≈ 0.446) ==')
ok = True
for px, py in [(5, 0), (10, 3), (-6, -4), (0, -10), (15, -8)]:
    place(px, py)
    for _ in range(200):
        data.ctrl[:] = gait.Q0
        mujoco.mj_step(model, data)
    dh = data.qpos[2] - float(h_fn(px, py))
    flag = 'OK' if abs(dh - 0.446) < 0.08 else '**偏离**'
    if abs(dh - 0.446) >= 0.08:
        ok = False
    print(f'  ({px:5.1f},{py:5.1f}): H={float(h_fn(px, py)):6.2f} 躯干z={data.qpos[2]:6.2f} '
          f'Δh={dh:.3f} 俯仰={pitch_of(data.qpos):+5.1f}° {flag}')

print('== 山地小跑 8 s (起点 1,0 直线前进) ==')
place(1.0, 0.0)
n = int(8 / 0.005)
zr, ps = [], []
x0 = data.qpos[0]
h0 = float(h_fn(1.0, 0.0))
for _ in range(n):
    data.ctrl[:] = gait.cpg(data.time, pitch=math.radians(pitch_of(data.qpos)))
    mujoco.mj_step(model, data)
    zr.append(data.qpos[2] - float(h_fn(data.qpos[0], data.qpos[1])))
    ps.append(pitch_of(data.qpos))
zr, ps = np.array(zr), np.array(ps)
print(f'  相对高度 均值 {zr.mean():.3f} / 最低 {zr.min():.3f} m | '
      f'前进 {data.qpos[0] - x0:.2f} m | 爬升 {float(h_fn(data.qpos[0], data.qpos[1])) - h0:.2f} m')
print(f'  俯仰 均值 {ps.mean():+.1f}° / 范围 [{ps.min():+.1f}, {ps.max():+.1f}]°')
print('TERRAIN_TEST_DONE', '(标定OK)' if ok else '(标定偏离, 需翻转高度图)')
