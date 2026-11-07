# -*- coding: utf-8 -*-
"""真实表面 vs 解析高度: 射线量测 + 半格偏移假设检验"""
import json
import os

import mujoco
import numpy as np

import generate_model as gm
import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

gm.generate(wheel=True)
model = mujoco.MjModel.from_xml_path('forest_dog_wheel.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with open('terrain_meta.json', encoding='utf-8') as f:
    meta = json.load(f)
h = terr.build_from_params(meta)

# 1) 引擎真实表面(射线)
geomid = np.zeros(1, dtype=np.int32)
pts = [(0.8, 0.0), (1.0, 0.5), (1.5, -0.5), (2.0, 1.0), (0.5, -1.5),
       (3.0, 0.0), (5.0, 2.0), (-2.0, -3.0)]
print('点(x,y)     射线表面     解析H      差值')
diffs = []
for x, y in pts:
    d = mujoco.mj_ray(model, data, np.array([x, y, 200.0]),
                      np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
    surf = 200.0 - d
    ha = float(h(x, y))
    diffs.append(surf - ha)
    print(f'({x:5.1f},{y:5.1f})  {surf:9.3f}  {ha:9.3f}  {surf - ha:+7.3f}')
diffs = np.array(diffs)
print(f'差值: 均值 {diffs.mean():+.3f} / 波动 ±{diffs.std():.3f}')

# 2) 半格偏移检验: 解析(x±半格) 是否解释差值
n, span = meta['n'], meta['span']
half = (span / (n - 1)) / 2
for name, sx, sy in (('x+半格', half, 0), ('x-半格', -half, 0),
                     ('y+半格', 0, half), ('y-半格', 0, -half)):
    dd = []
    for x, y in pts:
        d = mujoco.mj_ray(model, data, np.array([x, y, 200.0]),
                          np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        dd.append(200.0 - d - float(h(x + sx, y + sy)))
    dd = np.array(dd)
    print(f'{name}: 差值均值 {dd.mean():+.4f} / 波动 ±{dd.std():.4f}')
print('SURFACE_CAL_DONE')
