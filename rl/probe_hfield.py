# -*- coding: utf-8 -*-
"""hfield 探针：对比数据方向 + 射线测真实表面 → 得出正确映射"""
import json
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

import terrain as terr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path('forest_dog.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

with open('terrain_meta.json', encoding='utf-8') as f:
    meta = json.load(f)
h_fn = terr.build_from_params(meta)
n = meta['n']
ax = np.linspace(-meta['span'] / 2, meta['span'] / 2, n)
gx, gy = np.meshgrid(ax, ax)
want = h_fn(gx, gy)
want_norm = (want - meta['z_min']) / (meta['z_max'] - meta['z_min'])

hf = model.hfield_data.reshape(n, n).astype(float)
print('hfield_data 范围: [%.4f, %.4f] n=%d' % (hf.min(), hf.max(), model.hfield_data.size))

for name, cand in (('原样', want_norm), ('上下翻转', want_norm[::-1]),
                   ('转置', want_norm.T), ('转置+上下翻转', want_norm.T[::-1]),
                   ('左右翻转', want_norm[:, ::-1])):
    err = np.abs(cand - hf).mean()
    print(f'  {name}: 平均偏差 {err:.4f}')

print('\n== 射线探针 (从 z=60 垂直向下) ==')
geomid = np.zeros(1, dtype=np.int32)
for (x, y) in [(5, 0), (10, 3), (-6, -4), (0, -10), (15, -8), (1, 0),
               (-10, 8), (20, 0), (0, 12)]:
    pnt = np.array([x, y, 60.0])
    vec = np.array([0.0, 0.0, -1.0])
    dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geomid)
    surf = 60.0 - dist if dist >= 0 else None
    want_h = float(h_fn(x, y))
    v_at = float(hf[::-1][int(round((y + 15) / 30 * (n - 1))),
                          int(round((x + 15) / 30 * (n - 1)))])
    print(f'  ({x:5.1f},{y:5.1f}): 表面 {None if surf is None else round(float(surf), 3)} | '
          f'解析H {want_h:7.3f} | 归一值 {v_at:.3f}')
print('PROBE_DONE')
