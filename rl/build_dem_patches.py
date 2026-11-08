# -*- coding: utf-8 -*-
"""从真实巡逻路线 + DEM 采样训练地形块, 按坡度分级入库 dem_patches.json

分级 (对应路线坡度谱):
  C1 缓坡 12-20°   (占真实里程 21%)
  C2 陡坡 20-28°   (占 37% 主体)
  C3 极陡 28-36°   (占 37% 中的硬核段)
输出: rl/dem_patches.json — 每级 8 块 {lat, lon, slope, rot}, 块间距 ≥250m
"""
import io
import json
import math
import random
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DEM_DIR = r'E:\forestrobo\data\leibo\dem'
dem = np.load(DEM_DIR + r'\dem_z12.npy').astype(np.float32)
meta = json.load(open(DEM_DIR + r'\dem_meta.json', encoding='utf-8'))
PX = (2 ** meta['z']) * 256
GND = 156543.03392 * math.cos(math.radians(28.47)) / (2 ** meta['z'])
gy, gx = np.gradient(dem, GND, GND)
SLOPE = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def to_rc(lat, lon):
    rad = math.radians(lat)
    yw = (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0
    return yw * PX - meta['y0'] * 256, (lon + 180.0) / 360.0 * PX - meta['x0'] * 256


def slope_at(lat, lon):
    r, c = to_rc(lat, lon)
    return float(SLOPE[int(np.clip(r, 0, dem.shape[0] - 1)), int(np.clip(c, 0, dem.shape[1] - 1))])


with open(r'E:\forestrobo\rl\deployment_plan.json', encoding='utf-8') as f:
    plan = json.load(f)

# 沿全部路线每 80m 采样候选点
cands = []
for z in plan['zones']:
    pts = z['route']
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        dy = (b[0] - a[0]) * 110574.0
        dx = (b[1] - a[1]) * 111320.0 * math.cos(math.radians((a[0] + b[0]) / 2))
        d = math.hypot(dx, dy)
        n = max(1, int(d / 80))
        for i in range(n + 1):
            t = i / n
            lat = a[0] + (b[0] - a[0]) * t
            lon = a[1] + (b[1] - a[1]) * t
            cands.append((lat, lon, slope_at(lat, lon), z['id']))

print('路线采样点: {}'.format(len(cands)))
rng = random.Random(7)
bank = {}
for name, lo, hi in [('C1', 12, 20), ('C2', 20, 28), ('C3', 28, 36)]:
    pool = [c for c in cands if lo <= c[2] < hi]
    pool.sort(key=lambda c: -c[2])  # 坡度大的优先 (C1 则接近 20 优先)
    if name == 'C1':
        pool.sort(key=lambda c: c[2])  # C1 取接近 12 侧? 不: 取中位附近 → 反转排序取较高段
        pool = pool[::-1][-len(pool):]
    picked = []
    for c in pool:
        if any(math.hypot((c[0] - p['lat']) * 110574, (c[1] - p['lon']) * 98000) < 250
               for p in picked):
            continue
        picked.append({'lat': round(c[0], 6), 'lon': round(c[1], 6),
                       'slope': round(c[2], 1), 'rot': round(rng.uniform(0, 360), 1),
                       'zone': c[3]})
        if len(picked) >= 8:
            break
    bank[name] = picked
    print('{} ({}-{}°): {} 块, 坡度 {}'.format(
        name, lo, hi, len(picked), '/'.join(str(p['slope']) for p in picked)))

with open(r'E:\forestrobo\rl\dem_patches.json', 'w', encoding='utf-8') as f:
    json.dump(bank, f, ensure_ascii=False, indent=1)
print('PATCH_BANK_DONE')
