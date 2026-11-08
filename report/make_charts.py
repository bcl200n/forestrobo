# -*- coding: utf-8 -*-
"""生成部署报告所需图表 (离线 matplotlib, 中文字体 Microsoft YaHei)

输出 report/assets/:
  map_overview.png  部署总图 (山体阴影 + 271 林班边界 + 巡护路线 + 补给站)
  fleet_bars.png    各林场机队配置 (狗数/林班/面积)
  slope_hist.png    各林场林班坡度分布
  battery_bars.png  各林场出勤段/换电/能耗
"""
import io
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'E:\forestrobo\report\assets'
os.makedirs(OUT, exist_ok=True)

FARM_COLOR = {'211': '#e74c3c', '212': '#3498db', '213': '#2ecc71', '214': '#9b59b6'}
ACCENT = '#2A7A65'  # FG-1 表色

# ---------- 数据 ----------
with open(r'E:\forestrobo\rl\deployment_plan.json', encoding='utf-8') as f:
    plan = json.load(f)
zones = plan['zones']
with open(r'E:\forestrobo\林场范围哈\forest_compartments.geojson', encoding='utf-8') as f:
    gj = json.load(f)
comps = []
for ft in gj['features']:
    p = ft.get('properties', {}) or {}
    g = ft.get('geometry', {}) or {}
    c = g.get('coordinates')
    polys = [c] if g.get('type') == 'Polygon' and c else (c or [])
    best, bha = None, 0.0
    for poly in polys:
        ring = poly[0]
        x = [pt[0] for pt in ring]
        y = [pt[1] for pt in ring]
        s = 0.0
        for k in range(len(ring) - 1):
            s += math.radians(ring[k + 1][0] - ring[k][0]) * (
                2 + math.sin(math.radians(ring[k][1])) + math.sin(math.radians(ring[k + 1][1])))
        ha = abs(s) * 6378137.0 ** 2 / 2 / 10000
        if ha > bha:
            bha, best = ha, ring
    if best:
        comps.append({'farm': str(p.get('场')), 'ring': best,
                      'slope': None, 'mu': float(p.get('面积') or 0)})

# 坡度从 plan zones 的 member_slope 只到区级; 逐班坡度直接重算太重 — 用各区 member_slope 加权画分布
# 改为从 demo 数据(HTML)不取; 这里直接用 farm 级近似 + DEM 采样每班质心
dem = np.load(r'E:\forestrobo\data\leibo\dem\dem_z12.npy').astype(np.float32)
with open(r'E:\forestrobo\data\leibo\dem\dem_meta.json', encoding='utf-8') as f:
    DM = json.load(f)
PX = (2 ** DM['z']) * 256
X0PX, Y0PX = DM['x0'] * 256, DM['y0'] * 256
GND = 156543.03392 * math.cos(math.radians(28.47)) / (2 ** DM['z'])
gy, gx = np.gradient(dem, GND, GND)
SLOPE = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def lon_to_col(lon):
    return (lon + 180.0) / 360.0 * PX - X0PX


def lat_to_row(lat):
    rad = math.radians(lat)
    yw = (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0
    return yw * PX - Y0PX


for c in comps:
    ring = c['ring']
    sl = []
    step = max(1, len(ring) // 8)
    for pt in ring[::step]:
        r0, c0 = int(lat_to_row(pt[1])), int(lon_to_col(pt[0]))
        if 0 <= r0 < SLOPE.shape[0] and 0 <= c0 < SLOPE.shape[1]:
            sl.append(SLOPE[r0, c0])
    c['slope'] = float(np.mean(sl)) if sl else 0.0

# ---------- 图1: 部署总图 ----------
az, alt = math.radians(315), math.radians(45)
slope_r = np.arctan(np.hypot(gx, gy))
aspect = np.arctan2(-gx, gy)
shade = np.sin(alt) * np.cos(slope_r) + np.cos(alt) * np.sin(slope_r) * np.cos(az - aspect)
shade = np.clip((shade - shade.min()) / (np.ptp(shade) + 1e-6), 0, 1)

lon0 = -180.0 + 360.0 * X0PX / PX
lon1 = -180.0 + 360.0 * (X0PX + dem.shape[1]) / PX
yw_top = Y0PX / PX
yw_bot = (Y0PX + dem.shape[0]) / PX
lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yw_top))))
lat_bot = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yw_bot))))

fig, ax = plt.subplots(figsize=(10.6, 10.2), dpi=150)
ax.imshow(shade, extent=[lon0, lon1, lat_bot, lat_top], cmap='gray', alpha=0.55,
          origin='upper', aspect=1.0 / math.cos(math.radians(28.5)))

for c in comps:
    xy = [(pt[0], pt[1]) for pt in c['ring']]
    ax.add_patch(MplPolygon(xy, closed=True, facecolor=FARM_COLOR[c['farm']],
                            edgecolor=FARM_COLOR[c['farm']], alpha=0.30, linewidth=0.35))
for z in zones:
    la = [p[0] for p in z['route']]
    lo = [p[1] for p in z['route']]
    ax.plot(lo, la, color=FARM_COLOR[z['farm']], lw=0.9, alpha=0.85, zorder=5)
st_lat = [z['station'][0] for z in zones]
st_lon = [z['station'][1] for z in zones]
ax.scatter(st_lon, st_lat, marker='^', s=42, c='#111111', zorder=6, label='补给站 (31)')
ax.set_xlim(lon0, lon1)
ax.set_ylim(lat_bot, lat_top)
ax.set_xlabel('经度 (°E)', fontsize=11)
ax.set_ylabel('纬度 (°N)', fontsize=11)
ax.set_title('雷波林业局 211/212/213/214 场 · 271 林班机狗部署总图', fontsize=13, pad=12)
handles = [plt.Line2D([0], [0], color=FARM_COLOR[f], lw=3,
                      label='{}场 ({}区)'.format(f, sum(1 for z in zones if z['farm'] == f)))
           for f in sorted(FARM_COLOR)]
handles.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#111111',
                          markersize=9, label='补给站 (31)'))
handles.append(plt.Line2D([0], [0], color='#555', lw=1.2, label='巡护路线'))
ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.9)
ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'map_overview.png'), dpi=150)
plt.close(fig)
print('map_overview.png')

# ---------- 各场聚合 ----------
farms = sorted(FARM_COLOR)
agg = {}
for f in farms:
    fz = [z for z in zones if z['farm'] == f]
    fc = [c for c in comps if c['farm'] == f]
    agg[f] = {
        'dogs': len(fz), 'comps': len(fc),
        'mu': sum(c['mu'] for c in fc) / 10000,
        'km': sum(z['km'] for z in fz),
        'h': sum(z['hours'] for z in fz),
        'seg': sum(z['sorties'] for z in fz),
        'sw': sum(z['swaps'] for z in fz),
        'wh': sum(z['wh'] for z in fz) / 1000,
        'slopes': [c['slope'] for c in fc],
    }

# ---------- 图2: 机队配置 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=150)
x = np.arange(len(farms))
w = 0.36
axes[0].bar(x - w / 2, [agg[f]['dogs'] for f in farms], w, color=ACCENT, label='机器狗 (只)')
axes[0].bar(x + w / 2, [agg[f]['comps'] for f in farms], w, color='#B8CFC7', label='林班 (个)')
for i, f in enumerate(farms):
    axes[0].text(i - w / 2, agg[f]['dogs'] + 1.5, str(agg[f]['dogs']), ha='center', fontsize=9)
    axes[0].text(i + w / 2, agg[f]['comps'] + 1.5, str(agg[f]['comps']), ha='center', fontsize=9)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f + '场' for f in farms])
axes[0].set_title('(a) 机队与林班数', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].spines[['top', 'right']].set_visible(False)

bars = axes[1].bar(x, [agg[f]['mu'] for f in farms], 0.5, color=[FARM_COLOR[f] for f in farms])
for i, f in enumerate(farms):
    axes[1].text(i, agg[f]['mu'] + 0.03, '{:.2f}'.format(agg[f]['mu']), ha='center', fontsize=9)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f + '场' for f in farms])
axes[1].set_title('(b) 巡护面积 (万亩)', fontsize=11)
axes[1].spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fleet_bars.png'), dpi=150)
plt.close(fig)
print('fleet_bars.png')

# ---------- 图3: 坡度分布 ----------
fig, ax = plt.subplots(figsize=(10.4, 3.9), dpi=150)
bins = np.arange(0, 40, 2.5)
for f in farms:
    ax.hist(agg[f]['slopes'], bins=bins, histtype='step', lw=1.8,
            color=FARM_COLOR[f], label='{}场 (均{:.1f}°)'.format(f, np.mean(agg[f]['slopes'])))
ax.axvline(25, color='#888', ls='--', lw=1)
ax.text(25.4, ax.get_ylim()[1] * 0.92, '25° 陡坡参考线', fontsize=9, color='#555')
ax.set_xlabel('林班平均坡度 (°)', fontsize=11)
ax.set_ylabel('林班数量', fontsize=11)
ax.set_title('271 个林班坡度分布 (真实 DEM 30m 采样)', fontsize=12)
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'slope_hist.png'), dpi=150)
plt.close(fig)
print('slope_hist.png')

# ---------- 图4: 电池负荷 ----------
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), dpi=150)
metrics = [('seg', '出勤段 (段/循环)', ACCENT), ('sw', '换电次数 (次/循环)', '#D4875A'),
           ('wh', '循环能耗 (kWh)', '#5B8DB8')]
for axi, (k, title, col) in zip(axes, metrics):
    axi.bar(x, [agg[f][k] for f in farms], 0.5, color=col)
    for i, f in enumerate(farms):
        v = agg[f][k]
        axi.text(i, v + max(agg[f][k] for f in farms) * 0.02,
                 '{:.1f}'.format(v) if k == 'wh' else str(int(v)), ha='center', fontsize=9)
    axi.set_xticks(x)
    axi.set_xticklabels([f + '场' for f in farms])
    axi.set_title(title, fontsize=10.5)
    axi.spines[['top', 'right']].set_visible(False)
fig.suptitle('各林场电池保障负荷 (双包 1500 Wh × 80%)', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'battery_bars.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('battery_bars.png')

print('CHARTS_DONE')
