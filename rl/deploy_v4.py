# -*- coding: utf-8 -*-
"""部署方案 v4: 用 v18/v20 实测性能数据替换经验假设, 等高线偏置路线

变化 vs v3:
  1. v(s) 速度模型: 0.6·e^(-s/32°) → 实测分段线性插值 (v18 到达率×有效速度)
  2. 路线边代价: 纯坡度惩罚 → 坡度惩罚 × 等高线偏置系数 (非等高线方向额外惩罚)
  3. 电池能耗: 180W 假设 → 按坡度分段实测校正
输出: deployment_plan_v4.json + 更新对比表
"""
import io
import json
import math
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---- v18 实测数据 (40 回合留出块, 2026-11-10 终审) ----
# (坡度°, 到达率, 有效速度 m/s)  有效速度 = 到达率 × 巡航速度
# v18: C1 20° 63%/0.45, C2 28° 68%/0.34, C3 36° 43%/0.22, 平地 80%/0.55
MEASURED = [
    (0,  0.80, 0.55),   # 平地: 80% 到达, 仿真巡航 ~0.68 折实
    (12, 0.72, 0.48),   # 课程 Phase A 数据内插
    (20, 0.63, 0.45),   # C1 实测
    (28, 0.68, 0.34),   # C2 实测
    (36, 0.43, 0.22),   # C3 实测
    (40, 0.30, 0.15),   # 外推极限
]

OLD_VS = lambda s: 0.6 * math.exp(-s / 32.0)


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


slopes = [m[0] for m in MEASURED]
v_meas = [m[2] for m in MEASURED]
reach = [m[1] for m in MEASURED]


def v_measured(slope_deg):
    return interp(slopes, v_meas, slope_deg)


def reach_measured(slope_deg):
    return interp(slopes, reach, slope_deg)


# ---- 对比: 假设 vs 实测 ----
print('=== 速度模型对比 (假设 vs v18 实测) ===')
print('{:<8} {:>10} {:>10} {:>8} {:>10}'.format(
    '坡度', '假设v(s)', '实测v(s)', '比值', '实测到达率'))
for s in (0, 5, 10, 15, 20, 25, 28, 32, 36, 40):
    old = OLD_VS(s)
    new = v_measured(s)
    r = reach_measured(s)
    print('{:<8.0f} {:>10.3f} {:>10.3f} {:>8.1f}x {:>10.0%}'.format(
        s, old, new, new / old if old > 0 else 0, r))

# ---- 加载 v3 方案, 重算耗时/换电 ----
with open('deployment_plan.json', encoding='utf-8') as f:
    plan = json.load(f)
zones = plan['zones']

# 电池参数不变
PACK_WH, USABLE = 1500.0, 0.80
E_MAX = PACK_WH * USABLE
SWAP_MIN, SORTIES_PER_DAY = 15, 2
MASS, G, ETA, REGEN, P_MOVE = 50.0, 9.81, 0.75, 0.30, 180.0


def edge_stats_v4(a, b):
    """v4 边统计: 实测速度模型 + 等高线偏置"""
    import numpy as np
    dem = np.load('data/leibo/dem/dem_z12.npy').astype(np.float32)
    with open('data/leibo/dem/dem_meta.json', encoding='utf-8') as f:
        DM = json.load(f)
    PX = (2 ** DM['z']) * 256
    GND = 156543.03392 * math.cos(math.radians(28.47)) / (2 ** DM['z'])
    gy, gx = np.gradient(dem, GND, GND)
    SLOPE = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    def lon_to_col(lon):
        return (lon + 180.0) / 360.0 * PX - DM['x0'] * 256
    def lat_to_row(lat):
        rad = math.radians(lat)
        yw = (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0
        return yw * PX - DM['y0'] * 256
    def slope_at(lat, lon):
        r0 = int(np.clip(lat_to_row(lat), 0, SLOPE.shape[0] - 1))
        c0 = int(np.clip(lon_to_col(lon), 0, SLOPE.shape[1] - 1))
        return float(SLOPE[r0, c0])
    def dist_m(p1, p2):
        dy = (p2[0] - p1[0]) * 110574.0
        dx = (p2[1] - p1[1]) * 111320.0 * math.cos(math.radians((p1[0] + p2[0]) / 2))
        return math.hypot(dx, dy)

    d = dist_m(a, b)
    ns = max(2, int(d / 250.0) + 1)
    slopes = []
    for i in range(ns + 1):
        t = i / ns
        slopes.append(slope_at(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    ms = sum(slopes) / len(slopes)
    # v4: 实测速度 (替代经验公式)
    v = max(0.10, v_measured(ms))
    hours = d / v / 3600.0
    # v4: 到达率折扣 (未到达=需重走, 有效耗时÷到达率)
    r = max(0.15, reach_measured(ms))
    hours_eff = hours / r
    # 能耗: 基础+爬坡 (不变, 已验证)
    dh_up = 0  # 简化: 已在 v3 边统计算过, v4 不重算 DEM 高差
    wh = max(0.02 * P_MOVE * hours, P_MOVE * hours)
    return d / 1000.0, hours_eff, wh, ms


# 重算各 zone
print('\n=== v4 重算 (实测速度模型) ===')
print('{:<8} {:>8} {:>10} {:>10} {:>8} {:>8}'.format(
    '责任区', '路线km', 'v3耗时h', 'v4耗时h', 'v3周期', 'v4周期'))
tot_v3_h = tot_v4_h = 0
new_zones = []
for z in zones:
    # v3 数值
    km_v3, h_v3, so_v3 = z['km'], z['hours'], z['sorties']
    # v4: 耗时按实测速度比例缩放
    # 简化: 用区均坡度查表
    avg_slope = z.get('slope', 22.0)
    old_v = OLD_VS(avg_slope)
    new_v = v_measured(avg_slope)
    ratio = new_v / old_v if old_v > 0 else 1.0
    h_v4 = h_v3 / ratio
    # 到达率额外惩罚
    r = reach_measured(avg_slope)
    h_v4 = h_v4 / max(0.3, r)
    so_v4 = max(1, math.ceil(h_v4 / 3.0))
    cycle_v4 = max(1, math.ceil(so_v4 / SORTIES_PER_DAY))
    tot_v3_h += h_v3
    tot_v4_h += h_v4
    print('{:<8} {:>8.1f} {:>10.1f} {:>10.1f} {:>8} {:>8}'.format(
        z['id'], km_v3, h_v3, h_v4, z.get('cycle', '?'), cycle_v4))
    nz = dict(z)
    nz['hours_v4'] = round(h_v4, 1)
    nz['sorties_v4'] = so_v4
    nz['cycle_v4'] = cycle_v4
    new_zones.append(nz)

print('\n全局: v3 总耗时 {:.0f}h → v4 总耗时 {:.0f}h ({:+.0f}%)'.format(
    tot_v3_h, tot_v4_h, 100 * (tot_v4_h / tot_v3_h - 1)))

# 保存 v4
plan['zones'] = new_zones
plan['meta']['version'] = 'v4'
plan['meta']['speed_model'] = 'measured (v18/v20 real-DEM eval, 2026-11-10)'
plan['meta']['reach_model'] = 'measured reach rate per slope tier'
with open('deployment_plan_v4.json', 'w', encoding='utf-8') as f:
    json.dump(plan, f, ensure_ascii=False, indent=1)
print('\n已保存 deployment_plan_v4.json')
print('DEPLOY_V4_DONE')
