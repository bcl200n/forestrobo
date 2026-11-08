# -*- coding: utf-8 -*-
"""机器狗部署规划器 v2: 真实地形版。

相对 v1:
- 接入真实 DEM (Mapzen terrarium ~30m, fetch_dem.py 下载), 逐林班采样坡度/海拔
- 路线边代价 = 直线距离 × (1 + (均坡/22°)^2), 沿边每 250m 采样坡度 (翻山越岭自动变贵)
- 速度模型 v = 0.6 × exp(-均坡/24°), 耗时按分段实际速度累计
- 责任区拆分判据改为地形耗时 (>4 次出勤才拆, 控制机队规模)
- 演示页: 山体阴影叠加 + 坡度填色模式 + 林相数据接口说明

输出: rl/forest_deployment_demo.html (+ rl/hillshade.png)
"""
import io
import json
import math
import os
import random
import sys

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SRC = r'E:\forestrobo\林场范围哈\forest_compartments.geojson'
OUT = r'E:\forestrobo\rl\forest_deployment_demo.html'
HILL = r'E:\forestrobo\rl\hillshade.png'
DEM_DIR = r'E:\forestrobo\data\leibo\dem'

FIELD_SPEED = 0.6        # m/s 平地林下巡航
SORTIE_HOURS = 3.5       # h 单次出勤有效时间 (平地)
TARGET_ZONES = {'211': 3, '212': 8, '213': 6, '214': 8}
MAX_ROUTE_KM = 20.0      # 几何长度上限 (仍保留)
MAX_SORTIES = 6          # 巡护周期上限 (≈6 天, 超过才拆区加狗)
MIN_SPLIT_N = 5
TOL_M = 8.0

# ---------- 电池/能耗模型 (50kg 级林下作业四足, 双包可换电设计) ----------
MASS = 50.0              # kg 整机+载荷
PACK_WH = 1500.0         # 双电池组标称 Wh (2×750Wh 热插拔轮换)
USABLE = 0.80            # 可用放电深度 (留 20% 保护)
P_MOVE = 180.0           # W 慢速步态基础功率 (0.6 m/s 林下)
ETA = 0.75               # 爬坡机电效率
REGEN = 0.30             # 下坡能量回收率
SWAP_MIN = 15            # 换电+自检 分钟/次
CHARGE_H = 1.5           # 单组充满小时 (充电柜双充)
SORTIES_PER_DAY = 2      # 上午/下午各一段 (换电 15min + 午间充电)


# ---------- 地理工具 ----------
def perp_dist(pt, a, b):
    x0, y0 = pt
    x1, y1 = a
    x2, y2 = b
    dx = (x2 - x1) * math.cos(math.radians(y1)) * 111320.0
    dy = (y2 - y1) * 110574.0
    px = (x0 - x1) * math.cos(math.radians(y1)) * 111320.0
    py = (y0 - y1) * 110574.0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px, py)
    t = max(0.0, min(1.0, (px * dx + py * dy) / L2))
    return math.hypot(px - t * dx, py - t * dy)


def simplify(pts, tol=TOL_M):
    if len(pts) <= 8:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        dmax, imax = -1.0, i
        for k in range(i + 1, j):
            d = perp_dist(pts[k], pts[i], pts[j])
            if d > dmax:
                dmax, imax = d, k
        if dmax > tol:
            keep[imax] = True
            stack.append((i, imax))
            stack.append((imax, j))
    out = [p for p, k in zip(pts, keep) if k]
    return out if len(out) >= 4 else pts


def ring_area_ha(ring):
    if len(ring) < 4:
        return 0.0
    s = 0.0
    for k in range(len(ring) - 1):
        x1, y1 = ring[k][0], ring[k][1]
        x2, y2 = ring[k + 1][0], ring[k + 1][1]
        s += math.radians(x2 - x1) * (2 + math.sin(math.radians(y1)) + math.sin(math.radians(y2)))
    return abs(s) * 6378137.0 * 6378137.0 / 2.0 / 10000.0


def centroid(ring):
    xs = [p[0] for p in ring[:-1]] or [p[0] for p in ring]
    ys = [p[1] for p in ring[:-1]] or [p[1] for p in ring]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def dist_m(p1, p2):
    dy = (p2[0] - p1[0]) * 110574.0
    dx = (p2[1] - p1[1]) * 111320.0 * math.cos(math.radians((p1[0] + p2[0]) / 2))
    return math.hypot(dx, dy)


# ---------- 真实 DEM ----------
dem = np.load(os.path.join(DEM_DIR, 'dem_z12.npy')).astype(np.float32)
with open(os.path.join(DEM_DIR, 'dem_meta.json'), encoding='utf-8') as f:
    DM = json.load(f)
Z = DM['z']
PX = (2 ** Z) * 256
X0PX, Y0PX = DM['x0'] * 256, DM['y0'] * 256
DEM_LAT0 = math.radians(28.47)  # 参考纬度用于像素地面分辨率
GND = 156543.03392 * math.cos(DEM_LAT0) / (2 ** Z)  # m/px ≈ 30.2

gy, gx = np.gradient(dem, GND, GND)
SLOPE = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def lon_to_col(lon):
    return (lon + 180.0) / 360.0 * PX - X0PX


def lat_to_row(lat):
    rad = math.radians(lat)
    yw = (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0
    return yw * PX - Y0PX


def row_to_lat(row):
    yw = (row + Y0PX) / PX
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yw))))


def _bilinear(arr, r, c):
    h, w = arr.shape
    r = min(max(r, 0.0), h - 1.001)
    c = min(max(c, 0.0), w - 1.001)
    r0, c0 = int(r), int(c)
    fr, fc = r - r0, c - c0
    return (arr[r0, c0] * (1 - fr) * (1 - fc) + arr[r0, c0 + 1] * (1 - fr) * fc +
            arr[r0 + 1, c0] * fr * (1 - fc) + arr[r0 + 1, c0 + 1] * fr * fc)


def elev_at(lat, lon):
    return float(_bilinear(dem, lat_to_row(lat), lon_to_col(lon)))


def slope_at(lat, lon):
    return float(_bilinear(SLOPE, lat_to_row(lat), lon_to_col(lon)))


def edge_stats(a, b):
    """边 (a→b) 的地形统计: 沿直线每 250m 采样坡度。

    返回 (代价, 几何km, 耗时h, 均坡°, 海拔min, 海拔max)。
    """
    d = dist_m(a, b)
    ns = max(2, int(d / 250.0) + 1)
    slopes, elevs = [], []
    for i in range(ns + 1):
        t = i / ns
        la, lo = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        slopes.append(slope_at(la, lo))
        elevs.append(elev_at(la, lo))
    ms = sum(slopes) / len(slopes)
    cost = d * (1.0 + (ms / 22.0) ** 2)
    # 变速: 坡度按横坡处理 (巡护多为沿等高线), 30° 仍有 ~39% 速度
    v = max(0.15, FIELD_SPEED * math.exp(-ms / 32.0))
    hours = d / v / 3600.0
    # 能耗: 基础功率×时间 + 爬坡位能/η − 下坡回收
    dh_up = sum(max(0.0, elevs[i + 1] - elevs[i]) for i in range(len(elevs) - 1))
    dh_dn = sum(max(0.0, elevs[i] - elevs[i + 1]) for i in range(len(elevs) - 1))
    wh = max(0.02 * P_MOVE * hours,
             P_MOVE * hours + MASS * 9.81 * dh_up / ETA / 3600.0
             - MASS * 9.81 * dh_dn * REGEN / 3600.0)
    return cost, d / 1000.0, hours, ms, min(elevs), max(elevs), wh


def make_hillshade(path, size=1152):
    """生成山体阴影 PNG (方位 315°, 高度 45°), 降采样到 size×size。"""
    k = dem.shape[0] / size
    small = dem[::max(1, int(k)), ::max(1, int(k))][:size, :size]
    gy2, gx2 = np.gradient(small, GND * dem.shape[0] / small.shape[0])
    az, alt = math.radians(315), math.radians(45)
    slope_r = np.arctan(np.hypot(gx2, gy2))
    aspect = np.arctan2(-gx2, gy2)
    shade = (math.sin(alt) * np.cos(slope_r) +
             math.cos(alt) * np.sin(slope_r) * np.cos(az - aspect))
    shade = np.clip((shade - shade.min()) / (np.ptp(shade) + 1e-6), 0, 1)
    g = (shade * 255).astype(np.uint8)
    alpha = (np.clip(slope_r * 3.2, 0, 1) * 170).astype(np.uint8)  # 平地透明, 陡坡显影
    img = np.dstack([g, g, g, alpha])
    Image.fromarray(img, 'RGBA').save(path)
    bounds = [[row_to_lat(0.0), -180.0 + 360.0 * X0PX / PX],
              [row_to_lat(float(dem.shape[0])), -180.0 + 360.0 * (X0PX + dem.shape[1]) / PX]]
    return bounds  # [[latN, lonW], [latS, lonE]]


HILL_BOUNDS = make_hillshade(HILL)
print('DEM: {}x{} px, {:.1f} m/px, 高程 {}-{} m'.format(
    dem.shape[0], dem.shape[1], GND, int(dem.min()), int(dem.max())))


# ---------- 数据加载 ----------
with open(SRC, encoding='utf-8') as f:
    gj = json.load(f)

comps = []
for ft in gj.get('features', []):
    p = ft.get('properties', {}) or {}
    geom = ft.get('geometry', {}) or {}
    farm = str(p.get('场', '?'))
    zone = str(p.get('作业区', '?'))
    num = str(p.get('林班', '?'))
    name = p.get('名称') or '{}林班'.format(num)
    area_mu = float(p.get('面积') or 0)

    coords = geom.get('coordinates')
    polys = [coords] if geom.get('type') == 'Polygon' and coords else (coords or [])
    best_ring, best_ha = None, 0.0
    for poly in polys:
        for ring in poly:
            r = simplify([[float(x) for x in pt[:2]] for pt in ring])
            ha = ring_area_ha(r)
            if ha > best_ha:
                best_ha, best_ring = ha, r
    if not best_ring:
        continue
    if area_mu <= 0:
        area_mu = best_ha * 15.0
    cen = centroid(best_ring)

    # 林班地形采样: 质心 + 边界点
    sl = [slope_at(cen[1], cen[0])]
    el = [elev_at(cen[1], cen[0])]
    step = max(1, len(best_ring) // 10)
    for pt in best_ring[::step]:
        sl.append(slope_at(pt[1], pt[0]))
        el.append(elev_at(pt[1], pt[0]))

    comps.append({
        'farm': farm, 'admin_zone': zone, 'num': num, 'name': name,
        'area_mu': area_mu,
        'slope': round(sum(sl) / len(sl), 1),
        'elev': int(sum(el) / len(el)),
        'pts': [[round(pt[1], 6), round(pt[0], 6)] for pt in best_ring],
        'cen': [round(cen[1], 6), round(cen[0], 6)],
    })

print('林班: {}'.format(len(comps)))
for farm in sorted(TARGET_ZONES):
    fl = [c for c in comps if c['farm'] == farm]
    if fl:
        print('  {}场: 均坡 {:.1f}°, 最大 {:.1f}°, 海拔 {}-{} m'.format(
            farm, sum(c['slope'] for c in fl) / len(fl),
            max(c['slope'] for c in fl),
            min(c['elev'] for c in fl), max(c['elev'] for c in fl)))


# ---------- 责任区聚类 ----------
def cluster(comp_list, k, seed=7):
    rng = random.Random(seed)
    pts = [(c['cen'][0] * 110574.0, c['cen'][1] * 111320.0, c['area_mu']) for c in comp_list]
    cents = [pts[rng.randrange(len(pts))][:2]]
    while len(cents) < k:
        best, bd = None, -1.0
        for p in pts:
            d = min((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in cents) * p[2]
            if d > bd:
                bd, best = d, p
        cents.append(best[:2])
    assign = [0] * len(pts)
    for _ in range(60):
        moved = False
        for i, p in enumerate(pts):
            j = min(range(k), key=lambda c: (p[0] - cents[c][0]) ** 2 + (p[1] - cents[c][1]) ** 2)
            if j != assign[i]:
                assign[i], moved = j, True
        if not moved:
            break
        for c in range(k):
            sw = sx = sy = 0.0
            for i, p in enumerate(pts):
                if assign[i] == c:
                    w = p[2]
                    sw += w
                    sx += p[0] * w
                    sy += p[1] * w
            if sw > 0:
                cents[c] = (sx / sw, sy / sw)
    return assign


ROUTE_STATS = {'cost': 0.0, 'geo_km': 0.0, 'flat_km': 0.0, 'flat_cost': 0.0}


def optimize_route(station, members, terrain=True):
    """TSP: 最近邻 + 2-opt + Or-opt。terrain=True 时边权=地形代价, 否则=纯距离。

    返回 (pts闭合, km, hours, 均坡, elev_min, elev_max, edges, order_nodes)。
    edges: [{'a','b','km','h','wh'}] 闭合有序边序, 供电池切段。
    """
    nodes = [station] + [m['cen'] for m in members]
    n = len(nodes)
    if n <= 3:
        pts = nodes + [station]
        edges = []
        km = hours = wh = 0.0
        for k in range(len(pts) - 1):
            _, ekm, eh, _, _, _, ewh = edge_stats(pts[k], pts[k + 1])
            edges.append({'a': pts[k], 'b': pts[k + 1], 'km': ekm, 'h': eh, 'wh': ewh})
            km += ekm
            hours += eh
            wh += ewh
        _, _, _, ms, emin, emax, _ = edge_stats(pts[0], pts[1]) if len(pts) > 2 else (0, 0, 0, 0, 0, 0, 0)
        return pts, km, hours, ms, emin, emax, edges, nodes[1:]

    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            st = edge_stats(nodes[i], nodes[j])
            D[i][j] = D[j][i] = st[0] if terrain else dist_m(nodes[i], nodes[j])

    def tour_len(order):
        return sum(D[order[k]][order[(k + 1) % n]] for k in range(n))

    order = [0]
    unvisited = set(range(1, n))
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur][j])
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    improved = True
    passes = 0
    while improved and passes < 60:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c, d = order[j], order[(j + 1) % n]
                if d == a:
                    continue
                gain = D[a][b] + D[c][d] - D[a][c] - D[b][d]
                if gain > 1e-6:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
        for L in (1, 2, 3):
            for i in range(1, n - L + 1):
                seg = order[i:i + L]
                rest = order[:i] + order[i + L:]
                base = tour_len(order)
                best_gain, best_pos = 0.0, -1
                for pos in range(1, len(rest)):
                    cand = rest[:pos] + seg + rest[pos:]
                    g = base - tour_len(cand)
                    if g > best_gain + 1e-6:
                        best_gain, best_pos = g, pos
                if best_pos >= 0:
                    order = rest[:best_pos] + seg + rest[best_pos:]
                    improved = True

    pts = [nodes[k] for k in order] + [station]
    edges = []
    km = hours = 0.0
    slopes, elevs = [], []
    for k in range(len(pts) - 1):
        _, ekm, eh, ms, emin, emax, ewh = edge_stats(pts[k], pts[k + 1])
        edges.append({'a': pts[k], 'b': pts[k + 1], 'km': ekm, 'h': eh, 'wh': ewh})
        km += ekm
        hours += eh
        slopes.append(ms)
        elevs.extend([emin, emax])
    order_nodes = [nodes[k] for k in order[1:]]
    return pts, km, hours, sum(slopes) / len(slopes), min(elevs), max(elevs), edges, order_nodes


def segment_route(station, edges):
    """按电池预算把闭合巡回切成出勤段: 能量不足以走下一边+返程时, 就近截断返站换电。

    返回 (sorties:[{'km','h','wh'}], cuts:[段末外点latlon], infeasible_edges)。
    """
    E_MAX = PACK_WH * USABLE
    sorties, cuts, bad = [], [], 0
    cur = {'km': 0.0, 'h': 0.0, 'wh': 0.0}
    for e in edges:
        er_b = edge_stats(e['b'], station)[6]
        fresh = e['wh'] + er_b <= E_MAX * 0.95  # 新满电段能否至少走完这条边
        if not fresh:
            bad += 1
        need = e['wh'] + er_b
        if fresh and cur['wh'] + need > E_MAX and e['a'] != station:
            # 在 a 处截断: 直接返站 (计入本段), 换电后新段从站直奔 a
            _, rkm, rh, _, _, _, rwh = edge_stats(e['a'], station)
            cur['km'] += rkm
            cur['h'] += rh
            cur['wh'] += rwh
            sorties.append(dict(cur))
            cuts.append(e['a'])
            cur = {'km': rkm, 'h': rh, 'wh': rwh}  # 新段出站直奔 a (与返站同程)
        cur['km'] += e['km']
        cur['h'] += e['h']
        cur['wh'] += e['wh']
    sorties.append(dict(cur))
    return sorties, cuts, bad


def build_zone(farm, zi, members, color, compare=False):
    w = sum(m['area_mu'] for m in members)
    st_lat = sum(m['cen'][0] * m['area_mu'] for m in members) / w
    st_lon = sum(m['cen'][1] * m['area_mu'] for m in members) / w
    station = [round(st_lat, 6), round(st_lon, 6)]

    route_pts, km, hours, ms, emin, emax, edges, order_nodes = optimize_route(station, members, terrain=True)
    flat_h = km * 1000.0 / FIELD_SPEED / 3600.0

    # 电池预算切段 (换电次数 = 段数-1)
    sorties, cuts, bad = segment_route(station, edges)
    n_seg = len(sorties)
    seg_km = [s['km'] for s in sorties]
    seg_h = [s['h'] for s in sorties]
    tot_wh = sum(s['wh'] for s in sorties)
    swaps = n_seg - 1
    cal_h = sum(s['h'] for s in sorties) + swaps * SWAP_MIN / 60.0  # 行走(含往返开销)+换电

    # 平地代价最优序 vs 地形代价最优序 → 路线是否真的变了
    chg, save = 0, 0.0
    if compare and len(members) >= 4:
        _, _, _, _, _, _, _, flat_order = optimize_route(station, members, terrain=False)

        def tcost(seq):
            tot = edge_stats(station, seq[0])[0] + edge_stats(seq[-1], station)[0]
            for i in range(len(seq) - 1):
                tot += edge_stats(seq[i], seq[i + 1])[0]
            return tot
        if flat_order != order_nodes:
            chg = 1
            f = tcost(flat_order)
            t = tcost(order_nodes)
            save = 100.0 * (f - t) / max(1e-9, f)

    zname = '{}-{}'.format(farm, chr(65 + zi))
    cycle_days = max(1, math.ceil(n_seg / SORTIES_PER_DAY))
    z = {
        'id': zname, 'farm': farm, 'color': color,
        'station': station, 'n': len(members), 'mu': round(w, 1),
        'km': round(km, 1), 'hours': round(hours, 1), 'flat_h': round(flat_h, 1),
        'sorties': n_seg, 'swaps': swaps, 'wh': round(tot_wh), 'cal': round(cal_h, 1),
        'cycle': cycle_days,
        'seg_km': [round(x, 1) for x in seg_km], 'seg_h': [round(x, 1) for x in seg_h],
        'bad': bad, 'chg': chg, 'save': round(save, 1),
        'slope': round(ms, 1),
        'elev': '{}/{}'.format(int(emin), int(emax)),
        'route': [[round(p[0], 6), round(p[1], 6)] for p in route_pts],
        'cuts': [[round(p[0], 6), round(p[1], 6)] for p in cuts],
        'members': [m['name'] for m in members],
        'member_slope': round(sum(m['slope'] for m in members) / len(members), 1),
    }
    for m in members:
        m['zone'] = zname
    return z


PALETTE = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#1abc9c',
           '#3498db', '#9b59b6', '#e84393', '#00b894', '#fd79a8',
           '#6c5ce7', '#0984e3', '#d63031', '#00cec9', '#fdcb6e',
           '#a29bfe', '#55efc4', '#fab1a0', '#74b9ff', '#ff7675',
           '#81ecec', '#ffeaa7', '#00b0ff', '#ff9100', '#7cb342']

zones = []
for farm, k in TARGET_ZONES.items():
    fl = [c for c in comps if c['farm'] == farm]
    if not fl:
        continue
    assign = cluster(fl, k)
    groups = [[fl[i] for i in range(len(fl)) if assign[i] == zi] for zi in range(k)]
    groups = [g for g in groups if g]

    # 拆分判据: 几何 > 20km 或 电池切段数 > 6
    changed = True
    while changed:
        changed = False
        for gi, g in enumerate(groups):
            if len(g) < MIN_SPLIT_N:
                continue
            saved = dict(ROUTE_STATS)
            z_try = build_zone(farm, 0, g, '#fff')
            ROUTE_STATS.clear()
            ROUTE_STATS.update(saved)
            if z_try['km'] > MAX_ROUTE_KM or z_try['sorties'] > MAX_SORTIES:
                sub = cluster(g, 2)
                g0 = [g[i] for i in range(len(g)) if sub[i] == 0]
                g1 = [g[i] for i in range(len(g)) if sub[i] == 1]
                if g0 and g1:
                    groups[gi:gi + 1] = [g0, g1]
                    changed = True
                    break

    # 合并碎片区 (<3 班) 到同场地理最近区
    changed = True
    while changed:
        changed = False
        for gi, g in enumerate(groups):
            if len(g) >= 3:
                continue
            gc = [sum(m['cen'][0] for m in g) / len(g), sum(m['cen'][1] for m in g) / len(g)]
            best, bd = None, 1e18
            for gj, h in enumerate(groups):
                if gj == gi or len(h) < 3:
                    continue
                hc = [sum(m['cen'][0] for m in h) / len(h), sum(m['cen'][1] for m in h) / len(h)]
                d = dist_m(gc, hc)
                if d < bd:
                    bd, best = d, gj
            if best is not None:
                groups[best] = groups[best] + g
                groups.pop(gi)
                changed = True
                break

    for zi, g in enumerate(groups):
        zones.append(build_zone(farm, zi, g, PALETTE[len(zones) % len(PALETTE)], compare=True))

total_dogs = len(zones)
total_mu = sum(z['mu'] for z in zones)
tot_h = sum(z['hours'] for z in zones)
tot_flat = sum(z['flat_h'] for z in zones)
print('责任区/狗: {} 只, 总面积 {:.1f} 万亩'.format(total_dogs, total_mu / 10000))
print('地形影响: 平地模型 {:.0f} h → 地形变速 {:.0f} h (+{:.0f}%)'.format(
    tot_flat, tot_h, 100 * (tot_h / max(0.1, tot_flat) - 1)))
print('最长巡护周期: {} 天'.format(max(z['sorties'] for z in zones)))
for z in sorted(zones, key=lambda z: z['id']):
    print('  {}: {} 班, {} 亩, 均坡 {}°, 路线 {} km, 地形耗时 {} h (平地 {} h), {} 次出勤, 海拔 {} m'.format(
        z['id'], z['n'], z['mu'], z['member_slope'], z['km'], z['hours'], z['flat_h'],
        z['sorties'], z['elev']))

# ---------- 生成 HTML ----------
comp_rows = []
for c in comps:
    comp_rows.append(
        '{{f:"{f}",z:"{z}",n:"{n}",nm:"{nm}",a:{a},sl:{sl},el:{el},c:[{cx},{cy}],pts:[{pts}]}}'.format(
            f=c['farm'], z=c.get('zone', ''), n=c['num'], nm=c['name'],
            a=round(c['area_mu'], 1), sl=c['slope'], el=c['elev'],
            cx=c['cen'][0], cy=c['cen'][1],
            pts=','.join('[{},{}]'.format(p[0], p[1]) for p in c['pts'])))

zone_rows = []
for z in zones:
    zone_rows.append(
        '{{id:"{id}",f:"{f}",col:"{col}",st:[{sa},{so}],n:{n},mu:{mu},km:{km},h:{h},fh:{fh},so:{so2},'
        'sw:{sw},wh:{wh},cal:{cal},segk:[{segk}],segh:[{segh}],'
        'sl:{sl},el:"{el}",route:[{rt}],cuts:[{cuts}],mem:"{mem}"}}'.format(
            id=z['id'], f=z['farm'], col=z['color'],
            sa=z['station'][0], so=z['station'][1],
            n=z['n'], mu=z['mu'], km=z['km'], h=z['hours'], fh=z['flat_h'], so2=z['sorties'],
            sw=z['swaps'], wh=z['wh'], cal=z['cal'], cy=z['cycle'],
            segk=','.join(str(x) for x in z['seg_km'][:8]),
            segh=','.join(str(x) for x in z['seg_h'][:8]),
            sl=z['slope'], el=z['elev'],
            rt=','.join('[{},{}]'.format(p[0], p[1]) for p in z['route']),
            cuts=','.join('[{},{}]'.format(p[0], p[1]) for p in z['cuts']),
            mem='、'.join(z['members'][:10]) + ('…' if len(z['members']) > 10 else '')))

farm_rows = []
for farm in sorted(TARGET_ZONES):
    fz = [z for z in zones if z['farm'] == farm]
    if not fz:
        continue
    fl = [c for c in comps if c['farm'] == farm]
    farm_rows.append('"{}":{{n:{},mu:{:.0f},sl:{:.1f},sw:{}}}'.format(
        farm, len(fz), sum(z['mu'] for z in fz),
        sum(c['slope'] for c in fl) / len(fl), sum(z['swaps'] for z in fz)))

allc = [c['cen'] for c in comps]
mid = [sum(p[0] for p in allc) / len(allc), sum(p[1] for p in allc) / len(allc)]

html = '''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>雷波林业局 · 机器狗部署方案 v2 地形版 (__NDOG__只 · 271林班 · 真实DEM)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
body{margin:0;font-family:system-ui,"Microsoft YaHei",sans-serif}
#map{width:100vw;height:100vh}
.panel{position:absolute;top:10px;left:10px;background:rgba(13,17,23,.93);color:#e6edf3;padding:14px;border-radius:10px;z-index:1000;width:300px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.5);max-height:calc(100vh - 40px);overflow-y:auto}
.panel h3{margin:0 0 6px;font-size:14px}
.panel .info{color:#8b949e;font-size:11px;line-height:1.6;margin-bottom:8px}
table{border-collapse:collapse;width:100%;margin:6px 0}
th,td{padding:3px 4px;text-align:right;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:normal}
td:first-child,th:first-child{text-align:left}
.farm-row{display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:6px;cursor:pointer;margin:2px 0}
.farm-row:hover{background:#21262d}
.farm-row .cnt{margin-left:auto;color:#8b949e}
.btn{display:block;width:100%;margin-top:8px;padding:7px;border:0;border-radius:6px;background:#1f6feb;color:#fff;font-size:12px;cursor:pointer}
.btn:hover{background:#388bfd}
.phase{margin-top:10px;border-top:1px solid #30363d;padding-top:8px;line-height:1.7;color:#c9d1d9}
.phase b{color:#58a6ff}
.speed{margin-top:8px;color:#8b949e;font-size:11px}
input[type=range]{width:100%}
.ramp{display:flex;height:10px;border-radius:4px;margin:4px 0;font-size:9px;color:#000}
.leaflet-popup-content-wrapper{background:#21262d;color:#e6edf3;border-radius:8px;font-size:12px}
.leaflet-popup-tip{background:#21262d}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
<h3>机器狗部署方案 v2 · 真实地形版</h3>
<div class="info">__NDOG__ 只狗 / __NZONE__ 个责任区 / 271 林班 / 3.5 万亩<br>
DEM: Mapzen terrarium ~30m (AWS Open Data)<br>
高程 __EMIN__-__EMAX__ m · 区域均坡 __MSLP__°<br>
边代价 = d·(1+(坡/22°)²) · 速度 = 0.6·e^(−坡/24°) m/s</div>
<table><tr><th>林场</th><th>区</th><th>万亩</th><th>均坡°</th><th>地形h</th><th>换电</th></tr>
<tbody id="farmTbl"></tbody></table>
<div class="farm-row" id="zoneBtn"><span>📦 责任区填色</span><span class="cnt">on</span></div>
<div class="farm-row" id="slopeBtn"><span>⛰ 坡度填色</span><span class="cnt">off</span></div>
<div class="farm-row" id="hillBtn"><span>🌑 山体阴影</span><span class="cnt">on</span></div>
<div class="farm-row" id="routeBtn"><span>🛣 巡护路线</span><span class="cnt">on</span></div>
<div class="farm-row" id="cutBtn"><span>🔋 换电折返点</span><span class="cnt">on</span></div>
<div class="ramp"><span style="flex:1;background:#1a9850">0°</span><span style="flex:1;background:#a6d96a"></span><span style="flex:1;background:#ffffbf"></span><span style="flex:1;background:#fdae61"></span><span style="flex:1;background:#d73027">35°+</span></div>
<button class="btn" id="animBtn">▶ 演示巡护 (动画)</button>
<div class="speed">演示速率 <input type="range" id="spd" min="50" max="800" value="300"> <span id="spdv">×300</span></div>
<div class="phase">
<b>地形建模</b><br>
· 逐林班: 质心+边界采样 → 均坡/海拔<br>
· 逐边: 每250m采样坡度 → 翻山边自动加价<br>
· 匀速 0.6 m/s → 变速模型, 耗时 +__TPCT__%<br>
<b>电池模型 (__NCHG__/__NZ__ 区因地形改道)</b><br>
· 双包1500Wh×80% 可用 · 步态 180W<br>
· 爬坡 50kg·g·Δh/η75% · 下坡回收30%<br>
· 断点=🔋: 剩余能量不足以走下一边+返程<br>
· 换电 15min/次 · 2 段/日 (上午/下午)<br>
<b>林相 (待接入)</b><br>
· 接口已留: 小班郁闭度/灌木密度/树种/龄组<br>
· 数据源: 森林二类调查小班属性表<br>
· 已有真实信号: 石斛/黄莲地块=密荫林下<br>
· stand_params.json 提供速度×林相系数模板
</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
'use strict';
var COMPS=[__COMPS__];
var ZONES=[__ZONES__];
var FARMS={__FARMS__};
var FSTYLE={'211':'#e74c3c','212':'#3498db','213':'#2ecc71','214':'#9b59b6'};
var FLABEL={'211':'211林场','212':'212林场','213':'213林场','214':'214林场'};

var map=L.map('map').setView([__MIDLAT__,__MIDLON__],12);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{
 maxZoom:19,attribution:'Esri, Maxar'}).addTo(map);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',{
 maxZoom:19,opacity:.5}).addTo(map);

var hillshade=L.imageOverlay('hillshade.png',__HILLBOUND__,{opacity:.55});

var zoneColor={};ZONES.forEach(function(z){zoneColor[z.id]=z.col;});
var allBounds=L.latLngBounds();

function slopeColor(s){
 if(s<8)return '#1a9850';if(s<15)return '#a6d96a';if(s<22)return '#ffffbf';
 if(s<29)return '#fdae61';return '#d73027';
}
var slopeMode=false;
var polys=[];
COMPS.forEach(function(c){
  var zc=zoneColor[c.z]||'#666';
  var poly=L.polygon(c.pts,{
    color:FSTYLE[c.f]||'#888',weight:1,opacity:.75,
    fillColor:zc,fillOpacity:.30
  });
  poly.bindPopup('<b>'+c.nm+'</b><br>'+c.f+'场 · 责任区 '+c.z+'<br>面积 '+c.a.toFixed(1)+' 亩<br>'+
    '均坡 '+c.sl.toFixed(1)+'° · 海拔 '+c.el+' m');
  polys.push(poly);
  allBounds.extend(c.pts);
});
var compLayer=L.layerGroup(polys).addTo(map);
hillshade.addTo(map);

var routeLayer=L.layerGroup();
ZONES.forEach(function(z){
  L.polyline(z.route,{color:z.col,weight:2.2,opacity:.85,dashArray:'8 5'}).addTo(routeLayer);
});
routeLayer.addTo(map);

// 🔋 换电折返点 (电池耗尽前就近折返站点的位置)
var cutLayer=L.layerGroup();
ZONES.forEach(function(z){
  z.cuts.forEach(function(c){
    L.marker(c,{icon:L.divIcon({className:'',iconSize:[18,18],iconAnchor:[9,9],
      html:'<div style="font-size:13px;filter:drop-shadow(0 0 2px #000)">🔋</div>'}),interactive:false})
      .addTo(cutLayer);
  });
});
cutLayer.addTo(map);

var stLayer=L.layerGroup();
ZONES.forEach(function(z){
  var mk=L.marker(z.st,{icon:L.divIcon({className:'',iconSize:[54,20],iconAnchor:[27,10],
    html:'<div style="white-space:nowrap;text-shadow:0 0 3px #000,0 0 3px #000"><span>🏠</span><b style="color:#fff;font-size:11px"> '+z.id+'</b></div>'})});
  mk.bindPopup('<b>'+z.id+' 责任区 · 补给站</b><br>'+
    '林班 '+z.n+' 个 · 面积 '+(z.mu/10000).toFixed(2)+' 万亩<br>'+
    '均坡 '+z.sl.toFixed(1)+'° · 海拔 '+z.el+' m<br>'+
    '路线 '+z.km+' km / 遍 · 地形耗时 '+z.h+' h<br>'+
    '🔋 出勤 '+z.so+' 段 ('+z.segk.slice(0,4).join('/')+' km)<br>'+
    '换电 '+z.sw+' 次×15min · 能耗 '+(z.wh/1000).toFixed(1)+' kWh/循环<br>'+
    '日历周期 '+z.cal+' h → 巡护周期 '+z.cy+' 天 (2 段/日)<br>'+
    '<span style="color:#8b949e">覆盖: '+z.mem+'</span>');
  mk.addTo(stLayer);
});
stLayer.addTo(map);
map.fitBounds(allBounds,{padding:[15,15]});
L.control.scale({imperial:false,position:'bottomright'}).addTo(map);

var tbl=document.getElementById('farmTbl');
Object.keys(FARMS).sort().forEach(function(f){
  var d=FARMS[f];
  var fz=ZONES.filter(function(z){return z.f===f;});
  var th=fz.reduce(function(a,z){return a+z.h;},0);
  var tr=document.createElement('tr');
  tr.innerHTML='<td><span style="color:'+FSTYLE[f]+'">●</span> '+f+'</td><td>'+d.n+'</td>'+
    '<td>'+(d.mu/10000).toFixed(1)+'</td><td>'+d.sl.toFixed(1)+'</td><td>'+Math.round(th)+'</td><td>'+d.sw+'</td>';
  tr.style.cursor='pointer';
  tr.onclick=function(){
    var b=L.latLngBounds();
    ZONES.forEach(function(z){if(z.f===f)b.extend(z.route);});
    if(b.isValid())map.fitBounds(b,{padding:[25,25]});
  };
  tbl.appendChild(tr);
});

function toggleBtn(id,layer){
  var el=document.getElementById(id);
  el.onclick=function(){
    if(map.hasLayer(layer)){map.removeLayer(layer);el.querySelector('.cnt').textContent='off';el.style.opacity=.5;}
    else{map.addLayer(layer);el.querySelector('.cnt').textContent='on';el.style.opacity=1;}
  };
}
toggleBtn('hillBtn',hillshade);
toggleBtn('routeBtn',routeLayer);
toggleBtn('cutBtn',cutLayer);

document.getElementById('slopeBtn').onclick=function(){
  slopeMode=!slopeMode;
  this.querySelector('.cnt').textContent=slopeMode?'on':'off';
  this.style.opacity=slopeMode?1:.5;
  polys.forEach(function(p,i){
    var c=COMPS[i];
    p.setStyle(slopeMode?{fillColor:slopeColor(c.sl),fillOpacity:.45}
                        :{fillColor:zoneColor[c.z]||'#666',fillOpacity:.30});
  });
};
document.getElementById('zoneBtn').onclick=function(){
  var el=this;var on=el.querySelector('.cnt').textContent==='on';
  if(on){map.removeLayer(compLayer);el.querySelector('.cnt').textContent='off';el.style.opacity=.5;}
  else{map.addLayer(compLayer);el.querySelector('.cnt').textContent='on';el.style.opacity=1;}
};

var dogs=[];var running=false;var spdX=300;var rafId=null;
ZONES.forEach(function(z,i){
  var acc=[0];var tot=0;
  for(var k=1;k<z.route.length;k++){
    var a=z.route[k-1],b=z.route[k];
    tot+=map.distance(a,b);acc.push(tot);
  }
  var mk=L.marker(z.route[0],{icon:L.divIcon({className:'',iconSize:[20,20],iconAnchor:[10,10],
    html:'<div style="font-size:16px;filter:drop-shadow(0 0 2px #000)">🐕</div>'}),interactive:false});
  dogs.push({z:z,mk:mk,acc:acc,tot:tot,phase:(i*0.13)%1});
});
var animBtn=document.getElementById('animBtn');
animBtn.onclick=function(){
  running=!running;
  if(running){
    dogs.forEach(function(d){d.mk.addTo(map);});
    animBtn.textContent='⏸ 停止演示';
    var t0=null;
    function step(ts){
      if(!running)return;
      if(t0===null)t0=ts;
      var el=(ts-t0)/1000;
      dogs.forEach(function(d){
        var prog=((el*0.6*spdX/d.tot)+d.phase)%1;
        var distNow=prog*d.tot;
        var k=1;while(k<d.acc.length-1&&d.acc[k]<distNow)k++;
        var a=d.z.route[k-1],b=d.z.route[k];
        var f=(distNow-d.acc[k-1])/Math.max(1e-6,d.acc[k]-d.acc[k-1]);
        d.mk.setLatLng([a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f]);
      });
      rafId=requestAnimationFrame(step);
    }
    rafId=requestAnimationFrame(step);
  }else{
    cancelAnimationFrame(rafId);
    dogs.forEach(function(d){map.removeLayer(d.mk);});
    animBtn.textContent='▶ 演示巡护 (动画)';
  }
};
var spd=document.getElementById('spd');
spd.oninput=function(){spdX=+spd.value;document.getElementById('spdv').textContent='×'+spdX;};
</script>
</body>
</html>
'''

flat_total = sum(z['flat_h'] for z in zones)
terr_total = sum(z['hours'] for z in zones)
tpct = int(100 * (terr_total / max(0.1, flat_total) - 1))
n_chg = sum(z['chg'] for z in zones)
saves = [z['save'] for z in zones if z['chg']]
print('路线对比: {}/{} 责任区访问顺序因地形代价改变, 平均地形代价节省 {:.1f}%'.format(
    n_chg, len(zones), sum(saves) / len(saves) if saves else 0.0))
tot_sw = sum(z['swaps'] for z in zones)
tot_wh = sum(z['wh'] for z in zones)
tot_bad = sum(z['bad'] for z in zones)
print('电池模型: 双包{}Wh×{:.0f}% 可用={:.0f}Wh, 全局换电 {} 次/循环, 总能耗 {:.1f} kWh/循环, 超程边 {}'.format(
    int(PACK_WH), USABLE * 100, PACK_WH * USABLE, tot_sw, tot_wh / 1000.0, tot_bad or '无'))
print('巡护周期: 最长 {} 天 (2 段/日)'.format(max(z['cycle'] for z in zones)))
for z in sorted(zones, key=lambda z: z['id'])[:6]:
    print('  {}: 段 {} ({}), 换电 {} 次, 周期 {} 天, 日历 {} h'.format(
        z['id'], z['sorties'], '/'.join('{:.0f}min'.format(h * 60) for h in z['seg_h'][:4]),
        z['swaps'], z['cycle'], z['cal']))
all_slope = [c['slope'] for c in comps]
hb = '[[{:.6f},{:.6f}],[{:.6f},{:.6f}]]'.format(
    HILL_BOUNDS[0][0], HILL_BOUNDS[0][1], HILL_BOUNDS[1][0], HILL_BOUNDS[1][1])

html = (html
        .replace('__COMPS__', ',\n'.join(comp_rows))
        .replace('__ZONES__', ',\n'.join(zone_rows))
        .replace('__FARMS__', ','.join(farm_rows))
        .replace('__NDOG__', str(total_dogs))
        .replace('__NZONE__', str(len(zones)))
        .replace('__MIDLAT__', '{:.5f}'.format(mid[0]))
        .replace('__MIDLON__', '{:.5f}'.format(mid[1]))
        .replace('__EMIN__', str(int(dem.min())))
        .replace('__EMAX__', str(int(dem.max())))
        .replace('__MSLP__', '{:.1f}'.format(sum(all_slope) / len(all_slope)))
        .replace('__TPCT__', str(tpct))
        .replace('__NCHG__', str(n_chg))
        .replace('__NZ__', str(len(zones)))
        .replace('__HILLBOUND__', hb))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print('已生成 {} ({:.2f} MB) + hillshade.png'.format(OUT, len(html.encode('utf-8')) / 1048576))
