# -*- coding: utf-8 -*-
"""统计 forest_compartments.geojson 全部林场覆盖与坐标范围"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

p = r'E:\forestrobo\林场范围哈\forest_compartments.geojson'
with open(p, encoding='utf-8') as f:
    gj = json.load(f)

feats = gj.get('features', [])
print('总要素数: {}'.format(len(feats)))

# 按 场 → 作业区 → 林班 分组
by_farm = {}
lats, lons = [], []
prop_keys = set()
for ft in feats:
    props = ft.get('properties', {}) or {}
    prop_keys.update(props.keys())
    farm = str(props.get('场', '?'))
    zone = str(props.get('作业区', '?'))
    comp = str(props.get('林班', '?'))
    by_farm.setdefault(farm, {}).setdefault(zone, []).append(comp)

    geom = ft.get('geometry', {}) or {}
    c = geom.get('coordinates')
    rings = []
    if geom.get('type') == 'Polygon':
        rings = [c] if c else []
    elif geom.get('type') == 'MultiPolygon':
        rings = c or []
    for poly in rings:
        for ring in poly:  # ring: [[lon,lat],...]
            for pt in ring:
                if len(pt) >= 2:
                    lons.append(pt[0]); lats.append(pt[1])

print('\n属性字段: {}'.format(sorted(prop_keys)))
print('\n=== 林场结构 ===')
for farm in sorted(by_farm, key=lambda x: (len(x), x)):
    zones = by_farm[farm]
    total = sum(len(v) for v in zones.values())
    zstr = ', '.join('作业区{}({}林班)'.format(z, len(v)) for z, v in sorted(zones.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 99))
    print('场 {}: {}个林班  [{}]'.format(farm, total, zstr))

print('\n=== 坐标范围 ===')
if lats:
    print('经度: {:.6f} ~ {:.6f}'.format(min(lons), max(lons)))
    print('纬度: {:.6f} ~ {:.6f}'.format(min(lats), max(lats)))
    # 每个场的范围
    print('\n各场坐标范围:')
    for farm in sorted(by_farm, key=lambda x: (len(x), x)):
        f_lats, f_lons = [], []
        for ft in feats:
            props = ft.get('properties', {}) or {}
            if str(props.get('场')) != farm:
                continue
            geom = ft.get('geometry', {}) or {}
            c = geom.get('coordinates')
            rings = [c] if geom.get('type') == 'Polygon' and c else (c or [] if geom.get('type') == 'MultiPolygon' else [])
            for poly in rings:
                for ring in poly:
                    for pt in ring:
                        if len(pt) >= 2:
                            f_lons.append(pt[0]); f_lats.append(pt[1])
        if f_lats:
            dlon = (max(f_lons) - min(f_lons)) * 111.32 * 0.9
            dlat = (max(f_lats) - min(f_lats)) * 110.57
            print('场 {}: {:.5f}~{:.5f}E, {:.5f}~{:.5f}N  约 {:.1f}km x {:.1f}km, 中心 ({:.5f}, {:.5f})'.format(
                farm, min(f_lons), max(f_lons), min(f_lats), max(f_lats), dlon, dlat,
                (min(f_lons) + max(f_lons)) / 2, (min(f_lats) + max(f_lats)) / 2))

# 有没有面积字段? 检查第一个要素的完整属性
print('\n第一个要素完整属性:')
print(json.dumps(feats[0].get('properties', {}), ensure_ascii=False, indent=1))
