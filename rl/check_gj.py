# -*- coding: utf-8 -*-
"""检查 forest_compartments.geojson 的内容结构"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

p = r'E:\forestrobo\林场范围哈\forest_compartments.geojson'
with open(p, encoding='utf-8') as f:
    gj = json.load(f)

feats = gj.get('features', [])
print('type={} features={}'.format(gj.get('type'), len(feats)))

for i, ft in enumerate(feats[:50]):
    props = ft.get('properties', {}) or {}
    geom = ft.get('geometry', {}) or {}
    gtype = geom.get('type')
    name = props.get('name') or props.get('NAME') or props.get('title')
    if name is None:
        # 展示第一个属性值
        vals = list(props.values())[:3]
        name = ' | '.join(str(v) for v in vals if v)
    coords = geom.get('coordinates')
    npts = '?'
    if gtype == 'Polygon' and coords:
        npts = len(coords[0])
    elif gtype == 'MultiPolygon' and coords:
        npts = sum(len(p0[0]) for p0 in coords)
    print('[{:3d}] {} gtype={} pts={} props={}'.format(
        i, name, gtype, npts, dict(list(props.items())[:6])))
