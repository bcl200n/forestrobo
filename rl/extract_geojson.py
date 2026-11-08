# -*- coding: utf-8 -*-
"""从 KMZ 提取所有地块的经纬度坐标 → 生成 WebGIS GeoJSON"""
import json
import os
import sys
import zipfile
import xml.etree.ElementTree as ET


def parse_all_kmz(directory):
    """解析目录下所有 KMZ, 提取每个 Placemark 的坐标"""
    all_features = []

    for fn in sorted(os.listdir(directory)):
        if not fn.endswith('.kmz'):
            continue
        path = os.path.join(directory, fn)
        category = fn.replace('.kmz', '')

        with zipfile.ZipFile(path, 'r') as z:
            kmls = [n for n in z.namelist() if n.lower().endswith('.kml')]
            if not kmls:
                continue
            content = z.read(kmls[0])

        root = ET.fromstring(content)

        for pm in root.iter():
            if not pm.tag.endswith('Placemark'):
                continue

            # 获取名称
            name = category
            for el in pm.iter():
                if el.tag.endswith('name') and el.text and el.text.strip():
                    name = el.text.strip()
                    break

            # 获取坐标
            coords = []
            geom_type = None

            for el in pm.iter():
                if el.tag.endswith('coordinates') and el.text:
                    for line in el.text.strip().split():
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            try:
                                lon, lat = float(parts[0]), float(parts[1])
                                coords.append([lon, lat])
                            except ValueError:
                                pass
                    break

            # 判断几何类型
            for el in pm.iter():
                if el.tag.endswith('Polygon'):
                    geom_type = 'Polygon'
                    break
                elif el.tag.endswith('LineString'):
                    geom_type = 'LineString'
                    break
                elif el.tag.endswith('Point'):
                    geom_type = 'Point'
                    break

            if coords and len(coords) >= 3 and geom_type == 'Polygon':
                all_features.append({
                    'type': 'Feature',
                    'properties': {
                        'name': name,
                        'category': category,
                        'points': len(coords),
                    },
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [coords],
                    },
                })
            elif coords and len(coords) >= 2 and geom_type == 'LineString':
                all_features.append({
                    'type': 'Feature',
                    'properties': {
                        'name': name,
                        'category': category,
                        'points': len(coords),
                    },
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': coords,
                    },
                })

    return all_features


if __name__ == '__main__':
    directory = r'E:\forestrobo\林场范围哈'
    features = parse_all_kmz(directory)

    print('提取了 {} 个要素'.format(len(features)))
    for f in features:
        props = f['properties']
        coords = f['geometry']['coordinates']
        if f['geometry']['type'] == 'Polygon':
            ring = coords[0]
        else:
            ring = coords
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        print('  "{}" ({}): {} 点, 经度 {:.5f}~{:.5f}, 纬度 {:.5f}~{:.5f}'.format(
            props['name'], props['category'], props['points'],
            min(lons), max(lons), min(lats), max(lats)))

    # 输出 GeoJSON
    geojson = {
        'type': 'FeatureCollection',
        'features': features,
    }
    out = os.path.join(directory, 'forest_features.geojson')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print('\n已输出: {}'.format(out))
    print('EXTRACT_DONE')
