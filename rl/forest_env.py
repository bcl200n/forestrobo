# -*- coding: utf-8 -*-
"""林场 KMZ 解析 (适配 ArcGIS 导出格式) → 多狗搜索环境"""
import json
import math
import os
import sys
import zipfile
import xml.etree.ElementTree as ET


def parse_kmz(kmz_path):
    """解析 KMZ, 兼容 ArcGIS 导出格式"""
    with zipfile.ZipFile(kmz_path, 'r') as z:
        kmls = [n for n in z.namelist()
                if n.lower().endswith('.kml')]
        if not kmls:
            raise ValueError('KMZ 中没有 KML')
        content = z.read(kmls[0])

    # 不用 namespace (ArcGIS KML 常省略)
    root = ET.fromstring(content)

    features = {'boundaries': [], 'regions': [], 'paths': [], 'points': []}

    def get_coords(element):
        """从任何元素的 coordinates 子代提取坐标"""
        for coord in element.iter():
            if coord.tag.endswith('coordinates') and coord.text:
                pts = []
                for line in coord.text.strip().split():
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            pts.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            pass
                return pts
        return []

    def get_name(element):
        for el in element.iter():
            if el.tag.endswith('name') and el.text:
                return el.text.strip()
        return 'unnamed'

    for pm in root.iter():
        if not pm.tag.endswith('Placemark'):
            continue
        name = get_name(pm)

        for poly in pm.iter():
            if poly.tag.endswith('Polygon'):
                coords = get_coords(poly)
                if len(coords) >= 3:
                    # 判断内外: 面积
                    area = abs(sum(
                        coords[i][0] * coords[i+1][1] -
                        coords[i+1][0] * coords[i][1]
                        for i in range(len(coords)-1))) * 111000 * 111000 * 0.5
                    if area > 5e6:  # >5km² → 边界
                        features['boundaries'].append(
                            {'name': name, 'coords': coords, 'area_m2': area})
                    else:
                        features['regions'].append(
                            {'name': name, 'coords': coords, 'area_m2': area})
                break

        for ls in pm.iter():
            if ls.tag.endswith('LineString'):
                coords = get_coords(ls)
                if len(coords) >= 2:
                    length = sum(math.hypot(
                        coords[i+1][0]-coords[i][0],
                        coords[i+1][1]-coords[i][1]) * 111000
                        for i in range(len(coords)-1))
                    features['paths'].append(
                        {'name': name, 'coords': coords,
                         'length_m': length})
                break

        for pt in pm.iter():
            if pt.tag.endswith('Point'):
                coords = get_coords(pt)
                if coords:
                    features['points'].append(
                        {'name': name, 'lon': coords[0][0],
                         'lat': coords[0][1]})
                break

    return features


def point_in_polygon(lon, lat, polygon):
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > lat) != (polygon[j][1] > lat) and
                lon < (polygon[j][0] - polygon[i][0]) *
                (lat - polygon[i][1]) /
                (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
        j = i
    return inside


def to_search_env(features, grid_size=40, n_dogs=10):
    """转换为搜索环境"""
    # 收集所有坐标确定范围
    all_coords = []
    for b in features['boundaries']:
        all_coords.extend(b['coords'])
    for r in features['regions']:
        all_coords.extend(r['coords'])
    if not all_coords:
        all_coords = [[0, 0], [1, 1]]

    min_lon = min(c[0] for c in all_coords)
    max_lon = max(c[0] for c in all_coords)
    min_lat = min(c[1] for c in all_coords)
    max_lat = max(c[1] for c in all_coords)
    lon_span = max(max_lon - min_lon, 0.0001)
    lat_span = max(max_lat - min_lat, 0.0001)

    # 网格: 默认不可通行
    grid = [[True] * grid_size for _ in range(grid_size)]

    # 边界内 = 可通行
    boundary = features['boundaries'][0]['coords'] if features['boundaries'] \
        else all_coords
    for gy in range(grid_size):
        for gx in range(grid_size):
            lon = min_lon + (gx + 0.5) / grid_size * lon_span
            lat = min_lat + (gy + 0.5) / grid_size * lat_span
            if point_in_polygon(lon, lat, boundary):
                grid[gy][gx] = False

    # 地块也标为可通行 (如果边界为空)
    if not features['boundaries']:
        for region in features['regions']:
            for gy in range(grid_size):
                for gx in range(grid_size):
                    lon = min_lon + (gx + 0.5) / grid_size * lon_span
                    lat = min_lat + (gy + 0.5) / grid_size * lat_span
                    if point_in_polygon(lon, lat, region['coords']):
                        grid[gy][gx] = False

    # 在可通行区域内随机放置目标 (幸存者)
    free = [(x, y) for y in range(grid_size) for x in range(grid_size)
             if not grid[y][x]]
    rng = __import__('random').Random(42)
    n_targets = min(4, max(1, len(free) // 50))
    targets = rng.sample(free, n_targets) if len(free) > n_targets else free[:n_targets]

    return {
        'grid': grid,
        'size': grid_size,
        'bounds': {'min_lon': min_lon, 'max_lon': max_lon,
                   'min_lat': min_lat, 'max_lat': max_lat},
        'targets': [list(t) for t in targets],
        'n_dogs': n_dogs,
        'features': features,
    }


if __name__ == '__main__':
    base = r'E:\forestrobo\林场范围哈'
    for fn in sorted(os.listdir(base)):
        if not fn.endswith('.kmz'):
            continue
        path = os.path.join(base, fn)
        print('=' * 50)
        print('文件: ' + fn)
        try:
            features = parse_kmz(path)
            print('边界: {} 个'.format(len(features['boundaries'])))
            for b in features['boundaries']:
                print('  "{}" 点数={}'.format(
                    b['name'], len(b['coords'])))
            print('地块: {} 个'.format(len(features['regions'])))
            for r in features['regions'][:8]:
                print('  "{}" 点数={}'.format(
                    r['name'], len(r['coords'])))
            if len(features['regions']) > 8:
                print('  ... 还有 {} 个'.format(
                    len(features['regions']) - 8))
            print('路径: {} 条'.format(len(features['paths'])))
            print('标注: {} 个'.format(len(features['points'])))

            env = to_search_env(features)
            open_cnt = sum(1 for row in env['grid']
                           for c in row if not c)
            print('搜索环境: {}x{} 网格, 可通行 {} 格, {} 个目标'.format(
                env['size'], env['size'], open_cnt,
                len(env['targets'])))
            b = env['bounds']
            print('经度: {:.5f} ~ {:.5f}'.format(
                b['min_lon'], b['max_lon']))
            print('纬度: {:.5f} ~ {:.5f}'.format(
                b['min_lat'], b['max_lat']))

            out = fn.replace('.kmz', '_env.json')
            with open(out, 'w', encoding='utf-8') as f:
                json.dump({
                    'grid': env['grid'],
                    'bounds': env['bounds'],
                    'targets': env['targets'],
                    'region_names': [r['name'] for r in
                                     features['regions'][:30]],
                    'region_coords': [r['coords'] for r in
                                      features['regions'][:30]],
                }, f, ensure_ascii=False, indent=2)
            print('已输出: ' + out)
        except Exception as e:
            print('失败: ' + str(e))
        print()
    print('PARSE_ALL_DONE')
