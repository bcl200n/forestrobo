# -*- coding: utf-8 -*-
"""用真实地块边界和地理配准高程做区域尺度的坡度路线分析。

默认读取公开的 Copernicus GLO-30 地表模型，--synthetic 可生成仿真样例。
"""
import argparse
import hashlib
import heapq
import json
import math
from itertools import count
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARCELS = ROOT / '林场范围哈' / 'forest_features.geojson'
DSM_TILE = 'Copernicus_DSM_COG_10_N28_00_E103_00_DEM'
DEFAULT_DSM = ROOT / 'data' / 'leibo' / 'terrain' / (DSM_TILE + '.tif')
DSM_URL = f'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{DSM_TILE}/{DSM_TILE}.tif'
DSM_REFERENCE = 'https://registry.opendata.aws/copernicus-dem/'


# ============================================================
# 1. 合成 DEM (模拟雷波山区)
# ============================================================
def generate_dem(width, height, seed=42, base_elev=1200, relief=400):
    """生成雷波山区合成高程模型 (平滑版)"""
    rng = np.random.default_rng(seed)

    # 低频分形噪声 (只用低频, 避免格子级突变)
    noise = np.zeros((height, width))
    frequencies = [(width / 3, 1.0), (width / 5, 0.4), (width / 8, 0.2)]

    for freq, amp in frequencies:
        grid_w = max(2, int(freq))
        grid_h = max(2, int(grid_w * height / width))
        grid = rng.random((grid_h, grid_w))
        ys = np.linspace(0, grid_h - 1, height)
        xs = np.linspace(0, grid_w - 1, width)
        coarse = np.zeros((height, width))
        for yi in range(height):
            y0 = int(ys[yi])
            y1 = min(y0 + 1, grid_h - 1)
            fy = ys[yi] - y0
            for xi in range(width):
                x0 = int(xs[xi])
                x1 = min(x0 + 1, grid_w - 1)
                fx = xs[xi] - x0
                coarse[yi, xi] = (grid[y0, x0] * (1 - fy) * (1 - fx) +
                                  grid[y0, x1] * (1 - fy) * fx +
                                  grid[y1, x0] * fy * (1 - fx) +
                                  grid[y1, x1] * fy * fx)
        noise += amp * coarse

    # 归一化到 [0,1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)

    # 归一化
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)

    # 河谷下切
    elev = base_elev + relief * noise
    for y in range(height):
        river_x = width * (0.15 + 0.7 * (y / height) ** 0.7)
        for x in range(width):
            rd = abs(x - river_x) / width
            elev[y, x] -= 100 * math.exp(-rd * 4)

    # 高斯平滑 (使坡度合理)
    from scipy.ndimage import gaussian_filter
    elev = gaussian_filter(elev, sigma=3)

    return elev


# ============================================================
# 2. 坡度与成本
# ============================================================
def compute_slope(dem, cell_size_m=25):
    """内部使用中心差分，边界使用单侧差分。"""
    heights = np.asarray(dem, dtype=float)
    h, w = heights.shape
    if not math.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError('cell_size_m must be finite and positive')
    heights = np.where(np.isfinite(heights), heights, np.nan)
    dzdx = np.gradient(heights, cell_size_m, axis=1) if w > 1 else np.zeros_like(heights)
    dzdy = np.gradient(heights, cell_size_m, axis=0) if h > 1 else np.zeros_like(heights)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.degrees(np.arctan2(dzdy, dzdx))
    missing = np.isnan(heights)
    slope[missing] = aspect[missing] = np.nan
    return slope, aspect


def terrain_cost(slope_deg):
    """地形通行成本 (考虑轮足机器狗能力)

    0-15°: 轻松, 成本 1.0
    15-25°: 正常, 成本 1.5
    25-35°: 困难, 成本 3.0
    35-45°: 极难, 成本 6.0
    ≥45°或未知坡度: 不可通行
    """
    if not math.isfinite(slope_deg) or slope_deg < 0:
        return float('inf')
    if slope_deg < 15:
        return 1.0
    elif slope_deg < 25:
        return 1.0 + (slope_deg - 15) / 10 * 0.5
    elif slope_deg < 35:
        return 1.5 + (slope_deg - 25) / 10 * 1.5
    elif slope_deg < 45:
        return 3.0 + (slope_deg - 35) / 10 * 3.0
    else:
        return float('inf')


def cost_surface(slope, cell_size_m=25):
    """生成成本表面"""
    h, w = slope.shape
    cost = np.full((h, w), float('inf'))
    for y in range(h):
        for x in range(w):
            cost[y, x] = terrain_cost(slope[y, x])
    return cost


# ============================================================
# 3. 地形感知 A* 路径规划
# ============================================================
def a_star_terrain(cost, start, goal, slope, cell_size_m=25):
    """返回 (x, y) 格坐标路线。起终点无效或不可达时返回 None。"""
    cost = np.asarray(cost, dtype=float)
    slope = np.asarray(slope, dtype=float)
    if cost.ndim != 2 or slope.shape != cost.shape:
        raise ValueError('cost and slope must have the same two-dimensional shape')
    if not math.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError('cell_size_m must be finite and positive')
    h, w = cost.shape
    traversable = np.isfinite(cost) & (cost >= 0) & np.isfinite(slope)
    start, goal = tuple(start), tuple(goal)
    for cell in (start, goal):
        if len(cell) != 2:
            return None
        x, y = cell
        if not (isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer))
                and 0 <= x < w and 0 <= y < h and traversable[y, x]):
            return None

    # 距离乘以最小格代价，保证启发式仍是自定义权重下的代价下界。
    min_weight = float(cost[traversable].min())
    order = count()
    open_set = [(min_weight * heuristic(start, goal, cell_size_m), next(order), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, _, current_cost, current = heapq.heappop(open_set)
        if current_cost > g_score[current]:
            continue

        if current == goal:
            # 重建路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            nx, ny = neighbor
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if not traversable[ny, nx]:
                continue
            if dx and dy and not (traversable[current[1], nx] and traversable[ny, current[0]]):
                continue
            tentative_g = current_cost + step_cost(cost, slope, current, neighbor, cell_size_m)
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                priority = tentative_g + min_weight * heuristic(neighbor, goal, cell_size_m)
                heapq.heappush(open_set, (priority, next(order), tentative_g, neighbor))

    return None


def heuristic(a, b, cell_size_m=25):
    return cell_size_m * math.hypot(a[0] - b[0], a[1] - b[1])


def step_cost(cost, slope, start, end, cell_size_m):
    x, y = start
    nx, ny = end
    slope_penalty = 1 + max(0, slope[ny, nx] - slope[y, x]) / 10
    return heuristic(start, end, cell_size_m) * cost[ny, nx] * slope_penalty


# ============================================================
# 4. 地块内最优路径规划
# ============================================================
def parcel_grid(regions, cell_size_m):
    """将经纬度转换为局部网格，按场地中纬度近似东西向米制距离。"""
    points = np.concatenate([np.asarray(region['coords'], dtype=float) for region in regions])
    origin = points.min(axis=0)
    northeast = points.max(axis=0)
    latitude = (origin[1] + northeast[1]) / 2
    meters_per_degree = np.array([111320 * math.cos(math.radians(latitude)), 111320])
    degrees_per_cell = cell_size_m / meters_per_degree
    grid_size = np.maximum(2, np.ceil((northeast - origin) / degrees_per_cell)).astype(int)
    return origin, degrees_per_cell, grid_size


def parcel_window(region, origin, degrees_per_cell, shape):
    from forest_env import point_in_polygon

    polygon = (np.asarray(region['coords']) - origin) / degrees_per_cell
    holes = [(np.asarray(ring) - origin) / degrees_per_cell for ring in region.get('holes', [])]
    left, bottom = np.maximum(0, np.floor(polygon.min(axis=0))).astype(int)
    right, top = np.minimum(shape[::-1], np.ceil(polygon.max(axis=0))).astype(int)
    rows, cols = slice(bottom, top), slice(left, right)
    mask = np.zeros((max(0, top - bottom), max(0, right - left)), dtype=bool)
    for y in range(bottom, top):
        for x in range(left, right):
            mask[y - bottom, x - left] = (
                point_in_polygon(x + .5, y + .5, polygon)
                and not any(point_in_polygon(x + .5, y + .5, ring) for ring in holes)
            )
    return rows, cols, mask, polygon


def plan_region_paths(dem, slope, cost, regions, cell_size_m=25):
    """在每块地的栅格边界内，连接相隔较远的两个模拟端点。"""
    paths = []
    if not regions:
        return paths
    origin, degrees_per_cell, _ = parcel_grid(regions, cell_size_m)
    for region in regions:
        rows, cols, mask, polygon = parcel_window(region, origin, degrees_per_cell, dem.shape)
        local_cost = np.where(mask, cost[rows, cols], math.inf)
        local_slope = slope[rows, cols]
        cells = np.argwhere(np.isfinite(local_cost) & (local_cost >= 0) & np.isfinite(local_slope))[:, ::-1]
        if len(cells) < 2:
            continue

        offset = np.array([cols.start, rows.start])
        start = tuple(cells[np.argmin(np.sum((cells + offset + .5 - polygon[0]) ** 2, axis=1))])
        goal = tuple(cells[np.argmax(np.sum((cells - start) ** 2, axis=1))])
        path = a_star_terrain(local_cost, start, goal, local_slope, cell_size_m)
        if path is None:
            continue
        path = [(x + cols.start, y + rows.start) for x, y in path]
        segments = list(zip(path, path[1:]))
        total_dist = sum(heuristic(a, b, cell_size_m) for a, b in segments)
        total_climb = sum(max(0, dem[b[1], b[0]] - dem[a[1], a[0]]) for a, b in segments)
        total_cost = sum(step_cost(cost, slope, a, b, cell_size_m) for a, b in segments)
        paths.append({
            'region': region['name'],
            'region_id': region.get('id'),
            'path': [[int(x), int(y)] for x, y in path],
            'length_m': round(total_dist, 2),
            'climb_m': round(float(total_climb), 2),
            'weighted_cost': round(float(total_cost), 2),
            'cells': len(path),
        })

    return paths


def find_feasible(cost, point, max_radius=5):
    """找到 point 附近最近的可行格"""
    x, y = point
    h, w = cost.shape
    if 0 <= x < w and 0 <= y < h and cost[y, x] < float('inf'):
        return (x, y)
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and cost[ny, nx] < float('inf'):
                    return (nx, ny)
    return None


def load_elevation(path, origin, degrees_per_cell, width, height):
    try:
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:
        raise RuntimeError('Install rasterio to use the measured elevation raster') from exc

    dx, dy = degrees_per_cell
    transform = from_origin(origin[0], origin[1] + height * dy, dx, dy)
    destination = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(f'Elevation raster has no coordinate reference system: {path}')
        reproject(rasterio.band(source, 1), destination, src_nodata=source.nodata,
                  dst_transform=transform, dst_crs='EPSG:4326', dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        source_crs = str(source.crs)
        native_resolution = list(source.res)
    if not np.isfinite(destination).any():
        raise ValueError(f'Elevation raster does not cover the parcel grid: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    is_copernicus = digest == '83cbc2fc4aa39b85fd16db284ca032e94b42819fc247f4ff2ae8b24db43e7b40'
    metadata = {
        'kind': 'copernicus_dsm' if is_copernicus else 'user_raster',
        'label': 'Copernicus DEM GLO-30 地表高程' if is_copernicus else '输入高程栅格',
        'file': path.name,
        'sha256': digest,
        'crs': source_crs,
        'native_resolution': native_resolution,
        'reference': DSM_REFERENCE if is_copernicus else None,
        'source_url': DSM_URL if is_copernicus else None,
    }
    # 路线网格的 y 轴向北，GeoTIFF 行号向南。
    return destination[::-1], metadata


def web_data(regions, paths, dem, slope, origin, step, cell_size_m, metadata):
    by_id = {path['region_id']: path for path in paths}
    parcels = []
    for region in regions:
        rows, cols, inside, polygon = parcel_window(region, origin, step, dem.shape)
        route = by_id.get(region['id'])
        elevations = np.where(inside, dem[rows, cols], np.nan)
        slopes = np.where(inside, slope[rows, cols], np.nan)
        parcels.append({
            'id': region['id'],
            'name': region['name'],
            'source': region['source'],
            'functionZone': region['functionZone'],
            'extentLonLat': (origin + np.array([[cols.start, rows.start], [cols.stop, rows.stop]]) * step).tolist(),
            'boundary': (polygon - [cols.start, rows.start]).tolist(),
            'elevationM': [[round(float(x), 1) if math.isfinite(x) else None for x in row] for row in elevations],
            'slopeDeg': [[round(float(x), 2) if math.isfinite(x) else None for x in row] for row in slopes],
            'parcelCells': int(inside.sum()),
            'path': [[int(x - cols.start), int(y - rows.start)] for x, y in route['path']] if route else [],
            'metrics': {key: route[key] for key in ('length_m', 'climb_m', 'weighted_cost')} if route else None,
        })
    return {'schema': 'leibo-parcel-terrain-v1', 'cellSizeM': cell_size_m,
            'coordinateConvention': 'array rows and grid y increase northward; path uses cell indices',
            'elevation': metadata, 'parcels': parcels}


# ============================================================
# 5. 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dem', type=Path, default=DEFAULT_DSM, help='有坐标参考系的本地高程栅格')
    parser.add_argument('--synthetic', action='store_true', help='使用合成高程做仿真测试')
    parser.add_argument('--cell-size', type=float, default=30., help='规划格长，米')
    parser.add_argument('--output-dir', type=Path, default=Path.cwd())
    parser.add_argument('--web-output', type=Path, help='可选的本地网页数据脚本路径')
    args = parser.parse_args()
    if not math.isfinite(args.cell_size) or args.cell_size <= 0:
        parser.error('--cell-size must be finite and positive')

    data = json.loads(PARCELS.read_text(encoding='utf-8'))
    regions = [
        {'id': feature.get('id', str(index)), 'name': feature['properties']['name'],
         'source': feature['properties'].get('source', PARCELS.name),
         'functionZone': feature['properties'].get('functionZone'),
         'coords': feature['geometry']['coordinates'][0],
         'holes': feature['geometry']['coordinates'][1:]}
        for index, feature in enumerate(data['features']) if feature['geometry']['type'] == 'Polygon'
    ]
    if not regions:
        raise ValueError('GeoJSON 中没有可规划的地块')
    print('地块:')
    for r in regions:
        print('  "{}" 边界点={}'.format(r['name'], len(r['coords'])))

    CELL_M = args.cell_size
    origin, degrees_per_cell, (GRID_W, GRID_H) = parcel_grid(regions, CELL_M)
    if GRID_W * GRID_H > 4_000_000:
        parser.error('地块网格超过 400 万格，请使用较粗的 --cell-size')
    if args.synthetic:
        dem = generate_dem(GRID_W, GRID_H, seed=42)
        source = {'kind': 'synthetic', 'label': '合成高程'}
    else:
        if not args.dem.is_file():
            parser.error(f'高程文件缺失：{args.dem}。请提供 --dem GeoTIFF，或显式使用 --synthetic。')
        dem, source = load_elevation(args.dem, origin, degrees_per_cell, GRID_W, GRID_H)
    print(f'高程来源：{source["label"]}')
    slope, aspect = compute_slope(dem, CELL_M)
    cost = cost_surface(slope, CELL_M)

    print('\n地形统计:')
    print('  海拔: {:.0f} ~ {:.0f} m'.format(np.nanmin(dem), np.nanmax(dem)))
    print('  坡度: {:.1f}° ~ {:.1f}°'.format(
        np.nanmin(slope), np.nanmax(slope)))
    print('  平均坡度: {:.1f}°'.format(np.nanmean(slope)))

    # 坡度分布
    bins = [(0, 15, '平缓'), (15, 25, '缓坡'), (25, 35, '陡坡'),
            (35, 45, '极陡'), (45, 90, '不可通行')]
    flat_slope = slope[np.isfinite(slope)]
    print('\n坡度分布:')
    for lo, hi, label in bins:
        pct = np.sum((flat_slope >= lo) & (flat_slope < hi)) / len(flat_slope) * 100
        print('  {} ({}-{}°): {:.1f}%'.format(label, lo, hi, pct))

    # 通行性
    passable = np.sum(cost < float('inf'))
    total = cost.size
    print('\n通行性: {}/{} 格可通行 ({:.1f}%)'.format(
        passable, total, passable / total * 100))

    # 路径规划
    print('\n规划地块路径:')
    paths = plan_region_paths(dem, slope, cost, regions, CELL_M)
    for p in paths:
        print('  "{}": {} 格, 水平 {:.0f}m, 爬升 {:.0f}m, 加权代价 {:.1f}'.format(
            p['region'], p['cells'], p['length_m'], p['climb_m'], p['weighted_cost']))
    print('  找到路线的地块: {}/{}'.format(len(paths), len(regions)))

    # 输出
    output = {
        'grid_size': [int(GRID_W), int(GRID_H)],
        'cell_size_m': CELL_M,
        'grid_origin_lonlat': origin.tolist(),
        'degrees_per_cell': degrees_per_cell.tolist(),
        'terrain_source': source['kind'],
        'elevation_source': source,
        'dem_stats': {
            'min': round(float(np.nanmin(dem))),
            'max': round(float(np.nanmax(dem))),
            'mean': round(float(np.nanmean(dem))),
        },
        'slope_stats': {
            'min': round(float(np.nanmin(slope)), 1),
            'max': round(float(np.nanmax(slope)), 1),
            'mean': round(float(np.nanmean(slope)), 1),
        },
        'paths': paths,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'terrain_paths.json').open('w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 输出 DEM 数据用于可视化 (降采样到 40x40)
    rows = np.linspace(0, GRID_H - 1, min(40, GRID_H), dtype=int)
    columns = np.linspace(0, GRID_W - 1, min(40, GRID_W), dtype=int)
    sample = np.ix_(rows, columns)

    with (args.output_dir / 'terrain_vis.json').open('w', encoding='utf-8') as f:
        json.dump({
            'dem': [[round(float(x), 1) if math.isfinite(x) else None for x in row] for row in dem[sample]],
            'slope': [[round(float(x), 1) if math.isfinite(x) else None for x in row] for row in slope[sample]],
            'passable': np.isfinite(cost[sample]).astype(int).tolist(),
            'sample_rows': rows.tolist(),
            'sample_columns': columns.tolist(),
            'cell_m': CELL_M,
        }, f)

    if args.web_output:
        data = web_data(regions, paths, dem, slope, origin, degrees_per_cell, CELL_M, source)
        payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        args.web_output.write_text('window.FOREST_TERRAIN = ' + payload + ';\n', encoding='utf-8')
    print(f'\n已输出: {args.output_dir}')
    print('TERRAIN_DONE')


if __name__ == '__main__':
    main()
