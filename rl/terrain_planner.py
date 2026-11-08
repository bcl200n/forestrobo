# -*- coding: utf-8 -*-
"""林场地形感知规划器: KMZ 边界 + 合成 DEM + 坡度分析 + 最优路径

雷波县位于四川盆地向云贵高原过渡带, 海拔 400~3000m, 沟谷深切, 坡度 25~45°
本模块:
1. 解析 KMZ 获取地块边界
2. 生成区域特定合成地形(分形噪声 + 区域高程特征)
3. 计算坡度/坡向/通行成本
4. A* 加权最优路径规划(避开陡坡)
5. 输出地形感知的作业路径 GeoJSON
"""
import json
import math
import os
import random
import sys

import numpy as np


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
    """计算坡度(度)和坡向"""
    h, w = dem.shape
    slope = np.zeros((h, w))
    aspect = np.zeros((h, w))

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            dzdx = (dem[y, x + 1] - dem[y, x - 1]) / (2 * cell_size_m)
            dzdy = (dem[y + 1, x] - dem[y - 1, x]) / (2 * cell_size_m)
            slope[y, x] = math.degrees(math.atan(
                math.sqrt(dzdx ** 2 + dzdy ** 2)))
            aspect[y, x] = math.degrees(math.atan2(dzdy, dzdx))

    return slope, aspect


def terrain_cost(slope_deg):
    """地形通行成本 (考虑轮足机器狗能力)

    0-15°: 轻松, 成本 1.0
    15-25°: 正常, 成本 1.5
    25-35°: 困难, 成本 3.0
    35-45°: 极难, 成本 6.0
    >45°: 不可通行
    """
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
    """A* 加权路径规划, 考虑坡度和距离"""
    h, w = cost.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, cell_size_m)}

    while open_set:
        open_set.sort(key=lambda x: x[0])
        _, current = open_set.pop(0)

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
            if cost[ny, nx] == float('inf'):
                continue
            if dx and dy:
                if cost[current[1], current[0] + dx] == float('inf') or \
                   cost[current[1] + dy, current[0]] == float('inf'):
                    continue

            dist = cell_size_m * math.hypot(dx, dy)
            slope_penalty = 1 + max(0, slope[ny, nx] - slope[current[1], current[0]]) / 10
            tentative_g = g_score[current] + dist * cost[ny, nx] * slope_penalty

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal, cell_size_m)
                if neighbor not in [item[1] for item in open_set]:
                    open_set.append((f_score[neighbor], neighbor))

    return None


def heuristic(a, b, cell_size_m=25):
    return cell_size_m * math.hypot(a[0] - b[0], a[1] - b[1])


# ============================================================
# 4. 地块内最优路径规划
# ============================================================
def plan_region_paths(dem, slope, cost, regions, cell_size_m=25):
    """为每个地块规划从入口到出口的最优路径"""
    paths = []
    h, w = dem.shape

    for region in regions:
        # 找地块内的可通行格子
        region_cells = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if cost[y, x] < float('inf'):
                    # 简化: 用成本表面的可通行区域
                    region_cells.append((x, y))

        if len(region_cells) < 2:
            continue

        # 起点和终点: 选择距离最远的两个可通行点
        # (简化: 用地块的第一个和最后一个点)
        start = (int(region['coords'][0][0] * w / 100),
                 int(region['coords'][0][1] * h / 100))
        goal = (int(region['coords'][len(region['coords']) // 2][0] * w / 100),
                int(region['coords'][len(region['coords']) // 2][1] * h / 100))
        start = (max(1, min(w - 2, start[0])), max(1, min(h - 2, start[1])))
        goal = (max(1, min(w - 2, goal[0])), max(1, min(h - 2, goal[1])))

        # 找最近的可行点
        start = find_feasible(cost, start)
        goal = find_feasible(cost, goal)
        if not start or not goal:
            continue

        path = a_star_terrain(cost, start, goal, slope, cell_size_m)
        if path:
            total_dist = len(path) * cell_size_m
            total_climb = 0
            for i in range(1, len(path)):
                total_climb += abs(dem[path[i][1], path[i][0]] -
                                   dem[path[i - 1][1], path[i - 0][0]] if False else
                                   dem[path[i][1], path[i][0]] -
                                   dem[path[i - 1][1], path[i - 1][0]])
            paths.append({
                'region': region['name'],
                'path': [[int(p[0]), int(p[1])] for p in path],
                'length_m': round(total_dist),
                'climb_m': round(total_climb),
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


# ============================================================
# 5. 主程序
# ============================================================
def main():
    from forest_env import parse_kmz
    import zipfile
    import xml.etree.ElementTree as ET

    base = r'E:\forestrobo\林场范围哈'
    kmz_file = os.path.join(base, '林下经济范围(1).kmz')

    print('=== 林场地形感知路径规划 ===\n')

    # 解析 KMZ
    features = parse_kmz(kmz_file)
    print('地块:')
    for r in features['regions']:
        print('  "{}" 边界点={}'.format(r['name'], len(r['coords'])))

    # 生成地形
    GRID_W, GRID_H = 60, 60
    CELL_M = 25  # 每格 25m
    dem = generate_dem(GRID_W, GRID_H, seed=42)
    slope, aspect = compute_slope(dem, CELL_M)
    cost = cost_surface(slope, CELL_M)

    print('\n地形统计:')
    print('  海拔: {:.0f} ~ {:.0f} m'.format(dem.min(), dem.max()))
    print('  坡度: {:.1f}° ~ {:.1f}°'.format(
        slope[1:-1, 1:-1].min(), slope[1:-1, 1:-1].max()))
    print('  平均坡度: {:.1f}°'.format(slope[1:-1, 1:-1].mean()))

    # 坡度分布
    bins = [(0, 15, '平缓'), (15, 25, '缓坡'), (25, 35, '陡坡'),
            (35, 45, '极陡'), (45, 90, '不可通行')]
    flat_slope = slope[1:-1, 1:-1].flatten()
    print('\n坡度分布:')
    for lo, hi, label in bins:
        pct = np.sum((flat_slope >= lo) & (flat_slope < hi)) / len(flat_slope) * 100
        print('  {} ({}-{}°): {:.1f}%'.format(label, lo, hi, pct))

    # 通行性
    passable = np.sum(cost < float('inf'))
    total = (GRID_W - 2) * (GRID_H - 2)
    print('\n通行性: {}/{} 格可通行 ({:.1f}%)'.format(
        passable, total, passable / total * 100))

    # 路径规划
    print('\n规划地块路径:')
    paths = plan_region_paths(dem, slope, cost, features['regions'], CELL_M)
    for p in paths:
        print('  "{}": {} 格, {:.0f}m, 爬升 {:.0f}m'.format(
            p['region'], p['cells'], p['length_m'], p['climb_m']))

    # 输出
    output = {
        'grid_size': [GRID_W, GRID_H],
        'cell_size_m': CELL_M,
        'dem_stats': {
            'min': round(float(dem.min())),
            'max': round(float(dem.max())),
            'mean': round(float(dem.mean())),
        },
        'slope_stats': {
            'min': round(float(slope[1:-1, 1:-1].min()), 1),
            'max': round(float(slope[1:-1, 1:-1].max()), 1),
            'mean': round(float(slope[1:-1, 1:-1].mean()), 1),
        },
        'paths': paths,
    }
    with open('terrain_paths.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 输出 DEM 数据用于可视化 (降采样到 40x40)
    vis_dem = dem[::GRID_H // 40, ::GRID_W // 40][:40, :40]
    vis_slope = slope[::GRID_H // 40, ::GRID_W // 40][:40, :40]
    vis_cost = cost[::GRID_H // 40, ::GRID_W // 40][:40, :40]

    with open('terrain_vis.json', 'w', encoding='utf-8') as f:
        json.dump({
            'dem': vis_dem.round(1).tolist(),
            'slope': vis_slope.round(1).tolist(),
            'passable': (vis_cost < float('inf')).astype(int).tolist(),
            'cell_m': CELL_M,
        }, f)

    print('\n已输出: terrain_paths.json + terrain_vis.json')
    print('TERRAIN_DONE')


if __name__ == '__main__':
    main()
