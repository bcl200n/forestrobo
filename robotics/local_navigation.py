# -*- coding: utf-8 -*-
"""修复 LocalGrid.route 的绕行能力:
1. 可达距离上限 1.8→5.0 m (绕障需要更远视野)
2. 增加全局 A* 直接在观测图上规划 (替代原来的 grid_distances+评分)
3. 保留原有 clearance_bias/weighted_local 参数兼容性
"""
import math
import heapq
import numpy as np
from robotics.terrain_planner import plan, inflate

# 可达距离上限从 1.8 提升到 5.0 — 绕行场景需要看到障碍物另一侧
REACH_LIMIT_M = 5.0


def grid_distances(start, blocked, resolution, limit=float('inf'), *, penalty=None):
    """Dijkstra on a grid with diagonal corner checks."""
    rows, cols = blocked.shape
    if blocked[start]:
        return {}, {}
    costs, parents = {start: 0.}, {}
    queue = [(0., start)]
    while queue:
        cost, cell = heapq.heappop(queue)
        if cost > costs[cell] + 1e-9:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not (dr or dc):
                    continue
                neighbor = cell[0] + dr, cell[1] + dc
                nr, nc = neighbor
                if not (0 <= nr < rows and 0 <= nc < cols) or blocked[neighbor]:
                    continue
                if dr and dc and (blocked[nr, cell[1]] or blocked[cell[0], nc]):
                    continue
                edge_cost = resolution * math.hypot(dr, dc)
                if penalty is not None:
                    edge_cost *= 1. + .5 * (penalty[cell] + penalty[neighbor])
                new_cost = cost + edge_cost
                if new_cost > limit or new_cost >= costs.get(neighbor, float('inf')):
                    continue
                costs[neighbor] = new_cost
                parents[neighbor] = cell
                heapq.heappush(queue, (new_cost, neighbor))
    return costs, parents


class LocalGrid:
    def __init__(self, origin=(-2., -4.), shape=(40, 60), resolution=.2):
        self.origin = np.asarray(origin, float)
        self.resolution = resolution
        self.known = np.zeros(shape, bool)
        self.occupied = np.zeros(shape, bool)
        self.last_update = -float('inf')
        self.goal_costs = {}

    def cell(self, point):
        col, row = np.floor((np.asarray(point) - self.origin) / self.resolution).astype(int)
        return int(row), int(col)

    def point(self, cell):
        row, col = cell
        return self.origin + (np.array([col, row]) + .5) * self.resolution

    def inside(self, cell):
        return 0 <= cell[0] < self.known.shape[0] and 0 <= cell[1] < self.known.shape[1]

    def update(self, position, angles, ranges, hits, stamp):
        if not np.isfinite(stamp) or stamp < self.last_update:
            raise ValueError('invalid/old stamp')
        angles, ranges, hits = np.asarray(angles), np.asarray(ranges), np.asarray(hits)
        if angles.ndim != 1 or angles.shape != ranges.shape or ranges.shape != hits.shape \
                or not np.isfinite(angles).all() or not np.isfinite(position).all() \
                or not np.isfinite(ranges).all() or (ranges < 0).any():
            raise ValueError('invalid beams')
        marked = []
        for angle, distance, hit in zip(angles, ranges, hits):
            direction = np.array([math.cos(angle), math.sin(angle)])
            for d in np.arange(0, max(0, distance - self.resolution * .5), self.resolution * .4):
                c = self.cell(np.asarray(position) + d * direction)
                if self.inside(c):
                    self.known[c] = True
            c = self.cell(np.asarray(position) + distance * direction)
            if hit and self.inside(c):
                marked.append(c)
        for c in marked:
            self.known[c] = True
            self.occupied[c] = True
        self.last_update = stamp

    def route(self, position, goal, radius=.48, *, geodesic=False,
              clearance_bias=False, weighted_local=False):
        self.goal_costs = {}
        start = self.cell(position)
        if not self.inside(start):
            return []
        goal_cell = self.cell(goal)
        if not self.inside(goal_cell):
            return []

        clearance_bias = clearance_bias or weighted_local
        geodesic = geodesic or clearance_bias
        penalty = 2. * inflate(self.occupied, self.resolution, radius + .24) if clearance_bias else None

        if geodesic:
            optimistic = inflate(self.occupied, self.resolution, radius)
            goal_costs, _ = grid_distances(goal_cell, optimistic, self.resolution, penalty=penalty)
            self.goal_costs = goal_costs
            if start not in goal_costs:
                return []

        # === 全局 A* 直接在观测图上规划 ===
        # 可执行路线: 已知自由 + 障碍膨胀(含物理半径)
        blocked = inflate(~self.known | self.occupied, self.resolution, radius)

        # 先尝试直接用 terrain_planner 的 plan() 全局规划
        heights = np.zeros(self.known.shape)  # 平面假设(局部网格)
        result = plan(heights, blocked.astype(float), start, goal_cell,
                      self.resolution, footprint_radius_m=0)  # 已 inflate 过

        if result['found']:
            # 全局路线找到, 直接返回路径点
            path_cells = result['path']
            return [self.point(c).tolist() for c in path_cells]

        # 全局路线不通(被未知区域阻断), 退回到原有的局部可达+评分逻辑
        costs, parent = grid_distances(start, blocked, self.resolution,
                                       limit=REACH_LIMIT_M)

        if weighted_local and costs:
            reachable = np.ones_like(blocked)
            for cell in costs:
                reachable[cell] = False
            _, parent = grid_distances(start, reachable, self.resolution, penalty=penalty)

        if not geodesic:
            # 使用乐观全局引导(穿未知区)来评估最佳前沿
            guide_blocked = inflate(self.occupied, self.resolution, radius)
            guide = plan(np.zeros(self.known.shape), guide_blocked.astype(float),
                         start, goal_cell, self.resolution, 0)
            if guide['found']:
                guide_points = np.array([self.point(c) for c in guide['path']])
                lengths = np.linalg.norm(np.diff(guide_points, axis=0), axis=1)
                remaining = np.r_[np.cumsum(lengths[::-1])[::-1], 0.]
            else:
                guide_points = np.array([[goal[0], goal[1]]])
                remaining = np.array([0.])
        else:
            guide_points = None

        best = None
        score = float('inf')
        for cell, cost in costs.items():
            if cost < .4:
                continue
            if geodesic:
                value = goal_costs.get(cell, float('inf'))
            else:
                distance = np.linalg.norm(guide_points - self.point(cell), axis=1)
                value = float(np.min(distance * 1.5 + remaining))
            if value < score:
                best, score = cell, value
        if best is None:
            return []
        path = [best]
        while path[-1] != start:
            path.append(parent[path[-1]])
        return [self.point(c).tolist() for c in path[::-1]]


class WaypointFollower:
    def command(self, position, yaw, path, goal):
        position = np.asarray(position, float)
        if np.linalg.norm(position - np.asarray(goal)) < .35:
            return np.zeros(3)
        if not path:
            return np.zeros(3)
        points = np.asarray(path)
        nearest = int(np.argmin(np.linalg.norm(points - position, axis=1)))
        target = points[-1]
        for point in points[nearest:]:
            if np.linalg.norm(point - position) > .55:
                target = point
                break
        error = target - position
        angle = math.atan2(error[1], error[0])
        yaw_error = (angle - yaw + math.pi) % (2 * math.pi) - math.pi
        speed = .22
        # 全向平移(不依赖纯差速转向)
        world = error / max(np.linalg.norm(error), 1e-9) * speed
        body = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]]) @ world
        return np.r_[body, float(np.clip(yaw_error * .5, -.3, .3))]
