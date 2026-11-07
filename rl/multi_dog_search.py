# -*- coding: utf-8 -*-
"""多狗协同搜索: 市场竞拍任务分配 + 分布式前沿 + 通信中断容错

三个核心机制:
1. 任务竞拍: 每只狗对自己可达前沿出价(距离+已探区域), 最高者获得
2. 地图融合: 通信窗口内共享观测, 通信中断时独立决策
3. 角色降级: 个体失效时自动重新分配其任务
"""
import math
from collections import deque, defaultdict
from enum import Enum
import numpy as np


class Role(Enum):
    SEARCHER = 'searcher'    # 搜索
    RELAY = 'relay'          # 通信中继
    LEADER = 'leader'        # 协调者


class DogAgent:
    """单只搜索狗的状态与决策"""

    def __init__(self, dog_id, start_pos, comm_range=8.0, max_speed=0.6):
        self.id = dog_id
        self.pos = np.array(start_pos, float)
        self.alive = True
        self.role = Role.SEARCHER
        self.comm_range = comm_range
        self.max_speed = max_speed
        self.known = set()  # 已观测自由格
        self.occupied = set()  # 已观测障碍格
        self.assigned_target = None
        self.visits = defaultdict(int)
        self.battery = 100.0  # %
        self.found_targets = set()

    def scan(self, world_blocked, world_size, range_m=3.0, pos=None):
        """观测周围环境, 更新已知格"""
        pos = pos if pos is not None else self.pos
        x, y = int(pos[0]), int(pos[1])
        for dx in range(-int(range_m), int(range_m) + 1):
            for dy in range(-int(range_m), int(range_m) + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < world_size and 0 <= ny < world_size:
                    if dx * dx + dy * dy <= range_m * range_m:
                        cell = (nx, ny)
                        if world_blocked[nx][ny]:
                            self.occupied.add(cell)
                        else:
                            self.known.add(cell)
        self.pos = pos

    def bid(self, target, world_size):
        """对目标出价: 越近越好, 重复访问越少越好"""
        if not self.alive or self.battery < 10:
            return -1
        dist = np.linalg.norm(np.array(target) - self.pos)
        visit_penalty = self.visits[target] * 0.5
        exploration_bonus = 0 if target in self.known else 2.0
        return exploration_bonus + 10.0 / (1 + dist) - visit_penalty

    def move_toward(self, target, dt=0.1):
        """朝目标移动一步"""
        if self.assigned_target is None:
            return
        target = np.array(self.assigned_target, float)
        direction = target - self.pos
        dist = np.linalg.norm(direction)
        if dist < 0.5:
            self.visits[tuple(int(x) for x in self.pos)] += 1
            self.assigned_target = None
            return
        self.pos += direction / dist * self.max_speed * dt
        self.battery -= 0.01 * dt

    def share_map(self):
        """导出已知地图供融合"""
        return {'id': self.id, 'known': self.known.copy(),
                'occupied': self.occupied.copy(),
                'pos': self.pos.copy(), 'alive': self.alive,
                'found': self.found_targets.copy()}

    def merge_map(self, other):
        """融合队友的地图"""
        self.known |= other['known']
        self.occupied |= other['occupied']


class MultiDogCoordinator:
    """多狗协同控制器"""

    def __init__(self, world_size=20, n_obstacles=15, n_targets=4, seed=42,
                 n_dogs=10):
        self.world_size = world_size
        self.n_dogs = n_dogs
        rng = np.random.default_rng(seed)

        # 生成世界
        self.blocked = [[False] * world_size for _ in range(world_size)]
        for _ in range(n_obstacles):
            x, y = rng.integers(2, world_size - 2, 2)
            for dx in range(int(rng.integers(1, 3))):
                for dy in range(int(rng.integers(1, 3))):
                    if 0 <= x+dx < world_size and 0 <= y+dy < world_size:
                        self.blocked[x+dx][y+dy] = True

        # 目标(幸存者)
        free_cells = [(x, y) for x in range(2, world_size-2)
                      for y in range(2, world_size-2)
                      if not self.blocked[x][y]]
        target_indices = rng.choice(len(free_cells), n_targets, replace=False)
        self.targets = [free_cells[i] for i in target_indices]
        self.found = set()

        # 初始化狗 (分散起点: 环形均匀分布)
        radius = min(world_size // 4, 4)
        cx, cy = world_size // 2, world_size // 2
        starts = []
        for i in range(n_dogs):
            angle = 2 * math.pi * i / n_dogs
            sx = int(cx + radius * math.cos(angle))
            sy = int(cy + radius * math.sin(angle))
            sx = max(1, min(world_size - 2, sx))
            sy = max(1, min(world_size - 2, sy))
            # 避开障碍
            for _ in range(20):
                if not self.blocked[sx][sy]:
                    break
                sx = max(1, min(world_size - 2, sx + 1))
            starts.append((sx, sy))
        self.dogs = [DogAgent(i, starts[i]) for i in range(n_dogs)]

        # 通信拓扑
        self.comm_links = {}  # (a,b) -> bool
        self.step_count = 0
        self.reassignments = 0
        self.rng = rng

    def comm_matrix(self):
        """计算当前通信连接"""
        links = {}
        for i in range(len(self.dogs)):
            for j in range(i + 1, len(self.dogs)):
                dist = np.linalg.norm(
                    self.dogs[i].pos - self.dogs[j].pos)
                links[(i, j)] = dist <= self.dogs[i].comm_range
        return links

    def find_frontiers(self, dog):
        """找到已知/未知边界的前沿格"""
        frontiers = []
        for cell in dog.known:
            x, y = cell
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.world_size and 0 <= ny < self.world_size:
                    if (nx, ny) not in dog.known and \
                       not self.blocked[nx][ny]:
                        frontiers.append((nx, ny))
        return list(set(frontiers))

    def auction_targets(self):
        """市场竞拍分配前沿目标"""
        # 收集所有未分配的前沿
        all_frontiers = set()
        for dog in self.dogs:
            if dog.alive:
                all_frontiers |= set(self.find_frontiers(dog))
        # 也把未找到的目标加入
        remaining = set(self.targets) - self.found
        all_frontiers |= remaining

        # 已分配的目标跳过
        assigned = {d.assigned_target for d in self.dogs
                    if d.assigned_target is not None}
        candidates = [f for f in all_frontiers if f not in assigned]

        if not candidates:
            return

        # 每只空闲狗对所有候选出价
        for dog in self.dogs:
            if not dog.alive or dog.assigned_target is not None:
                continue
            best_bid, best_target = -1, None
            for target in candidates:
                bid = dog.bid(target, self.world_size)
                if bid > best_bid:
                    best_bid, best_target = bid, target
            if best_target is not None:
                dog.assigned_target = best_target
                candidates.remove(best_target)

    def handle_failure(self, failed_id):
        """处理个体失效: 重新分配其任务"""
        failed = self.dogs[failed_id]
        failed.alive = False
        failed.assigned_target = None
        self.reassignments += 1
        # 立即触发重新竞拍
        self.auction_targets()

    def simulate_comm_outage(self, duration_steps=10):
        """模拟通信中断"""
        self.outage_remaining = duration_steps

    def step(self, dt=0.5):
        """推进一步"""
        self.step_count += 1

        # 通信中断模拟
        if hasattr(self, 'outage_remaining') and self.outage_remaining > 0:
            self.outage_remaining -= 1
            comm_active = False
        else:
            comm_active = True

        # 通信窗口: 融合地图
        if comm_active:
            links = self.comm_matrix()
            for (i, j), connected in links.items():
                if connected and self.dogs[i].alive and self.dogs[j].alive:
                    self.dogs[i].merge_map(self.dogs[j].share_map())
                    self.dogs[j].merge_map(self.dogs[i].share_map())

        # 竞拍 (通信正常时全局竞拍, 中断时独立决策)
        if comm_active:
            self.auction_targets()
        else:
            for dog in self.dogs:
                if dog.alive and dog.assigned_target is None:
                    frontiers = self.find_frontiers(dog)
                    if frontiers:
                        dog.assigned_target = frontiers[0]

        # 移动
        for dog in self.dogs:
            if not dog.alive:
                continue
            dog.scan(self.blocked, self.world_size)
            dog.move_toward(dog.assigned_target, dt)

            # 检查目标发现
            for i, target in enumerate(self.targets):
                if i not in self.found:
                    dist = np.linalg.norm(dog.pos - np.array(target))
                    if dist < 1.0:
                        self.found.add(i)
                        dog.found_targets.add(i)

        return len(self.found) == len(self.targets)

    def status(self):
        return {
            'step': self.step_count,
            'found': f'{len(self.found)}/{len(self.targets)}',
            'alive': sum(d.alive for d in self.dogs),
            'reassignments': self.reassignments,
            'coverage': sum(len(d.known) for d in self.dogs) / (
                self.world_size * self.world_size),
        }


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    print('=== 多狗协同搜索测试 (10狗起步) ===\n')

    # 规模扫描: 4/6/8/10/12/16 只狗
    for n in (4, 6, 8, 10, 12, 16):
        sim = MultiDogCoordinator(seed=42, n_dogs=n)
        for _ in range(300):
            if sim.step():
                break
        s = sim.status()
        print(f'{n:2}狗: 找到{s["found"]} 存活{s["alive"]} '
              f'覆盖{min(s["coverage"]*100,100):.0f}% '
              f'步数{sim.step_count}')

    # 10狗 + 个体失效 (3只失效)
    print(f'\n--- 10狗 + 3只失效 ---')
    sim2 = MultiDogCoordinator(seed=42, n_dogs=10)
    for i in range(20):
        sim2.step()
    sim2.handle_failure(2)
    sim2.handle_failure(5)
    sim2.handle_failure(8)
    for _ in range(300):
        if sim2.step():
            break
    print(f'10狗失效3只后: {sim2.status()}')

    # 10狗 + 通信中断
    print(f'\n--- 10狗 + 通信中断(50步) ---')
    sim3 = MultiDogCoordinator(seed=42, n_dogs=10)
    for i in range(20):
        sim3.step()
    sim3.simulate_comm_outage(50)
    for _ in range(300):
        if sim3.step():
            break
    print(f'10狗+中断后: {sim3.status()}')

    # 16狗大规模
    print(f'\n--- 16狗大规模 ---')
    sim4 = MultiDogCoordinator(seed=42, n_dogs=16)
    for _ in range(300):
        if sim4.step():
            break
    print(f'16狗: {sim4.status()}')

    print('\nMULTI_DOG_DONE')
