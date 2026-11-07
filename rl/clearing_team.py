# -*- coding: utf-8 -*-
"""多狗协同清障: 角色分配 + 协同推拉力学 + 阻力评估

三种清障模式:
1. 单狗推: 轻型障碍 (木头/小石)
2. 双狗抬: 中型障碍 (混凝土板)
3. 三狗协同: 重型障碍 (倒塌梁)
"""
import math
import random
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ObstacleType(Enum):
    LIGHT = 'light'    # <20kg, 单狗可推
    MEDIUM = 'medium'  # 20-80kg, 需双狗
    HEAVY = 'heavy'    # >80kg, 需三狗
    IMMOVABLE = 'immovable'  # 无法移动


class Obstacle:
    def __init__(self, obs_id, pos, obs_type, mass_kg, friction=0.6):
        self.id = obs_id
        self.pos = np.array(pos, float)
        self.type = obs_type
        self.mass = mass_kg
        self.friction = friction
        self.cleared = False

    @property
    def force_needed(self):
        """移动所需推力 (N)"""
        return self.mass * 9.81 * self.friction

    @property
    def dogs_needed(self):
        """需要的狗数"""
        single_dog_force = 150  # N (单狗最大推力)
        n = math.ceil(self.force_needed / single_dog_force)
        return min(n, 3)


@dataclass
class ClearTask:
    obstacle: Obstacle
    assigned_dogs: list = field(default_factory=list)
    phase: str = 'pending'  # pending → approaching → positioning → pushing → cleared
    progress: float = 0.0


class ClearingDog:
    def __init__(self, dog_id, pos, max_push_force=150, max_lift_force=80):
        self.id = dog_id
        self.pos = np.array(pos, float)
        self.max_push = max_push_force
        self.max_lift = max_lift_force
        self.battery = 100
        self.current_task = None
        self.state = 'idle'  # idle / moving / pushing / lifting

    def move_to(self, target, dt=0.1, speed=0.5):
        direction = np.array(target) - self.pos
        dist = np.linalg.norm(direction)
        if dist < 0.3:
            return True
        self.pos += direction / dist * speed * dt
        self.battery -= 0.02 * dt
        return False


class ClearingTeam:
    """多狗清障协同控制器"""

    def __init__(self, obstacles_spec, dog_positions=None):
        self.obstacles = []
        for spec in obstacles_spec:
            self.obstacles.append(Obstacle(*spec))
        self.tasks = []
        self.dogs = []
        n_dogs = max(3, max((o.dogs_needed for o in self.obstacles), default=3))
        if dog_positions is None:
            dog_positions = [(0, i * 2) for i in range(n_dogs)]
        self.dogs = [ClearingDog(i, pos) for i, pos in
                     zip(range(len(dog_positions)), dog_positions)]
        self.total_cleared = 0
        self.total_steps = 0

    def assess_and_assign(self):
        """评估障碍并分配任务"""
        pending = [o for o in self.obstacles if not o.cleared]
        # 按需要狗数排序 (多狗的先处理, 因为占资源多)
        pending.sort(key=lambda o: -o.dogs_needed)

        for obstacle in pending:
            n_needed = obstacle.dogs_needed
            if n_needed > len(self.dogs):
                obstacle.type = ObstacleType.IMMOVABLE
                continue

            # 找空闲狗, 按距离排序
            idle = [d for d in self.dogs
                    if d.current_task is None and d.battery > 20]
            idle.sort(key=lambda d: np.linalg.norm(d.pos - obstacle.pos))

            if len(idle) >= n_needed:
                assigned = idle[:n_needed]
                task = ClearTask(obstacle, [d.id for d in assigned])
                for d in assigned:
                    d.current_task = task
                self.tasks.append(task)

    def step(self, dt=0.1):
        """推进一步清障"""
        self.total_steps += 1

        # 评估与分配
        self.assess_and_assign()

        # 执行任务
        for task in self.tasks:
            if task.phase == 'cleared':
                continue

            obstacle = task.obstacle
            dogs = [self.dogs[i] for i in task.assigned_dogs
                    if self.dogs[i].current_task == task]

            if not dogs:
                task.phase = 'pending'
                continue

            # 阶段1: 接近
            if task.phase in ('pending',):
                task.phase = 'approaching'
            elif task.phase == 'approaching':
                all_arrived = True
                for d in dogs:
                    arrived = d.move_to(obstacle.pos - np.array([1, 0]), dt)
                    if not arrived:
                        all_arrived = False
                    d.state = 'moving'
                if all_arrived:
                    task.phase = 'positioning'

            # 阶段2: 定位 (围绕障碍分布)
            elif task.phase == 'positioning':
                n = len(dogs)
                for i, d in enumerate(dogs):
                    angle = math.pi * i / max(n - 1, 1)
                    target = obstacle.pos - np.array(
                        [math.cos(angle), math.sin(angle)]) * 0.5
                    d.move_to(target, dt)
                task.phase = 'pushing'

            # 阶段3: 推/抬
            elif task.phase == 'pushing':
                total_force = sum(d.max_push for d in dogs)
                if total_force >= obstacle.force_needed:
                    # 可以移动
                    move_speed = 0.05 * (total_force /
                                          obstacle.force_needed)
                    obstacle.pos += np.array([move_speed, 0])
                    task.progress = min(1.0, task.progress + move_speed * 0.1)
                    for d in dogs:
                        d.state = 'pushing'
                        d.battery -= 0.05 * dt
                    if task.progress >= 1.0:
                        obstacle.cleared = True
                        task.phase = 'cleared'
                        self.total_cleared += 1
                        for d in dogs:
                            d.current_task = None
                            d.state = 'idle'
                else:
                    # 力不足, 尝试加入更多狗
                    task.phase = 'pending'
                    for d in dogs:
                        d.current_task = None
                        d.state = 'idle'

        # 检查是否全部清除
        return all(o.cleared or o.type == ObstacleType.IMMOVABLE
                   for o in self.obstacles)

    def status(self):
        return {
            'step': self.total_steps,
            'cleared': f'{self.total_cleared}/{len(self.obstacles)}',
            'dog_states': {d.id: d.state for d in self.dogs},
            'battery': {d.id: round(d.battery) for d in self.dogs},
            'active_tasks': sum(1 for t in self.tasks
                                if t.phase != 'cleared'),
        }


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    print('=== 多狗协同清障测试 ===\n')

    # 场景: 5个障碍, 不同重量
    obstacles = [
        (0, (5, 0), ObstacleType.LIGHT, 15),     # 木头, 单狗
        (1, (8, 2), ObstacleType.MEDIUM, 50),    # 混凝土块, 双狗
        (2, (12, -1), ObstacleType.HEAVY, 120),  # 倒塌梁, 三狗
        (3, (15, 3), ObstacleType.LIGHT, 10),    # 碎石, 单狗
        (4, (18, 0), ObstacleType.MEDIUM, 60),   # 板材, 双狗
    ]

    team = ClearingTeam(obstacles, dog_positions=[(0, i * 2) for i in range(4)])
    print(f'障碍评估:')
    for obs in team.obstacles:
        print(f'  #{obs.id} {obs.type.value:8} {obs.mass:5.0f}kg '
              f'→ 需{obs.dogs_needed}狗 力{obs.force_needed:.0f}N')

    print(f'\n狗群: {len(team.dogs)}只, 最大推力{team.dogs[0].max_push}N/只')

    for step_i in range(200):
        if team.step(dt=0.5):
            break
        if step_i % 40 == 0:
            print(f'  step {step_i}: {team.status()["cleared"]} '
                  f'活跃任务{team.status()["active_tasks"]}')

    print(f'\n最终: {team.status()}')

    # 场景2: 资源不足 (只有2狗, 有3狗障碍)
    print('\n=== 资源不足场景 (2狗 vs 3狗障碍) ===')
    obstacles2 = [
        (0, (3, 0), ObstacleType.HEAVY, 100),
        (1, (6, 2), ObstacleType.LIGHT, 10),
    ]
    team2 = ClearingTeam(obstacles2, dog_positions=[(0, 0), (0, 2)])
    for step_i in range(50):
        if team2.step():
            break
    print(f'结果: {team2.status()["cleared"]}')
    for o in team2.obstacles:
        if o.type == ObstacleType.IMMOVABLE:
            print(f'  障碍#{o.id} ({o.mass}kg) 被标记为不可移动(狗力不足)')

    print('\nCLEARING_TEST_DONE')
