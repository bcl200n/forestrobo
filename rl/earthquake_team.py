# -*- coding: utf-8 -*-
"""多狗抗震协同: 角色自适应 + 通信中继 + 个体失效降级 + 负载重分配

三种抗震机制:
1. 物理抗摔: 检测倾覆并触发自我恢复序列
2. 角色重组: 领队失效自动选举新领队
3. 通信自愈: 中继节点动态部署保持网络连通
"""
import math
import random
from collections import defaultdict, deque
from enum import Enum
import numpy as np


class FormationType(Enum):
    LINE = 'line'       # 一字队形(窄通道)
    WEDGE = 'wedge'     # 楔形(搜索)
    GRID = 'grid'       # 网格(排障)
    SCATTER = 'scatter' # 散开(地震躲避)


class DogState:
    NORMAL = 'normal'
    FALLEN = 'fallen'
    RECOVERING = 'recovering'
    RELAYING = 'relaying'
    OFFLINE = 'offline'


class EarthquakeDog:
    """抗震搜救狗个体"""

    def __init__(self, dog_id, pos):
        self.id = dog_id
        self.pos = np.array(pos, float)
        self.state = DogState.NORMAL
        self.tilt_deg = 0.0
        self.battery = 100
        self.payload_kg = 0
        self.communiction_ok = True
        self.gps_ok = True
        self.assigned_role = 'searcher'
        self.recovery_timer = 0
        self.msg_queue = deque()
        self.neighbors = set()

    def apply_shock(self, magnitude):
        """模拟地震冲击"""
        # 冲击造成随机倾斜和位移
        self.tilt_deg = random.uniform(0, magnitude * 8)
        self.pos += np.random.normal(0, magnitude * 0.1, 2)
        if self.tilt_deg > 45:
            self.state = DogState.FALLEN
            self.recovery_timer = int(self.tilt_deg / 10)
        elif self.tilt_deg > 20:
            self.state = DogState.RECOVERING
            self.recovery_timer = 2
        # 随机通信损坏
        if random.random() < magnitude * 0.1:
            self.communiction_ok = False

    def recover(self):
        """自我恢复序列"""
        if self.state == DogState.FALLEN or self.state == DogState.RECOVERING:
            self.recovery_timer -= 1
            self.tilt_deg = max(0, self.tilt_deg - 15)
            if self.recovery_timer <= 0:
                self.state = DogState.NORMAL
                self.tilt_deg = 0
                return True  # 恢复完成
        return False


class EarthquakeTeam:
    """多狗抗震协同控制器"""

    def __init__(self, n_dogs=4, area_size=30):
        self.n_dogs = n_dogs
        self.area = area_size
        self.dogs = [EarthquakeDog(i, (i * 3, 0)) for i in range(n_dogs)]
        self.formation = FormationType.WEDGE
        self.leader_id = 0
        self.network_partitions = []
        self.active_earthquake = False
        self.recovery_events = []
        self.role_reassignments = 0

    def elect_leader(self):
        """选丼新领队 (基于电池+状态)"""
        candidates = [d for d in self.dogs
                      if d.state == DogState.NORMAL and d.communiction_ok]
        if not candidates:
            return None
        # 电池最多的当领队
        best = max(candidates, key=lambda d: d.battery)
        old_leader = self.leader_id
        self.leader_id = best.id
        if old_leader != best.id:
            self.role_reassignments += 1
            self.recovery_events.append(
                f'step: leader {old_leader} → {best.id}')
        return best.id

    def deploy_relays(self):
        """部署通信中继修复网络分区"""
        # 找到通信孤岛
        online = [d for d in self.dogs
                  if d.state != DogState.OFFLINE]
        if len(online) < 2:
            return

        # 简单连通性检查
        connected = set()
        queue = [self.leader_id] if any(
            d.id == self.leader_id for d in online) else [online[0].id]
        connected.add(queue[0])
        while queue:
            dog = self.dogs[queue.pop()]
            for other in online:
                if other.id not in connected:
                    dist = np.linalg.norm(dog.pos - other.pos)
                    if dist < 10:
                        connected.add(other.id)
                        queue.append(other.id)

        # 未连接的狗需要中继
        disconnected = [d for d in online if d.id not in connected]
        if disconnected:
            # 指派正常狗作为中继
            for d in disconnected:
                # 找最近已连接的狗
                nearest = min(connected, key=lambda cid: np.linalg.norm(
                    self.dogs[cid].pos - d.pos))
                mid = (self.dogs[nearest].pos + d.pos) / 2
                # 找一只狗去中继位置
                relay_candidates = [d2 for d2 in online
                                    if d2.state == DogState.NORMAL
                                    and d2.assigned_role != 'relay']
                if relay_candidates:
                    relay = relay_candidates[0]
                    relay.assigned_role = 'relay'
                    relay.pos = mid  # 移动到中继位
                    self.recovery_events.append(
                        f'dog {relay.id} → relay at ({mid[0]:.1f},{mid[1]:.1f})')

    def reassign_roles(self):
        """重新分配角色 (搜索/中继/备用)"""
        available = [d for d in self.dogs
                     if d.state == DogState.NORMAL]
        n = len(available)
        if n == 0:
            return
        roles = ['searcher'] * max(1, n - 1) + ['relay'] * (1 if n > 2 else 0)
        # 备用角色给电池最多的
        for i, dog in enumerate(available):
            dog.assigned_role = roles[i % len(roles)]

    def earthquake_step(self, magnitude):
        """地震事件"""
        self.active_earthquake = True
        for dog in self.dogs:
            dog.apply_shock(magnitude)
            if dog.state == DogState.FALLEN:
                self.recovery_events.append(
                    f'dog {dog.id} fallen (tilt={dog.tilt_deg:.0f}°)')

    def step(self):
        """推进一步协同"""
        # 恢复过程
        for dog in self.dogs:
            if dog.state in (DogState.FALLEN, DogState.RECOVERING):
                if dog.recover():
                    self.recovery_events.append(f'dog {dog.id} recovered')

        # 领队选举 (如果领队不在线)
        leader = self.dogs[self.leader_id]
        if leader.state != DogState.NORMAL or not leader.communiction_ok:
            self.elect_leader()

        # 角色重分配
        self.reassign_roles()

        # 通信修复
        self.deploy_relays()

        # 移动逻辑 (简化)
        for dog in self.dogs:
            if dog.state == DogState.NORMAL:
                dog.battery -= 0.005

    def status(self):
        states = {}
        for dog in self.dogs:
            states[dog.id] = {
                'state': dog.state,
                'tilt': round(dog.tilt_deg),
                'battery': round(dog.battery),
                'role': dog.assigned_role,
                'comm': dog.communiction_ok,
            }
        return {
            'leader': self.leader_id,
            'formation': self.formation.value,
            'dogs': states,
            'reassignments': self.role_reassignments,
            'events': self.recovery_events[-5:],
        }


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    print('=== 多狗抗震协同测试 ===\n')

    # 场景: 4狗 + 地震(中等强度) + 恢复
    team = EarthquakeTeam(n_dogs=4)
    print('初始状态:', {k: v['state'] for k, v in team.status()['dogs'].items()})

    # 触发地震
    team.earthquake_step(magnitude=3.0)
    print('\n地震后:')
    for did, info in team.status()['dogs'].items():
        print(f'  狗{did}: {info["state"]} (倾角{info["tilt"]}°) 通信={info["comm"]}')

    # 恢复过程
    for i in range(10):
        team.step()
        if i == 3:
            print(f'\n恢复第4步:')
            for did, info in team.status()['dogs'].items():
                print(f'  狗{did}: {info["state"]} (倾角{info["tilt"]}°) 角色={info["role"]}')

    print(f'\n最终状态: 领队={team.status()["leader"]}')
    for did, info in team.status()['dogs'].items():
        print(f'  狗{did}: {info["state"]} 角色={info["role"]}')
    print(f'  角色重分配: {team.role_reassignments}次')
    print(f'  恢复事件: {team.status()["events"]}')

    # 强震场景
    print('\n=== 强震场景 (magnitude=6) ===')
    team2 = EarthquakeTeam(n_dogs=4)
    team2.earthquake_step(magnitude=6.0)
    fallen = sum(1 for d in team2.dogs if d.state == DogState.FALLEN)
    print(f'倒地: {fallen}/4')
    for i in range(15):
        team2.step()
    recovered = sum(1 for d in team2.dogs if d.state == DogState.NORMAL)
    print(f'15步后恢复: {recovered}/4')
    print(f'领队: {team2.status()["leader"]}')
    print(f'事件数: {len(team2.recovery_events)}')

    print('\nEARTHQUAKE_TEST_DONE')
