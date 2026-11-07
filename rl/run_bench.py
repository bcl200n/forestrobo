# -*- coding: utf-8 -*-
"""全场景自主基准 (修复后) + blocked 超时敏感性分析"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import json
import numpy as np
from forest_autonomy import ForestAutonomy

SCENARIOS = ('corridor', 'detour', 'slope', 'dropout', 'dynamic')
SEEDS = (100, 101)

print('=== 修复后全场景基准 ===')
print(f'{"场景":10} {"成功":4} {"摔倒":4} {"碰撞":4} {"阻塞":4} 剩余距离')
for scenario in SCENARIOS:
    outcomes = []
    for seed in SEEDS:
        sim = ForestAutonomy(scenario=scenario, seed=seed)
        try:
            for _ in range(sim.env.max_steps):
                done, result = sim.step()
                if done:
                    break
            outcomes.append(result)
        finally:
            sim.close()
    s = sum(r['success'] for r in outcomes)
    f = sum(r['fallen'] for r in outcomes)
    c = sum(r['collision'] for r in outcomes)
    b = sum(r['blocked'] for r in outcomes)
    dist = np.mean([r['distance_m'] for r in outcomes])
    print(f'{scenario:10} {s}/{len(outcomes):2}  {f:2}   {c:2}   {b:2}   {dist:5.2f}m')

# blocked 原因深入: detour 场景轨迹
print('\n=== detour 轨迹采样 (seed=100) ===')
sim = ForestAutonomy('detour', 100)
try:
    for i in range(sim.env.max_steps):
        done, result = sim.step()
        if i % 100 == 0 or done:
            pos = sim.env.data.qpos[:3]
            cmd = sim.env.command if hasattr(sim.env, 'command') else [0, 0, 0]
            print(f'  t={sim.env.data.time:5.1f}s pos=({pos[0]:5.2f},{pos[1]:5.2f}) '
                  f'cmd=({cmd[0]:5.2f},{cmd[1]:5.2f}) dist={result["distance_m"]:5.2f} '
                  f'{"DONE" if done else ""}')
        if done:
            break
finally:
    sim.close()
print('BENCH_DONE')
