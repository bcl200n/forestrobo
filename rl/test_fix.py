# -*- coding: utf-8 -*-
"""测试 detour 修复效果"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from forest_autonomy import ForestAutonomy

for sc in ('detour', 'dynamic'):
    for seed in (100, 101):
        sim = ForestAutonomy(scenario=sc, seed=seed)
        try:
            for _ in range(sim.env.max_steps):
                done, r = sim.step()
                if done:
                    break
            if r['success']:
                status = 'PASS'
            elif r['blocked']:
                status = 'FAIL(blocked)'
            elif r['fallen']:
                status = 'FAIL(fallen)'
            elif r['collision']:
                status = 'FAIL(collision)'
            else:
                status = 'FAIL(timeout)'
            print('%s/%d: %s dist=%.2f' % (sc, seed, status, r['distance_m']))
        finally:
            sim.close()
print('FIX_TEST_DONE')
