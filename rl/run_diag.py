# -*- coding: utf-8 -*-
"""detour 三模式快速诊断"""
import sys
sys.path.insert(0, '.')
from diagnose_forest_detours import DiagnosticForest

results = []
for mode in ('observed_map', 'truth_map', 'checked_route'):
    sim = DiagnosticForest('detour', 10110, mode)
    try:
        for _ in range(sim.env.max_steps):
            done, result = sim.step()
            if done:
                break
        results.append((mode, result['success'], result['blocked'],
                        round(result['distance_m'], 2), result.get('replans', 0)))
    finally:
        sim.close()

print('模式              成功   blocked  剩余距离  重规划')
for mode, ok, blk, dist, replans in results:
    print('%16s  %-6s %-7s %8.2f  %d' % (mode, ok, blk, dist, replans))
print('DIAG_DONE')
