# -*- coding: utf-8 -*-
"""汇总各代实验的成功率: forest_v3 / navigation_residual / rescue 系列"""
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1) forest_v3 首轮
p = 'models/forest_v3_experiment'
for e in ('foot', 'wheel'):
    f = os.path.join(p, e, 'evaluation.json')
    if os.path.exists(f):
        j = json.load(open(f, encoding='utf-8'))
        eps = j if isinstance(j, list) else j.get('episodes', [])
        n = len(eps)
        s = sum(1 for x in eps if x.get('is_success'))
        print(f'forest_v3 {e}: {s}/{n} 成功')

# 2) navigation_residual
p = 'models/navigation_residual'
for e in ('foot', 'wheel'):
    f = os.path.join(p, e, 'evaluation.json')
    bf = os.path.join(p, e, 'baseline.json')
    if os.path.exists(f):
        j = json.load(open(f, encoding='utf-8'))
        eps = j if isinstance(j, list) else j.get('episodes', [])
        n = len(eps)
        s = sum(1 for x in eps if x.get('is_success'))
        pg = [x.get('progress_m', 0) for x in eps]
        print(f'nav_residual {e}: {s}/{n} 成功 | 平均推进 '
              f'{sum(pg)/max(n,1):.2f} m')
        if os.path.exists(bf):
            bj = json.load(open(bf, encoding='utf-8'))
            beps = bj if isinstance(bj, list) else bj.get('episodes', [])
            bn = len(beps)
            bs = sum(1 for x in beps if x.get('is_success'))
            bpg = [x.get('progress_m', 0) for x in beps]
            print(f'  基线(零残差): {bs}/{bn} 成功 | 平均推进 '
                  f'{sum(bpg)/max(bn,1):.2f} m')

# 3) rescue 系列
for d in sorted(os.listdir('models')):
    if not d.startswith('rescue_') or 'smoke' in d:
        continue
    f = os.path.join('models', d, 'summary.json')
    if not os.path.exists(f):
        cands = [x for x in os.listdir(os.path.join('models', d))
                 if x.endswith('.json')]
        f = os.path.join('models', d, cands[0]) if cands else None
    if f:
        try:
            j = json.load(open(f, encoding='utf-8'))
            if isinstance(j, dict):
                keys = list(j.keys())[:8]
                print(f'{d}: {keys}')
        except Exception as ex:
            print(f'{d}: 解析失败 {ex}')
print('SUMMARY_DONE')
