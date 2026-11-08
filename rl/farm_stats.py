# -*- coding: utf-8 -*-
"""按林场统计机队需求"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open(r'E:\forestrobo\rl\deployment_plan.json', encoding='utf-8') as f:
    plan = json.load(f)

zones = plan['zones']
farms = {}
for z in zones:
    f = farms.setdefault(z['farm'], {
        'dogs': 0, 'comps': 0, 'mu': 0.0, 'km': 0.0, 'h': 0.0,
        'sorties': 0, 'swaps': 0, 'wh': 0, 'max_cycle': 0, 'zones': []})
    f['dogs'] += 1
    f['comps'] += z['n']
    f['mu'] += z['mu']
    f['km'] += z['km']
    f['h'] += z['hours']
    f['sorties'] += z['sorties']
    f['swaps'] += z['swaps']
    f['wh'] += z['wh']
    f['max_cycle'] = max(f['max_cycle'], z['cycle'])
    f['zones'].append(z['id'])

hdr = ('林场', '狗', '林班', '万亩', '均坡°', '路线km', '地形h', '出勤段', '换电', 'kWh', '周期d')
print('{:<6}{:>5}{:>6}{:>8}{:>8}{:>9}{:>9}{:>8}{:>7}{:>8}{:>8}'.format(*hdr))
print('-' * 82)
farm_slope = {f: m['slope'] for f, m in plan['meta']['farms'].items()}
tot = [0] * 9
for farm in sorted(farms):
    d = farms[farm]
    print('{:<6}{:>5}{:>6}{:>8.2f}{:>8.1f}{:>9.1f}{:>9.1f}{:>8}{:>7}{:>8.1f}{:>8}'.format(
        farm, d['dogs'], d['comps'], d['mu'] / 10000, farm_slope.get(farm, 0),
        d['km'], d['h'], d['sorties'], d['swaps'], d['wh'] / 1000, d['max_cycle']))
    tot[0] += d['dogs']; tot[1] += d['comps']; tot[2] += d['mu']
    tot[3] += d['km']; tot[4] += d['h']; tot[5] += d['sorties']
    tot[6] += d['swaps']; tot[7] += d['wh']
print('-' * 82)
print('{:<6}{:>5}{:>6}{:>8.2f}{:>8}{:>9.1f}{:>9.1f}{:>8}{:>7}{:>8.1f}{:>8}'.format(
    '合计', tot[0], tot[1], tot[2] / 10000, '', tot[3], tot[4], tot[5], tot[6], tot[7] / 1000,
    max(f['max_cycle'] for f in farms.values())))

print('\n=== 各场责任区明细 ===')
for farm in sorted(farms):
    print('{}场 ({} 只): {}'.format(farm, farms[farm]['dogs'], ', '.join(farms[farm]['zones'])))

print('\n=== 配套需求 ===')
print('电池组: {} 狗 × 2 组 = {} 组 1500Wh (轮换充电)'.format(
    tot[0], tot[0] * 2))
print('补给站: {} 个 (每责任区 1 个)'.format(tot[0]))
print('日充电量: 每循环 {:.0f} kWh / 周期 ≤3 天 → 日均 {:.0f} kWh'.format(
    tot[7] / 1000, tot[7] / 1000 / 3))
