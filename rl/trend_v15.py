# -*- coding: utf-8 -*-
"""v15 回合趋势 (logs/forest_v15_*.monitor.csv, 按时间分 8 段)"""
import csv
import glob
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

rows = []
for f in glob.glob(r'E:\forestrobo\rl\logs\forest_v15_*.monitor.csv'):
    with open(f, newline='') as fh:
        rd = csv.reader(fh)
        try:
            next(rd)
            next(rd)
        except StopIteration:
            continue
        for r in rd:
            if len(r) >= 3:
                rows.append((float(r[2]), float(r[0]), int(r[1])))
rows.sort()
n = len(rows)
print('v15 回合数: {}'.format(n))
if n < 16:
    sys.exit()
q = n // 8
for k in range(8):
    seg = rows[k * q:(k + 1) * q] if k < 7 else rows[7 * q:]
    r_mean = sum(x[1] for x in seg) / len(seg)
    l_mean = sum(x[2] for x in seg) / len(seg)
    print('段{}: 回报 {:+.0f}, 长度 {:.0f}'.format(k, r_mean, l_mean))
