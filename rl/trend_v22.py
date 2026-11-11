# -*- coding: utf-8 -*-
import csv
import glob
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

rows = []
for f in glob.glob(r'E:\forestrobo\rl\logs\forest_v22_*.monitor.csv'):
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
print('v22 回合数: {}'.format(n))
if n >= 12:
    q = n // 6
    for k in range(6):
        seg = rows[k * q:(k + 1) * q] if k < 5 else rows[5 * q:]
        print('段{}: 回报 {:+.0f}, 长度 {:.0f}'.format(
            k, sum(x[1] for x in seg) / len(seg), sum(x[2] for x in seg) / len(seg)))
