# -*- coding: utf-8 -*-
"""从 monitor CSV 看各 chunk 的回合回报/长度趋势 (纯 csv, 无 pandas)"""
import csv
import glob
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

for ci in range(4):
    rows = []
    for f in glob.glob(r'E:\forestrobo\rl\logs\forest_v14_{}_*.monitor.csv'.format(ci)):
        with open(f, newline='') as fh:
            rd = csv.reader(fh)
            next(rd)  # JSON 元数据行
            next(rd)  # 表头行
            for r in rd:
                if len(r) >= 3:
                    rows.append((float(r[2]), float(r[0]), int(r[1])))  # t, r, l
    if not rows:
        continue
    rows.sort()
    n = len(rows)
    if n < 24:
        print('chunk {}: 仅 {} 回合, 跳过'.format(ci, n))
        continue
    q = n // 6
    print('chunk {} ({} 回合):'.format(ci, n))
    for k in range(6):
        seg = rows[k * q:(k + 1) * q] if k < 5 else rows[5 * q:]
        r_mean = sum(x[1] for x in seg) / len(seg)
        l_mean = sum(x[2] for x in seg) / len(seg)
        print('  段{}: 回报 {:.0f}, 长度 {:.0f}'.format(k, r_mean, l_mean))
