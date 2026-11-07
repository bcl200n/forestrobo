# -*- coding: utf-8 -*-
"""解析林场 KMZ 文件"""
import json
import os
import sys
sys.path.insert(0, '.')

from forest_env import parse_kmz, summary, features_to_grid

base = r'E:\forestrobo\林场范围哈'
for fn in os.listdir(base):
    if not fn.endswith('.kmz'):
        continue
    path = os.path.join(base, fn)
    print('=' * 50)
    print('文件: %s' % fn)
    print('=' * 50)
    try:
        features = parse_kmz(path)
        summary(features)
        grid = features_to_grid(features)
        open_cnt = sum(1 for row in grid['grid'] for c in row if not c)
        print('网格: %dx%d 可通行 %d/%d 格' % (
            grid['size'], grid['size'], open_cnt, grid['size'] ** 2))
        out = fn.replace('.kmz', '_grid.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump({
                'grid': grid['grid'],
                'bounds': grid['bounds'],
                'points': grid['points'][:20],
            }, f, ensure_ascii=False, indent=2)
        print('已输出: %s' % out)
    except Exception as e:
        print('解析失败: %s' % e)
print('PARSE_DONE')
