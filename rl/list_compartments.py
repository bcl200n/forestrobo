# -*- coding: utf-8 -*-
"""列出所有林班编号和属性"""
import os
import re

path = os.path.join(r'E:\forestrobo\林场范围哈', '雷波林班(1).ovobj')
with open(path, 'rb') as f:
    data = f.read()
text = data.decode('utf-8', errors='replace')

# 提取每个林班的完整属性
comps = {}
for m in re.finditer(r'(\d+)林班', text):
    num = int(m.group(1))
    if num in comps:
        continue
    # 找附近的 <td> 属性
    ctx = text[m.start():m.start() + 500]
    attrs = re.findall(r'<td>([^<]+)</td>', ctx)
    # 过滤掉标签名本身
    vals = [a for a in attrs if a not in ('局', '场', '作业区', '林班', '名称', '面积')]
    if vals:
        comps[num] = vals

print('=== 雷波林班列表 (共 {} 个) ===\n')
print('编号  属性')
print('-' * 50)
for num in sorted(comps):
    attrs = comps[num]
    area = ''
    for a in attrs:
        if any(c.isdigit() for c in a) and ('亩' in a or '公顷' in a or 'm²' in a or '.' in a):
            area = a
            break
    if not area and len(attrs) > 0:
        area = attrs[0]
    print('{:>4}   {}'.format(num, area[:40]))

# 完整显示第一个林班的属性作为样例
print('\n=== 1林班 属性详情 ===')
m = re.search(r'1林班', text)
if m:
    ctx = text[m.start():m.start() + 1000]
    attrs = re.findall(r'<td>([^<]+)</td>', ctx)
    for i in range(0, len(attrs), 2):
        if i + 1 < len(attrs):
            print('  {}: {}'.format(attrs[i], attrs[i + 1]))
