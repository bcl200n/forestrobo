# -*- coding: utf-8 -*-
"""从渲染 PDF 提取各 H1 章节的正文页码, 生成 add_toc_placeholders --entries 参数"""
import io
import json
import sys

import pymupdf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

H1S = ["一、执行摘要", "二、项目背景与规划目标", "三、数据基础与数据源",
       "四、部署规划方法", "五、地形影响分析", "六、电池续航与换电保障",
       "七、机队配置与统计结果", "八、实施节奏与任务体系", "九、风险与限制",
       "十、结论与后续建议", "附录 A 责任区明细表", "附录 B 规划参数汇总",
       "附录 C 交付文件清单"]

pdf = pymupdf.open(r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.pdf')
pages = [pdf[i].get_text() for i in range(len(pdf))]
print('pdf pages:', len(pages))

# 正文起始 pdf 页 = 包含摘要正文首句的页 (目录页不含正文段落)
body_start = None
for i, t in enumerate(pages):
    if '本报告面向雷波林业局' in t:
        body_start = i
        break
print('body starts at pdf page', (body_start + 1) if body_start is not None else None)

entries = []
used = set()
for i in range(body_start, len(pages)):
    for h in H1S:
        if h in pages[i] and h not in used:
            used.add(h)
            entries.append({'level': 1, 'text': h, 'page': str(i - body_start + 1)})

# 按原顺序输出
ordered = [e for h in H1S for e in entries if e['text'] == h]
print(json.dumps(ordered, ensure_ascii=False))
with open(r'E:\forestrobo\report\toc_entries.json', 'w', encoding='utf-8') as f:
    json.dump(ordered, f, ensure_ascii=False)
print('MISSING:', [h for h in H1S if h not in used])
