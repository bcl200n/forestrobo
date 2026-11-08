# -*- coding: utf-8 -*-
"""检查所有 H1/H2 标题是否孤悬页底 (标题是页面最后一个正文行)"""
import io
import re
import sys

import pymupdf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HEADINGS = ['一、执行摘要', '二、项目背景与规划目标', '三、数据基础与数据源',
            '四、部署规划方法', '五、地形影响分析', '六、电池续航与换电保障',
            '七、机队配置与统计结果', '八、实施节奏与任务体系', '九、风险与限制',
            '十、结论与后续建议', '附录 A 责任区明细表', '附录 B 规划参数汇总',
            '附录 C 交付文件清单',
            '2.1 项目背景', '2.2 规划目标与范围', '4.1 责任区划分', '4.2 补给站选址',
            '4.3 巡护路线优化', '6.1 能耗模型', '6.2 出勤切段策略']

doc = pymupdf.open(r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.pdf')
bad = 0
for i in range(2, len(doc)):  # 跳过封面/目录
    lines = [ln.strip() for ln in doc[i].get_text().split('\n') if ln.strip()]
    # 去掉页脚页码行 (纯数字或罗马数字在末尾)
    while lines and re.fullmatch(r'[0-9IVX]+', lines[-1]):
        lines.pop()
    if not lines:
        continue
    last = lines[-1]
    for h in HEADINGS:
        if last.startswith(h) and len(last) <= len(h) + 2:
            print('孤悬页底: p{:02d} 末行= {}'.format(i + 1, last))
            bad += 1
print('ORPHANS:', bad)
print('PAGES:', len(doc))
