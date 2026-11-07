# -*- coding: utf-8 -*-
"""检查 TOC 区域：sdt / fldSimple / instrText 全量"""
import re
import zipfile

d = zipfile.ZipFile('林下作业机器狗专用机械臂设计方案.docx').read(
    'word/document.xml').decode('utf-8')

print('sdt 块数:', len(re.findall(r'<w:sdt>', d)))
print('fldSimple 数:', len(re.findall(r'<w:fldSimple', d)))
print('instrText 数:', len(re.findall(r'<w:instrText', d)))
print('TOC 字样 instrText:', re.findall(r'<w:instrText[^>]*>([^<]*)</w:instrText>', d)[:5])
i = d.find('TOC1')
print('--- 第一个 TOC1 前 1600 字符 ---')
print(d[max(0, i - 1600):i + 100].replace('><', '>\n<'))
