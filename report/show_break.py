# -*- coding: utf-8 -*-
import io
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with zipfile.ZipFile(r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.docx') as z:
    text = z.read('word/document.xml').decode('utf-8')

# 错误: line 2, column 7935 → 定位附近文本 (line 2 的列 7935)
lines = text.split('\n')
col = 7935
pos = sum(len(l) + 1 for l in lines[:1]) + col
print('总长', len(text))
print('--- 损伤点前后 700 字 ---')
print(text[pos - 500:pos + 200])
