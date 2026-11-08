# -*- coding: utf-8 -*-
"""检查 document.xml 中 TOC 域结构, 定位纯点线残影行"""
import io
import re
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with zipfile.ZipFile(r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.docx') as z:
    doc = z.read('word/document.xml').decode('utf-8')

# TOC 域: begin -> instrText TOC -> separate -> [结果] -> end
m = re.search(r'<w:instrText[^>]*>[^<]*TOC[^<]*</w:instrText>', doc)
print('instrText:', m.group(0) if m else None)
sep = doc.find('separate"', m.start()) if m else -1
end = doc.find('fldCharType="end"', sep)
seg = doc[sep:end]
print('seg len:', len(seg))
print('--- RAW around field ---')
print(doc[max(0, m.start() - 300):end + 2600])
paras = re.findall(r'<w:p[ >].*?</w:p>', seg, re.S)
print('paragraphs:', len(paras))
for i, p in enumerate(paras):
    texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p)
    style = re.search(r'<w:pStyle w:val="([^"]+)"', p)
    tab = 'tab' if 'leader' in p else ''
    print('[{:2d}] style={} {} text={!r}'.format(
        i, style.group(1) if style else '-', tab, ''.join(texts)[:60]))
    if i > 20:
        break
