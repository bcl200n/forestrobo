# -*- coding: utf-8 -*-
"""按最新 PDF 实测页码修补 TOC 占位条目 + 删除无页码的重复首条目"""
import re
import shutil
import sys
import zipfile

import pypdfium2 as pdfium

docx_path = sys.argv[1]
pdf_path = sys.argv[2]

# 1) 实测: 标题(去空格) → 正文页码 (正文从 PDF 第 4 页起 = 正文第 1 页)
pdf = pdfium.PdfDocument(pdf_path)
page_txt = [re.sub(r'\s+', '', pdf[i].get_textpage().get_text_bounded())
            for i in range(len(pdf))]
pdf.close()

def norm(s):
    return re.sub(r'\s+', '', s)

with zipfile.ZipFile(docx_path) as z:
    names = z.namelist()
    data = {n: z.read(n) for n in names}
doc = data['word/document.xml'].decode('utf-8')

paras = re.findall(r'<w:p\b.*?</w:p>', d_ := doc, re.S)
fixed, removed = 0, 0

# 先收集全部 TOC 样式条目：key -> [(有页码数字?, 段落)]
entries = []
for m in re.finditer(r'<w:p\b.*?</w:p>', doc, re.S):
    p_ = m.group(0)
    if not re.search(r'w:pStyle w:val="TOC\d"', p_):
        continue
    texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p_)
    if not texts:
        continue
    has_digit = any(t.strip().isdigit() for t in texts[1:])
    entries.append((norm(texts[0]), has_digit, p_))

# 同标题多条目时，删除无页码数字的条目；若段内含 TOC 主域字符（begin/TOC instrText/separate），
# 先把它们转移到下一条目段落开头，避免破坏域链
from collections import Counter
key_ok = {k for k, h, _ in entries if h}
dup_pages = [p_ for k, h, p_ in entries if not h and k in key_ok]

MAIN_FIELD_RUN = re.compile(
    r'<w:r>(?:(?!</w:r>).)*?(?:fldCharType="begin"|fldCharType="separate"|'
    r'<w:instrText[^>]*>[^<]*TOC[^<]*</w:instrText>)(?:(?!</w:r>).)*?</w:r>', re.S)

for p_ in dup_pages:
    field_runs = MAIN_FIELD_RUN.findall(p_)
    doc = doc.replace(p_, '', 1)
    if field_runs:
        # 插入到删除段之后的第一个 TOC 条目段落开头
        mnext = re.search(r'<w:p\b(?:(?!</w:p>).)*?w:pStyle w:val="TOC\d"(?:(?!</w:p>).)*?</w:p>',
                          doc, re.S)
        if mnext:
            nxt = mnext.group(0)
            head = ''.join(field_runs)
            doc = doc.replace(nxt, head + nxt, 1)
    removed += 1

# 页码修正
for m in re.finditer(r'<w:p\b.*?</w:p>', doc, re.S):
    p_ = m.group(0)
    if not re.search(r'w:pStyle w:val="TOC\d"', p_):
        continue
    texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p_)
    if not texts:
        continue
    key = norm(texts[0])
    actual = None
    for i, t in enumerate(page_txt):
        if i >= 3 and key in t:
            actual = i + 1 - 3
            break
    if actual is not None and len(texts) >= 2 and texts[-1].strip().isdigit():
        if str(actual) != texts[-1].strip():
            new_p = re.sub(r'(<w:t[^>]*>)' + re.escape(texts[-1]) + r'(</w:t>)(?!.*<w:t)',
                           r'\g<1>' + str(actual) + r'\g<2>', p_, flags=re.S)
            doc = doc.replace(p_, new_p, 1)
            fixed += 1

data['word/document.xml'] = doc.encode('utf-8')
tmp = docx_path + '.tmp'
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in names:
        z.writestr(n, data[n])
shutil.move(tmp, docx_path)
print(f'TOC_PATCH_OK: 修正页码 {fixed} 条, 删除重复条目 {removed} 条')
