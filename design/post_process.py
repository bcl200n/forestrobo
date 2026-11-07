# -*- coding: utf-8 -*-
"""docx 后处理:
1. 向 TOC 的 sdt 块注入完整域字符链 (begin + instrText + separate ... end)
2. 按节修补页脚 PAGE 域格式开关 (罗马/阿拉伯) — WPS 兼容
3. 移除封面节空 <w:pgNumType/>
"""
import re
import shutil
import sys
import zipfile

src = sys.argv[1]
tmp = src + '.tmp'

with zipfile.ZipFile(src, 'r') as z:
    names = z.namelist()
    data = {n: z.read(n) for n in names}

doc = data['word/document.xml'].decode('utf-8')

# 0) 向 TOC sdt 注入域字符链（若缺失）
if 'fldCharType="begin"' not in doc:
    # 先清理孤儿 end 段（此前误删 begin 后残留）
    doc = doc.replace('<w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>', '', 1)
    m = re.search(r'(<w:sdt><w:sdtPr>(?:(?!</w:sdtPr>).)*?Table of Contents'
                  r'(?:(?!</w:sdtPr>).)*?</w:sdtPr>)<w:sdtContent>', doc, re.S)
    if m:
        field_start = (m.group(0) +
                       '<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
                       '<w:r><w:instrText xml:space="preserve">'
                       ' TOC \\o &quot;1-3&quot; \\h \\z \\u '
                       '</w:instrText></w:r>'
                       '<w:r><w:fldChar w:fldCharType="separate"/></w:r></w:p>')
        doc = doc[:m.start()] + field_start + doc[m.end():]
        # 在 sdtContent 结束前插入 end
        e2 = doc.find('</w:sdtContent>', m.start())
        doc = (doc[:e2] + '<w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
               + doc[e2:])
        print('已注入 TOC 域字符链 (begin/instrText/separate/end)')

# 1) 移除空 pgNumType
doc = re.sub(r'<w:pgNumType/>', '', doc)

# 2) 按出现顺序找 sectPr, 提取 default footerReference r:id
sects = re.findall(r'<w:sectPr[ >].*?</w:sectPr>', doc, re.S)
print('节数量:', len(sects))
rels = data['word/_rels/document.xml.rels'].decode('utf-8')
rid2file = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(footer\d+\.xml)"', rels))

sect_footers = []
for i, s in enumerate(sects):
    m = re.search(r'w:footerReference[^>]*w:type="default"[^>]*r:id="(rId\d+)"', s)
    sect_footers.append(rid2file.get(m.group(1)) if m else None)
    print(f'  节{i + 1}: footer = {sect_footers[-1]}')

# 节 2 → ROMAN, 节 3 → arabic（1 节封面无页脚）
fmt_map = {}
if len(sect_footers) >= 3:
    if sect_footers[1]:
        fmt_map['word/' + sect_footers[1]] = 'ROMAN'
    if sect_footers[2] and sect_footers[2] != sect_footers[1]:
        fmt_map['word/' + sect_footers[2]] = 'arabic'

for fname, fmt in fmt_map.items():
    xml = data[fname].decode('utf-8')
    xml2 = re.sub(r'(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)',
                  r'\g<1> PAGE \\* ' + fmt + r' \\* MERGEFORMAT \g<2>', xml)
    data[fname] = xml2.encode('utf-8')
    print(f'  已修补 {fname} -> {fmt}')

data['word/document.xml'] = doc.encode('utf-8')

with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in names:
        z.writestr(n, data[n])
shutil.move(tmp, src)
print('POSTPROCESS_OK')
