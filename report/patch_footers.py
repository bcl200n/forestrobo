# -*- coding: utf-8 -*-
"""后处理: 页脚 PAGE 域格式开关 (ROMAN/arabic) + 清除封面空 pgNumType (WPS 兼容)"""
import io
import re
import shutil
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DOCX = r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.docx'
TMP = DOCX + '.tmp'

with zipfile.ZipFile(DOCX, 'r') as zin:
    names = zin.namelist()
    items = {n: zin.read(n) for n in names}

# 1. document.xml: 按 section 顺序找 footerReference default 的 rId
doc = items['word/document.xml'].decode('utf-8')
doc = doc.replace('<w:pgNumType/>', '')
refs = re.findall(r'<w:footerReference w:type="default" r:id="(rId\d+)"/>', doc)
print('footer default refs (按节序):', refs)

rels = items['word/_rels/document.xml.rels'].decode('utf-8')
rid2file = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(footer\d+\.xml)"', rels))
print('rId -> footer:', rid2file)

# 第一个 default footer = 目录节(ROMAN), 其余 = 正文(arabic)
fmt_map = {}
for i, rid in enumerate(refs):
    f = rid2file.get(rid)
    if not f:
        continue
    fmt_map['word/' + f] = 'ROMAN' if i == 0 else 'arabic'
print('格式映射:', fmt_map)

# 2. 修补各 footer 的 PAGE instrText
for fname, fmt in fmt_map.items():
    xml = items[fname].decode('utf-8')
    new = re.sub(r'(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)',
                 r'\1 PAGE \\* ' + fmt + r' \\* MERGEFORMAT \2', xml)
    if new != xml:
        items[fname] = new.encode('utf-8')
        print('patched {} -> {}'.format(fname, fmt))

items['word/document.xml'] = doc.encode('utf-8')

# 3. TOC 幻影行修复:
#    域首段 PAGEREF 子域结果为空 (separate 紧跟 end) → 渲染为纯点线无页码行。
#    修复 = 给 PAGEREF 补缓存结果 "1", 并删除紧随的第一条重复静态行 (一、执行摘要...1)。
phantom_pat = (r'(<w:instrText xml:space="preserve">PAGEREF _Toc1 \\h</w:instrText>'
               r'<w:fldChar w:fldCharType="separate"/>)(<w:fldChar w:fldCharType="end"/>)')
new_doc, n1 = re.subn(phantom_pat, r'\1<w:t>1</w:t>\2', doc, count=1)
if n1:
    doc = new_doc
    print('PAGEREF 缓存页码已补: 1')
    # 删除第一条静态重复行: 静态行的 <w:t> 无 xml:space 属性, 可与域内超链接行区分
    marker = '<w:t>一、执行摘要</w:t>'
    idx = doc.find(marker)
    if idx > 0:
        p_start = doc.rfind('<w:p>', 0, idx)
        p_end = doc.find('</w:p>', idx) + len('</w:p>')
        if p_start > 0 and p_end > p_start:
            doc = doc[:p_start] + doc[p_end:]
            print('已删除重复静态目录行: 一、执行摘要...1 (精确切片 {}-{})'.format(p_start, p_end))

# XML 良构校验 (防回归: 补丁若破坏结构立即失败)
from xml.etree import ElementTree as ET
try:
    ET.fromstring(doc.encode('utf-8'))
except ET.ParseError as e:
    raise SystemExit('FATAL: patch 破坏了 document.xml: {}'.format(e))

items['word/document.xml'] = doc.encode('utf-8')

# 4. settings.xml: 剥离 updateFields (LibreOffice 导出 PDF 时会重算 TOC 域, 与注入的静态条目叠加产生重复行; Word 中用户仍可手动更新域)
st = items.get('word/settings.xml', b'').decode('utf-8')
st2 = st.replace('<w:updateFields w:val="true"/>', '')
if st2 != st:
    items['word/settings.xml'] = st2.encode('utf-8')
    print('stripped updateFields from settings.xml')

with zipfile.ZipFile(TMP, 'w', zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, items[n])
shutil.move(TMP, DOCX)
print('FOOTER_PATCH_DONE')
