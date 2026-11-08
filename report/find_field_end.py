# -*- coding: utf-8 -*-
import io
import re
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with zipfile.ZipFile(r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.docx') as z:
    doc = z.read('word/document.xml').decode('utf-8')

m = re.search(r'<w:instrText[^>]*>[^<]*TOC[^<]*</w:instrText>', doc)
sep = doc.find('fldCharType="separate"', m.start())
# 找 sdtContent 结束前的所有 fldChar end
sdt_end = doc.find('</w:sdtContent>', sep)
seg = doc[sep:sdt_end]
ends = [mm.start() for mm in re.finditer(r'fldCharType="end"', seg)]
print('field separate at', sep, ' sdtContent end at', sdt_end, ' seg len', len(seg))
print('end fldChars in sdt:', len(ends), ' last at seg+', ends[-1] if ends else None)
# 最后一个静态条目 (附录 C) 之后是否紧跟 end?
lastc = seg.rfind('附录 C')
print('附录C at seg+', lastc)
print('--- 附录C 前后 300 字 ---')
print(seg[max(0, lastc - 100):lastc + 500])
