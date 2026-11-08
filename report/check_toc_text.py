# -*- coding: utf-8 -*-
import io
import sys

import pymupdf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

doc = pymupdf.open(r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.pdf')
t = doc[1].get_text()
lines = [ln for ln in t.split('\n') if ln.strip()]
for ln in lines:
    print(repr(ln))
