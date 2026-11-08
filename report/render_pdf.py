# -*- coding: utf-8 -*-
import io
import os
import sys

import pymupdf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

doc = pymupdf.open(r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.pdf')
print('pages:', len(doc))
out_dir = r'E:\forestrobo\report\render'
os.makedirs(out_dir, exist_ok=True)
for i in range(len(doc)):
    pix = doc[i].get_pixmap(dpi=72)
    pix.save(os.path.join(out_dir, 'p{:02d}.png'.format(i + 1)))
print('RENDER_DONE')
