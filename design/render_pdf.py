# -*- coding: utf-8 -*-
"""PDF → 页面 PNG 渲染（pypdfium2）"""
import os
import sys

import pypdfium2 as pdfium

src = sys.argv[1]
out_dir = os.path.join(os.path.dirname(os.path.abspath(src__ := src)), 'render') \
    if False else os.path.join(os.path.dirname(os.path.abspath(src)), 'render')
os.makedirs(out_dir, exist_ok=True)

pdf = pdfium.PdfDocument(src)
n = len(pdf)
print('总页数:', n)
for i in range(n):
    page = pdf[i]
    w, h = page.get_size()
    scale = 2.0 if w < 700 else 1.5
    bmp = page.render(scale=scale)
    img = bmp.to_pil()
    img.save(os.path.join(out_dir, f'page_{i + 1:02d}.png'))
pdf.close()
print('RENDER_OK', out_dir)
