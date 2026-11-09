# -*- coding: utf-8 -*-
"""v19 巡航视频 (render_v16 模板: v17 模型 → v19)"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
t = open('render_v16.py', encoding='utf-8').read()
t = t.replace('models/forest_v16/final', 'models/forest_v19/final')
t = t.replace('forest_v16_showcase.mp4', 'forest_v19_showcase.mp4')
t = t.replace('v16 巡航实录', 'v19 门控双专家巡航实录')
open('render_v19.py', 'w', encoding='utf-8').write(t)
print('render_v19.py written')
