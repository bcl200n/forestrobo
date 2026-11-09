# -*- coding: utf-8 -*-
"""v19 出炉自动验收: 轮询 final.zip → 三方评估 → 绕行考场 → 冠军视频"""
import io
import os
import subprocess
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FINAL = 'models/forest_v19/final.zip'
PY = sys.executable

while not os.path.exists(FINAL):
    time.sleep(120)
print('v19 final detected', flush=True)

steps = [
    ('三方评估 v17/v18/v19', ['eval_v19.py']),
    ('绕行考场', ['render_detour.py', 'models/forest_v19/final']),
    ('巡航视频', ['render_v19.py']),
]
for title, script in steps:
    print('==== ' + title + ' ====', flush=True)
    r = subprocess.run([PY, '-I'] + script, capture_output=True)
    out = (r.stdout + r.stderr).decode('utf-8', errors='replace')
    tail = [ln for ln in out.split('\n') if ('到达' in ln or 'DONE' in ln or 'OK' in ln or '绕行' in ln or 'VIDEO' in ln)]
    print('\n'.join(tail[-12:]), flush=True)
print('AUTO_ACCEPTANCE_DONE', flush=True)
