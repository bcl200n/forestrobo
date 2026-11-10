# -*- coding: utf-8 -*-
"""v20 出炉自动验收: 轮询 final.zip → 冠军对照 (v18 vs v20) → 绕行考场 → 视频"""
import io
import os
import subprocess
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FINAL = 'models/forest_v20/final.zip'
PY = sys.executable

while not os.path.exists(FINAL):
    time.sleep(120)
print('v20 final detected', flush=True)

# 生成 v20 评估与视频脚本
t = open('eval_v18.py', encoding='utf-8').read()
t = (t.replace("{'v17': PPO.load('models/forest_v17/final', device='cpu'),\n          "
               "'v18': PPO.load('models/forest_v18/final', device='cpu')}",
               "{'v18': PPO.load('models/forest_v18/final', device='cpu'),\n          "
               "'v20': PPO.load('models/forest_v20/final', device='cpu')}")
        .replace("for name in ('v17', 'v18'):", "for name in ('v18', 'v20'):")
        .replace('v18_eval.json', 'v20_eval.json')
        .replace('== v18 验收', '== v20 验收 (冠军对照)')
        .replace('V18_EVAL_DONE', 'V20_EVAL_DONE'))
open('eval_v20.py', 'w', encoding='utf-8').write(t)
t = open('render_v16.py', encoding='utf-8').read()
t = (t.replace('models/forest_v16/final', 'models/forest_v20/final')
      .replace('forest_v16_showcase.mp4', 'forest_v20_showcase.mp4')
      .replace('v16 巡航实录', 'v20 前瞻观测巡航实录'))
open('render_v20.py', 'w', encoding='utf-8').write(t)
print('eval_v20.py / render_v20.py written', flush=True)

steps = [
    ('冠军对照 v18 vs v20', ['eval_v20.py']),
    ('绕行考场', ['render_detour.py', 'models/forest_v20/final']),
    ('巡航视频', ['render_v20.py']),
]
for title, script in steps:
    print('==== ' + title + ' ====', flush=True)
    r = subprocess.run([PY, script[0]] + script[1:], capture_output=True)
    out = (r.stdout + r.stderr).decode('utf-8', errors='replace')
    tail = [ln for ln in out.split('\n')
            if ('到达' in ln or 'DONE' in ln or 'OK' in ln or '绕行' in ln or 'VIDEO' in ln)]
    print('\n'.join(tail[-12:]), flush=True)
print('AUTO_ACCEPT_V20_DONE', flush=True)
