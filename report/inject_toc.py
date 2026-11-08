# -*- coding: utf-8 -*-
"""调用 add_toc_placeholders.py 注入真实页码目录条目"""
import io
import json
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open(r'E:\forestrobo\report\toc_entries.json', encoding='utf-8') as f:
    entries = json.load(f)

r = subprocess.run([
    sys.executable,
    r'C:\Users\user\.zcode\cli\plugins\cache\zcode-plugins-official\documents\0.2.1\skills\docx\scripts\add_toc_placeholders.py',
    r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.docx',
    '--entries', json.dumps(entries, ensure_ascii=False),
], capture_output=True)
sys.stdout.write(r.stdout.decode('utf-8', errors='replace'))
sys.stderr.write(r.stderr.decode('utf-8', errors='replace'))
print('EXIT', r.returncode)
