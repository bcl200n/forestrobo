# -*- coding: utf-8 -*-
"""一键构建链: 生成 docx -> 分页探测 -> 注入目录 -> 页脚补丁 -> postcheck -> 终版 PDF -> 渲染验证"""
import io
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

NODE = r'E:\forestrobo\tools\node-v22.23.4-win-x64\node.exe'
PY = sys.executable
SOFFICE = r'E:\forestrobo\tools\libreoffice\program\soffice.exe'
DOCX = '雷波林业局机器狗巡护部署方案报告.docx'
SKILL = r'C:\Users\user\.zcode\cli\plugins\cache\zcode-plugins-official\documents\0.2.1\skills\docx\scripts'


def run(title, *cmd, env=None, check=True):
    print('==== ' + title + ' ====')
    r = subprocess.run(list(cmd), capture_output=True, env=env, cwd=r'E:\forestrobo\report')
    out = (r.stdout + r.stderr).decode('utf-8', errors='replace')
    print(out[-1800:] if len(out) > 1800 else out)
    if check and r.returncode != 0:
        print('FAILED at ' + title)
        sys.exit(1)
    return r


import os
env = dict(os.environ, NODE_PATH=r'E:\forestrobo\design\node_modules')

run('1. generate.js', NODE, 'generate.js', env=env)
run('2. pdf#1 (分页探测)', SOFFICE, '--headless', '--convert-to', 'pdf', '--outdir', '.', DOCX)
run('3. 提取章节页码', PY, '-I', 'toc_pages.py')
run('4. 注入目录条目', PY, '-X', 'utf8', '-I', 'inject_toc.py')
run('5. 页脚补丁+剥离updateFields', PY, '-X', 'utf8', '-I', 'patch_footers.py')
run('6. postcheck', PY, '-X', 'utf8', os.path.join(SKILL, 'postcheck.py'), DOCX)
run('7. 终版 PDF', SOFFICE, '--headless', '--convert-to', 'pdf', '--outdir', '.', DOCX)
run('8. 渲染', PY, '-I', 'render_pdf.py')
run('9. 目录页文本核验', PY, '-I', 'check_toc_text.py')
print('CHAIN_DONE')
