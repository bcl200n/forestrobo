# -*- coding: utf-8 -*-
"""验证部署演示页占位符已全部替换且 JS 数据完整"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

h = open(r'E:\forestrobo\rl\forest_deployment_demo.html', encoding='utf-8').read()
ph = re.findall(r'__[A-Z]+__', h)
print('未替换占位符:', ph if ph else '无')
print('COMPS:', h.count('{f:"'), ' ZONES:', h.count('{id:"'))
print('标题:', re.search(r'<title>(.*?)</title>', h).group(1))
# 基本语法检查: 大括号配对
print('braces balance:', h.count('{') - h.count('}'))
print('VERIFY_DONE')
