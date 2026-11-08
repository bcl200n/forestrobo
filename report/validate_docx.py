# -*- coding: utf-8 -*-
"""验证 docx zip 完整性与 XML 良构性"""
import io
import sys
import zipfile
from xml.etree import ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

P = r'E:\forestrobo\report\雷波林业局机器狗巡护部署方案报告.docx'
try:
    zf = zipfile.ZipFile(P)
    bad = zf.testzip()
    print('zip ok, bad entry:', bad)
    for n in zf.namelist():
        if n.endswith('.xml') or n.endswith('.rels'):
            data = zf.read(n)
            try:
                ET.fromstring(data)
            except ET.ParseError as e:
                print('XML ERROR in {}: {}'.format(n, e))
                # 打印错误位置附近
                text = data.decode('utf-8', errors='replace')
                print('  ...len={}'.format(len(text)))
    print('XML_VALIDATE_DONE')
except Exception as e:
    print('FATAL:', e)
