# -*- coding: utf-8 -*-
"""直接查看 KMZ 内部 KML 原始内容"""
import zipfile
import os

base = r'E:\forestrobo\林场范围哈'
for fn in os.listdir(base):
    if not fn.endswith('.kmz'):
        continue
    path = os.path.join(base, fn)
    print('=== %s ===' % fn)
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            print('文件列表: %s' % names[:10])
            kmls = [n for n in names if n.endswith('.kml') or n.endswith('.KML')]
            if kmls:
                content = z.read(kmls[0])[:2000]
                print(content.decode('utf-8', errors='replace')[:1000])
            else:
                # 看第一个文件内容
                if names:
                    content = z.read(names[0])[:2000]
                    print('首个文件 (%s):' % names[0])
                    print(content.decode('utf-8', errors='replace')[:800])
    except Exception as e:
        print('错误: %s' % e)
    print()
print('INSPECT_DONE')
