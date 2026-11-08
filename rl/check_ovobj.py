# -*- coding: utf-8 -*-
"""检查 ovobj 文件格式"""
import os
import zipfile

path = os.path.join(r'E:\forestrobo\林场范围哈', '雷波林班(1).ovobj')
size = os.path.getsize(path)
print('文件: 雷波林班(1).ovobj')
print('大小: {} bytes ({:.1f} KB)'.format(size, size / 1024))

with open(path, 'rb') as f:
    header = f.read(32)
print('文件头 hex:', header[:16].hex())
print('文件头 ascii:', header[:16].decode('ascii', errors='replace'))

# 尝试作为 ZIP 打开
try:
    with zipfile.ZipFile(path, 'r') as z:
        names = z.namelist()
        print('是 ZIP 文件! 包含:')
        for n in names[:20]:
            print('  {}'.format(n))
        # 读取第一个文件
        if names:
            content = z.read(names[0])[:2000]
            print('首个文件内容预览:')
            print(content.decode('utf-8', errors='replace')[:1000])
except Exception as e:
    print('不是 ZIP: {}'.format(e))
    # 尝试读取更多内容找文本
    with open(path, 'rb') as f:
        content = f.read(5000)
    # 搜索数字 212, 213, 214
    text = content.decode('utf-8', errors='replace')
    for keyword in ('212', '213', '214', '林班', 'polygon', 'Polygon', '坐标', 'coordinate'):
        idx = text.find(keyword)
        if idx >= 0:
            print('找到 "{}" at offset {}: ...{}...'.format(
                keyword, idx, text[max(0, idx - 20):idx + 50]))
