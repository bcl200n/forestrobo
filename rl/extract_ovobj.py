# -*- coding: utf-8 -*-
"""从奥维地图 .ovobj 二进制中提取林班名称和坐标"""
import math
import os
import struct
import re

path = os.path.join(r'E:\forestrobo\林场范围哈', '雷波林班(1).ovobj')

with open(path, 'rb') as f:
    data = f.read()

print('文件大小: {} bytes'.format(len(data)))

# 提取所有 UTF-8 文本段
texts = []
i = 0
while i < len(data):
    # 尝试解码 UTF-8 连续段
    try:
        start = i
        s = b''
        while i < len(data):
            b = data[i]
            if 0x20 <= b < 0x7f or b >= 0xC0:  # ASCII 可打印或 UTF-8 多字节起始
                s += bytes([b])
                i += 1
            elif 0x80 <= b < 0xC0 and s:  # UTF-8 续字节
                s += bytes([b])
                i += 1
            else:
                break
        if len(s) >= 4:
            try:
                text = s.decode('utf-8')
                if any(c > '\u4e00' for c in text):  # 含中文
                    texts.append((start, text))
            except UnicodeDecodeError:
                pass
    except Exception:
        pass
    i += 1

print('\n找到 {} 段含中文的文本'.format(len(texts)))
for offset, text in texts[:30]:
    text_clean = text.strip()
    if text_clean:
        print('  offset={}: "{}"'.format(offset, text_clean[:60]))

# 搜索数字 212/213/214 附近的浮点数 (可能是坐标)
print('\n搜索 212/213/214 相关数据:')
for target in (212, 213, 214):
    # 搜索字符串 "212" 等
    for encoding in ('utf-8', 'gbk', 'utf-16-le'):
        encoded = str(target).encode(encoding)
        idx = 0
        while True:
            idx = data.find(encoded, idx)
            if idx < 0:
                break
            context = data[max(0, idx - 20):idx + 30]
            # 尝试在附近找浮点数
            nearby_doubles = []
            for off in range(max(0, idx - 200), min(len(data) - 8, idx + 200)):
                try:
                    val = struct.unpack('<d', data[off:off + 8])[0]
                    if 28.0 < val < 29.0:  # 纬度范围
                        nearby_doubles.append(('lat', off, val))
                    elif 103.0 < val < 104.0:  # 经度范围
                        nearby_doubles.append(('lon', off, val))
                except Exception:
                    pass
            if nearby_doubles:
                print('  "{}" at offset {} → 附近坐标: {}'.format(
                    target, idx,
                    ['{}={:.5f}'.format(t, v) for t, _, v in nearby_doubles[:5]]))
            idx += len(encoded)
            if idx > len(data):
                break

# 搜索浮点数对 (lat ~28.x, lon ~103.x)
print('\n扫描文件中的经纬度浮点数对:')
coords_found = []
for i in range(0, len(data) - 8, 4):
    try:
        lat = struct.unpack('<d', data[i:i + 8])[0]
        if 28.0 < lat < 29.0:
            # 在附近找经度
            for j in range(i - 100, min(i + 100, len(data) - 8)):
                if j < 0:
                    continue
                lon = struct.unpack('<d', data[j:j + 8])[0]
                if 103.0 < lon < 104.0:
                    coords_found.append((i, lat, lon))
                    break
    except Exception:
        pass

print('找到 {} 个坐标对'.format(len(coords_found)))
for offset, lat, lon in coords_found[:20]:
    print('  offset={}: lat={:.6f}, lon={:.6f}'.format(offset, lat, lon))

print('\nEXTRACT_DONE')
