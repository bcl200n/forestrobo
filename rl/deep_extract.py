# -*- coding: utf-8 -*-
"""深度解析 ovobj: 提取林班编号和坐标"""
import os
import re
import struct

path = os.path.join(r'E:\forestrobo\林场范围_Ha'.replace('_Ha', '\u54c8'),
                    '雷波林班(1).ovobj')
with open(path, 'rb') as f:
    data = f.read()

# 1. 提取所有林班编号
print('=== 林班列表 ===')
text = data.decode('utf-8', errors='replace')
# 找所有 "N林班" 模式
comps = sorted(set(re.findall(r'(\d+)林班', text)), key=int)
print('林班编号: {}'.format(', '.join(comps[:30])))
print('共 {} 个林班'.format(len(comps)))

# 2. 找 212, 213, 214
for target in ('212', '213', '214'):
    pattern = r'{}林班'.format(target)
    if re.search(pattern, text):
        print('\n✓ 找到 {}林班'.format(target))
        # 找上下文
        for m in re.finditer(pattern, text):
            ctx = text[max(0, m.start() - 50):m.end() + 200]
            # 提取属性
            attrs = re.findall(r'<td>([^<]+)</td>', ctx)
            print('  属性: {}'.format(attrs[:10]))
    else:
        print('\n✗ 未找到 {}林班'.format(target))

# 3. 尝试 float32 坐标
print('\n=== 扫描 float32 坐标 ===')
coords = []
for i in range(0, len(data) - 4, 4):
    try:
        lon = struct.unpack('<f', data[i:i + 4])[0]
        if 103.0 < lon < 104.0 and 1e5 < lon:
            # 在前后找纬度
            for j in range(max(0, i - 32), min(len(data) - 4, i + 32)):
                lat = struct.unpack('<f', data[j:j + 4])[0]
                if 28.0 < lat < 29.0:
                    coords.append((i, lat, lon))
                    break
    except Exception:
        pass

# 也尝试 (lat, lon) 顺序
coords2 = []
for i in range(0, len(data) - 4, 4):
    try:
        lat = struct.unpack('<f', data[i:i + 4])[0]
        if 28.0 < lat < 29.0:
            for j in range(max(0, i - 32), min(len(data) - 4, i + 32)):
                lon = struct.unpack('<f', data[j:j + 4])[0]
                if 103.0 < lon < 104.0:
                    coords2.append((i, lat, lon))
                    break
    except Exception:
        pass

print('float32 (lon,lat) 对: {}'.format(len(coords)))
print('float32 (lat,lon) 对: {}'.format(len(coords2)))
for c in (coords2 or coords)[:15]:
    print('  offset={}: lat={:.6f}, lon={:.6f}'.format(c[0], c[1], c[2]))

# 4. 搜索 int 坐标 (Web Mercator 或米制坐标)
print('\n=== 扫描整数坐标 (米制/CGCS2000) ===')
# CGCS2000 / Gauss-Kruger ~ 3520000N, 530000E
int_coords = []
for i in range(0, len(data) - 4, 4):
    try:
        val = struct.unpack('<i', data[i:i + 4])[0]
        if 3500000 < val < 3600000:  # 纬度 (米)
            for j in range(max(0, i - 32), min(len(data) - 4, i + 32)):
                val2 = struct.unpack('<i', data[j:j + 4])[0]
                if 520000 < val2 < 540000:  # 经度 (米)
                    int_coords.append((i, val, val2))
                    break
    except Exception:
        pass
print('int (CGCS2000) 对: {}'.format(len(int_coords)))
for c in int_coords[:15]:
    print('  offset={}: N={}m, E={}m'.format(c[0], c[1], c[2]))

# 5. 如果以上都没有, 打印林班名称附近的原始数据
if not coords and not coords2 and not int_coords:
    print('\n未找到浮点/整数坐标, 尝试解析二进制结构...')
    # 找第一个林班的数据开始位置
    m = re.search(r'1林班', text)
    if m:
        byte_offset = len(text[:m.start()].encode('utf-8', errors='replace'))
        print('1林班 文本偏移: {}'.format(byte_offset))
        # 打印之后的 hex dump
        chunk = data[byte_offset:byte_offset + 200]
        for row in range(0, min(200, len(chunk)), 16):
            hex_str = ' '.join('{:02x}'.format(b) for b in chunk[row:row + 16])
            ascii_str = ''.join(chr(b) if 0x20 <= b < 0x7f else '.'
                               for b in chunk[row:row + 16])
            print('  {:06x}: {}  {}'.format(byte_offset + row, hex_str, ascii_str))

print('\nDEEP_EXTRACT_DONE')
