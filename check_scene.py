# -*- coding: utf-8 -*-
"""场景几何验证：足端落地 / 机身水平 / 锯片对准目标 / 驱动器统计"""
import bpy
import math

def gullyF(x, y):
    zc = 2.0 + 0.55 * math.sin(x * 0.33)
    d = (y - zc) / 1.7
    return math.exp(-d * d)

def H(x, y):
    h = (1.35 * math.sin(x * 0.16) * math.cos(y * 0.13)
         + 0.55 * math.sin(x * 0.42 + 1.7) * math.sin(y * 0.36 + 0.6)
         + 0.28 * math.sin(x * 0.9 - 0.4) * math.cos(y * 1.1 + 1.3))
    h += 1.15 * math.exp(-((x + 6) ** 2 + (y - 4) ** 2) / 9.0)
    h -= 1.05 * gullyF(x, y)
    h += 0.5 * math.exp(-((x - 11) ** 2 + (y - 3) ** 2) / 14.0)
    return h

print('=== 集合统计 ===')
for c in ['机器人', '地形', '植被', '岩石与倒木']:
    coll = bpy.data.collections.get(c)
    print(c, ':', len(coll.objects) if coll else '缺失!')

print('=== 四足足端落地检查（离地应为足端球半径 ~0.048）===')
for nm in ['左前', '右前', '左后', '右后']:
    foot = bpy.data.objects.get('腿·%s·足端' % nm)
    if foot:
        wp = foot.matrix_world.translation
        print('%s 足端离地: %.3f m' % (nm, wp.z - H(wp.x, wp.y)))

print('=== 机身姿态（应为水平: 俯仰/侧倾≈0）===')
root = bpy.data.objects.get('机器人')
if root:
    e = root.matrix_world.to_euler()
    print('机身 俯仰 %.2f° 侧倾 %.2f° 航向 %.1f°'
          % (math.degrees(e.y), math.degrees(e.x), math.degrees(e.z)))

print('=== 锯片与目标树距离 ===')
saw = bpy.data.objects.get('圆盘锯片')
if saw:
    sw = saw.matrix_world.translation
    best, bd = None, 9e9
    for ob in bpy.data.objects:
        if ob.name.startswith('T-') and '·' not in ob.name and ob.type == 'EMPTY':
            p = ob.matrix_world.translation
            d = math.hypot(p.x - sw.x, p.y - sw.y)
            if d < bd:
                bd, best = d, ob.name
    print('锯片位置 (%.2f, %.2f, %.2f)，距最近候选木 %s 水平 %.2f m'
          % (sw.x, sw.y, sw.z, best, bd))

print('=== 爆炸分解驱动器 ===')
cnt = 0
for ob in bpy.data.objects:
    if ob.animation_data and ob.animation_data.drivers:
        cnt += len(ob.animation_data.drivers)
print('驱动器总数:', cnt)

print('=== 动画 ===')
print('帧范围:', bpy.context.scene.frame_start, '-', bpy.context.scene.frame_end)
acts = [a.name for a in bpy.data.actions]
print('动作:', acts)
print('CHECK_DONE')
