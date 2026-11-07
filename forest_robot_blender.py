# -*- coding: utf-8 -*-
"""林下作业机器人 · Blender 三维场景生成脚本
====================================================
功能：一键生成“四足机器狗式林下作业机器人”完整三维概念场景
  - 机器狗（参考宇树 Go2/B1 躯体比例 + Z1 机械臂布局、云深处绝影 X20 工业传感配置）：
    躯干 + 4×3 自由度电驱动腿（髋侧摆/髋前摆/膝），四足按地形高度逐腿逆解站立，
    机身始终保持水平 —— 天然适应复杂地形；
    背部 Z1 风格三自由度疏伐机械臂 + 圆盘锯、激光雷达、RTK 天线、深度相机、警示灯
  - 起伏林地地形（丘陵 + 冲刷沟，按坡度分草地/裸土/沙土材质）
  - 疏伐目标簇（橙色标记木，其中 1 株已伐倒 + 留桩，机械臂正对准下一株）
  - 散生木、林缘大径木、岩石与倒木
  - 相机 / 太阳光 / 世界背景，激光雷达与锯片循环旋转动画
  - ★ 爆炸分解视图：播放动画自动“展开 → 合拢”；也可选中“机器人”根物体，
    在 N 面板 → 自定义属性 中拖动「爆炸系数」滑杆（0–1）手动控制分解程度

用法：
  1. 打开 Blender（2.83 ~ 4.x 均可）
  2. 顶部切换到“Scripting / 脚本”工作区
  3. 点 Open 打开本文件，点击 ▶ Run Script（或按 Alt+P）
  4. 自动切到材质预览即可查看；重复运行会先清理旧场景，可放心反复执行

导出：生成后 File → Export → 选择 FBX / OBJ / glTF(GLB) 即可
====================================================
"""

import bpy
import math
from mathutils import Matrix, Vector, Euler

# ---------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------
_seed = 7
def rnd():
    global _seed
    _seed = (_seed * 1664525 + 1013904223) & 0xFFFFFFFF
    return _seed / 4294967296.0

def clamp(v, a, b):
    return a if v < a else (b if v > b else v)

def lerp(a, b, t):
    return a + (b - a) * t

# 地形高度场（Blender 坐标：X-Y 平面，Z 向上）
def gullyF(x, y):
    zc = 2.0 + 0.55 * math.sin(x * 0.33)
    d = (y - zc) / 1.7
    return math.exp(-d * d)

def H(x, y):
    h = (1.35 * math.sin(x * 0.16) * math.cos(y * 0.13)
         + 0.55 * math.sin(x * 0.42 + 1.7) * math.sin(y * 0.36 + 0.6)
         + 0.28 * math.sin(x * 0.9 - 0.4) * math.cos(y * 1.1 + 1.3))
    h += 1.15 * math.exp(-((x + 6) ** 2 + (y - 4) ** 2) / 9.0)     # 概览台地
    h -= 1.05 * gullyF(x, y)                                        # 冲刷沟
    h += 0.5 * math.exp(-((x - 11) ** 2 + (y - 3) ** 2) / 14.0)     # 东南缓丘
    return h

# ---------------------------------------------------------------
# 场景清理（幂等：重复运行先删除旧场景）
# ---------------------------------------------------------------
COLL_NAME = '林下作业机器人场景'
old = bpy.data.collections.get(COLL_NAME)
if old:
    for ob in list(old.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for ch in list(old.children):
        for ob in list(ch.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.collections.remove(ch)
    bpy.data.collections.remove(old)

master = bpy.data.collections.new(COLL_NAME)
bpy.context.scene.collection.children.link(master)

def sub_collection(name):
    c = bpy.data.collections.new(name)
    master.children.link(c)
    return c

COL_ROBOT = sub_collection('机器人')
COL_TERRAIN = sub_collection('地形')
COL_FLORA = sub_collection('植被')
COL_OBS = sub_collection('岩石与倒木')

# ---------------------------------------------------------------
# 材质
# ---------------------------------------------------------------
def make_mat(name, color, rough=0.8, metal=0.05, emissive=None, e_strength=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    if b:
        b.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
        b.inputs['Roughness'].default_value = rough
        b.inputs['Metallic'].default_value = metal
        if emissive:
            for nm in ('Emission Color', 'Emission'):   # 4.x / 旧版 名称兼容
                if nm in b.inputs:
                    b.inputs[nm].default_value = (emissive[0], emissive[1], emissive[2], 1.0)
                    if 'Emission Strength' in b.inputs:
                        b.inputs['Emission Strength'].default_value = e_strength
                    break
    return m

M_ORANGE  = make_mat('机器人橙', (0.85, 0.44, 0.18), rough=0.55)
M_ORANGE2 = make_mat('深橙', (0.75, 0.35, 0.12), rough=0.6)
M_CHAR    = make_mat('炭黑', (0.17, 0.18, 0.20), rough=0.9)
M_STEEL   = make_mat('钢灰', (0.60, 0.64, 0.67), rough=0.4, metal=0.7)
M_DARK    = make_mat('深色胶带', (0.11, 0.13, 0.14), rough=0.85)
M_WOOD    = make_mat('树干', (0.48, 0.32, 0.22), rough=0.9)
M_WOOD2   = make_mat('深树干', (0.42, 0.29, 0.20), rough=0.9)
M_ROCK    = make_mat('岩石', (0.55, 0.55, 0.53), rough=0.95)
M_LEAF_A  = make_mat('叶绿A', (0.18, 0.49, 0.23), rough=0.9)
M_LEAF_B  = make_mat('叶绿B', (0.25, 0.60, 0.31), rough=0.9)
M_LEAF_Y  = make_mat('疏伐候选叶', (0.60, 0.63, 0.29), rough=0.9)
M_GRASS   = make_mat('草地', (0.33, 0.42, 0.24), rough=1.0)
M_DIRT    = make_mat('裸土', (0.42, 0.35, 0.23), rough=1.0)
M_SAND    = make_mat('沟谷沙土', (0.55, 0.48, 0.34), rough=1.0)
M_BLADE   = make_mat('锯片钢', (0.72, 0.76, 0.79), rough=0.3, metal=0.85)
M_BAND    = make_mat('疏伐标记', (0.91, 0.47, 0.17), rough=0.5,
                     emissive=(0.85, 0.40, 0.06), e_strength=0.4)
M_BEACON  = make_mat('警示灯', (1.0, 0.55, 0.10), rough=0.4,
                     emissive=(1.0, 0.48, 0.0), e_strength=2.5)
M_LIDAR   = make_mat('雷达窗', (0.18, 0.90, 0.42), rough=0.3,
                     emissive=(0.18, 0.90, 0.42), e_strength=1.5)
M_GLASS   = make_mat('镜头', (0.05, 0.06, 0.08), rough=0.15, metal=0.6)

# ---------------------------------------------------------------
# 物体创建辅助（局部坐标：X=前进, Y=左侧, Z=向上）
# ---------------------------------------------------------------
def _xform(loc, rot, scale):
    m = Matrix.Translation(Vector(loc))
    if rot:
        m = m @ Euler(rot, 'XYZ').to_matrix().to_4x4()
    if scale:
        m = m @ Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0)).to_4x4()
    return m

def _finish(ob, name, parent, loc, rot, coll, scale=None):
    ob.name = name
    ob.parent = parent
    ob.matrix_local = _xform(loc, rot, scale)
    coll.objects.link(ob)
    for c in list(ob.users_collection):
        if c != coll:
            c.objects.unlink(ob)
    return ob

def box(name, parent, loc, size, mat, rot=None, coll=COL_ROBOT):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    ob = bpy.context.active_object
    _finish(ob, name, parent, loc, rot, coll, scale=size)
    ob.data.materials.append(mat)
    return ob

def cyl(name, parent, loc, r, h, mat, seg=14, rot=None, r2=None, coll=COL_ROBOT):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, vertices=seg)
    ob = bpy.context.active_object
    _finish(ob, name, parent, loc, rot, coll)
    ob.data.materials.append(mat)
    return ob

def cone(name, parent, loc, r, h, mat, seg=9, rot=None, coll=COL_FLORA):
    bpy.ops.mesh.primitive_cone_add(radius1=r, radius2=0.0, depth=h, vertices=seg)
    ob = bpy.context.active_object
    _finish(ob, name, parent, loc, rot, coll)
    ob.data.materials.append(mat)
    return ob

def torus(name, parent, loc, R, r, mat, rot=None, coll=COL_ROBOT):
    bpy.ops.mesh.primitive_torus_add(mode='MAJOR_MINOR', major_radius=R, minor_radius=r,
                                     major_segments=24, minor_segments=8)
    ob = bpy.context.active_object
    _finish(ob, name, parent, loc, rot, coll)
    ob.data.materials.append(mat)
    return ob

def sphere(name, parent, loc, r, mat, seg=12, coll=COL_ROBOT):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=seg, ring_count=max(6, seg - 4))
    ob = bpy.context.active_object
    _finish(ob, name, parent, loc, None, coll)
    ob.data.materials.append(mat)
    return ob

def joint(name, parent, loc, rot=None, size=0.14):
    bpy.ops.object.empty_add(type='PLAIN_AXES')
    ob = bpy.context.active_object
    ob.empty_display_size = size
    return _finish(ob, name, parent, loc, rot, COL_ROBOT)

# ---------------------------------------------------------------
# 地形（网格数据直接生成，按坡度分材质）
# ---------------------------------------------------------------
def build_terrain(n=80, size=40.0):
    verts = []
    faces = []
    half = size / 2.0
    step = size / n
    for j in range(n + 1):
        for i in range(n + 1):
            x = -half + i * step
            y = -half + j * step
            verts.append((x, y, H(x, y)))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + n + 2, a + n + 1))

    me = bpy.data.meshes.new('地形网格')
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    me.materials.append(M_GRASS)
    me.materials.append(M_DIRT)
    me.materials.append(M_SAND)

    ob = bpy.data.objects.new('地形', me)
    COL_TERRAIN.objects.link(ob)

    e = 0.35
    for p in me.polygons:
        cx = (verts[p.vertices[0]][0] + verts[p.vertices[2]][0]) / 2.0
        cy = (verts[p.vertices[0]][1] + verts[p.vertices[2]][1]) / 2.0
        gx = (H(cx + e, cy) - H(cx - e, cy)) / (2 * e)
        gy = (H(cx, cy + e) - H(cx, cy - e)) / (2 * e)
        slope = math.hypot(gx, gy)
        g = gullyF(cx, cy)
        if g > 0.35:
            p.material_index = 2
        elif slope > 0.55:
            p.material_index = 1
        else:
            p.material_index = 0
    me.update()
    return ob

build_terrain()

# ---------------------------------------------------------------
# 场地布局参数（与概念设计一致）
# ---------------------------------------------------------------
WP = [(-9, 7.5), (-11, -2.5), (-4, -8.5), (4, -9.5), (11, -5), (12, 3), (6, 8.5), (-2, 10)]
C = (6.0, -4.0)     # 疏伐目标簇中心
A0 = (3.4, -2.8)    # 名义进场方向参考点（用于选定作业目标与停靠位）
DET = (0.0, 1.0)    # 探测站位

def dist_seg(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    t = clamp((apx * abx + apy * aby) / (abx * abx + aby * aby), 0.0, 1.0)
    return math.hypot(px - (ax + abx * t), py - (ay + aby * t))

def near_path(px, py, d):
    for i in range(len(WP)):
        a = WP[i]
        b = WP[(i + 1) % len(WP)]
        if dist_seg(px, py, a[0], a[1], b[0], b[1]) < d:
            return True
    return False

# ---------------------------------------------------------------
# 树木
# ---------------------------------------------------------------
trees = []   # (empty_obj, x, y, s, marked)

def make_tree(x, y, s, marked=False, big=False):
    idx = len(trees) + 1
    base = joint('T-%02d' % idx, None, (x, y, H(x, y)),
                 rot=(0.0, 0.0, rnd() * 6.28), size=0.1)
    # joint 默认放进机器人集合，移到植被集合
    COL_FLORA.objects.link(base)
    for c in list(base.users_collection):
        if c != COL_FLORA:
            c.objects.unlink(base)
    trunk_m = M_WOOD2 if big else M_WOOD
    cyl('T-%02d·树干' % idx, base, (0, 0, 0.85 * s),
        0.125 * s, 1.7 * s, trunk_m, seg=7)
    leaf = M_LEAF_Y if marked else (M_LEAF_A if rnd() > 0.5 else M_LEAF_B)
    cone('T-%02d·冠1' % idx, base, (0, 0, 1.65 * s), 0.74 * s, 1.2 * s, leaf)
    cone('T-%02d·冠2' % idx, base, (0, 0, 2.35 * s), 0.56 * s, 1.0 * s, leaf)
    cone('T-%02d·冠3' % idx, base, (0, 0, 3.0 * s), 0.38 * s, 0.85 * s, leaf)
    if marked:
        cyl('T-%02d·标记环' % idx, base, (0, 0, 0.95 * s),
            0.135 * s, 0.12 * s, M_BAND, seg=10)
    trees.append((base, x, y, s, marked))
    return base

# 疏伐目标簇：过密幼树（T-01 已伐倒，T-02 为当前目标）
base_ang = math.atan2(A0[1] - C[1], A0[0] - C[0])
cluster = []
for k in range(7):
    ang = base_ang + (k - 3) / 3.0 * 0.8
    rr = 1.05 + ((k * 37) % 40) / 40.0 * 0.6
    tx = C[0] + math.cos(ang) * rr
    ty = C[1] + math.sin(ang) * rr
    s = 0.55 + ((k * 13) % 30) / 100.0
    cluster.append(make_tree(tx, ty, s, marked=True))

# 选定当前作业目标：最贴近进场方向的候选木；机器狗停靠在其外侧 0.95 m
approach = Vector((A0[0] - C[0], A0[1] - C[1]))
approach.normalize()
best_i, best_dot = 0, -9.0
for i in range(7):
    td = Vector((trees[i][1] - C[0], trees[i][2] - C[1]))
    td.normalize()
    if td.dot(approach) > best_dot:
        best_dot, best_i = td.dot(approach), i
tgt_tree = trees[best_i]
outward = Vector((tgt_tree[1] - C[0], tgt_tree[2] - C[1]))
outward.normalize()
A = (tgt_tree[1] + outward.x * 0.95, tgt_tree[2] + outward.y * 0.95)

# T-01 伐倒（向远离机器人一侧倒下），并留桩
t0, t0x, t0y, t0s, _ = trees[0]
fd = Vector((t0x - A[0], t0y - A[1], 0.0))
fd.normalize()
axis = Vector((fd.y, -fd.x, 0.0))
t0.rotation_mode = 'AXIS_ANGLE'
t0.rotation_axis_angle = (math.radians(80), axis.x, axis.y, axis.z)
cyl('T-01·伐桩', None, (t0x, t0y, H(t0x, t0y) + 0.12 * t0s),
    0.11 * t0s, 0.24 * t0s, M_WOOD, seg=8, coll=COL_FLORA)

# 散生木
placed, tries = 0, 0
while placed < 26 and tries < 900:
    tries += 1
    px = (rnd() * 2 - 1) * 13
    py = (rnd() * 2 - 1) * 12
    if near_path(px, py, 2.6):
        continue
    if math.hypot(px + 6, py - 4) < 2.9:
        continue
    if math.hypot(px - DET[0], py - DET[1]) < 3.4:
        continue
    if math.hypot(px - C[0], py - C[1]) < 3.2:
        continue
    if math.hypot(px - A[0], py - A[1]) < 1.5:
        continue
    if any(math.hypot(px - t[1], py - t[2]) < 1.35 for t in trees):
        continue
    make_tree(px, py, 0.9 + rnd() * 0.6)
    placed += 1

# 林缘大径木
for i in range(9):
    ba = i / 9.0 * math.pi * 2 + 0.3
    bx = math.cos(ba) * (15.5 + rnd() * 2.5)
    by = math.sin(ba) * (15.0 + rnd() * 2.5)
    make_tree(bx, by, 1.6 + rnd() * 0.5, big=True)

# ---------------------------------------------------------------
# 岩石与倒木
# ---------------------------------------------------------------
orp, otries = 0, 0
while orp < 9 and otries < 400:
    otries += 1
    ox = (rnd() * 2 - 1) * 12
    oy = (rnd() * 2 - 1) * 11
    if near_path(ox, oy, 1.9):
        continue
    if math.hypot(ox + 6, oy - 4) < 2.6:
        continue
    if math.hypot(ox - C[0], oy - C[1]) < 2.6:
        continue
    if math.hypot(ox - DET[0], oy - DET[1]) < 3.0:
        continue
    if math.hypot(ox - A[0], oy - A[1]) < 1.7:
        continue
    if rnd() > 0.4:
        r = 0.25 + rnd() * 0.32
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=r)
        ob = bpy.context.active_object
        _finish(ob, '岩石%02d' % (orp + 1), None,
                (ox, oy, H(ox, oy) - r * 0.2),
                (rnd() * 3, rnd() * 3, rnd() * 3), COL_OBS,
                scale=(1.0, 1.0, 0.72))
        ob.data.materials.append(M_ROCK)
    else:
        ob = cyl('倒木%02d' % (orp + 1), None, (ox, oy, H(ox, oy) + 0.12),
                 0.14, 1.7, M_WOOD2, seg=8, coll=COL_OBS)
        ob.rotation_euler = (math.pi / 2 + (rnd() - 0.5) * 0.2, 0, rnd() * 3.14)
    orp += 1

# ---------------------------------------------------------------
# 机器狗（参考宇树 Go2/B1 + Z1、云深处绝影 X20）
#   四足按地形高度逐腿逆解站立，机身保持水平
# ---------------------------------------------------------------
yaw = math.atan2(tgt_tree[2] - A[1], tgt_tree[1] - A[0])
cy, sy = math.cos(yaw), math.sin(yaw)

Lt1, Lt2 = 0.30, 0.34          # 大腿 / 小腿长度
HIP_OFF = 0.07                 # 髋侧摆轴 → 髋前摆轴 的侧向偏移
legs_def = [
    ('左前', ( 0.27,  0.155, -0.02), ( 0.30,  0.19), +HIP_OFF),
    ('右前', ( 0.27, -0.155, -0.02), ( 0.30, -0.19), -HIP_OFF),
    ('左后', (-0.27,  0.155, -0.02), (-0.30,  0.19), +HIP_OFF),
    ('右后', (-0.27, -0.155, -0.02), (-0.30, -0.19), -HIP_OFF),
]
foot_h = []
for _, _, fl, _d in legs_def:
    wx = A[0] + fl[0] * cy - fl[1] * sy
    wy = A[1] + fl[0] * sy + fl[1] * cy
    foot_h.append(H(wx, wy))
ride = sum(foot_h) / 4.0 + 0.50    # 站立高度：机身中心离地约 0.5 m

root = joint('机器人', None, (A[0], A[1], ride), rot=(0.0, 0.0, yaw), size=0.3)
body = joint('躯干', root, (0, 0, 0))

def norm_pi(t):
    while t > math.pi:
        t -= 2 * math.pi
    while t < -math.pi:
        t += 2 * math.pi
    return t

def solve_leg(u, d):
    """四足腿逆解：u = 足端相对髋侧摆轴的体坐标向量，d = 髋侧向偏移（左+ / 右-）
    返回（髋侧摆角, 髋前摆角, 膝弯角），膝向后折（宇树式站姿）"""
    ux, uy, uz = u
    rg = max(math.hypot(uy, uz), 1e-6)
    s = clamp(d / rg, -0.99, 0.99)
    a0 = math.asin(s)
    phi = math.atan2(uy, uz)
    th_r = min((norm_pi(a0 - phi), norm_pi(math.pi - a0 - phi)), key=abs)
    cs, sn = math.cos(th_r), math.sin(th_r)
    pz = -uy * sn + uz * cs
    px = ux
    r = clamp(math.hypot(px, pz), abs(Lt1 - Lt2) + 0.02, Lt1 + Lt2 - 0.01)
    cos_k = clamp((r * r - Lt1 * Lt1 - Lt2 * Lt2) / (2 * Lt1 * Lt2), -1.0, 1.0)
    th_k = -math.acos(cos_k)
    th_p = math.atan2(px, -pz) - math.atan2(Lt2 * math.sin(th_k), Lt1 + Lt2 * math.cos(th_k))
    return th_r, th_p, th_k

# ---- 四条 3 自由度电驱动腿 ----
for i, (nm, hip, fl, dsg) in enumerate(legs_def):
    fz = foot_h[i] + 0.045 - ride
    u = (fl[0] - hip[0], fl[1] - hip[1], fz - hip[2])
    th_r, th_p, th_k = solve_leg(u, dsg)
    roll = joint('腿·%s·髋侧摆' % nm, body, hip, rot=(th_r, 0.0, 0.0), size=0.07)
    pitch = joint('腿·%s·髋前摆' % nm, roll, (0, dsg, 0), rot=(0.0, -th_p, 0.0), size=0.07)
    cyl('腿·%s·髋电机' % nm, pitch, (0, 0, 0), 0.075, 0.15, M_CHAR, seg=14, rot=(math.pi / 2, 0, 0))
    box('腿·%s·大腿' % nm, pitch, (0, 0, -Lt1 / 2), (0.085, 0.115, Lt1), M_CHAR)
    knee = joint('腿·%s·膝' % nm, pitch, (0, 0, -Lt1), rot=(0.0, -th_k, 0.0), size=0.06)
    cyl('腿·%s·膝电机' % nm, knee, (0, 0, 0), 0.052, 0.115, M_ORANGE2, seg=12, rot=(math.pi / 2, 0, 0))
    box('腿·%s·小腿' % nm, knee, (0, 0, -Lt2 / 2), (0.05, 0.062, Lt2), M_DARK)
    sphere('腿·%s·足端' % nm, knee, (0, 0, -Lt2), 0.048, M_DARK, seg=10)

# ---- 躯干与顶部载荷 ----
box('主躯干', body, (0, 0, 0), (0.78, 0.30, 0.17), M_ORANGE)
box('上甲板', body, (0, 0, 0.105), (0.66, 0.26, 0.05), M_CHAR)
box('侧护板-左', body, (0, 0.163, -0.015), (0.56, 0.028, 0.115), M_CHAR)
box('侧护板-右', body, (0, -0.163, -0.015), (0.56, 0.028, 0.115), M_CHAR)
box('前脸传感器盒', body, (0.40, 0, 0.01), (0.12, 0.24, 0.12), M_CHAR)
box('电池舱', body, (-0.10, 0, 0.165), (0.30, 0.20, 0.08), M_STEEL)

# 深度相机（前置双目）
cam_g = joint('深度相机', body, (0.475, 0, 0.02))
box('相机支架', cam_g, (0, 0, 0), (0.06, 0.20, 0.09), M_CHAR)
cyl('左镜头', cam_g, (0.045, 0.05, 0.01), 0.032, 0.07, M_GLASS, rot=(0, 0, math.pi / 2), seg=10)
cyl('右镜头', cam_g, (0.045, -0.05, 0.01), 0.032, 0.07, M_GLASS, rot=(0, 0, math.pi / 2), seg=10)

# 激光雷达（前顶置，Livox 风格）
cyl('雷达基座', body, (0.28, 0, 0.14), 0.055, 0.08, M_STEEL, seg=12)
lidar = joint('激光雷达', body, (0.28, 0, 0.205))
cyl('雷达外壳', lidar, (0, 0, 0), 0.10, 0.09, M_CHAR, seg=14)
lidar_spin = joint('雷达扫描旋转', lidar, (0, 0, 0.05))
cyl('雷达转鼓', lidar_spin, (0, 0, 0), 0.105, 0.045, M_CHAR, seg=14)
for w in range(4):
    wa = w * math.pi / 2
    box('雷达窗口%d' % w, lidar_spin,
        (math.cos(wa) * 0.106, math.sin(wa) * 0.106, 0),
        (0.018, 0.018, 0.02), M_LIDAR, rot=(0, 0, -wa))

# 尾部 RTK 双天线与警示灯
box('RTK横杆', body, (-0.335, 0, 0.135), (0.05, 0.26, 0.04), M_STEEL)
cyl('RTK天线-1', body, (-0.335, 0.10, 0.19), 0.022, 0.10, M_DARK, seg=8)
cyl('RTK天线-2', body, (-0.335, -0.10, 0.19), 0.022, 0.10, M_DARK, seg=8)
cyl('警示灯', body, (-0.36, 0.155, 0.15), 0.04, 0.08, M_BEACON, seg=10)
sphere('警示灯罩', body, (-0.36, 0.155, 0.20), 0.042, M_BEACON, seg=10)

# ---- 背部 Z1 风格三自由度疏伐机械臂 + 圆盘锯（对准目标候选木）----
La1, La2 = 0.52, 0.66          # 大臂长 / 小臂+锯 有效长度
shW = Vector((A[0] + 0.10 * cy, A[1] + 0.10 * sy, ride + 0.20 + 0.09))
tz = H(tgt_tree[1], tgt_tree[2]) + 0.55 * tgt_tree[3] + 0.35
dx, dy = tgt_tree[1] - shW.x, tgt_tree[2] - shW.y
dzv = tz - shW.z
d_h = math.hypot(dx, dy)
r = clamp(math.hypot(d_h, dzv), abs(La1 - La2) + 0.06, La1 + La2 - 0.03)
a = math.atan2(dzv, max(d_h, 0.001))
cosE = clamp((La1 * La1 + La2 * La2 - r * r) / (2 * La1 * La2), -1, 1)
cosB = clamp((La1 * La1 + r * r - La2 * La2) / (2 * La1 * r), -1, 1)
th1 = a + math.acos(cosB)          # 大臂抬升角
bend = math.pi - math.acos(cosE)   # 肘部下弯角
rel_yaw = math.atan2(dy, dx) - yaw

arm_base = joint('机械臂·基座回转', body, (0.10, 0, 0.20), rot=(0, 0, rel_yaw))
cyl('基座回转体', arm_base, (0, 0, 0.045), 0.085, 0.10, M_ORANGE2, seg=16)
shoulder = joint('机械臂·肩关节', arm_base, (0, 0, 0.09), rot=(0, -th1, 0))
box('大臂', shoulder, (La1 / 2, 0, 0), (La1, 0.09, 0.10), M_ORANGE)
cyl('肩关节销', shoulder, (0, 0, 0), 0.05, 0.13, M_CHAR, seg=10, rot=(math.pi / 2, 0, 0))
elbow = joint('机械臂·肘关节', shoulder, (La1, 0, 0), rot=(0, bend, 0))
box('小臂', elbow, (0.26, 0, 0), (0.52, 0.075, 0.085), M_CHAR)
cyl('肘关节销', elbow, (0, 0, 0), 0.042, 0.11, M_ORANGE2, seg=10, rot=(math.pi / 2, 0, 0))
wrist = joint('机械臂·腕部', elbow, (0.52, 0, 0))
saw_spin = joint('锯片旋转', wrist, (0.14, 0, 0))
cyl('圆盘锯片', saw_spin, (0, 0, 0), 0.16, 0.016, M_BLADE, seg=26, rot=(math.pi / 2, 0, 0))
torus('锯齿圈', saw_spin, (0, 0, 0), 0.158, 0.013, M_DARK, rot=(math.pi / 2, 0, 0))
cyl('锯毂', wrist, (0.14, 0, 0), 0.036, 0.09, M_STEEL, seg=10, rot=(math.pi / 2, 0, 0))
torus('锯防护罩', wrist, (0.14, 0, 0.01), 0.185, 0.026, M_ORANGE2, rot=(math.pi / 2, 0, 0))

# ---------------------------------------------------------------
# 动画：LiDAR 旋转 / 锯片旋转（96 帧时间轴，两圈无缝循环）
# ---------------------------------------------------------------
def spin_anim(ob, axis, turns, f_end=48):
    """在无其他姿态的空物体上创建匀速旋转循环动画。"""
    for f, v in ((1, 0.0), (f_end, turns * 2 * math.pi)):
        if axis == 'X':
            ob.rotation_euler = Euler((v, 0.0, 0.0), 'XYZ')
        elif axis == 'Y':
            ob.rotation_euler = Euler((0.0, v, 0.0), 'XYZ')
        else:
            ob.rotation_euler = Euler((0.0, 0.0, v), 'XYZ')
        ob.keyframe_insert(data_path='rotation_euler', frame=f)
    # 设为线性插值，保证匀速
    if ob.animation_data and ob.animation_data.action:
        for fc in ob.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

spin_anim(lidar_spin, 'Z', 2, f_end=96)   # 激光雷达扫描旋转
spin_anim(saw_spin, 'Y', 6, f_end=96)     # 圆盘锯旋转（独立子空物体，不影响臂部姿态）

# ---------------------------------------------------------------
# 爆炸分解视图（Exploded View）
#   根物体「机器人」上的自定义属性「爆炸系数」(0–1) 通过驱动器
#   控制各部件沿装配方向平移：躯干固定为基准，四条腿向四角分开、
#   雷达/天线向上、机械臂链式向前上抬升、覆盖件逐层脱离……
# ---------------------------------------------------------------
root['爆炸系数'] = 0.0
try:
    root.id_properties_ui('爆炸系数').update(
        min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, subtype='FACTOR')
except Exception:
    pass   # 旧版 Blender 无滑杆 UI 数据，仍可输入数值

EXPLODE = {
    # 四条腿总成：向四角分开
    '腿·左前·髋侧摆': ( 0.12,  0.62, 0.05),
    '腿·右前·髋侧摆': ( 0.12, -0.62, 0.05),
    '腿·左后·髋侧摆': (-0.12,  0.62, 0.05),
    '腿·右后·髋侧摆': (-0.12, -0.62, 0.05),
    # 躯干覆盖件
    '上甲板': ( 0.00,  0.00, 0.42),
    '电池舱': (-0.05,  0.00, 0.72),
    '侧护板-左': ( 0.00,  0.42, 0.00),
    '侧护板-右': ( 0.00, -0.42, 0.00),
    '前脸传感器盒': ( 0.50,  0.00, 0.08),
    # 传感部件
    '深度相机': ( 0.55,  0.00, 0.25),
    '激光雷达': ( 0.05,  0.00, 0.70),
    'RTK横杆': ( 0.00,  0.00, 0.50),
    'RTK天线-1': ( 0.00,  0.10, 0.66),
    'RTK天线-2': ( 0.00, -0.10, 0.66),
    '警示灯': (-0.08,  0.18, 0.48),
    '警示灯罩': (-0.08,  0.18, 0.62),
    # 背部机械臂（链式渐进分离）
    '机械臂·基座回转': ( 0.00, 0.00, 0.45),
    '机械臂·肩关节':   ( 0.20, 0.00, 0.12),
    '机械臂·肘关节':   ( 0.18, 0.00, 0.06),
    '机械臂·腕部':     ( 0.16, 0.00, 0.02),
    '锯防护罩': ( 0.14, 0.00, 0.10),
    '锯毂':     ( 0.07, 0.00, 0.00),
}

def add_explode_driver(ob, offset):
    """在物体 location 通道上挂驱动器：位置 = 基准 + 分解方向 × 爆炸系数"""
    base = ob.location.copy()
    for i in range(3):
        off = float(offset[i])
        if abs(off) < 1e-9:
            continue
        fc = ob.driver_add('location', i)
        drv = fc.driver
        var = drv.variables.new()
        var.name = 'x'
        tgt = var.targets[0]
        tgt.id_type = 'OBJECT'
        tgt.id = root
        tgt.data_path = '["爆炸系数"]'
        drv.expression = '%.6f + %.6f * x' % (base[i], off)

for ob in list(COL_ROBOT.all_objects):
    if ob == root or ob == body:
        continue          # 机器人根 / 躯干锚点保持固定
    if ob.name in EXPLODE:
        add_explode_driver(ob, EXPLODE[ob.name])

# 演示动画：合拢 → 展开 → 合拢（与旋转动画共用 96 帧时间轴，无缝循环）
for f, v in ((1, 0.0), (36, 0.0), (60, 1.0), (96, 0.0)):
    root['爆炸系数'] = v
    try:
        root.keyframe_insert(data_path='["爆炸系数"]', frame=f)
    except Exception:
        break   # 极旧版本不支持自定义属性关键帧，跳过（滑杆仍可用）
root['爆炸系数'] = 0.0

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 96

# ---------------------------------------------------------------
# 灯光 / 相机 / 世界
# ---------------------------------------------------------------
def aim_rotation(pos, target):
    d = Vector(target) - Vector(pos)
    return d.to_track_quat('-Z', 'Y').to_euler()

sun_data = bpy.ops.object.light_add(type='SUN', location=(14, 10, 20))
sun = bpy.context.active_object
sun.name = '太阳光'
sun.rotation_euler = aim_rotation((14, 10, 20), (0, 0, 0))
sun.data.energy = 3.2
sun.data.angle = math.radians(8)
master.objects.link(sun)
for c in list(sun.users_collection):
    if c != master:
        c.objects.unlink(sun)

fill_data = bpy.ops.object.light_add(type='SUN', location=(-12, -8, 10))
fillL = bpy.context.active_object
fillL.name = '补光'
fillL.rotation_euler = aim_rotation((-12, -8, 10), (0, 0, 0))
fillL.data.energy = 0.7
master.objects.link(fillL)
for c in list(fillL.users_collection):
    if c != master:
        c.objects.unlink(fillL)

bpy.ops.object.camera_add(location=(A[0] + 4.2, A[1] + 3.4, ride + 1.8))
cam = bpy.context.active_object
cam.name = '概念展示相机'
cam.rotation_euler = aim_rotation(cam.location, (A[0], A[1], ride + 0.35))
cam.data.lens = 42
master.objects.link(cam)
for c in list(cam.users_collection):
    if c != master:
        c.objects.unlink(cam)
bpy.context.scene.camera = cam

try:
    w = bpy.context.scene.world
    if w is None:
        w = bpy.data.worlds.new('世界')
        bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0.72, 0.80, 0.76, 1.0)
        bg.inputs[1].default_value = 1.0
except Exception:
    pass

# 选中机器人根节点，尝试切到材质预览
try:
    bpy.ops.object.select_all(action='DESELECT')
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    break
            break
except Exception:
    pass

print('✅ 四足机器狗式林下作业机器人场景已生成：机器人 / 地形 / 植被 / 岩石与倒木 四个集合。')
print('✅ 四足按地形高度逐腿逆解站立，机身水平；机械臂正对准疏伐候选木。')
print('✅ 播放动画（空格）：雷达与锯片旋转 + 整机爆炸分解演示（第 36–60 帧展开）。')
print('✅ 手动控制分解：选中“机器人”→ N 面板 → 自定义属性 → 拖动「爆炸系数」滑杆（0–1）。')
