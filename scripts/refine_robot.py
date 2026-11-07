"""Generate a robot-only, modular concept assembly from the original scene.

Blender: --background original.blend --python scripts/refine_robot.py
The original file is never overwritten. Proposed specifications are labelled.
"""
import bpy
import json
import math
from pathlib import Path
from mathutils import Vector

BASE = Path(__file__).resolve().parents[1]
catalog = json.loads((BASE/'design/configs/platforms.json').read_text(encoding='utf-8'))
root = bpy.data.objects['机器人']
root.animation_data_clear()
root['爆炸系数'] = 0.
root.location = (0, 0, .55)
root.rotation_euler = (0, 0, 0)
robot = bpy.data.collections['机器人']


def descendant(ob, parent):
    while ob:
        if ob == parent:
            return True
        ob = ob.parent
    return False


old_arm = bpy.data.objects.get('机械臂·基座回转')
for ob in list(bpy.data.objects):
    if not descendant(ob, root) or (old_arm and descendant(ob, old_arm)):
        bpy.data.objects.remove(ob, do_unlink=True)
body = bpy.data.objects['躯干']


def mat(name, color, metallic=0., rough=.45):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Metallic'].default_value = metallic
    p.inputs['Roughness'].default_value = rough
    return m


orange = mat('作业橙', (.72,.24,.055), .35)
black = mat('阳极氧化炭灰', (.045,.065,.07), .45)
steel = mat('拉丝金属', (.47,.56,.59), .8, .28)
rubber = mat('足端与轮胎橡胶', (.025,.035,.032), 0, .85)
glass = mat('光学玻璃', (.025,.19,.21), .6, .15)
green = mat('导航标识绿', (.20,.64,.41), .3)


def meta(ob, group, spec, note='', variants='patrol,survey,work', tool=''):
    ob['group'] = group
    ob['spec'] = spec
    ob['note'] = note
    ob['variants'] = variants
    ob['tool'] = tool
    return ob


def finish(ob, name, parent, loc, material, group, spec, **kwargs):
    ob.name = name
    ob.parent = parent
    ob.location = loc
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    robot.objects.link(ob)
    ob.data.materials.clear()
    ob.data.materials.append(material)
    return meta(ob, group, spec, **kwargs)


def box(name, parent, loc, size, material=black, group='躯干', spec='', **kwargs):
    bpy.ops.mesh.primitive_cube_add(size=1)
    ob = bpy.context.object
    ob.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = ob.modifiers.new('制造圆角示意', 'BEVEL')
    bevel.width = min(size)*.16
    bevel.segments = 3
    ob.modifiers.new('加权法线', 'WEIGHTED_NORMAL')
    return finish(ob, name, parent, loc, material, group, spec, **kwargs)


def cyl(name, parent, loc, radius, length, material=steel, axis='Z', group='机械臂', spec='', **kwargs):
    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=radius, depth=length)
    ob = finish(bpy.context.object, name, parent, loc, material, group, spec, **kwargs)
    ob.rotation_euler = (math.pi/2,0,0) if axis=='Y' else (0,math.pi/2,0) if axis=='X' else (0,0,0)
    for p in ob.data.polygons:
        p.use_smooth = len(p.vertices)==4
    bevel = ob.modifiers.new('边缘倒角', 'BEVEL')
    bevel.width = min(radius*.08, .003)
    bevel.segments = 2
    ob.modifiers.new('端面法线', 'WEIGHTED_NORMAL')
    return ob


def joint(name, parent, loc, rotation=(0,0,0)):
    ob = bpy.data.objects.new(name, None)
    robot.objects.link(ob)
    ob.parent, ob.location, ob.rotation_euler = parent, loc, rotation
    return ob


def explode(ob, vector):
    for i, offset in enumerate(vector):
        if not offset:
            continue
        base = ob.location[i]
        driver = ob.driver_add('location', i).driver
        driver.expression = f'{base}+{offset}*e'
        var = driver.variables.new()
        var.name = 'e'
        var.targets[0].id = root
        var.targets[0].data_path = '["爆炸系数"]'


def bolts(parent, prefix, x=0., radius=.045, axis='Y', variants='patrol,survey,work', group='机械臂'):
    for i in range(6):
        a = i*math.tau/6
        pos = (radius*math.cos(a),x,radius*math.sin(a)) if axis=='Y' else (x,radius*math.cos(a),radius*math.sin(a))
        cyl(f'{prefix}·紧固件{i+1}', parent, pos, .004, .007, black, axis,
            group=group, spec='M4外形示意 · 紧固等级/预紧力待设计', variants=variants)


# Refine retained body/legs; normalize stance independently of the old terrain.
for ob in list(robot.all_objects):
    if ob.type != 'MESH':
        continue
    group = '腿' if ob.name.startswith('腿') else '导航' if any(s in ob.name for s in ('雷达','相机','镜头','RTK','天线')) else '躯干'
    dims = ' × '.join(str(round(float(v)*1000)) for v in ob.dimensions)
    meta(ob, group, f'外形包围盒约 {dims} mm · 原始概念模型')
    if ob.name in ('上甲板','电池舱','主躯干') or '护板' in ob.name or '大腿' in ob.name or '小腿' in ob.name:
        bevel = ob.modifiers.new('结构倒角', 'BEVEL')
        bevel.width, bevel.segments = .07, 3  # local geometry coordinates
        ob.modifiers.new('结构法线', 'WEIGHTED_NORMAL')
    if '足端' in ob.name:
        ob['variants'] = 'patrol,survey'
    if '镜头' in ob.name:
        ob.rotation_euler = (0,math.pi/2,0)
    if '雷达' in ob.name:
        ob['note'] = '现有概念外形；扫描范围、点频、量程与防护等级待选型。不能保证机械臂不遮挡。'

for side in ('左前','右前','左后','右后'):
    bpy.data.objects[f'腿·{side}·髋侧摆'].rotation_euler = (0,0,0)
    bpy.data.objects[f'腿·{side}·髋前摆'].rotation_euler = (0,-.65,0)
    knee = bpy.data.objects[f'腿·{side}·膝']
    knee.rotation_euler = (0,1.3,0)
    parent = bpy.data.objects[f'腿·{side}·髋前摆']
    cyl(f'腿·{side}·髋编码器盖', parent, (0,.081,0), .058,.012, steel,'Y',group='腿',spec='双编码器安装盖示意 · 编码器待选型')
    bolts(parent, f'腿·{side}·髋', x=.089, radius=.046,group='腿')
    foot = bpy.data.objects[f'腿·{side}·足端']
    wheel = joint(f'腿·{side}·轮足模组', knee, tuple(foot.location))
    cyl(f'腿·{side}·主动轮胎', wheel,(0,0,.027),.075,.065,rubber,'Y',group='腿',spec='轮半径75 mm · 与RL通用轮足参数一致',variants='work')
    cyl(f'腿·{side}·轮毂电机',wheel,(0,.037,.027),.041,.012,orange,'Y',group='腿',spec='轮毂外形示意 · 扭矩/功率待选型',variants='work')
    for i in range(10):
        a=i*math.tau/10
        tread=box(f'腿·{side}·防滑胎纹{i+1}',wheel,(.074*math.cos(a),0,.027+.074*math.sin(a)),(.016,.069,.009),rubber,'腿','胎纹外形示意',variants='work')
        tread.rotation_euler[1]=-a

# Navigation module: additional sensors are proposals, not a tested configuration.
nav = joint('导航·后视模组', body,(-.42,0,.02), (0,0,math.pi))
box('导航·后视双目壳体',nav,(0,0,0),(.055,.15,.065),black,'导航','新增建议 · 外形55×150×65 mm',note='倒退及工具作业盲区观测；视场需实测')
for y in (-.045,.045):
    cyl(f'导航·后视镜头{y}',nav,(.033,y,0),.018,.018,glass,'X',group='导航',spec='全局快门镜头外形 · 焦距/FOV待选型')
for side in (-1,1):
    mount=joint(f'导航·侧向模组{side}',body,(.12,side*.19,.015),(0,0,side*math.pi/2))
    box(f'导航·侧向深度壳体{side}',mount,(0,0,0),(.045,.10,.05),black,'导航','新增建议 · 侧向近场感知',note='低矮障碍与树干侧向距离；未实现硬件驱动')
    cyl(f'导航·侧向深度窗口{side}',mount,(.026,0,0),.016,.012,glass,'X',group='导航',spec='深度/ToF视窗示意 · 量程待选型')
    explode(mount,(0,side*.18,.05))
imu=box('导航·IMU隔振底座',body,(-.06,0,-.035),(.075,.06,.028),green,'导航','新增建议 · 75×60×28 mm',note='靠近躯干中心；需硬件时间同步和外参标定')
explode(imu,(0,0,-.28))
for side in (-1,1):
    for i in range(8):
        box(f'躯干·散热鳍片{side}-{i}',body,(-.16+i*.037,side*.182,-.012),(.014,.012,.078),steel,'躯干','散热外形示意 · 热设计待验证')
box('躯干·急停底座',body,(-.28,.135,.235),(.065,.055,.028),black,'躯干','新增建议 · 独立安全回路接口待实现')
cyl('躯干·硬件急停蘑菇头',body,(-.28,.135,.261),.025,.025,mat('急停红',(.65,.025,.02)),group='躯干',spec='硬件急停外形 · 不等同已接线安全功能')


def make_arm(key, variant):
    spec = catalog['arms'][key]
    length1,length2,tip = spec['links_m']
    n = spec['dof']
    note = spec['source']
    base = joint(f'{key}·J1基座回转',body,(-.16,0,.205),(0,0,-.25))
    cyl(f'{key}·J1回转法兰',base,(0,0,.035),.078,.07,orange,group='机械臂',spec=f'J1 · {spec["rated_torque_nm"][0]} N·m额定目标',note=note,variants=variant)
    shoulder=joint(f'{key}·J2肩俯仰',base,(0,0,.075),(0,-1.05,0))
    cyl(f'{key}·J2肩驱动器',shoulder,(0,0,0),.060,.11,black,'Y',group='机械臂',spec=f'J2 · {spec["rated_torque_nm"][1]} N·m额定目标',note=note,variants=variant)
    bolts(shoulder,f'{key}·肩',x=.06,variants=variant)
    box(f'{key}·大臂碳管',shoulder,(length1/2,0,0),(length1,.065,.075),orange,'机械臂',f'L1 = {length1*1000:.0f} mm · 概念尺寸',variants=variant)
    box(f'{key}·大臂走线槽',shoulder,(length1/2,-.043,0),(length1*.8,.017,.024),black,'机械臂','内走线槽/检修盖示意',variants=variant)
    elbow=joint(f'{key}·J3肘俯仰',shoulder,(length1,0,0),(0,1.25,0))
    cyl(f'{key}·J3肘驱动器',elbow,(0,0,0),.05,.095,orange,'Y',group='机械臂',spec=f'J3 · {spec["rated_torque_nm"][2]} N·m额定目标',note=note,variants=variant)
    bolts(elbow,f'{key}·肘',x=.052,radius=.035,variants=variant)
    box(f'{key}·小臂碳管',elbow,(length2/2,0,0),(length2,.052,.06),black,'机械臂',f'L2 = {length2*1000:.0f} mm · 概念尺寸',variants=variant)
    wrist=joint(f'{key}·腕俯仰座',elbow,(length2,0,0),(0,-.2,0))
    if n==5:
        cyl(f'{key}·J4腕俯仰驱动器',wrist,(0,0,0),.039,.08,steel,'Y',group='机械臂',spec='J4 · 18 N·m额定目标',note=note,variants=variant)
    box(f'{key}·腕连接臂',wrist,(tip/2,0,0),(tip,.045,.045),orange,'机械臂',f'腕段 = {tip*1000:.0f} mm',variants=variant)
    roll=joint(f'{key}·J{n}腕回转',wrist,(tip,0,0))
    cyl(f'{key}·J{n}腕回转模组',roll,(0,0,0),.041,.052,black,'X',group='机械臂',spec=f'J{n} · {spec["rated_torque_nm"][-1]} N·m额定目标',note=note,variants=variant)
    cyl(f'{key}·六维力传感器',roll,(.04,0,0),.039,.023,green,'X',group='机械臂',spec='量程待选型 · 工作载荷须留余量',note='原方案500 N量程用于500 N工况无裕度，需重新选型',variants=variant)
    cyl(f'{key}·电动快换盘',roll,(.065,0,0),.045,.022,steel,'X',group='机械臂',spec='外径90 mm概念尺寸 · 锁紧/供电待设计',variants=variant)
    bolts(roll,f'{key}·法兰',x=.08,radius=.034,axis='X',variants=variant)
    box(f'{key}·腕部相机',roll,(.015,-.062,.025),(.045,.035,.045),black,'导航','新增腕部视觉 · 焦距/近距标定待选型',variants=variant)
    cyl(f'{key}·腕部镜头',roll,(.043,-.062,.025),.012,.014,glass,'X',group='导航',spec='腕部视觉窗口',variants=variant)
    explode(base,(0,0,.4))
    explode(shoulder,(.1,0,.04))
    explode(elbow,(.12,0,.03))
    explode(wrist,(.09,0,0))
    explode(roll,(.12,0,0))
    return joint(f'{key}·工具接口',roll,(.09,0,0))


light_tcp=make_arm('light4','survey')
work_tcp=make_arm('work5','work')


def tool_box(name, parent, loc, size, tool, material=black):
    variant='survey' if tool in ('caliper','sampler') else 'work'
    return box(name,parent,loc,size,material,'工具',catalog['tools'][tool]['limit'],variants=variant,tool=tool,note=catalog['tools'][tool]['source'])


# Distinct interchangeable tools occupy the same TCP; viewer displays one at a time.
for y in (-.22,.22):
    tool_box('T3·卡尺测臂'+str(y),light_tcp,(.15,y,0),(.30,.018,.045),'caliper',steel)
tool_box('T3·400mm测径导轨',light_tcp,(.015,0,0),(.04,.46,.055),'caliper',orange)
tool_box('T3·磁栅与电机壳',light_tcp,(.02,0,.046),(.075,.13,.04),'caliper')
for y in (-.22,.22):
    tool_box('T3·恒力接触测头'+str(y),light_tcp,(.285,y,0),(.035,.032,.055),'caliper',black)
for y in (-.036,.036):
    tool_box('T2·样品夹持指'+str(y),light_tcp,(.10,y,0),(.16,.014,.025),'sampler',steel)
cyl('T2·密封样品管',light_tcp,(.12,0,0),.022,.14,green,'Z',group='工具',spec='样品管外径44 mm示意 · 非土钻',variants='survey',tool='sampler')
tool_box('T4·夹爪电机壳',work_tcp,(.045,0,0),(.09,.11,.07),'gripper',orange)
for side in (-1,1):
    tool_box('T4·180mm夹持指'+str(side),work_tcp,(.14,side*.09,0),(.17,.022,.065),'gripper',steel)
    tool_box('T4·防滑接触垫'+str(side),work_tcp,(.20,side*.076,0),(.055,.009,.055),'gripper',rubber)
tool_box('T1·主轴驱动壳',work_tcp,(.075,0,0),(.15,.085,.085),'saw',orange)
cyl('T1·200mm锯片',work_tcp,(.17,0,0),.10,.005,steel,'Y',group='工具',spec='Φ200 mm · 500 W/4500 rpm为方案目标，非已验证能力',variants='work',tool='saw')
for i in range(32):
    a=i*math.tau/32
    tooth=tool_box(f'T1·锯齿{i+1}',work_tcp,(.17+.101*math.cos(a),0,.101*math.sin(a)),(.014,.008,.009),'saw',steel)
    tooth.rotation_euler[1]=-a
# Half-ring guard covers the upper blade, leaving the work sector visible.
for i in range(12):
    a=(i+.5)*math.pi/12
    guard=tool_box(f'T1·分段防护罩{i+1}',work_tcp,(.17+.115*math.cos(a),0,.115*math.sin(a)),(.035,.045,.025),'saw',orange)
    guard.rotation_euler[1]=-a
for side in (-1,1):
    tool_box(f'T1·集成抱夹指{side}',work_tcp,(.23,side*.075,-.035),(.15,.018,.025),'saw',black)

# Robot-only file: no tree, floor, camera or lights. Web derives framing from mesh.
root['model_version']='modular-concept-v2'
root['catalog_path']='design/configs/platforms.json'
bpy.context.scene.frame_set(1)
root['爆炸系数']=0.
bpy.context.view_layer.update()
output=BASE/'design/forest_robot_modular.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(output))
print('Saved detailed modular robot:', output)
import runpy
runpy.run_path(str(BASE/'scripts/export_robot_web.py'),run_name='__main__')
