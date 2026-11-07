"""Add two earthquake-response concept variants without changing forest models."""
import bpy
import json
import math
import runpy
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
root=bpy.data.objects['机器人'];body=bpy.data.objects['躯干'];robot=bpy.data.collections['机器人']
root.animation_data_clear();root['爆炸系数']=0.
# Reuse only modelling helpers, not the scene-generation side effects.
source=(BASE/'scripts/refine_robot.py').read_text(encoding='utf-8')
helpers=source[source.index('def mat('):source.index('# Refine retained body/legs')]
exec(helpers,globals())
for ob in list(robot.all_objects):
    if ob.type!='MESH':continue
    variants=ob.get('variants','patrol,survey,work').split(',')
    if 'patrol' in variants or 'survey' in variants:variants.append('rescue_scout')
    if ('work' in variants or ob.name.startswith('work5')) and ob.get('group')!='腿' and not ob.get('tool'):
        variants.append('rescue_clear')
    # Existing light arm is shared with scout; forestry tools remain exclusive.
    if ob.name.startswith('light4') and not ob.get('tool'):variants.append('rescue_scout')
    if ob.get('tool'):variants=[v for v in variants if not v.startswith('rescue_')]
    ob['variants']=','.join(dict.fromkeys(variants))

all_rescue='rescue_scout,rescue_clear'
nav=joint('地震·多模态探测塔',body,(.20,0,.20))
box('地震·探测塔减振底座',nav,(0,0,0),(.16,.20,.035),black,'导航','新增概念 · 160×200 mm安装底座',variants=all_rescue,note='移动时数据作线索；微动检测需停稳和隔振')
box('地震·热像可见光云台',nav,(0,0,.09),(.12,.16,.11),orange,'导航','双通道壳体示意 · 分辨率/温标待选型',variants=all_rescue)
for side in (-1,1):cyl(f'地震·双通道窗口{side}',nav,(.068,side*.045,.09),.024,.025,glass,'X',group='导航',spec='热像/可见光窗口 · 热像不能直接透视瓦砾',variants=all_rescue)
box('地震·雷达微动阵列',nav,(.015,0,.195),(.065,.19,.095),green,'导航','雷达罩190×95 mm外形 · 频段/探测距离待选型',variants=all_rescue,note='金属、含水介质和环境运动可能干扰；未实现生命检测算法')
for i in range(4):cyl(f'地震·拾音阵列{i}',nav,(.053,-.075+i*.05,.195),.005,.012,black,'X',group='导航',spec='拾音孔位 · 停止电机/切削后采集',variants=all_rescue)
box('地震·多气体采样舱',body,(.02,.205,.09),(.12,.08,.07),orange,'导航','气体检测安装舱 · 组分/阈值按实际选型标定',variants=all_rescue,note='未知/超时不得自动允许切割；外形不代表防爆认证')
box('地震·通信中继与光纤卷盘',body,(-.31,0,.25),(.15,.21,.09),steel,'躯干','通信/系留回传建议模块 · 带宽/卷盘扭矩待设计',variants=all_rescue,note='系留线存在缠绕风险，需张力与牵引点校核')
explode(nav,(.1,0,.65))

track_parent=joint('地震·履带底盘总成',body,(0,0,-.28))
for side in (-1,1):
    group=joint(f'地震·履带{side}',track_parent,(0,side*.30,0))
    box(f'地震·履带承载连接{side}',body,(0,side*.20,-.15),(.64,.12,.18),steel,'躯干','概念承载连接 · 未经结构强度校核',variants='rescue_clear')
    verts=[];faces=[];n=48
    for radius,y in ((.17,-.085),(.17,.085),(.14,-.085),(.14,.085)):
        for i in range(n):
            a=i*math.tau/n;x=(.32 if math.cos(a)>=0 else -.32)+radius*math.cos(a)
            verts.append((x,y,radius*math.sin(a)))
    for i in range(n):
        j=(i+1)%n
        for a,b in ((0,n),(2*n,3*n),(0,2*n),(n,3*n)):faces.append((a+i,a+j,b+j,b+i))
    mesh=bpy.data.meshes.new('履带闭环网格');mesh.from_pydata(verts,[],faces);mesh.update()
    ob=bpy.data.objects.new(f'地震·连续履带{side}',mesh);robot.objects.link(ob)
    ob.parent=group;ob.data.materials.append(rubber)
    meta(ob,'腿','外形约980×170×340 mm/侧 · 无动力学验证',variants='rescue_clear',note='轻型履带建议，不代表瓦砾/楼梯通过能力')
    for i,x in enumerate((-.32,-.1,.1,.32)):
        cyl(f'地震·承重轮{side}-{i}',group,(x,0,0),.132,.15,orange,'Y',group='腿',spec='承重轮外径264 mm外形示意',variants='rescue_clear')
    for top in (-1,1):
        for i in range(9):box(f'地震·履带齿{side}-{top}-{i}',group,(-.30+i*.075,0,top*.176),(.026,.175,.025),rubber,'腿','履带齿外形 · 附着力待试验',variants='rescue_clear')
    explode(group,(0,side*.35,-.08))
for side in (-1,1):
    box(f'地震·驻车支撑臂{side}',body,(.38,side*.28,-.25),(.16,.08,.30),steel,'躯干','新增概念支撑脚 · 可调高度机构待设计',variants='rescue_clear')
    box(f'地震·支撑脚垫{side}',body,(.38,side*.28,-.415),(.17,.15,.035),rubber,'躯干','支撑面积示意 · 不等同地面锚固',variants='rescue_clear')

def toolpart(name,parent,loc,size,tool,variant,material=orange):
    return box(name,parent,loc,size,material,'工具','新增地震工具概念 · 参数待选型',variants=variant,tool=tool)

light=bpy.data.objects['light4·工具接口'];work=bpy.data.objects['work5·工具接口']
toolpart('地震·可部署探头主机',light,(.09,0,0),(.16,.09,.065),'life_probe','rescue_scout',green)
cyl('地震·缝隙摄像探杆',light,(.24,0,0),.012,.20,steel,'X',group='工具',spec='Φ24×200 mm外形建议 · 可换柔性探头',variants='rescue_scout',tool='life_probe')
cyl('地震·探头镜窗',light,(.348,0,0),.011,.012,glass,'X',group='工具',spec='近距照明/成像探头 · 不自动诊断生命体征',variants='rescue_scout',tool='life_probe')
toolpart('地震·往复锯驱动壳',work,(.095,0,0),(.19,.09,.09),'rescue_cut','rescue_clear')
toolpart('地震·往复锯条',work,(.27,0,-.02),(.19,.008,.028),'rescue_cut','rescue_clear',steel)
toolpart('地震·切割防护与夹持座',work,(.13,0,-.07),(.20,.12,.035),'rescue_cut','rescue_clear',black)
for side in (-1,1):toolpart(f'地震·切割抱夹指{side}',work,(.24,side*.055,-.055),(.14,.014,.035),'rescue_cut','rescue_clear',steel)
for arm,parent,variant in [('轻臂',light,'rescue_scout'),('作业臂',work,'rescue_clear')]:
    toolpart(f'地震·{arm}清障夹爪座',parent,(.05,0,0),(.10,.10,.06),'rescue_gripper',variant)
    for side in (-1,1):toolpart(f'地震·{arm}柔顺夹指{side}',parent,(.16,side*.065,0),(.17,.018,.065),'rescue_gripper',variant,steel)

root['model_version']='modular-forest-rescue-concept-v3'
root['爆炸系数']=0.;bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(BASE/'design/forest_robot_rescue.blend'))
runpy.run_path(str(BASE/'scripts/export_robot_web.py'),run_name='__main__')
