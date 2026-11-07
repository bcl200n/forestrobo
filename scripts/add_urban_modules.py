"""Two urban-renewal concept assemblies reusing the modular platform."""
import bpy
import math
import runpy
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
root=bpy.data.objects['机器人'];body=bpy.data.objects['躯干'];robot=bpy.data.collections['机器人']
root.animation_data_clear();root['爆炸系数']=0.
source=(BASE/'scripts/refine_robot.py').read_text(encoding='utf-8')
exec(source[source.index('def mat('):source.index('# Refine retained body/legs')],globals())
for ob in list(robot.all_objects):
    if ob.type!='MESH':continue
    variants=ob.get('variants','').split(',')
    if not ob.get('tool'):
        if 'work' in variants and not ob.name.startswith('work5'):variants.append('urban_scan')
        if 'survey' in variants and not ob.name.startswith('work5'):variants.append('urban_inspect')
    ob['variants']=','.join(dict.fromkeys(variants))

mast=joint('城市·扫描桅杆',body,(-.04,0,.20))
box('城市·桅杆锁紧底座',mast,(0,0,.025),(.18,.17,.05),black,'导航','新增概念安装座 · 停车扫描需锁紧',variants='urban_scan')
cyl('城市·扫描支柱',mast,(0,0,.23),.025,.38,steel,group='导航',spec='固定380 mm支柱外形 · 不是已验证升降机构',variants='urban_scan')
cyl('城市·三维扫描仪基座',mast,(0,0,.45),.10,.08,orange,group='导航',spec='仪器安装外形 · 实际量程/角误差待选型',variants='urban_scan')
box('城市·测绘扫描仪主机',mast,(0,0,.55),(.14,.17,.15),black,'导航','外形140×170×150 mm · 未选型扫描仪',variants='urban_scan',note='移动扫描用于覆盖；测绘成果需控制点和独立检查点验证')
for side in (-1,1):cyl(f'城市·扫描视窗{side}',mast,(side*.075,0,.55),.038,.018,glass,'X',group='导航',spec='扫描视窗外形示意',variants='urban_scan')
explode(mast,(0,0,.65))
for side in (-1,1):
    arm=joint(f'城市·全景相机支架{side}',body,(-.22,side*.21,.19))
    box(f'城市·全景相机{side}',arm,(0,0,0),(.07,.06,.07),orange,'导航','视觉补色/纹理记录建议模块',variants='urban_scan,urban_inspect')
    cyl(f'城市·全景镜头{side}',arm,(0,side*.037,0),.024,.018,glass,'Y',group='导航',spec='镜头外形 · 内参/外参/时间同步需标定',variants='urban_scan,urban_inspect')
box('城市·本地数据与同步舱',body,(-.27,0,.255),(.18,.19,.10),steel,'躯干','本地存储/时钟同步接口 · 具体容量待任务预算',variants='urban_scan,urban_inspect')
tcp=bpy.data.objects['light4·工具接口']
box('城市·接触检测头壳体',tcp,(.075,0,0),(.13,.065,.055),green,'工具','新增概念 · 质量预算0.8 kg，实际载荷待验证',variants='urban_inspect',tool='urban_probe')
cyl('城市·柔顺检测探针',tcp,(.17,0,0),.018,.06,steel,'X',group='工具',spec='接触探针外形 · 检测原理及力阈值待选型',variants='urban_inspect',tool='urban_probe')
box('城市·标尺成像模组',tcp,(.07,0,0),(.12,.09,.06),orange,'工具','新增概念 · 质量预算0.6 kg，未声称裂缝测宽精度',variants='urban_inspect',tool='urban_camera')
cyl('城市·近距成像镜头',tcp,(.14,0,0),.022,.025,glass,'X',group='工具',spec='标定后用于病害候选成像',variants='urban_inspect',tool='urban_camera')
for side in (-1,1):box(f'城市·成像补光与标尺{side}',tcp,(.12,side*.05,0),(.025,.012,.07),steel,'工具','补光/标尺示意 · 非自动结构鉴定',variants='urban_inspect',tool='urban_camera')
root['model_version']='modular-forest-rescue-urban-v4'
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(BASE/'design/forest_robot_urban.blend'))
runpy.run_path(str(BASE/'scripts/export_robot_web.py'),run_name='__main__')
