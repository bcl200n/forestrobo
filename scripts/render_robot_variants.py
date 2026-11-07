"""Studio previews and glTF assemblies of the three concept configurations."""
import bpy
print('PREVIEW: script entered', flush=True)
import json
from pathlib import Path
from mathutils import Vector

BASE=Path(__file__).resolve().parents[1]
catalog=json.loads((BASE/'design/configs/platforms.json').read_text(encoding='utf-8'))
robot=bpy.data.collections['机器人']
root=bpy.data.objects['机器人']
root.animation_data_clear()
root['爆炸系数']=0.
scene=bpy.context.scene
print('PREVIEW: configuring renderer', flush=True)
scene.render.engine='CYCLES'
scene.cycles.samples=24
scene.cycles.use_denoising=True
scene.render.resolution_x=1400
scene.render.resolution_y=1000
scene.render.resolution_percentage=100
scene.world.color=(.15,.19,.18)
scene.view_settings.view_transform='AgX'
scene.render.image_settings.file_format='PNG'
out=BASE/'design/previews'
out.mkdir(exist_ok=True)
print('PREVIEW: adding camera', flush=True)
bpy.ops.object.camera_add()
camera=bpy.context.object
scene.camera=camera
camera.data.type='ORTHO'
for position,power,size in [((1,-3,4),600,4),((-3,-1,2),500,3),((0,3,4),800,3)]:
    bpy.ops.object.light_add(type='AREA',location=position)
    ob=bpy.context.object
    ob.data.energy=power
    ob.data.shape='DISK'
    ob.data.size=size
    ob.rotation_euler=(Vector((0,0,.6))-ob.location).to_track_quat('-Z','Y').to_euler()
for variant,entry in catalog['variants'].items():
    print('PREVIEW: variant', variant, flush=True)
    tool=entry['tools'][0]
    visible=[]
    for ob in list(robot.all_objects):
        if ob.type!='MESH':
            continue
        show=variant in ob.get('variants','patrol,survey,work').split(',') and (not ob.get('tool','') or ob['tool']==tool)
        ob.hide_render=not show
        if show:
            visible.append(ob)
    bpy.context.view_layer.update()
    verts=[ob.matrix_world@Vector(v) for ob in visible for v in ob.bound_box]
    lo=Vector([min(v[i] for v in verts) for i in range(3)])
    hi=Vector([max(v[i] for v in verts) for i in range(3)])
    center=(lo+hi)/2
    camera.location=center+Vector((2.2,-3.1,1.6))
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.ortho_scale=max(1.4,(hi-lo).length*1.25)
    scene.render.filepath=str(out/f'{variant}.png')
    print('PREVIEW: rendering', scene.render.filepath, flush=True)
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.select_all(action='DESELECT')
    for ob in list(robot.all_objects):
        if ob.type=='EMPTY' or ob in visible:
            ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(BASE/'design'/f'robot_{variant}.glb'),
        export_format='GLB',use_selection=True,export_animations=False,
        export_extras=True,export_apply=True)
print(f'Rendered {len(catalog["variants"])} variants and GLB assemblies')
