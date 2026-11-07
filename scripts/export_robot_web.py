"""Blender background export of the existing robot, including driver explosion."""
import bpy
import json
from pathlib import Path

out = Path(__file__).resolve().parents[1] / 'web' / 'robot.json'
robot = bpy.data.collections.get('机器人')
if robot is None:
    raise RuntimeError('Robot collection missing; generate the Blender scene first')
root = bpy.data.objects['机器人']
root.animation_data_clear()
meshes = {}


def belongs_to_robot(ob):
    # Some tree decorations were assigned to the robot collection by helpers.
    # Assembly membership follows the parent hierarchy, not the collection.
    parent = ob
    while parent is not None:
        if parent == root:
            return True
        parent = parent.parent
    return False


for factor in (0, 1):
    root['爆炸系数'] = float(factor)
    root.update_tag()
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    for ob in robot.all_objects:
        if ob.type != 'MESH' or not belongs_to_robot(ob):
            continue
        ev = ob.evaluated_get(deps)
        mesh = ev.to_mesh()
        mesh.calc_loop_triangles()
        coords = [tuple(round(v, 5) for v in (ev.matrix_world @ vert.co)) for vert in mesh.vertices]
        if factor == 0:
            colors = []
            for mat in mesh.materials:
                node = mat.node_tree.nodes.get('Principled BSDF') if mat and mat.use_nodes else None
                colors.append(list(node.inputs['Base Color'].default_value[:3]) if node else [0.5]*3)
            normal_matrix = ev.matrix_world.to_3x3().inverted().transposed()
            normals = [[tuple(round(v, 5) for v in (normal_matrix @ mesh.corner_normals[i].vector).normalized())
                        for i in t.loops] for t in mesh.loop_triangles]
            dimensions = [round(float(v)*1000, 1) for v in ob.dimensions]
            label = ob.name.replace('light4', '轻型臂').replace('work5', '作业臂')
            meshes[ob.name] = dict(name=ob.name, label=label, vertices=coords,
                triangles=[[list(t.vertices), t.material_index] for t in mesh.loop_triangles],
                normals=normals, colors=colors, dimensions_mm=dimensions,
                group=ob.get('group', ''), spec=ob.get('spec', '原始概念外形，未选型'),
                note=ob.get('note', ''), variants=ob.get('variants', 'patrol,survey,work').split(','),
                tool=ob.get('tool', ''))
        else:
            meshes[ob.name]['exploded'] = coords
        ev.to_mesh_clear()
catalog_path = Path(__file__).resolve().parents[1]/'design/configs/platforms.json'
catalog = json.loads(catalog_path.read_text(encoding='utf-8')) if catalog_path.exists() else {}
out.write_text(json.dumps(dict(source=Path(bpy.data.filepath).name, catalog=catalog,
    version=root.get('model_version', 'original'), parts=list(meshes.values())), ensure_ascii=False), encoding='utf-8')
print('Exported', len(meshes), 'parts to', out)
import runpy
runpy.run_path(str(Path(__file__).with_name('build_robot_web.py')), run_name='__main__')
