# -*- coding: utf-8 -*-
"""Blender 无头运行包装：执行机器狗场景生成脚本并保存 .blend 文件"""
import bpy

exec(open(r'E:\forestrobo\forest_robot_blender.py', encoding='utf-8').read())

bpy.ops.wm.save_as_mainfile(filepath=r'E:\forestrobo\林下作业机器狗场景.blend')
print('BLEND_SAVED')
