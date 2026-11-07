# -*- coding: utf-8 -*-
"""多狗编队演示：4×4 十六只机器狗对角小跑编队
用法:
  python swarm.py video   → 渲染 forest_swarm_demo.mp4
  python swarm.py view    → 交互式 3D 窗口（鼠标旋转/缩放）
"""
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import gait

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

ROWS, COLS = 4, 4
N = ROWS * COLS
LANE_DY = 1.35          # 横向车间距
ROW_DX = 2.8            # 纵向排间距
X0 = 3.0
PH = [i * 0.83 for i in range(N)]   # 每狗随机化相位


def dog_xml(tag, x, y):
    legs = ''
    for px, py, fw, sd, nm in ((0.27, -0.155, 0.03, -1, 'FR'),
                               (0.27, 0.155, 0.03, +1, 'FL'),
                               (-0.27, -0.155, -0.03, -1, 'RR'),
                               (-0.27, 0.155, -0.03, +1, 'RL')):
        y0, y1 = 0.03 * sd, 0.07 * sd
        legs += f"""
    <body name="{tag}_{nm}_hip" pos="{px} {py} 0">
      <joint name="{tag}_{nm}_abd" class="leg" axis="1 0 0" range="-0.6 0.6"/>
      <geom type="capsule" fromto="0 {y0} 0 0 {y1} 0" size="0.045" mass="0.7"/>
      <body name="{tag}_{nm}_thigh" pos="0 {y1} 0">
        <joint name="{tag}_{nm}_hip" class="leg" axis="0 1 0" range="-0.8 2.4"/>
        <geom type="capsule" fromto="0 0 0 {fw} 0 -0.30" size="0.048" mass="1.3"/>
        <body name="{tag}_{nm}_calf" pos="{fw} 0 -0.30">
          <joint name="{tag}_{nm}_knee" class="leg" axis="0 1 0" range="-2.8 -0.4"/>
          <geom type="capsule" fromto="0 0 0 {-abs(fw) * 0.7:.2f} 0 -0.32" size="0.036" mass="0.32"/>
          <geom name="{tag}_{nm}_foot" type="sphere" pos="{-abs(fw) * 0.7:.2f} 0 -0.32"
                size="0.026" mass="0.06" friction="1.3 0.5 0.5"/>
        </body>
      </body>
    </body>"""
    return f"""
  <body name="{tag}_torso" pos="{x} {y} 0.45">
    <freejoint name="{tag}_root"/>
    <geom name="{tag}_torso_geom" type="box" size="0.39 0.15 0.085" mass="9.0"
          rgba="0.85 0.44 0.18 1"/>
    <geom name="{tag}_deck" type="box" pos="0 0 0.105" size="0.33 0.13 0.022"
          mass="2.2" rgba="0.20 0.20 0.22 1"/>
    <geom name="{tag}_lidar" type="cylinder" pos="0.28 0 0.19" size="0.05 0.045"
          mass="0.4" rgba="0.13 0.13 0.15 1"/>{legs}
  </body>"""


def swarm_xml():
    bodies, acts = '', ''
    i = 0
    for r in range(ROWS):
        for c in range(COLS):
            tag = f'D{i:02d}'
            bodies += dog_xml(tag, X0 + r * ROW_DX, (c - (COLS - 1) / 2) * LANE_DY)
            for nm in ('FR', 'FL', 'RR', 'RL'):
                for j in ('abd', 'hip', 'knee'):
                    acts += f'    <position name="{tag}_{nm}_{j}" joint="{tag}_{nm}_{j}" class="servo"/>\n'
            i += 1
    return f"""<mujoco model="swarm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" iterations="50" solver="Newton" integrator="implicitfast"/>
  <default>
    <geom friction="1.1 0.4 0.4" condim="3" density="800" margin="0.002"/>
    <default class="leg"><joint type="hinge" damping="0.5" armature="0.015" frictionloss="0.04"/></default>
    <default class="servo"><position kp="300" kv="15" forcerange="-45 45"/></default>
  </default>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/>
    <map znear="0.02"/>
    <global offwidth="1600" offheight="900"/>
  </visual>
  <worldbody>
    <light pos="6 -4 8" dir="-0.4 0.3 -1" diffuse="0.9 0.9 0.85"/>
    <geom name="floor" type="plane" size="60 60 2" friction="1.2 0.5 0.5"/>
{bodies}
  </worldbody>
  <actuator>
{acts}  </actuator>
</mujoco>"""


def make_model():
    path = os.path.join(BASE, 'forest_swarm.xml')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(swarm_xml())
    m = mujoco.MjModel.from_xml_path(path)
    m.opt.timestep = 0.005
    return m


def reset(model, data):
    mujoco.mj_resetData(model, data)
    for i in range(N):
        tag_x = X0 + (i // COLS) * ROW_DX
        tag_y = (i % COLS - (COLS - 1) / 2) * LANE_DY
        q = slice(i * 19, (i + 1) * 19)
        data.qpos[q] = np.concatenate([[tag_x, tag_y, 0.45, 1, 0, 0, 0], gait.Q0])
        data.ctrl[i * 12:(i + 1) * 12] = gait.Q0
    mujoco.mj_forward(model, data)


def step(model, data):
    for i in range(N):
        data.ctrl[i * 12:(i + 1) * 12] = gait.cpg(data.time, ph0=PH[i])
    mujoco.mj_step(model, data)


def lead_x(data):
    return max(data.qpos[i * 19] for i in range(N))


def main_video():
    import imageio.v2 as imageio
    model = make_model()
    data = mujoco.MjData(model)
    reset(model, data)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    frames = []
    FPS = 30
    target = 0.0
    TOTAL = 24.0
    k = 0
    while data.time < TOTAL:
        target += 1.0 / FPS
        while data.time < target:
            step(model, data)
        p = data.time / TOTAL
        if p < 0.33:                       # 段1: 高空环绕
            cam.lookat[:] = [X0 + 1.5 * ROW_DX, 0, 0.3]
            cam.distance = 17 - 4 * (p / 0.33)
            cam.azimuth = 40 + 70 * (p / 0.33)
            cam.elevation = -52
        else:                              # 段2: 侧后跟随
            cx = float(np.mean([data.qpos[i * 19] for i in range(N)]))
            cam.lookat[:] = [cx, 0, 0.55]
            cam.distance = 7.5
            cam.azimuth = 105
            cam.elevation = -11
        renderer.update_scene(data, cam)
        frames.append(renderer.render())
        k += 1
    renderer.close()
    out = os.path.join(BASE, 'forest_swarm_demo.mp4')
    imageio.mimsave(out, frames, fps=FPS, quality=9, macro_block_size=None)
    print(f'VIDEO_OK {out} {len(frames)} 帧 {os.path.getsize(out) // 1024} KB')


def main_view():
    model = make_model()
    data = mujoco.MjData(model)
    reset(model, data)
    print('已打开编队窗口：16 只机器狗对角小跑编队前进；鼠标旋转/缩放，关窗退出。')
    with mujoco.viewer.launch_passive(model, data) as v:
        last = time.time()
        while v.is_running():
            now = time.time()
            n = int((now - last) / model.opt.timestep)
            last += n * model.opt.timestep
            for _ in range(min(n, 60)):
                step(model, data)
            v.sync()
            time.sleep(0.01)
            if lead_x(data) > 26 or data.qpos[2] < 0.12:
                reset(model, data)
    print('查看器已退出')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'view'
    main_video() if mode == 'video' else main_view()
