# -*- coding: utf-8 -*-
"""林下机器狗环境演示视频（步态修复版）：
段 1 — 场景环视（站立） 段 2 — 跟随小跑
输出: forest_dog_demo.mp4 (1280x720, 30fps)
"""
import json
import math
import os

import imageio.v2 as imageio
import mujoco

import gait
import terrain as terr

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

if not os.path.exists('forest_dog.xml'):
    from generate_model import generate
    generate()

model = mujoco.MjModel.from_xml_path('forest_dog.xml')
model.opt.timestep = 0.005
data = mujoco.MjData(model)
with open('terrain_meta.json', encoding='utf-8') as f:
    h = terr.build_from_params(json.load(f))
FPS = 30


def pitch_of():
    w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))


def reset():
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [1.0, 0.0, float(h(1.0, 0.0)) + 0.45]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = gait.Q0
    data.ctrl[:] = gait.Q0
    mujoco.mj_forward(model, data)


def main():
    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    frames = []

    def advance(seconds, use_cpg):
        target = data.time + seconds
        while data.time < target:
            data.ctrl[:] = (gait.cpg(data.time, pitch=math.radians(pitch_of()))
                            if use_cpg else gait.Q0)
            mujoco.mj_step(model, data)

    def snap(lookat, dist, azim, elev):
        cam.lookat[:] = lookat
        cam.distance = dist
        cam.azimuth = azim
        cam.elevation = elev
        renderer.update_scene(data, cam)
        frames.append(renderer.render())

    # 段 1: 山地场景环视 7 s（站立）
    reset()
    for k in range(7 * FPS):
        advance(1.0 / FPS, use_cpg=False)
        p = k / (7 * FPS)
        cx, cy = 3.2 + 1.5 * p, 0.5
        snap([cx, cy, float(h(cx, cy)) + 1.0], 10.0 - 3.0 * p,
             35 + 80 * p, -24 - 8 * p)

    # 段 2: 跟随小跑 16 s
    reset()
    for k in range(16 * FPS):
        advance(1.0 / FPS, use_cpg=True)
        x, y = data.qpos[0], data.qpos[1]
        yaw = math.atan2(2 * (data.qpos[3] * data.qpos[6]),
                         1 - 2 * (data.qpos[6] ** 2))
        snap([x, y, float(h(x, y)) + 0.72], 3.2, math.degrees(yaw) + 135, -13)

    renderer.close()
    out = os.path.join(BASE, 'forest_terrain_demo.mp4')
    imageio.mimsave(out, frames, fps=FPS, quality=9, macro_block_size=None)
    print(f'VIDEO_OK {out} {len(frames)} 帧 {os.path.getsize(out) // 1024} KB')


if __name__ == '__main__':
    main()
