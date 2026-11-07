# -*- coding: utf-8 -*-
"""自主基准全场景实录视频: 5 场景 × 1 种子, 含成功/失败标注"""
import math
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, '..')

import imageio
FPS = 30
W, H = 1280, 720

# 场景描述 (中文标注)
SCENARIOS = [
    ('corridor', '走廊导航 — 通过'),
    ('slope', '3°斜坡 — 通过'),
    ('dropout', '传感器中断 — 通过'),
    ('detour', '绕行 — 阻塞(物理宽度限制)'),
    ('dynamic', '动态障碍 — 阻塞'),
]

frames_all = []
from forest_autonomy import ForestAutonomy

for scenario, label in SCENARIOS:
    sim = ForestAutonomy(scenario=scenario, seed=100)
    try:
        renderer = mujoco.Renderer(sim.env.model, height=H, width=W)
        cam = mujoco.MjvCamera()
        scenario_frames = []
        outcome = ''
        for step_i in range(sim.env.max_steps):
            done, result = sim.step()
            if step_i % 2 == 0:  # 每2步渲染1帧 (60→30fps)
                # 跟随相机
                pos = sim.env.data.qpos[:3]
                yaw = math.atan2(
                    sim.env.data.xmat[1].reshape(3, 3)[1, 0],
                    sim.env.data.xmat[1].reshape(3, 3)[0, 0])
                cam.lookat[:] = [pos[0], pos[1], pos[2] + 0.3]
                cam.distance = 4.5
                cam.azimuth = yaw * 180 / math.pi + 135
                cam.elevation = -15
                renderer.update_scene(sim.env.data, cam)
                frame = renderer.render()
                scenario_frames.append(frame)
            if done:
                if result['success']:
                    outcome = '✓ 到达目标'
                elif result['blocked']:
                    outcome = '✗ 阻塞 (剩余 %.1fm)' % result['distance_m']
                elif result['fallen']:
                    outcome = '✗ 摔倒'
                elif result['collision']:
                    outcome = '✗ 碰撞'
                else:
                    outcome = '✗ 超时'
                break
        if not outcome:
            outcome = '✗ 超时'

        # 生成标题帧 (2秒)
        title = np.full((H, W, 3), 240, dtype=np.uint8)
        import cv2
        cv2.putText(title, label, (W//2 - 200, H//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2)
        cv2.putText(title, outcome, (W//2 - 150, H//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 140, 0) if '✓' in outcome else (180, 30, 30), 2)
        for _ in range(FPS * 2):
            frames_all.append(title)

        # 加场景帧 (最多10秒)
        frames_all.extend(scenario_frames[:FPS * 10])

        # 结尾帧 (1秒)
        ending = np.full((H, W, 3), 240, dtype=np.uint8)
        cv2.putText(ending, outcome, (W//2 - 150, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 140, 0) if '✓' in outcome else (180, 30, 30), 2)
        for _ in range(FPS):
            frames_all.append(ending)

        renderer.close()
        print(f'{scenario}: {outcome} ({len(scenario_frames)} 帧)')
    except Exception as e:
        print(f'{scenario}: 渲染失败 {e}')
    finally:
        sim.close()

if frames_all:
    imageio.mimsave('autonomy_benchmark.mp4', frames_all, fps=FPS,
                    quality=9, macro_block_size=None)
    print(f'VIDEO_OK autonomy_benchmark.mp4 {len(frames_all)} 帧 '
          f'{os.path.getsize("autonomy_benchmark.mp4") // 1024} KB')
