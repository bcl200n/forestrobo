# -*- coding: utf-8 -*-
"""训练结果可视化回放（MuJoCo GUI 视图，单环境 CPU 仿真）

用法:
  .venv\Scripts\python.exe enjoy.py --model models/forest_dog_ppo/final.zip
"""
import argparse
import math
import os

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from newton_vecenv import ACT_DIM, ACTION_SCALE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='models/forest_dog_ppo/final.zip')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--fixed-cam', action='store_true')
    args = p.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    model = PPO.load(args.model, device='cpu')
    leg0 = np.array([0.0, 0.9, -1.8])
    q0 = np.tile(leg0, 4)

    m = mujoco.MjModel.from_xml_path('forest_dog.xml')
    m.opt.timestep = 0.005
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as v:
        for ep in range(args.episodes):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            hd = np.random.uniform(-math.pi, math.pi)
            d.qpos[:3] = [np.random.uniform(0.4, 1.6),
                          np.random.uniform(-0.9, 0.9), 0.44]
            d.qpos[3:7] = [np.cos(hd / 2), 0, 0, np.sin(hd / 2)]
            d.qpos[7:] = q0
            ga = np.random.uniform(-math.pi, math.pi)
            gd = np.random.uniform(10, 18)
            goal = d.qpos[:2] + np.array([np.cos(ga), np.sin(ga)]) * gd
            last_act = np.zeros(ACT_DIM)
            prev_pos, prev_yaw = d.qpos[:3].copy(), hd
            print(f'--- 回合 {ep + 1}: 目标 ({goal[0]:.1f}, {goal[1]:.1f}) ---')

            for step in range(500):
                pos, quat = d.qpos[:3], d.qpos[3:7]
                w, x, y, z = quat
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                vel = (pos - prev_pos) / 0.02
                cy, sy = math.cos(yaw), math.sin(yaw)
                v_body = np.array([cy * vel[0] + sy * vel[1],
                                   -sy * vel[0] + cy * vel[1],
                                   vel[2]])
                # 重力投影（用 MuJoCo xmat: 躯干 id=1）
                Rx = d.xmat[1].reshape(3, 3)
                grav_body = Rx.T @ np.array([0, 0, -1.0])
                yaw_rate = ((yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi) / 0.02
                gv = goal - pos[:2]
                gdn = np.linalg.norm(gv)
                gun = gv / max(gdn, 1e-6)
                goal_body = np.array([cy * gun[0] + sy * gun[1],
                                      -sy * gun[0] + cy * gun[1]])
                laser = np.full(3, 6.0)
                for i in range(m.ngeom):
                    if m.geom(i).type != mujoco.mjtGeom.mjGEOM_CYLINDER:
                        continue
                    gp = d.geom_xpos[i]
                    rx, ry = gp[0] - pos[0], gp[1] - pos[1]
                    r_ = m.geom(i).size[0]
                    dist = math.hypot(rx, ry) - r_
                    if dist >= 6.0:
                        continue
                    ang = math.atan2(ry, rx)
                    rel = (ang - yaw + math.pi) % (2 * math.pi) - math.pi
                    if abs(rel) < math.radians(30):
                        laser[0] = min(laser[0], max(0.0, dist))
                    elif math.radians(30) <= rel < math.radians(120):
                        laser[1] = min(laser[1], max(0.0, dist))
                    elif -math.radians(120) < rel <= -math.radians(30):
                        laser[2] = min(laser[2], max(0.0, dist))
                obs = np.concatenate([v_body, grav_body, [yaw_rate], goal_body,
                                      laser, [pos[2]], d.qpos[7:] - q0,
                                      d.qvel[6:], last_act]).astype(np.float32)
                act, _ = model.predict(obs, deterministic=True)
                last_act = act
                d.ctrl[:] = q0 + act * ACTION_SCALE
                for _ in range(4):
                    mujoco.mj_step(m, d)
                prev_pos, prev_yaw = pos.copy(), yaw
                if gdn < 1.0:
                    print(f'  ✓ 到达目标! 步数 {step}')
                    break
                if step == 499:
                    print(f'  ✗ 超时, 剩余距离 {gdn:.1f} m')
                v.sync()
    print('回放结束')


if __name__ == '__main__':
    main()
