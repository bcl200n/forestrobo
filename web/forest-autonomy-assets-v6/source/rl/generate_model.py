# -*- coding: utf-8 -*-
"""生成"林下作业机器狗" MJCF 模型与障碍布局（山地地形版）
- 程序化高度场山地（多倍频正弦 + 全局倾斜 + 出生区整平）
- 障碍物（树干/岩石/倒木）贴合地形高度，陡坡处自动避让
输出: forest_dog.xml + forest_layout.json + terrain_hf.png + terrain_meta.json
"""
import argparse
import json
import math
import os

import numpy as np

import terrain as terr

LEG_DEFAULT = (0.0, 0.9, -1.72)   # (髋侧摆, 髋前摆, 膝) 站立姿态
SPAWN_CLEAR_R = 2.2


def leg_xml(prefix: str, px: float, py: float, thigh_fwd: float, side: int,
            wheel: bool = False) -> str:
    """一条腿: 髋侧摆(x轴) → 髋前摆(y轴) → 膝(y轴); side=+1 左 / -1 右（髋部横杆向外）
    wheel=True: 足端换主动轮 + 髋横杆加长(0.07→0.12, 加轮距) + leg_w 阻尼类"""
    jc = 'leg_w' if wheel else 'leg'
    if wheel:
        y0, y1 = 0.05 * side, 0.12 * side
    else:
        y0, y1 = 0.03 * side, 0.07 * side
    if wheel:
        # 轮足: 小腿壳缩短(止于轮心上方, 不触地), 末端为主动轮
        calf_end = '-0.015 0 -0.20'
        end = f"""<body name="{prefix}_wheelbody" pos="-{abs(thigh_fwd)*0.7:.2f} 0 -0.295">
          <joint name="{prefix}_wheel" type="hinge" axis="0 1 0" damping="0.05" armature="0.02"/>
          <geom name="{prefix}_wheel" type="capsule" fromto="0 -0.032 0 0 0.032 0"
                size="0.075" mass="0.45" friction="1.6 0.02 0.02"
                rgba="0.15 0.15 0.17 1"/>
        </body>"""
    else:
        calf_end = f'-{abs(thigh_fwd)*0.7:.2f} 0 -0.295'
        end = f"""<geom name="{prefix}_foot" type="sphere" pos="-{abs(thigh_fwd)*0.7:.2f} 0 -0.32"
                size="0.026" mass="0.06" friction="1.3 0.5 0.5"/>"""
    return f"""
    <body name="{prefix}_hip" pos="{px} {py} 0">
      <joint name="{prefix}_abd" class="{jc}" axis="1 0 0" range="-0.6 0.6"/>
      <geom type="capsule" fromto="0 {y0} 0 0 {y1} 0" size="0.045" mass="0.7"/>
      <body name="{prefix}_thigh" pos="0 {y1} 0">
        <joint name="{prefix}_hip" class="{jc}" axis="0 1 0" range="-0.8 2.4"/>
        <geom type="capsule" fromto="0 0 0 {thigh_fwd} 0 -0.30" size="0.048" mass="1.3"/>
        <body name="{prefix}_calf" pos="{thigh_fwd} 0 -0.30">
          <joint name="{prefix}_knee" class="{jc}" axis="0 1 0" range="-2.8 -0.4"/>
          <geom type="capsule" fromto="0 0 0 {calf_end}" size="0.036" mass="0.32"/>
          {end}
        </body>
      </body>
    </body>"""


PROFILES = {
    # 课程学习地形剖面: 剖面 → (amp1, amp2, amp3, rough_amp, tilt_deg)
    'std':    (1.10, 0.45, 0.14, 0.030, 6.0),
    'smooth': (0.45, 0.18, 0.05, 0.008, 3.0),
}


def generate(seed=7, n_trees=26, n_rocks=42, n_logs=8, tilt_deg=None,
             h_fn=None, wheel=False, hf_n=513, profile='std', out_dir=None):
    rng = np.random.default_rng(seed)
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    xml_name = 'forest_dog_wheel.xml' if wheel else 'forest_dog.xml'

    # ── 地形高度场（可外部传入, 如坡面测试; 或按课程剖面生成）──
    if h_fn is None:
        if profile in PROFILES:
            a1, a2, a3, ra, td = PROFILES[profile]
            h_fn = terr.make_h_fn(seed=seed, amp1=a1, amp2=a2, amp3=a3,
                                  rough_amp=ra, tilt_deg=td)
        else:
            h_fn = terr.make_h_fn(seed=seed, tilt_deg=tilt_deg or 6.0)
    shift, elevation, z_max = terr.write_hfield(h_fn, os.path.join(out_dir, 'terrain_hf.bin'),
                                                os.path.join(out_dir, 'terrain_meta.json'), n=hf_n)

    def h_sim(x, y):
        return h_fn(x, y) + shift

    def slope_ok(x, y, lim=0.55):
        e = 0.25
        gx = (float(h_fn(x + e, y)) - float(h_fn(x - e, y))) / (2 * e)
        gy = (float(h_fn(x, y + e)) - float(h_fn(x, y - e))) / (2 * e)
        return math.hypot(gx, gy) < lim

    # ── 树干：前方扇形 3~21m ──
    trees = []
    tries = 0
    while len(trees) < n_trees and tries < 4000:
        tries += 1
        ang = rng.uniform(-1.36, 1.36)
        d = rng.uniform(3.0, 21.0)
        x, y = math.cos(ang) * d, math.sin(ang) * d
        if math.hypot(x, y) < SPAWN_CLEAR_R + 1.0 or not slope_ok(x, y):
            continue
        r = rng.uniform(0.09, 0.16)
        if any((t[0] - x) ** 2 + (t[1] - y) ** 2 < 1.5 for t in trees):
            continue
        trees.append((round(x, 3), round(y, 3), round(r, 3),
                      round(float(h_sim(x, y)), 3)))

    # ── 岩石：埋地半球 ──
    rocks = []
    tries = 0
    while len(rocks) < n_rocks and tries < 4000:
        tries += 1
        x, y = rng.uniform(0.5, 20.0), rng.uniform(-8.0, 8.0)
        if math.hypot(x, y) < SPAWN_CLEAR_R or not slope_ok(x, y, 0.8):
            continue
        r = rng.uniform(0.10, 0.26)
        if any((o[0] - x) ** 2 + (o[1] - y) ** 2 < 1.0 for o in rocks):
            continue
        rocks.append((round(x, 3), round(y, 3), round(r, 3),
                      round(float(h_sim(x, y)), 3)))

    # ── 倒木：贴地横卧 ──
    logs = []
    tries = 0
    while len(logs) < n_logs and tries < 3000:
        tries += 1
        x, y = rng.uniform(2.0, 19.0), rng.uniform(-7.5, 7.5)
        if math.hypot(x, y) < SPAWN_CLEAR_R + 0.5 or not slope_ok(x, y):
            continue
        ang = rng.uniform(0, math.pi)
        ln = rng.uniform(1.2, 1.9)
        dx, dy = math.cos(ang) * ln, math.sin(ang) * ln
        logs.append((round(x, 3), round(y, 3), round(dx, 3), round(dy, 3),
                     round(float(h_sim(x + dx / 2, y + dy / 2)), 3)))

    # ── 障碍 XML（z 贴合地形）──
    obs_xml = []
    for i, (x, y, r, z) in enumerate(trees):
        obs_xml.append(
            f'    <geom name="tree_{i:02d}" type="cylinder" pos="{x} {y} {z + 1.45:.3f}" '
            f'size="{r} 1.5" rgba="0.26 0.42 0.20 1" friction="0.9"/>')
    for i, (x, y, r, z) in enumerate(rocks):
        h = rng.uniform(0.02, 0.10)
        obs_xml.append(
            f'    <geom name="rock_{i:02d}" type="sphere" pos="{x} {y} {z + r - h:.3f}" '
            f'size="{r}" rgba="0.55 0.55 0.53 1" friction="1.0"/>')
    for i, (x, y, dx, dy, z) in enumerate(logs):
        obs_xml.append(
            f'    <geom name="log_{i:02d}" type="capsule" '
            f'fromto="{x} {y} {z + 0.10:.3f} {x + dx:.3f} {y + dy:.3f} {z + 0.10:.3f}" '
            f'size="0.10" rgba="0.40 0.28 0.18 1" friction="1.0"/>')

    q_legs = ' '.join(f'{LEG_DEFAULT[0]} {LEG_DEFAULT[1]} {LEG_DEFAULT[2]}' + (' 0' if wheel else '')
                      for _ in range(4))
    q_wheels = ''  # wheel coordinates already interleaved per leg
    wheel_acts = ''
    if wheel:
        wheel_acts = ('\n' + '\n'.join(
            f'    <velocity name="{p}_wheel" joint="{p}_wheel" kv="1.2" '
            f'forcerange="-12 12" ctrllimited="true" ctrlrange="-40 40"/>'
            for p in ('FR', 'FL', 'RR', 'RL')))
    torso_z = 0.47 if wheel else 0.42
    n_grid = hf_n
    radius = h_fn.params['span'] / 2

    xml = f"""<mujoco model="forest_dog">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" iterations="50" solver="Newton" integrator="implicitfast"
          gravity="0 0 -9.81"/>
  <asset>
    <hfield name="terrain" nrow="{n_grid}" ncol="{n_grid}"
            size="{radius:.1f} {radius:.1f} {elevation:.3f} 0.3"
            file="terrain_hf.bin"/>
  </asset>
  <default>
    <geom friction="1.1 0.4 0.4" condim="3" density="800" margin="0.002"/>
    <default class="leg">
      <joint type="hinge" damping="0.5" armature="0.015" frictionloss="0.04"/>
    </default>
    <default class="servo">
      <position kp="300" kv="15" forcerange="-45 45"/>
    </default>
    <default class="servo_w">
      <position kp="40" kv="9" forcerange="-45 45"/>
    </default>
    <default class="leg_w">
      <joint type="hinge" damping="1.2" armature="0.015" frictionloss="0.08"/>
    </default>
  </default>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/>
    <map znear="0.02"/>
    <global offwidth="1600" offheight="900"/>
  </visual>
  <worldbody>
    <light pos="6 -4 8" dir="-0.4 0.3 -1" diffuse="0.9 0.9 0.85"/>
    <geom name="floor" type="hfield" hfield="terrain" friction="1.2 0.5 0.5"/>
    <body name="torso" pos="0 0 0.42">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.39 0.15 0.085" mass="9.0"
            rgba="0.85 0.44 0.18 1"/>
      <geom name="deck_geom" type="box" pos="0 0 0.105" size="0.33 0.13 0.022"
            mass="0.3" rgba="0.20 0.20 0.22 1"/>
      <geom name="battery_geom" type="box" pos="-0.02 0 -0.115" size="0.26 0.11 0.05"
            mass="2.6" rgba="0.35 0.55 0.35 1"/> <!-- 电池下置底盘: 降重心 -->
      <geom name="lidar_geom" type="cylinder" pos="0.28 0 0.19" size="0.05 0.045"
            mass="0.4" rgba="0.13 0.13 0.15 1"/>
      <site name="imu" pos="0 0 0" size="0.01"/>{leg_xml('FR', 0.27, -0.155, 0.03, -1, wheel)}{leg_xml('FL', 0.27, 0.155, 0.03, +1, wheel)}{leg_xml('RR', -0.27, -0.155, -0.03, -1, wheel)}{leg_xml('RL', -0.27, 0.155, -0.03, +1, wheel)}
    </body>
{chr(10).join(obs_xml)}
  </worldbody>
  <actuator>
    <position name="FR_abd" joint="FR_abd" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="FR_hip" joint="FR_hip" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="FR_knee" joint="FR_knee" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="FL_abd" joint="FL_abd" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="FL_hip" joint="FL_hip" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="FL_knee" joint="FL_knee" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="RR_abd" joint="RR_abd" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="RR_hip" joint="RR_hip" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="RR_knee" joint="RR_knee" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="RL_abd" joint="RL_abd" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="RL_hip" joint="RL_hip" class="{'servo_w' if wheel else 'servo'}"/>
    <position name="RL_knee" joint="RL_knee" class="{'servo_w' if wheel else 'servo'}"/>{wheel_acts}
  </actuator>
  <keyframe>
    <key name="stance" qpos="0 0 {torso_z} 1 0 0 0 {q_legs}{q_wheels}"/>
  </keyframe>
</mujoco>
"""
    with open(os.path.join(out_dir, xml_name), 'w', encoding='utf-8') as f:
        f.write(xml)
    with open(os.path.join(out_dir, 'forest_layout.json'), 'w', encoding='utf-8') as f:
        json.dump({'seed': seed, 'tilt_deg': tilt_deg, 'trees': trees,
                   'rocks': rocks, 'logs': logs, 'leg_default': list(LEG_DEFAULT),
                   'z_range': [0.0, round(z_max, 3)]}, f)
    print(f'✅ 模型已生成（{"轮足" if wheel else "足式"}山地版）: {out_dir}\\{xml_name}')
    print(f'   仿真高程 0 ~ {z_max:.2f} m (平移 {shift:.2f}) | 树 {len(trees)} | '
          f'岩 {len(rocks)} | 倒木 {len(logs)} | 全局倾角 {tilt_deg}°')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--trees', type=int, default=26)
    p.add_argument('--rocks', type=int, default=42)
    p.add_argument('--logs', type=int, default=8)
    p.add_argument('--tilt', type=float, default=6.0, help='全局爬升倾角(度)')
    a = p.parse_args()
    generate(a.seed, a.trees, a.rocks, a.logs, a.tilt)

