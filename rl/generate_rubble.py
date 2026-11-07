# -*- coding: utf-8 -*-
"""生成建筑废墟搜救场景（足式机器狗 + 找幸存者）
- 高度场: 瓦砾堆地形 (terrain_rubble.bin, 独立文件防并行冲突)
- 障碍: 混凝土块 / 倾斜楼板 / 露筋柱
- 幸存者: 4 个绿色人形胶囊 (静态, 训练中解析式接近判定)
输出: rubble_dog.xml + rubble_layout.json
"""
import argparse
import json
import math
import os

import numpy as np

import terrain as terr

LEG_DEFAULT = (0.0, 0.9, -1.72)
WHEEL_LEG_DEFAULT = (0.0, 0.55, -1.15)
SPAWN_CLEAR = 2.4


def leg_xml(prefix, px, py, fw, side, wheel=False):
    y0, y1 = 0.03 * side, 0.07 * side
    if wheel:
        end = f"""<body name="{prefix}_wheelbody" pos="-{abs(fw)*0.7:.2f} 0 -0.295">
          <joint name="{prefix}_wheel" type="hinge" axis="0 1 0" damping="0.05" armature="0.02"/>
          <geom name="{prefix}_wheel" type="capsule" fromto="0 -0.032 0 0 0.032 0"
                size="0.075" mass="0.45" friction="1.6 0.02 0.02" rgba="0.15 0.15 0.17 1"/>
        </body>"""
        calf = '-0.015 0 -0.20'
    else:
        end = f"""<geom name="{prefix}_foot" type="sphere" pos="-{abs(fw)*0.7:.2f} 0 -0.32"
                size="0.026" mass="0.06" friction="1.3 0.5 0.5"/>"""
        calf = f'-{abs(fw)*0.7:.2f} 0 -0.32'
    return f"""
    <body name="{prefix}_hip" pos="{px} {py} 0">
      <joint name="{prefix}_abd" class="leg" axis="1 0 0" range="-0.6 0.6"/>
      <geom type="capsule" fromto="0 {y0} 0 0 {y1} 0" size="0.045" mass="0.7"/>
      <body name="{prefix}_thigh" pos="0 {y1} 0">
        <joint name="{prefix}_hip" class="leg" axis="0 1 0" range="-0.8 2.4"/>
        <geom type="capsule" fromto="0 0 0 {fw} 0 -0.30" size="0.048" mass="1.3"/>
        <body name="{prefix}_calf" pos="{fw} 0 -0.30">
          <joint name="{prefix}_knee" class="leg" axis="0 1 0" range="-2.8 -0.4"/>
          <geom type="capsule" fromto="0 0 0 {calf}" size="0.036" mass="0.32"/>
          {end}
        </body>
      </body>
    </body>"""


def generate(seed=11, n_blocks=None, n_slabs=None, n_rebar=None, n_surv=4,
             profile='full', wheel=False, out_dir=None):
    WHEEL = wheel
    rng = np.random.default_rng(seed)
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(out_dir)
    if profile == 'flat':                     # 课程阶段1: 平地少障碍 + 超近幸存者
        n_mounds, n_blocks, n_slabs, n_rebar = 0, 6, 0, 4
        surv_range = (1.5, 3.0)
    else:                                     # 完整废墟 + 中近距幸存者
        n_mounds = 16
        n_blocks = 16 if n_blocks is None else n_blocks
        n_slabs = 8 if n_slabs is None else n_slabs
        n_rebar = 8 if n_rebar is None else n_rebar
        surv_range = (3.0, 6.0)

    h_fn = terr.make_rubble_fn(seed=seed, n_mounds=n_mounds)
    shift, elevation, z_max = terr.write_hfield(
        h_fn, 'terrain_rubble.bin', 'terrain_rubble_meta.json', n=513)

    def hs(x, y):                                   # 仿真高程(含平移)
        return h_fn(x, y) + shift

    def place(min_r, lo, hi, keep, min_gap):
        """在 x∈[lo,hi], y∈[-9,9] 放一个点, 与 keep 内点距 > min_gap"""
        for _ in range(300):
            x = rng.uniform(lo, hi)
            y = rng.uniform(-9.0, 9.0)
            if math.hypot(x - 1.0, y) < SPAWN_CLEAR + min_r:
                continue
            if all(math.hypot(x - k[0], y - k[1]) > min_gap for k in keep):
                return float(x), float(y)
        return None

    # ── 幸存者: 极坐标径向放置(距离=课程距离), 彼此 ≥2.2m ──
    survivors, keep = [], []
    for _ in range(n_surv):
        for _try in range(600):
            ang = rng.uniform(-math.pi, math.pi)
            rr_ = rng.uniform(surv_range[0], surv_range[1])
            x = 1.0 + math.cos(ang) * rr_          # 出生点约 (1,0)
            y = math.sin(ang) * rr_
            if abs(y) > 9.0 or x < 0.5 or x > 19.0:
                continue
            if all(math.hypot(x - k[0], y - k[1]) > 2.2 for k in keep):
                survivors.append((round(x, 3), round(y, 3),
                                  round(float(hs(x, y)), 3)))
                keep.append((x, y))
                break

    # ── 混凝土块 ──
    blocks = []
    for _ in range(n_blocks):
        p = place(0.8, 2.5, 18.0, keep + [b[:2] for b in blocks], 2.0)
        if p:
            sx, sy, sz = (rng.uniform(0.35, 0.8), rng.uniform(0.35, 0.8),
                          rng.uniform(0.25, 0.6))
            blocks.append((p[0], p[1], round(sx, 3), round(sy, 3),
                           round(sz, 3), round(float(rng.uniform(0, math.pi)), 3),
                           round(float(hs(*p)), 3)))

    # ── 倾斜楼板 ──
    slabs = []
    for _ in range(n_slabs):
        p = place(1.0, 3.0, 18.0, keep + [b[:2] for b in blocks]
                  + [s[:2] for s in slabs], 2.2)
        if p:
            slabs.append((p[0], p[1], round(rng.uniform(0.9, 1.4), 3),
                          round(rng.uniform(0.6, 0.9), 3),
                          round(rng.uniform(0.18, 0.5), 3),          # 倾角
                          round(rng.uniform(0, math.pi), 3),
                          round(float(hs(*p)), 3)))

    # ── 露筋柱 ──
    rebars = []
    for _ in range(n_rebar):
        p = place(0.3, 2.5, 18.0, keep, 1.2)
        if p:
            rebars.append((p[0], p[1], round(float(hs(*p)), 3)))

    # ── XML 组装 ──
    obs = []
    for i, (x, y, sx, sy, sz, yaw, z) in enumerate(blocks):
        obs.append(
            f'    <geom name="block_{i:02d}" type="box" pos="{x} {y} {z + sz * 0.35:.3f}" '
            f'size="{sx} {sy} {sz}" euler="0 0 {yaw:.3f}" '
            f'rgba="0.62 0.60 0.56 1" friction="1.0"/>')
    for i, (x, y, sx, sy, tilt, yaw, z) in enumerate(slabs):
        obs.append(
            f'    <geom name="slab_{i:02d}" type="box" pos="{x} {y} {z + 0.12:.3f}" '
            f'size="{sx} {sy} 0.05" euler="{tilt:.3f} 0 {yaw:.3f}" '
            f'rgba="0.55 0.53 0.50 1" friction="0.9"/>')
    for i, (x, y, z) in enumerate(rebars):
        obs.append(
            f'    <geom name="rebar_{i:02d}" type="cylinder" pos="{x} {y} {z + 0.55:.3f}" '
            f'size="0.025 0.55" rgba="0.45 0.30 0.20 1" friction="0.8"/>')
    for i, (x, y, z) in enumerate(survivors):
        obs.append(
            f'    <geom name="survivor_{i}" type="capsule" '
            f'fromto="{x} {y} {z + 0.08:.3f} {x} {y} {z + 0.46:.3f}" '
            f'size="0.09" rgba="0.10 0.80 0.30 1"/>')

    q_legs = ' '.join(f'{LEG_DEFAULT[0]} {LEG_DEFAULT[1]} {LEG_DEFAULT[2]}'
                      for _ in range(4))
    q_wlegs = ' '.join(f'{WHEEL_LEG_DEFAULT[0]} {WHEEL_LEG_DEFAULT[1]} '
                       f'{WHEEL_LEG_DEFAULT[2]}' for _ in range(4))
    wheel_acts = ('\n' + '\n'.join(
        f'    <velocity name="{p}_wheel" joint="{p}_wheel" kv="1.2" '
        f'forcerange="-15 15" ctrllimited="true" ctrlrange="-40 40"/>'
        for p in ('FR', 'FL', 'RR', 'RL'))) if WHEEL else ''
    radius = h_fn.params['span'] / 2
    xml = f"""<mujoco model="rubble_dog">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" iterations="50" solver="Newton" integrator="implicitfast"
          gravity="0 0 -9.81"/>
  <asset>
    <hfield name="terrain" nrow="513" ncol="513"
            size="{radius:.1f} {radius:.1f} {elevation:.3f} 0.3"
            file="terrain_rubble.bin"/>
  </asset>
  <default>
    <geom friction="1.1 0.4 0.4" condim="3" density="800" margin="0.002"/>
    <default class="leg"><joint type="hinge" damping="0.5" armature="0.015"
        frictionloss="0.04"/></default>
    <default class="servo"><position kp="220" kv="12" forcerange="-120 120"/></default>
  </default>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/>
    <map znear="0.02"/>
    <global offwidth="1600" offheight="900"/>
  </visual>
  <worldbody>
    <light pos="6 -4 10" dir="-0.4 0.3 -1" diffuse="0.85 0.85 0.9"/>
    <geom name="floor" type="hfield" hfield="terrain" friction="1.2 0.5 0.5"/>
    <body name="torso" pos="0 0 {0.46 if WHEEL else 0.42}">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.39 0.15 0.085" mass="9.0"
            rgba="0.85 0.44 0.18 1"/>
      <geom name="deck_geom" type="box" pos="0 0 0.105" size="0.33 0.13 0.022"
            mass="2.2" rgba="0.20 0.20 0.22 1"/>
      <geom name="lidar_geom" type="cylinder" pos="0.28 0 0.19" size="0.05 0.045"
            mass="0.4" rgba="0.13 0.13 0.15 1"/>
      <site name="imu" pos="0 0 0" size="0.01"/>{leg_xml('FR', 0.27, -0.155, 0.03, -1, WHEEL)}{leg_xml('FL', 0.27, 0.155, 0.03, +1, WHEEL)}{leg_xml('RR', -0.27, -0.155, -0.03, -1, WHEEL)}{leg_xml('RL', -0.27, 0.155, -0.03, +1, WHEEL)}
    </body>
{chr(10).join(obs)}
  </worldbody>
  <actuator>
    <position name="FR_abd" joint="FR_abd" class="servo"/>
    <position name="FR_hip" joint="FR_hip" class="servo"/>
    <position name="FR_knee" joint="FR_knee" class="servo"/>
    <position name="FL_abd" joint="FL_abd" class="servo"/>
    <position name="FL_hip" joint="FL_hip" class="servo"/>
    <position name="FL_knee" joint="FL_knee" class="servo"/>
    <position name="RR_abd" joint="RR_abd" class="servo"/>
    <position name="RR_hip" joint="RR_hip" class="servo"/>
    <position name="RR_knee" joint="RR_knee" class="servo"/>
    <position name="RL_abd" joint="RL_abd" class="servo"/>
    <position name="RL_hip" joint="RL_hip" class="servo"/>
    <position name="RL_knee" joint="RL_knee" class="servo"/>{wheel_acts}
  </actuator>
  <keyframe>
    <key name="stance" qpos="0 0 {0.46 if WHEEL else 0.42} 1 0 0 0 {q_wlegs if WHEEL else q_legs}{' 0 0 0 0' if WHEEL else ''}"/>
  </keyframe>
</mujoco>
"""
    with open('rubble_dog.xml', 'w', encoding='utf-8') as f:
        f.write(xml)

    # 危险物(激光雷达用): 露筋 + 混凝土块(等效半径) + 楼板中心
    hazards = [[x, y, 0.05] for x, y, _ in rebars]
    hazards += [[b[0], b[1], max(b[2], b[3])] for b in blocks]
    hazards += [[s[0], s[1], 0.6] for s in slabs]
    with open('rubble_layout.json', 'w', encoding='utf-8') as f:
        json.dump({'seed': seed, 'survivors': survivors, 'hazards': hazards,
                   'leg_default': list(LEG_DEFAULT)}, f)
    print(f'✅ 废墟场景已生成: rubble_dog.xml')
    print(f'   幸存者 {len(survivors)} | 混凝土块 {len(blocks)} | 楼板 {len(slabs)} | '
          f'露筋 {len(rebars)} | 高程 0~{z_max:.2f} m')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=11)
    a = p.parse_args()
    generate(a.seed)

