# -*- coding: utf-8 -*-
"""程序化地形：解析高度场函数 + MuJoCo hfield PNG 生成
- make_h_fn: 多倍频正弦山地 + 全局倾斜 + 出生区整平
- make_ramp_fn: 纯坡面（陡坡测试用）
- write_hfield_png: 16-bit 灰度 PNG + 元数据
"""
import json
import math
import os

import imageio.v2 as imageio
import numpy as np


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def make_h_fn(seed=7, span=60.0, amp1=1.1, amp2=0.45, amp3=0.14,
              rough_amp=0.03, tilt_deg=6.0):
    """山地高度场 H(x,y)：三层倍频正弦 + 粗糙度 + 向 +x 爬升 + 出生区整平到 0"""
    rng = np.random.default_rng(seed)
    octs = []
    for wl, a in ((span * 0.92, amp1), (span * 0.36, amp2),
                  (span * 0.15, amp3), (1.7, rough_amp)):
        k = 2 * np.pi / wl
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        octs.append((k * d[0], k * d[1], rng.uniform(0, 2 * np.pi), a))
    tilt = np.tan(np.radians(tilt_deg))

    def h(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = np.zeros_like(x)
        for kx, ky, ph, a in octs:
            out = out + a * np.sin(kx * x + ky * y + ph)
        out = out + tilt * x
        d = np.maximum(np.abs(x - 1.0) / 3.2, np.abs(y) / 2.6)
        return out * _smoothstep(d)          # 出生区平滑归零

    h.params = dict(kind='hills', seed=seed, span=span, amp1=amp1, amp2=amp2,
                    amp3=amp3, rough_amp=rough_amp, tilt_deg=tilt_deg)
    return h


def make_rubble_fn(seed=11, span=44.0, n_mounds=16, mound_h=(0.35, 1.15),
                   mound_r=(1.4, 3.6), rough_amp=0.02):
    """建筑废墟高度场: 平地基础 + 高斯瓦砾堆 + 细糙度, 出生区整平"""
    rng = np.random.default_rng(seed)
    mounds = []
    while len(mounds) < n_mounds:
        x = rng.uniform(2.0, span * 0.55)
        y = rng.uniform(-span * 0.3, span * 0.3)
        if math.hypot(x - 1.0, y) < 3.2:      # 出生区留空
            continue
        mounds.append((x, y, rng.uniform(*mound_r), rng.uniform(*mound_h)))

    def h(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = 0.02 * np.sin(1.9 * x + 0.7) * np.cos(1.6 * y - 0.4)
        for mx, my, mr, mh in mounds:
            d2 = (x - mx) ** 2 + (y - my) ** 2
            out = out + mh * np.exp(-d2 / (2 * (mr / 2) ** 2))
        d = np.maximum(np.abs(x - 1.0) / 3.2, np.abs(y) / 2.6)
        return out * _smoothstep(d)

    h.params = dict(kind='rubble', seed=seed, span=span, n_mounds=n_mounds,
                    mound_h=list(mound_h), mound_r=list(mound_r),
                    rough_amp=rough_amp)
    return h


def make_ramp_fn(slope_deg=24.2, span=24.0):
    """纯坡面：沿 +x 匀速爬升（24.2° = 45% 坡度），出生区整平"""
    k = np.tan(np.radians(slope_deg))

    def h(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = k * x
        d = np.maximum(np.abs(x - 1.0) / 3.2, np.abs(y) / 2.6)
        return out * _smoothstep(d)

    h.params = dict(kind='ramp', slope_deg=slope_deg, span=span)
    return h


def build_from_params(p):
    """从元数据重建仿真用高度函数（含平移量, 与 MuJoCo 表面严格一致）"""
    if p['kind'] == 'ramp':
        base = make_ramp_fn(p['slope_deg'], p.get('span', 24.0))
    elif p['kind'] == 'rubble':
        base = make_rubble_fn(p['seed'], p['span'], p['n_mounds'],
                              tuple(p['mound_h']), tuple(p['mound_r']),
                              p['rough_amp'])
    else:
        base = make_h_fn(p['seed'], p['span'], p['amp1'], p['amp2'], p['amp3'],
                         p['rough_amp'], p['tilt_deg'])
    shift = float(p.get('shift', 0.0))

    def h(x, y):
        return base(x, y) + shift

    h.base = base
    h.shift = shift
    return h


def write_hfield(h_fn, path_bin, path_meta, n=1025):
    """生成 MuJoCo/Newton 双端精确一致的二进制高度场
    格式: int32 nrow,ncol + float32 行主序 (行0 = y 最小; 两引擎二进制行序约定相同)
    地形整体平移至非负区间, 平移量记入元数据。返回 (shift, elevation, z_max)"""
    span = h_fn.params['span']
    ax = np.linspace(-span / 2, span / 2, n)
    gx, gy = np.meshgrid(ax, ax)             # 行=y 升序, 列=x 升序
    v = h_fn(gx, gy)
    shift = float(-v.min())
    hp = v + shift                            # ≥ 0
    z_max = float(hp.max())
    elevation = z_max * 1.001 + 0.1
    # 经验校准: MuJoCo hfield 表面 = 数据×elevation + 0.0049×elevation(均匀抬升)
    grid = (hp / elevation - 0.0049).astype('<f4')    # 行0 = y 最小
    grid = np.clip(grid, 0.0, 1.0)
    with open(path_bin, 'wb') as f:
        f.write(np.array([n, n], dtype='<i4').tobytes())
        f.write(grid.tobytes())
    meta = dict(h_fn.params)
    meta.update(n=n, shift=round(shift, 4), elevation=round(elevation, 4),
                z_max=round(z_max, 4), png=os.path.basename(path_bin))
    with open(path_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    return shift, elevation, z_max
