# -*- coding: utf-8 -*-
"""机械臂设计方案配图（精修版 v2）
风格: 工程制图 + 卡片式信息设计, FG-1 森林绿主题
图 1 总体构型 / 图 2 工作空间包络 / 图 3 双场景作业流程
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

INK = '#1A2E29'
INK_SOFT = '#3D524B'
ORANGE_D = '#8A4B1E'
TEAL = '#2A7A65'
TEAL_D = '#1F6350'
TEAL_L = '#EDF5F2'
ORANGE = '#C9702F'
ORANGE_L = '#FBF1E9'
GREEN = '#7BA05B'
SAND = '#E8DFD3'
GRAY = '#8A9490'
GRID = '#E4ECE8'

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
os.makedirs(OUT, exist_ok=True)
DPI = 200

CHIP = dict(boxstyle='round,pad=0.28', fc='white', ec='#B8C8C0', lw=0.9)
SHADOW = dict(boxstyle='round,pad=0.02', fc='#000000', ec='none', alpha=0.10)


def soft_box(ax, x, y, w, h, fc, ec, lw=1.4, rad=0.03, shadow=True, z=3):
    if shadow:
        ax.add_patch(patches.FancyBboxPatch((x + 0.014, y - 0.016), w, h,
                                            boxstyle=f'round,pad=0.02,rounding_size={rad}',
                                            fc='#101815', ec='none', alpha=0.10, zorder=z - 1))
    b = patches.FancyBboxPatch((x, y), w, h,
                               boxstyle=f'round,pad=0.02,rounding_size={rad}',
                               fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(b)
    return b


def chip_label(ax, text, xy, xytext, color=INK, fs=9, ha='left', arrow=True, z=7):
    if arrow:
        ax.annotate('', xy=xy, xytext=xytext,
                    arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.9,
                                    shrinkA=2, shrinkB=4), zorder=z - 1)
    ax.plot(*xy, marker='o', ms=3.2, color=TEAL_D, zorder=z)
    ax.text(xytext[0], xytext[1], text, fontsize=fs, color=color, ha=ha,
            va='center', bbox=CHIP, zorder=z,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


# ═══════════════════════════ 图 1 总体构型 ═══════════════════════════
def fig1_configuration():
    fig, ax = plt.subplots(figsize=(10.4, 6.6), dpi=DPI)
    ax.set_xlim(-0.85, 1.68)
    ax.set_ylim(-0.34, 1.44)
    ax.axis('off')
    ax.set_aspect('equal')

    # 底纹网格（工程图纸感）
    for gx in np.arange(-0.8, 1.7, 0.25):
        ax.plot([gx, gx], [-0.34, 1.44], color=GRID, lw=0.5, zorder=0)
    for gy in np.arange(-0.25, 1.45, 0.25):
        ax.plot([-0.85, 1.68], [gy, gy], color=GRID, lw=0.5, zorder=0)

    GROUND = -0.12
    ax.plot([-0.8, 1.65], [GROUND, GROUND], color=INK_SOFT, lw=1.6, zorder=2)
    for x in np.arange(-0.78, 1.65, 0.13):
        ax.plot([x, x - 0.055], [GROUND, GROUND - 0.05], color='#AEB8B2', lw=0.8, zorder=2)

    # ── 四足平台（侧视, 朝右）──
    # 远侧腿（浅色）
    for hx, kx, fx in ((0.235, 0.265, 0.215), (-0.185, -0.135, -0.225)):
        ax.plot([hx, kx], [0.03, -0.155], color='#93A39B', lw=6.2,
                solid_capstyle='round', zorder=2)
        ax.plot([kx, fx], [-0.155, -0.295], color='#93A39B', lw=4.2,
                solid_capstyle='round', zorder=2)
        ax.add_patch(patches.FancyBboxPatch((fx - 0.035, -0.315), 0.07, 0.026,
                                            boxstyle='round,pad=0.005,rounding_size=0.012',
                                            fc='#7C8C84', ec='none', zorder=2))
    # 躯干 + 甲板
    soft_box(ax, -0.38, 0.0, 0.68, 0.185, SAND, INK, lw=1.7, rad=0.045, z=4)
    soft_box(ax, -0.30, 0.185, 0.54, 0.042, INK, INK, lw=1.0, rad=0.02, shadow=False, z=5)
    # 头部
    ax.add_patch(patches.Circle((0.345, 0.115), 0.056, fc=SAND, ec=INK, lw=1.5, zorder=4))
    ax.add_patch(patches.Circle((0.345, 0.163), 0.021, fc=TEAL, ec=INK, lw=0.9, zorder=5))
    ax.plot([0.39, 0.425], [0.10, 0.085], color=INK, lw=3.2,
            solid_capstyle='round', zorder=4)
    # 近侧腿
    for hx, kx, fx in ((0.16, 0.19, 0.14), (-0.265, -0.315, -0.245)):
        ax.plot([hx, kx], [0.03, -0.155], color=INK, lw=6.8,
                solid_capstyle='round', zorder=3)
        ax.plot([kx, fx], [-0.155, -0.295], color=INK, lw=4.4,
                solid_capstyle='round', zorder=3)
        ax.add_patch(patches.FancyBboxPatch((fx - 0.036, -0.318), 0.072, 0.028,
                                            boxstyle='round,pad=0.005,rounding_size=0.012',
                                            fc=INK_SOFT, ec='none', zorder=3))
    # 前顶激光雷达 + 尾部 RTK
    ax.add_patch(patches.Rectangle((0.088, 0.227), 0.024, 0.085, fc='#5A6B64',
                                   ec=INK, lw=0.9, zorder=5))
    soft_box(ax, 0.062, 0.312, 0.076, 0.042, INK, INK, lw=0.8, rad=0.014, shadow=False, z=5)
    ax.add_patch(patches.Circle((0.1, 0.333), 0.012, fc='#3DDBB5', ec='none', zorder=6))
    ax.plot([-0.36, -0.38], [0.227, 0.30], color=INK, lw=1.4, zorder=4)
    ax.plot([-0.33, -0.345], [0.227, 0.275], color=INK, lw=1.4, zorder=4)

    chip_label(ax, '四足移动平台 · 站高 ≈ 0.5 m', (-0.25, 0.05),
               (-0.82, -0.26), color=INK_SOFT, fs=9)

    # ── 机械臂（5 自由度）──
    base_x = 0.02
    soft_box(ax, base_x - 0.055, 0.227, 0.11, 0.052, '#5A6B64', INK, lw=1.1, rad=0.012, z=5)
    j1 = np.array([base_x, 0.302])
    sh = np.array([base_x, 0.362])
    a2 = np.deg2rad(58)
    el = sh + 0.35 * np.array([np.cos(a2), np.sin(a2)])
    a3 = a2 - np.deg2rad(98)
    wr = el + 0.40 * np.array([np.cos(a3), np.sin(a3)])
    td = a3 + np.deg2rad(88)
    tool = wr + 0.12 * np.array([np.cos(td), np.sin(td)])

    def link(p, q, color, lw):
        ax.plot([p[0], q[0]], [p[1] + 0.012, q[1] + 0.012], color='#101815',
                lw=lw + 2, alpha=0.10, solid_capstyle='round', zorder=5)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw,
                solid_capstyle='round', zorder=6)
        ax.plot([p[0], q[0]], [p[1] + lw * 0.0035, q[1] + lw * 0.0035],
                color='white', lw=lw * 0.28, alpha=0.28,
                solid_capstyle='round', zorder=6)

    link(j1, sh, INK, 6.5)
    link(sh, el, TEAL, 10)
    link(el, wr, INK, 7.5)
    link(wr, tool, GRAY, 5.5)

    def joint(p, r, ring):
        ax.add_patch(patches.Circle(p + (0.01, -0.012), r, fc='#101815', ec='none',
                                    alpha=0.12, zorder=6))
        ax.add_patch(patches.Circle(p, r, fc='white', ec=ring, lw=2.0, zorder=7))
        ax.add_patch(patches.Circle(p, r * 0.32, fc=ring, ec='none', zorder=8))

    joint(j1, 0.030, INK)
    ax.plot([j1[0] - 0.017, j1[0] + 0.017], [j1[1], j1[1]], color=INK, lw=1.1, zorder=9)
    ax.plot([j1[0], j1[0]], [j1[1] - 0.017, j1[1] + 0.017], color=INK, lw=1.1, zorder=9)
    joint(sh, 0.036, TEAL_D)
    joint(el, 0.032, TEAL_D)
    joint(wr, 0.028, INK)
    ax.text(j1[0], j1[1] - 0.085, 'J1', ha='center', fontsize=8.5, color=INK,
            fontweight='bold', zorder=9)
    ax.text(sh[0] - 0.105, sh[1] - 0.030, 'J2', ha='center', fontsize=8.5, color=INK,
            fontweight='bold', zorder=9)
    ax.text(el[0] + 0.088, el[1] + 0.035, 'J3', ha='center', fontsize=8.5, color=INK,
            fontweight='bold', zorder=9)
    ax.text(wr[0] + 0.045, wr[1] - 0.115, 'J4/J5', ha='center', fontsize=8.5,
            color=INK, fontweight='bold', zorder=9)

    # 末端: 快换盘 + 圆锯(带齿) + 剪砧提示
    ax.plot([tool[0]], [tool[1]], marker='s', ms=9, mfc=ORANGE, mec=INK, mew=1.4, zorder=9)
    ax.text(tool[0] + 0.012, tool[1] + 0.055, 'QC', fontsize=7.5, color=ORANGE_D,
            fontweight='bold', ha='left', zorder=10)
    saw = tool + 0.105 * np.array([np.cos(np.deg2rad(22)), np.sin(np.deg2rad(22))])
    th = np.linspace(0, 2 * np.pi, 27)
    rr = np.where(np.arange(27) % 2 == 0, 0.107, 0.093)
    ring = np.stack([saw[0] + rr * np.cos(th), saw[1] + rr * np.sin(th)], axis=1)
    ax.add_patch(patches.Polygon(ring, closed=True, fc='#F4F7F5', ec=ORANGE,
                                 lw=2.0, zorder=8, joinstyle='round'))
    ax.add_patch(patches.Circle(saw, 0.106, fc='none', ec='#D8E0DC', lw=0.7, zorder=8))
    ax.add_patch(patches.Circle(saw, 0.03, fc=ORANGE, ec=INK, lw=1.0, zorder=9))
    shear = saw + 0.135 * np.array([np.cos(np.deg2rad(-35)), np.sin(np.deg2rad(-35))])
    ax.plot([saw[0], shear[0]], [saw[1], shear[1]], color=INK, lw=3.0,
            solid_capstyle='round', zorder=9)

    # 工作半径弧
    arc_th = np.linspace(np.deg2rad(-12), np.deg2rad(116), 80)
    ax.plot(sh[0] + 0.87 * np.cos(arc_th), sh[1] + 0.87 * np.sin(arc_th),
            ls=(0, (6, 4)), color=GRAY, lw=1.3, alpha=0.9, zorder=4)
    chip_label(ax, '最大工作半径 0.87 m',
               (sh[0] + 0.87 * np.cos(np.deg2rad(112)), sh[1] + 0.87 * np.sin(np.deg2rad(112))),
               (sh[0] - 0.02, sh[1] + 0.95), color=INK_SOFT, fs=9, ha='center')

    # 尺寸标注（自动法向偏移 + 白底牌）
    def dim(p, q, text, side=1, dist=0.10):
        m = (p + q) / 2
        d = (q - p) / np.linalg.norm(q - p)
        n = np.array([-d[1], d[0]]) * side
        ax.annotate('', xy=q, xytext=p,
                    arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=1.3,
                                    shrinkA=0, shrinkB=0), zorder=7)
        ax.text(*(m + n * dist), text, fontsize=9, color=ORANGE, ha='center',
                va='center', bbox=CHIP, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    dim(sh, el, '大臂 L1 = 350 mm', side=1)
    dim(el, wr, '小臂 L2 = 400 mm', side=1)
    dim(wr, tool, '腕-法兰 120 mm', side=1, dist=0.105)

    # 关节标注
    chip_label(ax, 'J1 基座回转 ±170°', tuple(j1), (j1[0] - 0.045, j1[1] + 0.135),
               color=INK, fs=9, ha='right')
    chip_label(ax, 'J2 肩关节 −30°~120°', tuple(sh), (sh[0] - 0.42, sh[1] + 0.15),
               color=INK, fs=9)
    chip_label(ax, 'J3 肘关节 ±140°', tuple(el), (el[0] + 0.10, el[1] + 0.30),
               color=INK, fs=9)
    chip_label(ax, 'J4/J5 腕俯仰 ±120° · 腕回转 ±170°', tuple(wr),
               (wr[0] + 0.13, wr[1] - 0.265), color=INK, fs=9)

    # ── 工具卡片 ──
    card_x, card_y, card_w, card_h = 0.90, 0.52, 0.74, 0.90
    soft_box(ax, card_x, card_y, card_w, card_h, TEAL_L, TEAL, lw=1.5, rad=0.045, z=6)
    ax.text(card_x + card_w / 2, card_y + card_h - 0.11, '快换工具头 · QC 盘',
            fontsize=10.5, color=INK, ha='center', fontweight='bold', zorder=7)
    ax.plot([card_x + 0.10, card_x + card_w - 0.10],
            [card_y + card_h - 0.185, card_y + card_h - 0.185], color=TEAL,
            lw=1.1, alpha=0.6, zorder=7)
    tools = [
        ('T1', '修枝锯剪复合模块', '剪枝 · 疏枝 · 疏伐  Ø2~80', ORANGE),
        ('T2', '取样钻模块', '生长锥 · 土钻取样', TEAL_D),
        ('T3', '胸径卡尺模块', 'DBH 双向计量 0.1 mm', INK_SOFT),
        ('T4', '多功能夹爪', '接枝 · 集枝 · 拾取', GREEN),
    ]
    for i, (tag, name, desc, c) in enumerate(tools):
        cy = card_y + card_h - 0.285 - i * 0.165
        ax.add_patch(patches.Circle((card_x + 0.105, cy), 0.043, fc=c, ec='none', zorder=7))
        ax.text(card_x + 0.105, cy, tag, ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold', zorder=8)
        ax.text(card_x + 0.175, cy + 0.024, name, fontsize=9, color=INK,
                fontweight='bold', va='center', zorder=7)
        ax.text(card_x + 0.175, cy - 0.038, desc, fontsize=7.6, color=INK_SOFT,
                va='center', zorder=7)
    ax.annotate('', xy=tool + np.array([0.045, 0.035]), xytext=(card_x, card_y + 0.42),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5,
                                connectionstyle='arc3,rad=-0.2'), zorder=7)

    # 比例尺
    ax.plot([1.02, 1.52], [-0.27, -0.27], color=INK_SOFT, lw=2.0, zorder=6)
    for e in (1.02, 1.27, 1.52):
        ax.plot([e, e], [-0.295, -0.245], color=INK_SOFT, lw=1.6, zorder=6)
    ax.text(1.27, -0.315, '0.5 m', fontsize=8.5, color=INK_SOFT, ha='center', zorder=6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_configuration.png'), dpi=DPI,
                facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════ 图 2 工作空间包络 ═══════════════════════════
def fig2_workspace():
    fig, ax = plt.subplots(figsize=(8.8, 6.0), dpi=DPI)
    ax.set_xlim(-0.32, 1.52)
    ax.set_ylim(-0.06, 2.58)

    # 地面
    ax.plot([-0.30, 1.50], [0, 0], color=INK_SOFT, lw=1.6, zorder=3)
    for x in np.arange(-0.28, 1.5, 0.14):
        ax.plot([x, x - 0.05], [0, -0.045], color='#AEB8B2', lw=0.8, zorder=2)

    base = np.array([0.0, 0.65])
    # 可达包络（三层渐变扇环）
    for r_in, r_out, al in ((0.15, 0.87, 0.15), (0.35, 0.87, 0.09), (0.55, 0.87, 0.06)):
        ax.add_patch(patches.Wedge(base, r_out, -58, 181, width=r_out - r_in,
                                   fc=TEAL, alpha=al, ec='none', zorder=1))
    for r_ in (0.15, 0.87):
        t = np.linspace(np.deg2rad(-58), np.deg2rad(180), 90)
        ax.plot(base[0] + r_ * np.cos(t), base[1] + r_ * np.sin(t),
                ls=(0, (5, 4)), color=TEAL_D, lw=1.1, alpha=0.85, zorder=2)
    chip_label(ax, 'R = 0.15 m', (base[0] + 0.15 * np.cos(np.deg2rad(160)),
                                  base[1] + 0.15 * np.sin(np.deg2rad(160))),
               (-0.06, 0.94), color=TEAL_D, fs=8.8, ha='center')
    chip_label(ax, 'R = 0.87 m', (base[0] + 0.87 * np.cos(np.deg2rad(-38)),
                                  base[1] + 0.87 * np.sin(np.deg2rad(-38))),
               (0.78, 0.06), color=TEAL_D, fs=8.8, ha='center')

    # 平台剪影 + 基座
    ax.add_patch(patches.FancyBboxPatch((-0.10, 0.50), 0.20, 0.09,
                                        boxstyle='round,pad=0.01,rounding_size=0.03',
                                        fc=SAND, ec=INK, lw=1.3, zorder=4))
    ax.add_patch(patches.Rectangle((-0.07, 0.59), 0.14, 0.022, fc=INK, ec='none', zorder=4))
    for hx, fx in ((0.06, 0.09), (-0.06, -0.10)):
        ax.plot([hx, fx], [0.50, 0.0], color=INK, lw=4.5, solid_capstyle='round', zorder=3)
    ax.plot([base[0]], [base[1]], marker='s', ms=9, mfc=ORANGE, mec=INK, mew=1.4, zorder=6)
    chip_label(ax, '臂基座 · 离地 0.65 m', tuple(base), (0.14, 0.80),
               color=INK, fs=9.2)

    # 任务高度带（右侧彩色带 + 标签）
    bands = [
        (0.05, 0.30, ORANGE, '贴地锯切带', '伐根 5~10 cm 斜口'),
        (0.40, 1.60, TEAL, '碳汇取样带', '生长锥 · 土钻 · 凋落物'),
        (1.80, 2.40, GREEN, '高位修枝带', '枝 Ø2~8 cm（中下部与取样带重叠）'),
    ]
    BX0, BX1 = 0.30, 1.02
    for lo, hi, c, t1, t2 in bands:
        ax.add_patch(patches.FancyBboxPatch((BX0, lo), BX1 - BX0, hi - lo,
                                            boxstyle='round,pad=0.012,rounding_size=0.02',
                                            fc=c, alpha=0.16, ec=c, lw=1.1, zorder=2))
        ax.add_patch(patches.Rectangle((BX0, lo + (hi - lo) / 2 - 0.0075),
                                       0.055, 0.015, fc=c, ec='none', zorder=3))
        ax.text(1.06, (lo + hi) / 2 + 0.035, t1, fontsize=9.4, color=c,
                fontweight='bold', va='center', zorder=4)
        ax.text(1.06, (lo + hi) / 2 - 0.062, t2, fontsize=8.0, color=INK_SOFT,
                va='center', zorder=4)

    ax.plot([-0.10, 1.02], [1.30, 1.30], ls=(0, (6, 4)), color=INK, lw=1.4, zorder=3)
    ax.add_patch(patches.FancyBboxPatch((1.06, 1.235), 0.44, 0.13,
                                        boxstyle='round,pad=0.02,rounding_size=0.02',
                                        fc=INK, ec=INK, zorder=4))
    ax.text(1.28, 1.30, 'DBH 1.30 m', fontsize=8.6, color='white',
            ha='center', va='center', fontweight='bold', zorder=5)

    ax.set_xlabel('距臂基座水平距离 (m)', fontsize=10.5, color=INK)
    ax.set_ylabel('离地高度 (m)', fontsize=10.5, color=INK)
    ax.grid(ls=':', color=GRID, lw=0.7, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=GRAY, labelsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_workspace.png'), dpi=DPI,
                facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════ 图 3 双场景作业流程 ═══════════════════════════
def fig3_workflow():
    fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=DPI)
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 6.0)
    ax.axis('off')

    A_C, A_FILL, A_D = '#C9702F', '#FDF6EF', '#8A4B1E'
    B_C, B_FILL, B_D = '#2A7A65', '#F0F7F4', '#17493B'

    def lane_header(y, badge, title, timing, color, fill):
        soft_box(ax, 0.15, y, 6.6, 0.52, fill, color, lw=1.5, rad=0.10, z=4)
        ax.add_patch(patches.Circle((0.48, y + 0.26), 0.17, fc=color, ec='none', zorder=5))
        ax.text(0.48, y + 0.26, badge, ha='center', va='center', fontsize=12,
                color='white', fontweight='bold', zorder=6)
        ax.text(0.78, y + 0.26, title, fontsize=11.5, color=INK,
                fontweight='bold', va='center', zorder=5)
        soft_box(ax, 7.05, y + 0.055, 3.78, 0.41, 'white', color, lw=1.1,
                 rad=0.10, shadow=False, z=4)
        ax.text(8.94, y + 0.26, timing, fontsize=9.2, color=color,
                ha='center', va='center', fontweight='bold', zorder=5)

    def step_cards(y, steps, color, fill, dark):
        W, H, GAP, X0 = 1.62, 1.34, 0.185, 0.15
        centers = []
        for i, (t1, t2) in enumerate(steps):
            x = X0 + i * (W + GAP)
            soft_box(ax, x, y, W, H, fill, color, lw=1.3, rad=0.08, z=4)
            ax.add_patch(patches.FancyBboxPatch((x + 0.02, y + H - 0.10), W - 0.04, 0.075,
                                                boxstyle='round,pad=0.008,rounding_size=0.03',
                                                fc=color, ec='none', zorder=5))
            ax.add_patch(patches.Circle((x + 0.155, y + H - 0.115), 0.135,
                                        fc=color, ec='white', lw=1.6, zorder=6))
            ax.text(x + 0.155, y + H - 0.117, str(i + 1), ha='center', va='center',
                    fontsize=9.5, color='white', fontweight='bold', zorder=7)
            ax.text(x + W / 2 + 0.05, y + H - 0.38, t1, fontsize=9.6, color=dark,
                    ha='center', va='center', fontweight='bold', zorder=5)
            ax.text(x + W / 2, y + 0.315, t2, fontsize=8.2, color=INK_SOFT,
                    ha='center', va='center', zorder=5, linespacing=1.5)
            centers.append((x, x + W))
            if i < len(steps) - 1:
                ax.annotate('', xy=(x + W + GAP - 0.015, y + H / 2),
                            xytext=(x + W + 0.015, y + H / 2),
                            arrowprops=dict(arrowstyle='-|>', color=color, lw=1.9,
                                            mutation_scale=13), zorder=6)
        return centers

    # 场景 A
    lane_header(5.18, 'A', '林下抚育作业 · 修枝 / 疏枝 / 疏伐',
                '修枝 ≈ 2.5 min/株 · 疏伐 ≈ 90 s/株', A_C, A_FILL)
    a_steps = [
        ('激光雷达建图', '标记作业木清单'),
        ('导航至株前', '0.6 m · 站姿锁定'),
        ('冠层选枝识别', '枯枝 · 过密枝 · 竞争枝'),
        ('视觉定位节痕', '枝条基部切口界定'),
        ('锯剪复合切割', 'Ø2~80 mm 平切口'),
        ('接枝防砸归堆', '夹爪接枝 · 集枝转位'),
    ]
    ca = step_cards(3.55, a_steps, A_C, A_FILL, A_D)
    # 循环回环（自下而上逐枝 → 下一株）
    ax.annotate('', xy=(4.15, 5.06), xytext=(9.90, 5.06),
                arrowprops=dict(arrowstyle='-|>', color=A_C, lw=1.7, ls=(0, (5, 3)),
                                mutation_scale=13), zorder=5)
    ax.text(7.0, 5.06, '自下而上逐枝 · 完成后转下一株', fontsize=8.6,
            color=A_D, ha='center', va='center', zorder=6,
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec=A_C, lw=0.8))

    ax.plot([0, 10.95], [3.28, 3.28], ls=(0, (4, 4)), color='#C8D4CE', lw=1.1)

    # 场景 B
    lane_header(2.24, 'B', '森林碳汇调查 · 样地计量',
                '≈ 1.5 h / 样地（20 m×20 m · 约 50 株）', B_C, B_FILL)
    b_steps = [
        ('RTK 进入样地', '环视建图 · 样线规划'),
        ('逐木检尺', '卡尺 1.30 m 双向 DBH'),
        ('抽样生长锥取芯', '3~5 株 · 年轮生长量'),
        ('五点土壤取样', '土钻 0~30 cm 有机碳'),
        ('凋落物收集', '夹爪拾取 · 装袋编号'),
        ('碳储量解算上传', '异速方程 · 含碳系数'),
    ]
    step_cards(0.66, b_steps, B_C, B_FILL, B_D)

    # 底部数据链
    soft_box(ax, 0.15, 0.05, 10.68, 0.44, TEAL_L, TEAL, lw=1.1, rad=0.10,
             shadow=False, z=4)
    ax.text(5.49, 0.27, '数据链：RTK 厘米级定位 ＋ 腕部六维力传感 ＋ 切口质量抽检 ＋ 作业日志云端同步',
            fontsize=9.0, color=INK, ha='center', va='center', zorder=5)

    fig.savefig(os.path.join(OUT, 'fig3_workflow.png'), dpi=DPI,
                facecolor='white', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig1_configuration()
    fig2_workspace()
    fig3_workflow()
    for f in sorted(os.listdir(OUT)):
        print('生成:', f, os.path.getsize(os.path.join(OUT, f)) // 1024, 'KB')
    print('DIAGRAMS_OK')
