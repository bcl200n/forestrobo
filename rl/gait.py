# -*- coding: utf-8 -*-
"""共享步态模块：站立姿态 + 对角小跑 CPG（摆动相折叠/支撑相伸展）"""
import math

import numpy as np

# 站立关节角 (abd, hip, knee)；knee 预补偿静态下垂
LEG0 = (0.0, 0.90, -1.72)
Q0 = np.tile(np.array(LEG0), 4)          # FR FL RR RL
PHASE = (0.0, math.pi, math.pi, 0.0)     # 对角配对: FR+RL / FL+RR

FREQ = 2.2        # 步频 Hz (与 A_HIP 符号组合经基元扫描验证: 前进+42°/s转向)
A_HIP = -0.26     # 髋摆幅
A_KNEE = 0.44     # 摆动相折叠幅
K_PUSH = 0.08     # 支撑相伸展助力

WHEEL_R = 0.075   # 轮半径 m
# 轮足专用站姿: 直腿(小腿近乎竖直, 让开地面), 躯干高 ≈0.57
WLEG0 = (0.0, 0.55, -1.15)
WQ0 = np.tile(np.array(WLEG0), 4)
WSTAND_Z = 0.575


def wheel_drive(v_ref: float = 1.5, pitch: float = 0.0,
                pitch_target: float = 0.0) -> np.ndarray:
    """轮足行驶控制: 直腿站姿 + 前髋调平(实测符号: 前髋+ → 抬头减小) + 轮毂速度伺服
    动作布局 = [abd,hip,knee]×4 + [FR,FL,RR,RL] 轮速(rad/s); pitch/pitch_target 单位: 弧度"""
    err_deg = math.degrees(pitch_target - pitch)     # >0 表示需要增加俯仰(压低鼻头)
    front_bias = float(np.clip(0.012 * err_deg, -0.30, 0.30))
    ctrl = np.zeros(16)
    for i in range(4):
        s = 1.0 if i < 2 else 0.0                    # 只调前髋
        ctrl[3 * i + 1] = WLEG0[1] + front_bias * s
        ctrl[3 * i + 2] = WLEG0[2]
    ctrl[12:] = v_ref / WHEEL_R
    return ctrl


def cpg(t: float, freq: float = FREQ, ph0: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """对角小跑 + 前庭反射: 上坡(俯仰<0)自动后摆加力蹬地, 下坡前倾限速
    ph0 为整狗相位偏移; pitch 为机体俯仰角(弧度, 鼻上仰为负)"""
    uphill = float(np.clip(-pitch, 0.0, 0.6))       # 上坡助力增益
    downhill = float(np.clip(pitch, 0.0, 0.6))      # 下坡限速增益
    hip_bias = 0.70 * uphill - 0.15 * downhill
    a_hip = A_HIP * (1.0 - 0.45 * uphill)           # 上坡缩短步幅增抓地
    knee_push = K_PUSH + 0.80 * uphill
    ctrl = np.zeros(12)
    w = 2 * math.pi * freq
    for i in range(4):
        th = w * t + PHASE[i] + ph0
        s, c = math.sin(th), math.cos(th)
        swing = max(0.0, -c)              # 腿由后向前摆的半周期
        ctrl[3 * i + 0] = LEG0[0]
        ctrl[3 * i + 1] = LEG0[1] + hip_bias + a_hip * s
        ctrl[3 * i + 2] = LEG0[2] - A_KNEE * swing + knee_push * max(0.0, c)
    return ctrl
