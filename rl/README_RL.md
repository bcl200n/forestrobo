# 林下机器狗强化学习训练环境（NVIDIA Isaac Newton）

在 GPU 批量物理仿真中训练四足机器狗完成林下导航任务：
穿绕树干、跨越岩石与倒木、走向随机目标点。

## 环境构成

| 组件 | 版本 | 说明 |
|---|---|---|
| Python 虚拟环境 | 3.12.10 | `E:\forestrobo\rl\.venv` |
| **NVIDIA Isaac Newton** | 1.7.0 | 物理引擎（`newton` 包，PyPI 正式名） |
| mujoco-warp | 3.15.0 | Newton 的 MuJoCo-Warp GPU 后端（版本必须 ~=3.15） |
| mujoco | 3.15.0 | 模型生态 / CPU 回放 |
| PyTorch | 2.15 (CPU) | 策略网络 |
| Stable-Baselines3 | 2.9.2 | PPO 实现 |
| 硬件 | RTX 5080 16GB | Warp GPU 并行仿真（CUDA 13.x 驱动即可，无需装工具包） |

## 文件说明

| 文件 | 作用 |
|---|---|
| `generate_model.py` | 生成机器狗 MJCF（12 DOF 四足 + 顶部载荷）与林下障碍布局 |
| `forest_dog.xml` | 生成的模型（树干 26 / 岩石 42 / 倒木 8） |
| `newton_vecenv.py` | Newton GPU 批量向量化环境（SB3 VecEnv 接口） |
| `train_forest_dog.py` | PPO 训练入口 |
| `enjoy.py` | MuJoCo GUI 回放训练好的策略 |
| `probe*.py` | API 探测脚本（可删） |

## 快速开始

```bat
cd E:\forestrobo\rl

:: 快速验证（~3 分钟）
.venv\Scripts\python.exe train_forest_dog.py --worlds 64 --steps 150000 --name quick

:: 正式训练（约 1.5 小时, 500 万步）
.venv\Scripts\python.exe train_forest_dog.py --worlds 64 --steps 5000000 --name run1

:: 更大规模（128 世界, 显存够时吞吐更高）
.venv\Scripts\python.exe train_forest_dog.py --worlds 128 --steps 10000000 --name run2

:: GUI 可视化回放
.venv\Scripts\python.exe enjoy.py --model models\run1\final.zip
```

模型保存在 `models\<实验名>\final.zip`。

## 任务与奖励设计

- **观测 (49 维)**：本体系线速度、重力投影、偏航角速率、目标方向（本体系）、
  虚拟激光雷达三扇区最近树距（前/左/右，量程 6 m）、躯干高度、
  12 关节位置偏差、12 关节速度、上一步动作
- **动作 (12 维)**：关节位置目标 = 站立姿态 + 0.25 × 动作（PD 伺服 kp=60）
- **奖励**：沿目标方向前进 +2.5·v、存活 +1.2、姿态平稳 −1.5·倾角、
  能量/平滑惩罚、树近距惩罚（<0.55 m）；到达目标 +10 并结束
- **终止**：倾覆（>53°）、躯干塌陷/弹飞、撞树（净距 <0.12 m）、超时 10 s

## 训练预期

- 150k 步：回报 ~80，成功率 1–5%（已在冒烟测试中验证）
- 1–2M 步：机器狗学会稳定行走并初步绕树
- 5–10M 步：成功率应显著提升（林下穿行、上下岩石地形）
- 吞吐约 1000 steps/s（64 世界；受每步 GPU↔CPU 观测同步限制，增大世界数可摊薄）

## 常见问题

- **`AttributeError: ... 'stat'` / MuJoCo 版本不匹配**：Newton 1.7 要求
  `mujoco~=3.15.0`，若 pip 装了更新版本请执行
  `.venv\Scripts\python.exe -m pip install "mujoco~=3.15.0" "mujoco-warp~=3.15.0"`
- **接触溢出警告**：已在代码里按世界数放大 `nconmax/njmax`；若仍出现可在
  `newton_vecenv.py` 中进一步调大
- **换成自己的地形**：改 `generate_model.py` 的树木/岩石/倒木数量与种子后重新生成
- **改任务**：奖励/终止逻辑集中在 `newton_vecenv.py` 的 `_compute_rew_done`
