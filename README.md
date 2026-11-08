# ForestRobo 林下模块化机器人

当前项目是仿真、结构概念与离线展示原型，尚未验证实机森林自主作业能力。

**演示入口：[研发Demo](web/demo.html)**。解压ForestRobo-Demo.zip后也可双击demo.html；无需联网。含七机型爆炸图、真实MuJoCo旧/新基元回放、合成地图A*路径示意和机械臂静态反力校核。回放不使用未达标的学习策略。

- [开发记录与真实训练结果](DEVELOPMENT.md)
- [两轮训练报告](TRAINING_REPORT.md)
- [底层速度控制第一阶段](design/VELOCITY_MILESTONE.md)
- [森林闭环基线与全场景结果](design/FOREST_AUTONOMY.md)
- [地震应急模块化方案](design/EARTHQUAKE_DESIGN.md)
- [城市更新测绘扫描与巡检方案](design/URBAN_RENEWAL_DESIGN.md)
- [与ETH野外运动控制研究的差距](design/LOCOMOTION_GAP.md)
- [首轮代码与硬件评审](REVIEW.md)
- [强化学习第三版使用说明](rl/README_FOREST_V3.md)
- [机器人交互爆炸图](web/index.html)：可双击本地打开；无需联网或安装依赖。
- [独立单文件网页](web/robot-offline.html)：方便转移展示。
- [共享机型与机械臂规格](design/configs/platforms.json)

七种配置覆盖森林巡检/调查/作业、地震侦察/轻型清障、城市测绘/近距巡检。网页包含360个可按机型组合的实体零件，支持零件拾取、名称与规格、工具切换和爆炸分解。新增结构均为设计建议，未选型的品牌参数不作虚构。

## 模型修改与重新导出

```powershell
# 从原场景生成独立的精细模块化模型，不覆盖原始场景
tools/blender-4.5.15-windows-x64/blender.exe --background '林下作业机器狗场景.blend' --python scripts/refine_robot.py

# 依次加入地震、城市更新模块，最终得到完整七机型文件
tools/blender-4.5.15-windows-x64/blender.exe --background design/forest_robot_modular.blend --python scripts/add_rescue_modules.py
tools/blender-4.5.15-windows-x64/blender.exe --background design/forest_robot_rescue.blend --python scripts/add_urban_modules.py

# 从精细模型重新导出网页网格、规格和独立离线页
tools/blender-4.5.15-windows-x64/blender.exe --background design/forest_robot_urban.blend --python scripts/export_robot_web.py

# 输出三种静态预览和标准GLB
tools/blender-4.5.15-windows-x64/blender.exe --background design/forest_robot_urban.blend --python scripts/render_robot_variants.py

# 只更新网页数据包装和离线单文件
rl/.venv/Scripts/python.exe scripts/build_robot_web.py
```

## 校核与回归

```powershell
rl/.venv/Scripts/python.exe rl/test_forest_contract.py
rl/.venv/Scripts/python.exe rl/test_forest_v3.py
rl/.venv/Scripts/python.exe -m unittest robotics.test_planning robotics.test_samples robotics.test_terrain_planner -v
tools/node-v22.23.4-win-x64/node.exe scripts/test_web_viewer.cjs
rl/.venv/Scripts/python.exe scripts/planning_examples.py
```

机器人控制与样品日志在 `robotics/`，不调用真实硬件。旧训练/回放脚本保留用于追溯；新实验请使用第三版入口，旧checkpoint的观测定义不兼容。

## 当前状态（2026-11-08）

### 已验证组件

| 模块 | 状态 | 说明 |
|---|---|---|
| Cartesian 步态 + 解析 IK | ✅ 可用 | 全向速度指令 → 关节伺服，无瞬移 |
| 速度跟踪残差 (PPO) | 仿真候选 | 第一阶段 PPO 在60个留出工况降低跟踪误差，低摩擦和加重组合下优势未保持 |
| 合成激光雷达建图 | ✅ 可用 | 3 高度×120 线, LocalGrid 0.2m 格 |
| 局部路径规划 | ✅ 可用 | A* + 膨胀障碍 + 前沿引导 |
| 任务级搜救搜索 | ✅ 100% | coverage 控制器 48/48 (detour_holdout) |
| 传感器故障处理 | ✅ 可用 | dropout 场景下安全停车 |

### 11 月 8 日配对仿真

| 工况 | 规则基线 | 规则加第一阶段 PPO | 规则加最终阶段 PPO |
|---|---:|---:|---:|
| 坡面速度跟踪，60 回合平均误差评分 | 0.227 | 0.209 | 0.220 |
| 坡面低摩擦和 2 kg 等效载荷，60 回合平均误差评分 | 0.254 | 0.257 | 0.251 |
| 加权路线规划，10 回合到达数 | 10/10 | 9/10 | 未参评 |

坡面为 ±5°、±10° 平面，包含五种固定速度指令和三个评测种子。误差评分为回合内纵向、横向、转向 RMSE 按 `1, 1, 0.4` 加权后的范数，跌倒另加 1。各组无跌倒。第一阶段策略在留出集主要改善转向，零指令下纵向 RMSE 从约 0.017 增至 0.093 m/s。最终阶段多训练了约 300 万步，常规坡面跟踪成绩弱于第一阶段。

路线基准使用新种子 `10240/10241`，覆盖通道、静态绕障、3° 坡面、短时激光中断和途中新增障碍。两组均无碰撞、无跌倒。PPO 组在一个静态绕障回合受阻。旧版不加权路线的受阻情况见 [规划诊断](design/research/siqi-shen/README.md)。仿真中的缩小障碍膨胀半径和后退回退仍需按实测机体净空独立验收。

[跟踪评测](rl/models/sim2real_review_20261108/tracking/summary.json) · [规则路线](rl/models/sim2real_review_20261108/route-baseline/summary.json) · [PPO 路线](rl/models/sim2real_review_20261108/route-phase1/summary.json)

### 真机迁移进度

第一阶段 PPO 已导出 NumPy 推理权重。`robotics/tracking_shadow.py` 以 50 Hz 机身速度、单位重力向量、步态相位和参考指令回放，输出供比对的速度残差。与原策略比对 4,346 个观测，归一化动作最大绝对差约 `4.8e-7`。异常时间戳、超时、零速度指令及外部禁用都会抑制残差。此入口仅写日志，当前没有真机运行记录。

```powershell
$trial = 'rl/models/sim2real_review_20261108/tracking'
python robotics/tracking_shadow.py "$trial/tracking_actor.npz" "$trial/shadow_frames.jsonl"
rl/.venv/Scripts/python.exe rl/evaluate_tracking_transfer.py --output rl/models/tracking_transfer_repeat
rl/.venv/Scripts/python.exe rl/render_tracking_review.py --output rl/models/tracking_video_repeat
```

实机适配依赖底盘型号、可读步态相位、时钟同步、机身速度估计和独立急停。若底盘仅开放高层速度接口，需用实测回放验证外环残差，现有相位相关权重仅用于影子比对。坡面滑移、载荷变化、静止保持和失联停车须在候选底盘上分开验收。

### 机械改进（v11-v13, 2026-11-05）

三处机械改动（电池下置/加轮距/悬架软化）使全速 1.36 m/s 直线稳定不摔（旧平台必摔）。轮足森林导航最终成绩：山地 65% / 平滑 80% 存活率。
