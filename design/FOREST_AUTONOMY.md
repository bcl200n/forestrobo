# 森林闭环自主行走基线

日期：2026-11-06。当前为通用四足、模拟激光和理想里程计下的物理闭环，不是已部署森林机器人，也不使用此前未达标的PPO策略。

|场景|到达目标|碰撞|跌倒|受阻保护|
|---|---|---|---|---|
|林间通道|2/2|0|0|0|
|中央障碍绕行|0/2|0|0|2|
|3°坡面|2/2|0|0|0|
|两秒感知中断|2/2|0|0|0|
|途中新增障碍|0/2|0|0|2|

合计 6/10 到达0.35 m终点范围。只覆盖两种布局种子和五种简化场景，不能作为实际森林总体成功率。

## 实现
1. cartesian_gait.py：足端二维轨迹和实际模型几何逆解，关节位置伺服执行；没有改写机身位姿。
2. local_navigation.py：按模拟多高度激光逐步建立占据图；全局未知区域只用于乐观方向引导，实际局部路径严格禁止未知单元。
3. forest_autonomy.py：局部重规划、跟踪、滤波速度反馈和模型专用前向偏差补偿；扫描超时0.2 s撤销运动指令，恢复后重规划。
4. 连续10 s未取得0.1 m目标进展或8 s运动范围不足0.25 m时触发受阻闭锁，撤销运动并用0.5 s稳定动作结束案例。保守门槛为项目建议值，可能拒绝需要长时间后退的路径，不代表正确完成自主恢复。
5. 保存全场景结果、控制/实际轨迹、模型XML/地形资产、源文件快照与SHA256。此前v1-v5失败和中断记录保留，不覆盖为成功。

## 尚未解决
中央障碍及途中新增障碍仍可能进入局部规划/跟踪停滞，受阻停车不计成功。必须继续完善前沿探索、局部子目标保持与恢复策略。
当前激光来自MuJoCo射线，位姿直接来自仿真，没有真实LIO/VIO/SLAM；局部高度为平面先验。未覆盖松软地面、植被误读、沟坎、倒木跨越或动态机械臂。占据图保守累积，不能正确清除已移除的障碍。
机械能耗为绝对机械功积分，不等同电池能耗。PID/步态参数仍依赖通用模型，不能直接发送到真实机器人。

## 复查
```powershell
rl/.venv/Scripts/python.exe rl/run_autonomy_benchmark.py --output rl/models/autonomy_recheck
rl/.venv/Scripts/python.exe rl/run_autonomy_benchmark.py --scenarios corridor --seeds 100 --render --output rl/models/autonomy_replay
rl/.venv/Scripts/python.exe -m unittest robotics.test_local_navigation robotics.test_terrain_planner
rl/.venv/Scripts/python.exe -m unittest discover -s rl -p test_cartesian_gait.py
```

演示页第05项加入真实物理回放。回放与评测均对应v6通道基线。
