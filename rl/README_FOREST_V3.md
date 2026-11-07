# 森林机器人第三版实验

本版是可复现仿真实验，不是可部署实机导航栈。首轮正式实验已完成，足式和轮足各262,144步，两者在36个独立回合中均0次成功。这轮结果否定了“修好接口即可得到强泛化”的假设，保留为失败基线，不能用训练回报或模型文件存在来证明性能。

## 环境与协议

`mujoco_forest_env.py` 支持 `embodiment='foot'|'wheel'`，分别12/16维动作；关节状态按名称映射。成功、失败和时间截断分离，返回机械能耗、路径长度与失败类别。机械能耗为积分 `Σ|τ q̇|dt`，不等于电池实际耗电，不含算力、驱动损耗与工具用电。

`forest_v3.py` 提供四帧历史、12方向几何射线、距离有效标志与数据年龄、相对地面测距、模拟里程计/IMU/编码器；随机化0–7.5 kg躯干等效载荷、重心、惯量、摩擦、驱动强度、阻尼与0–60 ms延迟。故障场景包含正常、噪声/掉帧、连续0.4–2 s量测冻结。载荷惯量是对角近似，不是可动机械臂的完整耦合动力学。

策略的速度、重力和目标方向仍由仿真状态合成，再经过故障模型，未实现真实LIO/VIO/GNSS融合。前方射线不是完整深度图，水平射线也可能漏掉低矮树根。通信年龄是合成通道，不能据此声称实现真实无线通信或断网任务恢复。

训练地形种子100–103/200–203，两阶段分别为smooth+nominal和std+degraded。评测种子10000/10001，两个地形剖面×三个故障场景×两种地形种子×三个回合种子，每个模型36回合。基线使用相同物理条件。输出分组Wilson区间；这是固定场景的条件回合区间，不能推断所有森林类型的成功率。还需独立生成规则、更多训练随机种子、实测地形和实机试验。

首轮已完成实验的第二阶段没有显式向新VecEnv传入域随机种子，不能保证逐步重现训练权重；保存模型后的评测是固定种子且可重跑的。后续训练入口已修复为逐阶段显式seed，协议增加domain_seed_by_stage。不要把新入口的可复现性修复追认给旧实验。

## 后续分层实验

`navigation_residual.py` 使用局部目标与避障启发式产生参考速度，再映射到差速轮速或CPG步态；PPO只学习两个修正量。它的零动作已有目标控制能力，因此 `navigation_residual/*/baseline.json` 比较的是“同一控制器且无学习修正”，绝不能与第一轮站立基线混为一谈。

这套底层仍是启发式，不保证足式转向准确或轮足打滑时的跟踪，不是已验证的全身控制器。感知失效超过0.2 s时参考速度置零。真实落地仍须接入经验证的底盘速度接口与独立安全控制器。

## 运行

从项目根目录执行，模型输出路径固定在项目内：

```powershell
rl/.venv/Scripts/python.exe rl/test_forest_contract.py
rl/.venv/Scripts/python.exe rl/test_forest_v3.py
rl/.venv/Scripts/python.exe -m unittest robotics.test_planning robotics.test_samples robotics.test_terrain_planner -v

# 第一轮重跑，会写同名输出；保留历史实验时请指定新的 --output
rl/.venv/Scripts/python.exe rl/run_forest_experiment.py --steps 262144 --worlds 4 --episodes 3 --output rl/models/forest_v3_experiment_new

# 分层控制后续实验
rl/.venv/Scripts/python.exe rl/train_navigation_residual.py --steps 65536 --worlds 4 --episodes 3 --output rl/models/navigation_residual_new
```

`requirements-forest-v3.txt` 是当前成功运行环境的版本记录；不是跨平台CUDA安装指南。不覆盖已有环境或模型，不自动下载新依赖。源目录没有Git元数据，因此本次修改没有提交记录。

## 结果文件

- `models/forest_v3_experiment/status.json`：首轮完成状态与实际步数。
- 每个机型的 `final.zip`、`protocol.json`、`evaluation.json`、`baseline.json`：策略、协议、逐回合评测与基线。
- `models/navigation_residual/`：第二轮分层控制实验结果，与第一轮schema不同。
- `v3_training.log`、`navigation_training.log`：原始训练日志。

禁止将旧版checkpoint传入第三版：观测含义/维数发生变化。第三版直接关节策略与分层策略也不兼容。没有成功评测前不部署，不将模拟性能宣传为实机性能。
