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
